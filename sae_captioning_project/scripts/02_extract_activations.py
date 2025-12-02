#!/usr/bin/env python3
"""
Script 02: Activation Extraction
================================

Extracts intermediate activations from the vision-language model
for both English and Arabic prompts.

Usage:
    python scripts/02_extract_activations.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import gc
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hooks import HookConfig, ActivationCache
from src.data import CrossLingualCaptionDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    """Load the vision-language model and processor."""
    model_name = config['model']['name']
    dtype_str = config['model'].get('dtype', 'float16')
    
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)
    
    logger.info(f"Loading model: {model_name}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    logger.info(f"Model loaded with dtype {dtype}")
    return model, processor


def get_model_hidden_size(model) -> int:
    """Get the hidden size from model config."""
    if hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        return model.config.d_model
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        return model.config.text_config.hidden_size
    else:
        # Try to infer from model architecture
        try:
            # Try different model structures
            if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
                return model.language_model.layers[0].mlp.gate_proj.in_features
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return model.model.layers[0].mlp.gate_proj.in_features
            else:
                raise ValueError("Could not find model layers")
        except Exception as e:
            logger.error(f"Error inferring hidden size: {e}")
            logger.error(f"Model structure: {type(model)}")
            logger.error(f"Model config: {model.config}")
            raise ValueError("Could not determine model hidden size")


def extract_activations_batch(
    model,
    processor,
    images,
    prompts,
    layers: list,
    device: str = "cuda"
) -> dict:
    """Extract activations for a batch of image-prompt pairs."""

    # For Gemma-3, we need to use apply_chat_template
    # Format each prompt as a chat message with image
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        formatted_prompts.append(text)

    # Prepare inputs
    inputs = processor(
        images=images,
        text=formatted_prompts,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Setup hooks
    config = HookConfig(
        layers=layers,
        component="residual",
        detach=True,
        to_cpu=True,
        dtype=torch.float32
    )
    
    activations = {layer: [] for layer in layers}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            activations[layer_idx].append(act.detach().cpu())
        return hook
    
    # Register hooks
    for layer_idx in layers:
        # Access language model layers for Gemma-3
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            layer = model.language_model.layers[layer_idx]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[layer_idx]
        else:
            raise ValueError(f"Could not access layer {layer_idx} from model")
        h = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Stack activations
    result = {
        layer: torch.cat(acts, dim=0) if acts else None
        for layer, acts in activations.items()
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract activations from VLM")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Override layers to extract')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize wandb if enabled
    wandb_run = None
    if config.get('logging', {}).get('use_wandb', False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging']['wandb_entity'],
                name=f"activation-extraction-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    'model': config['model']['name'],
                    'layers': args.layers or config['layers']['extraction'],
                    'num_samples': config['data'].get('num_samples'),
                    'batch_size': args.batch_size or config['data'].get('batch_size', 4),
                },
                tags=['activation-extraction', 'gemma-3']
            )
            logger.info(f"Wandb initialized: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_run = None

    # Setup paths
    processed_dir = Path(config['paths']['processed_data'])
    checkpoints_dir = Path(config['paths']['checkpoints'])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for processed data
    csv_path = processed_dir / 'samples.csv'
    if not csv_path.exists():
        logger.error("No processed data found. Run 01_prepare_data.py first.")
        return 1
    
    # Load model
    model, processor = load_model(config)
    device = config['model'].get('device', 'cuda')
    
    # Get model info
    hidden_size = get_model_hidden_size(model)
    logger.info(f"Model hidden size: {hidden_size}")
    
    # Determine layers
    layers = args.layers or config['layers']['extraction']
    logger.info(f"Extracting layers: {layers}")
    
    # Load dataset
    batch_size = args.batch_size or config['data'].get('batch_size', 4)
    
    dataset = CrossLingualCaptionDataset(
        data_dir=processed_dir,
        csv_path=csv_path,
        max_samples=config['data'].get('num_samples')
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Extract for both languages
    checkpoint_interval = 500  # Save every 500 samples to avoid OOM

    for language in ['english', 'arabic']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Extracting activations for {language.upper()}")
        logger.info(f"{'='*50}")

        # Lists to accumulate chunks
        chunk_activations = {layer: [] for layer in layers}
        chunk_genders = []
        chunk_image_ids = []

        # Final accumulated data (will load from checkpoints at the end)
        all_checkpoint_files = []

        start_time = time.time()
        samples_processed = 0

        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Processing {language}"):
            # Get batch
            batch_indices = range(i, min(i + batch_size, len(dataset)))
            batch_samples = [dataset[j] for j in batch_indices]

            images = [s['image'] for s in batch_samples]
            prompts = [s[f'{language}_prompt'] for s in batch_samples]
            genders = [s['ground_truth_gender'] for s in batch_samples]
            image_ids = [s['image_id'] for s in batch_samples]

            # Extract
            try:
                batch_activations = extract_activations_batch(
                    model, processor, images, prompts, layers, device
                )

                for layer in layers:
                    if batch_activations[layer] is not None:
                        chunk_activations[layer].append(batch_activations[layer])

                chunk_genders.extend(genders)
                chunk_image_ids.extend(image_ids)
                samples_processed += len(batch_samples)

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue

            # Save checkpoint periodically to avoid OOM
            if samples_processed >= checkpoint_interval or i + batch_size >= len(dataset):
                if chunk_activations[layers[0]]:  # Check if we have any data
                    # Stack current chunk
                    stacked_chunk = {}
                    for layer in layers:
                        if chunk_activations[layer]:
                            stacked_chunk[layer] = torch.cat(chunk_activations[layer], dim=0)

                    # Save checkpoint
                    checkpoint_idx = len(all_checkpoint_files)
                    checkpoint_path = checkpoints_dir / f'activations_{language}_chunk_{checkpoint_idx}.pt'
                    torch.save({
                        'activations': stacked_chunk,
                        'genders': chunk_genders,
                        'image_ids': chunk_image_ids,
                    }, checkpoint_path)

                    all_checkpoint_files.append(checkpoint_path)
                    logger.info(f"Saved checkpoint {checkpoint_idx} with {len(chunk_image_ids)} samples")

                    # Clear chunk data
                    chunk_activations = {layer: [] for layer in layers}
                    chunk_genders = []
                    chunk_image_ids = []
                    samples_processed = 0

                    # Clear memory
                    del stacked_chunk
                    gc.collect()
                    torch.cuda.empty_cache()

            # Log to wandb periodically
            if wandb_run and i % (batch_size * 10) == 0 and i > 0:
                elapsed = time.time() - start_time
                total_processed = i + batch_size
                samples_per_sec = total_processed / elapsed if elapsed > 0 else 0

                log_dict = {
                    f'{language}/samples_processed': total_processed,
                    f'{language}/samples_per_sec': samples_per_sec,
                    f'{language}/progress': total_processed / len(dataset) * 100,
                }

                # Log GPU memory if available
                if torch.cuda.is_available():
                    log_dict[f'{language}/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                    log_dict[f'{language}/gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9

                try:
                    import wandb
                    wandb.log(log_dict)
                except:
                    pass

            # Clear GPU memory periodically
            if i % (batch_size * 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()

        logger.info(f"Merging {len(all_checkpoint_files)} checkpoint files...")

        # Load and merge all checkpoints
        all_genders = []
        all_image_ids = []
        stacked_activations = {layer: [] for layer in layers}

        for checkpoint_path in all_checkpoint_files:
            checkpoint_data = torch.load(checkpoint_path)
            for layer in layers:
                if layer in checkpoint_data['activations']:
                    stacked_activations[layer].append(checkpoint_data['activations'][layer])
            all_genders.extend(checkpoint_data['genders'])
            all_image_ids.extend(checkpoint_data['image_ids'])

        # Stack all activations
        final_activations = {}
        for layer in layers:
            if stacked_activations[layer]:
                final_activations[layer] = torch.cat(stacked_activations[layer], dim=0)
                logger.info(f"Layer {layer}: {final_activations[layer].shape}")

        # Clear memory
        del stacked_activations
        gc.collect()

        # Save
        output_path = checkpoints_dir / f'activations_{language}.pt'
        torch.save({
            'activations': stacked_activations,
            'genders': all_genders,
            'image_ids': all_image_ids,
            'layers': layers,
            'hidden_size': hidden_size,
            'timestamp': datetime.now().isoformat(),
            'model': config['model']['name'],
        }, output_path)

        logger.info(f"Saved {language} activations to {output_path}")

        # Log final stats to wandb
        if wandb_run:
            elapsed = time.time() - start_time
            log_dict = {
                f'{language}/total_samples': len(all_image_ids),
                f'{language}/total_time_sec': elapsed,
                f'{language}/avg_samples_per_sec': len(all_image_ids) / elapsed if elapsed > 0 else 0,
            }

            # Log activation shapes
            for layer, act in stacked_activations.items():
                log_dict[f'{language}/layer_{layer}_shape'] = str(act.shape)

            # Log gender distribution
            gender_counts = {}
            for gender in all_genders:
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
            for gender, count in gender_counts.items():
                log_dict[f'{language}/gender_{gender}_count'] = count

            try:
                import wandb
                wandb.log(log_dict)
            except:
                pass

        # Clear memory before next language
        del all_activations
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("\nActivation extraction complete!")

    # Finish wandb run
    if wandb_run:
        try:
            import wandb
            wandb.finish()
        except:
            pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
