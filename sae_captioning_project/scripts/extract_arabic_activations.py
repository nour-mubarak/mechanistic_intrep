#!/usr/bin/env python3
"""
Extract Arabic activations only (English already extracted)
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
    dtype_str = config['model'].get('dtype', 'float32')

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Override to float32 for stability
    if dtype != torch.float32:
        logger.warning(f"Overriding dtype from {dtype} to float32 for numerical stability")
        dtype = torch.float32

    logger.info(f"Loading model: {model_name}")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load model in float32
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
            if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
                return model.language_model.layers[0].mlp.gate_proj.in_features
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return model.model.layers[0].mlp.gate_proj.in_features
            else:
                raise ValueError("Could not find model layers")
        except Exception as e:
            logger.error(f"Error inferring hidden size: {e}")
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

    # Format prompts as chat messages
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
    activations = {layer: [] for layer in layers}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            # Detach, move to CPU, and convert to float32
            act = act.detach().cpu().float()

            # Check for NaN values
            if torch.isnan(act).any():
                logger.error(f"NaN detected in layer {layer_idx} activations!")
                logger.error(f"NaN count: {torch.isnan(act).sum().item()} / {act.numel()}")
                # Replace NaN with zeros to prevent corruption
                act = torch.nan_to_num(act, nan=0.0)
                logger.warning("Replaced NaN values with zeros")

            activations[layer_idx].append(act)
        return hook

    # Register hooks
    for layer_idx in layers:
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
    parser = argparse.ArgumentParser(description="Extract Arabic activations from VLM")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

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
    layers = config['layers']['extraction']
    logger.info(f"Extracting layers: {layers}")

    # Load dataset
    batch_size = args.batch_size or config['data'].get('batch_size', 4)

    dataset = CrossLingualCaptionDataset(
        data_dir=processed_dir,
        csv_path=csv_path,
        max_samples=config['data'].get('num_samples')
    )

    logger.info(f"Dataset size: {len(dataset)}")

    # Extract Arabic only
    checkpoint_interval = 50

    logger.info(f"\n{'='*50}")
    logger.info(f"Extracting activations for ARABIC")
    logger.info(f"{'='*50}")

    # Lists to accumulate chunks
    chunk_activations = {layer: [] for layer in layers}
    chunk_genders = []
    chunk_image_ids = []

    # Final accumulated data
    all_checkpoint_files = []

    start_time = time.time()
    samples_processed = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Processing arabic"):
        # Get batch
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_samples = [dataset[j] for j in batch_indices]

        images = [s['image'] for s in batch_samples]
        prompts = [s['arabic_prompt'] for s in batch_samples]
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

        # Save checkpoint periodically
        if samples_processed >= checkpoint_interval or i + batch_size >= len(dataset):
            if chunk_activations[layers[0]]:
                # Stack current chunk
                stacked_chunk = {}
                for layer in layers:
                    if chunk_activations[layer]:
                        stacked_chunk[layer] = torch.cat(chunk_activations[layer], dim=0)

                # Validate activations before saving
                has_nan = False
                for layer, act in stacked_chunk.items():
                    if torch.isnan(act).any():
                        nan_count = torch.isnan(act).sum().item()
                        logger.error(f"Checkpoint {len(all_checkpoint_files)} layer {layer} contains {nan_count} NaN values!")
                        has_nan = True

                if has_nan:
                    logger.error("Skipping checkpoint due to NaN values")
                    chunk_activations = {layer: [] for layer in layers}
                    chunk_genders = []
                    chunk_image_ids = []
                    samples_processed = 0
                    continue

                # Save checkpoint
                checkpoint_idx = len(all_checkpoint_files)
                checkpoint_path = checkpoints_dir / f'activations_arabic_chunk_{checkpoint_idx}.pt'

                torch.save({
                    'activations': stacked_chunk,
                    'genders': chunk_genders,
                    'image_ids': chunk_image_ids,
                    'hidden_size': hidden_size,
                    'model': config['model']['name'],
                }, checkpoint_path)

                all_checkpoint_files.append(checkpoint_path)
                logger.info(f"Saved checkpoint {checkpoint_idx} with {len(chunk_image_ids)} samples")

                # Log activation statistics
                for layer, act in stacked_chunk.items():
                    logger.info(f"  Layer {layer}: shape={act.shape}, mean={act.mean():.6f}, std={act.std():.6f}, min={act.min():.6f}, max={act.max():.6f}")

                # Clear chunk data
                chunk_activations = {layer: [] for layer in layers}
                chunk_genders = []
                chunk_image_ids = []
                samples_processed = 0

                # Clear memory
                del stacked_chunk
                gc.collect()
                torch.cuda.empty_cache()

        # Clear GPU memory periodically
        if i % (batch_size * 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info(f"\nArabic extraction completed! Created {len(all_checkpoint_files)} checkpoint files.")
    logger.info(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"Checkpoints saved in: {checkpoints_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
