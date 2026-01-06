#!/usr/bin/env python3
"""
Script 18: Full Layer Activation Extraction with NCC
====================================================

Extracts activations from ALL layers (0-33) of Gemma-3-4B on the full dataset
using Neural Corpus Compilation (NCC) methodology for efficient large-scale extraction.

NCC Key Principles:
- Process activations in streaming fashion to minimize memory
- Save layer-wise checkpoints for recovery from failures
- Use efficient tensor storage formats
- Enable parallel downstream processing

Usage:
    # Extract all 34 layers on full dataset
    python scripts/18_extract_full_activations_ncc.py --config configs/config.yaml

    # Extract specific layer ranges
    python scripts/18_extract_full_activations_ncc.py --config configs/config.yaml --layer-ranges 0-10 20-33

    # Extract with custom batch size
    python scripts/18_extract_full_activations_ncc.py --config configs/config.yaml --batch-size 2
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
import json
from typing import List, Dict, Tuple, Optional

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


def parse_layer_ranges(ranges_str: List[str], max_layer: int = 33) -> List[int]:
    """
    Parse layer range specifications.

    Args:
        ranges_str: List of range strings like ['0-10', '20-33']
        max_layer: Maximum layer index

    Returns:
        Sorted list of layer indices
    """
    if not ranges_str:
        # Default: all layers
        return list(range(max_layer + 1))

    layers = set()
    for range_spec in ranges_str:
        if '-' in range_spec:
            start, end = range_spec.split('-')
            start, end = int(start), int(end)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(range_spec))

    return sorted(list(layers))


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

    # Get model architecture info
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        num_layers = len(model.language_model.layers)
        logger.info(f"Model has {num_layers} layers (language_model.layers)")
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        logger.info(f"Model has {num_layers} layers (model.layers)")
    else:
        logger.warning("Could not determine number of layers")
        num_layers = 34  # Default for Gemma-3-4B

    return model, processor, num_layers


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


def extract_activations_batch_ncc(
    model,
    processor,
    images,
    prompts,
    layers: List[int],
    device: str = "cuda",
    save_intermediate: bool = False,
    checkpoint_dir: Optional[Path] = None,
    batch_idx: int = 0
) -> Dict[int, torch.Tensor]:
    """
    Extract activations for a batch using NCC methodology.

    NCC principles:
    - Stream activations directly to CPU/disk to minimize GPU memory
    - Use detached tensors to prevent gradient graph buildup
    - Clear GPU cache aggressively
    - Optional intermediate saves for recovery
    """

    # Format prompts for Gemma-3
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

    # Setup activation storage
    activations = {layer: [] for layer in layers}
    hooks = []

    def make_hook(layer_idx):
        """Create hook that immediately moves activations to CPU and converts to float32."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            # NCC: Detach, move to CPU, convert to float32 immediately
            act = act.detach().cpu().float()

            # Check for NaN values
            if torch.isnan(act).any():
                logger.error(f"NaN detected in layer {layer_idx} activations!")
                logger.error(f"NaN count: {torch.isnan(act).sum().item()} / {act.numel()}")
                # Replace NaN with zeros to prevent corruption
                act = torch.nan_to_num(act, nan=0.0)
                logger.warning("Replaced NaN values with zeros")

            activations[layer_idx].append(act)

            # NCC: Clear GPU cache after each layer extraction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hook

    # Register hooks for all requested layers
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

    # Remove hooks immediately
    for h in hooks:
        h.remove()

    # NCC: Stack activations immediately and clear list
    result = {}
    for layer in layers:
        if activations[layer]:
            result[layer] = torch.cat(activations[layer], dim=0)
            activations[layer].clear()  # Clear list to free memory

    # NCC: Optional intermediate checkpoint for large batches
    if save_intermediate and checkpoint_dir:
        checkpoint_path = checkpoint_dir / f'batch_{batch_idx}_intermediate.pt'
        torch.save(result, checkpoint_path)
        logger.debug(f"Saved intermediate checkpoint: {checkpoint_path}")

    # Final cleanup
    del inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_layer_checkpoint(
    layer_idx: int,
    activations: torch.Tensor,
    genders: List[str],
    image_ids: List[str],
    checkpoint_dir: Path,
    language: str,
    metadata: dict
) -> Path:
    """
    Save checkpoint for a single layer (NCC layer-wise storage).

    This enables:
    - Parallel processing of different layers
    - Recovery from failures
    - Selective layer loading
    """
    checkpoint_path = checkpoint_dir / f'layer_{layer_idx}_{language}.pt'

    torch.save({
        'layer': layer_idx,
        'activations': activations,
        'genders': genders,
        'image_ids': image_ids,
        'timestamp': datetime.now().isoformat(),
        **metadata
    }, checkpoint_path)

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from all VLM layers using NCC methodology"
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--layer-ranges', type=str, nargs='+', default=None,
                       help='Layer ranges to extract (e.g., 0-10 20-33). Default: all layers')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (default: 1 for stability)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Save checkpoint every N samples')
    parser.add_argument('--languages', type=str, nargs='+', default=['english', 'arabic'],
                       help='Languages to process')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
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
                name=f"full-activation-extraction-ncc-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    'model': config['model']['name'],
                    'layer_ranges': args.layer_ranges,
                    'num_samples': config['data'].get('num_samples'),
                    'batch_size': args.batch_size or config['data'].get('batch_size', 1),
                    'checkpoint_interval': args.checkpoint_interval,
                    'methodology': 'NCC (Neural Corpus Compilation)',
                },
                tags=['activation-extraction', 'ncc', 'full-layers', 'gemma-3']
            )
            logger.info(f"Wandb initialized: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_run = None

    # Setup paths
    processed_dir = Path(config['paths']['processed_data'])

    if args.output_dir:
        checkpoints_dir = Path(args.output_dir)
    else:
        checkpoints_dir = Path(config['paths']['checkpoints']) / 'full_layers_ncc'

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for layer-wise storage (NCC)
    layer_checkpoints_dir = checkpoints_dir / 'layer_checkpoints'
    layer_checkpoints_dir.mkdir(exist_ok=True)

    # Check for processed data
    csv_path = processed_dir / 'samples.csv'
    if not csv_path.exists():
        logger.error("No processed data found. Run 01_prepare_data.py first.")
        return 1

    # Load model
    model, processor, num_layers = load_model(config)
    device = config['model'].get('device', 'cuda')

    # Get model info
    hidden_size = get_model_hidden_size(model)
    logger.info(f"Model hidden size: {hidden_size}")
    logger.info(f"Model total layers: {num_layers}")

    # Determine layers to extract
    layers = parse_layer_ranges(args.layer_ranges, max_layer=num_layers - 1)
    logger.info(f"Extracting {len(layers)} layers: {layers}")

    # Load dataset
    batch_size = args.batch_size or config['data'].get('batch_size', 1)
    logger.info(f"Batch size: {batch_size}")

    dataset = CrossLingualCaptionDataset(
        data_dir=processed_dir,
        csv_path=csv_path,
        max_samples=config['data'].get('num_samples')
    )

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Total tokens to process: ~{len(dataset) * 278} (estimated)")

    # Save extraction metadata
    metadata = {
        'model': config['model']['name'],
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'layers_extracted': layers,
        'num_samples': len(dataset),
        'batch_size': batch_size,
        'checkpoint_interval': args.checkpoint_interval,
        'extraction_date': datetime.now().isoformat(),
        'methodology': 'NCC (Neural Corpus Compilation)',
    }

    metadata_path = checkpoints_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Extract for each language
    checkpoint_interval = args.checkpoint_interval

    for language in args.languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting activations for {language.upper()}")
        logger.info(f"{'='*60}")

        # NCC: Store layer-wise accumulations separately
        layer_wise_data = {
            layer: {
                'activations': [],
                'genders': [],
                'image_ids': []
            }
            for layer in layers
        }

        # Checkpoint tracking
        chunk_activations = {layer: [] for layer in layers}
        chunk_genders = []
        chunk_image_ids = []
        chunk_files = []

        start_time = time.time()
        samples_processed = 0

        for batch_idx in tqdm(range(0, len(dataset), batch_size), desc=f"Processing {language}"):
            # Get batch
            batch_indices = range(batch_idx, min(batch_idx + batch_size, len(dataset)))
            batch_samples = [dataset[j] for j in batch_indices]

            images = [s['image'] for s in batch_samples]
            prompts = [s[f'{language}_prompt'] for s in batch_samples]
            genders = [s['ground_truth_gender'] for s in batch_samples]
            image_ids = [s['image_id'] for s in batch_samples]

            # Extract with NCC methodology
            try:
                batch_activations = extract_activations_batch_ncc(
                    model=model,
                    processor=processor,
                    images=images,
                    prompts=prompts,
                    layers=layers,
                    device=device,
                    save_intermediate=False,  # Disable for now
                    checkpoint_dir=layer_checkpoints_dir,
                    batch_idx=batch_idx
                )

                # Accumulate to chunk
                for layer in layers:
                    if layer in batch_activations:
                        chunk_activations[layer].append(batch_activations[layer])

                chunk_genders.extend(genders)
                chunk_image_ids.extend(image_ids)
                samples_processed += len(batch_samples)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

            # Save checkpoint periodically (NCC: layer-wise saves)
            if samples_processed >= checkpoint_interval or batch_idx + batch_size >= len(dataset):
                if chunk_activations[layers[0]]:  # Check if we have any data
                    logger.info(f"\nSaving checkpoint at {samples_processed} samples...")

                    # NCC: Process and save each layer separately
                    for layer in layers:
                        if chunk_activations[layer]:
                            # Stack activations for this layer
                            layer_act = torch.cat(chunk_activations[layer], dim=0)

                            # Validate
                            if torch.isnan(layer_act).any():
                                nan_count = torch.isnan(layer_act).sum().item()
                                logger.error(f"Layer {layer} contains {nan_count} NaN values! Skipping.")
                                continue

                            # Accumulate to layer-wise storage
                            layer_wise_data[layer]['activations'].append(layer_act)

                    # Store metadata (same for all layers in this chunk)
                    layer_wise_data[layers[0]]['genders'].extend(chunk_genders)
                    layer_wise_data[layers[0]]['image_ids'].extend(chunk_image_ids)

                    # Log statistics
                    logger.info(f"Checkpoint: {len(chunk_image_ids)} samples accumulated")
                    for layer in layers[:3]:  # Log first 3 layers
                        if chunk_activations[layer]:
                            act = torch.cat(chunk_activations[layer], dim=0)
                            logger.info(
                                f"  Layer {layer}: shape={act.shape}, "
                                f"mean={act.mean():.6f}, std={act.std():.6f}"
                            )

                    # Clear chunk data
                    chunk_activations = {layer: [] for layer in layers}
                    chunk_genders = []
                    chunk_image_ids = []
                    samples_processed = 0

                    # Aggressive memory cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Log to wandb periodically
            if wandb_run and batch_idx % (batch_size * 10) == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                total_processed = batch_idx + batch_size
                samples_per_sec = total_processed / elapsed if elapsed > 0 else 0

                log_dict = {
                    f'{language}/samples_processed': total_processed,
                    f'{language}/samples_per_sec': samples_per_sec,
                    f'{language}/progress': total_processed / len(dataset) * 100,
                }

                # Log GPU memory
                if torch.cuda.is_available():
                    log_dict[f'{language}/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                    log_dict[f'{language}/gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9

                try:
                    import wandb
                    wandb.log(log_dict)
                except:
                    pass

            # Periodic GPU cleanup
            if batch_idx % (batch_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # NCC: Final merge and save - process each layer independently
        logger.info(f"\nMerging and saving {len(layers)} layers for {language}...")

        # Use metadata from first layer (they're all the same)
        final_genders = layer_wise_data[layers[0]]['genders']
        final_image_ids = layer_wise_data[layers[0]]['image_ids']

        logger.info(f"Total samples: {len(final_image_ids)}")

        # Save each layer separately (NCC principle: layer-wise storage)
        for layer in tqdm(layers, desc="Saving layers"):
            if layer_wise_data[layer]['activations']:
                # Merge all chunks for this layer
                layer_activations = torch.cat(layer_wise_data[layer]['activations'], dim=0)

                logger.info(f"Layer {layer}: {layer_activations.shape}")

                # Save layer checkpoint
                save_layer_checkpoint(
                    layer_idx=layer,
                    activations=layer_activations,
                    genders=final_genders,
                    image_ids=final_image_ids,
                    checkpoint_dir=layer_checkpoints_dir,
                    language=language,
                    metadata={
                        'hidden_size': hidden_size,
                        'model': config['model']['name'],
                    }
                )

                # Clear from memory
                del layer_activations
                layer_wise_data[layer]['activations'].clear()
                gc.collect()

        # Also save a combined file for convenience (optional)
        logger.info(f"Creating combined checkpoint for {language}...")
        combined_activations = {}

        for layer in tqdm(layers, desc="Loading layers for combined save"):
            layer_checkpoint_path = layer_checkpoints_dir / f'layer_{layer}_{language}.pt'
            if layer_checkpoint_path.exists():
                layer_data = torch.load(layer_checkpoint_path)
                combined_activations[layer] = layer_data['activations']

        # Save combined
        combined_path = checkpoints_dir / f'activations_{language}_all_layers.pt'
        torch.save({
            'activations': combined_activations,
            'genders': final_genders,
            'image_ids': final_image_ids,
            'layers': layers,
            'hidden_size': hidden_size,
            'timestamp': datetime.now().isoformat(),
            'model': config['model']['name'],
            'num_layers_extracted': len(layers),
        }, combined_path)

        logger.info(f"Saved combined checkpoint to {combined_path}")

        # Log final stats to wandb
        if wandb_run:
            elapsed = time.time() - start_time
            log_dict = {
                f'{language}/total_samples': len(final_image_ids),
                f'{language}/total_layers': len(layers),
                f'{language}/total_time_sec': elapsed,
                f'{language}/avg_samples_per_sec': len(final_image_ids) / elapsed if elapsed > 0 else 0,
            }

            # Log layer shapes
            for layer in layers[:10]:  # Log first 10 layers
                if layer in combined_activations:
                    log_dict[f'{language}/layer_{layer}_shape'] = str(combined_activations[layer].shape)

            # Log gender distribution
            gender_counts = {}
            for gender in final_genders:
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
            for gender, count in gender_counts.items():
                log_dict[f'{language}/gender_{gender}_count'] = count

            try:
                import wandb
                wandb.log(log_dict)
            except:
                pass

        # Clear memory before next language
        del combined_activations
        del layer_wise_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\n" + "="*60)
    logger.info("Full layer activation extraction complete!")
    logger.info(f"Extracted {len(layers)} layers for {len(args.languages)} language(s)")
    logger.info(f"Layer-wise checkpoints: {layer_checkpoints_dir}")
    logger.info(f"Combined checkpoints: {checkpoints_dir}")
    logger.info("="*60)

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
