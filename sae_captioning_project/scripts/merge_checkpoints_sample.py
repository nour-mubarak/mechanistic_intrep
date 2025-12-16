#!/usr/bin/env python3
"""
Merge a sample of activation checkpoint files for analysis
(to avoid OOM issues with full dataset)
"""
import torch
import gc
from pathlib import Path
import logging
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_checkpoint_sample(checkpoints_dir, language, layers, max_samples=500):
    """Merge a sample of checkpoint files to avoid memory issues."""

    checkpoints_dir = Path(checkpoints_dir)

    # Find all checkpoint files for this language
    checkpoint_files = sorted(checkpoints_dir.glob(f'activations_{language}_chunk_*.pt'))

    if not checkpoint_files:
        logger.error(f"No checkpoint files found for {language}")
        return False

    logger.info(f"Found {len(checkpoint_files)} checkpoint files for {language}")
    logger.info(f"Sampling up to {max_samples} samples...")

    # Determine how many chunks to use
    samples_per_chunk = 50  # from inspection
    num_chunks_needed = min(len(checkpoint_files), (max_samples + samples_per_chunk - 1) // samples_per_chunk)

    logger.info(f"Using {num_chunks_needed} chunks for ~{num_chunks_needed * samples_per_chunk} samples")

    # Load metadata from first checkpoint
    first_checkpoint = torch.load(checkpoint_files[0], weights_only=False)
    all_genders = []
    all_image_ids = []
    hidden_size = first_checkpoint.get('hidden_size')
    model_name = first_checkpoint.get('model')

    # Collect metadata from selected chunks
    selected_files = checkpoint_files[:num_chunks_needed]
    for checkpoint_path in selected_files:
        checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        all_genders.extend(checkpoint_data['genders'])
        all_image_ids.extend(checkpoint_data['image_ids'])
        del checkpoint_data
        gc.collect()

    logger.info(f"Total samples: {len(all_genders)}")

    # Process one layer at a time
    final_activations = {}

    for layer in layers:
        logger.info(f"Processing layer {layer}...")
        layer_activations = []

        # Load this layer from selected checkpoints
        for checkpoint_path in selected_files:
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            if layer in checkpoint_data['activations']:
                layer_activations.append(checkpoint_data['activations'][layer])
            del checkpoint_data
            gc.collect()

        # Concatenate this layer
        if layer_activations:
            final_activations[layer] = torch.cat(layer_activations, dim=0)
            logger.info(f"Layer {layer}: {final_activations[layer].shape}")

        # Clear layer data
        del layer_activations
        gc.collect()
        torch.cuda.empty_cache()

    # Save merged file
    output_path = checkpoints_dir / f'activations_{language}_sample.pt'
    torch.save({
        'activations': final_activations,
        'genders': all_genders,
        'image_ids': all_image_ids,
        'layers': layers,
        'hidden_size': hidden_size,
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'num_samples': len(all_genders),
    }, output_path)

    logger.info(f"Saved sampled activations to {output_path}")
    file_size_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"File size: {file_size_gb:.2f} GB")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='both',
                       choices=['english', 'arabic', 'both'],
                       help='Which language to merge')
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum number of samples to include')
    args = parser.parse_args()

    checkpoints_dir = Path('checkpoints')
    layers = [2, 6, 10, 14, 18, 22, 26, 30]

    if args.language in ['english', 'both']:
        # Merge English checkpoints
        logger.info("="*50)
        logger.info(f"Merging English checkpoints (sample)...")
        logger.info("="*50)
        success_en = merge_checkpoint_sample(checkpoints_dir, 'english', layers, args.max_samples)

        if success_en:
            logger.info("English merge completed successfully!")
        else:
            logger.error("English merge failed!")

    if args.language in ['arabic', 'both']:
        # Merge Arabic checkpoints
        logger.info("\n" + "="*50)
        logger.info(f"Merging Arabic checkpoints (sample)...")
        logger.info("="*50)
        success_ar = merge_checkpoint_sample(checkpoints_dir, 'arabic', layers, args.max_samples)

        if success_ar:
            logger.info("Arabic merge completed successfully!")
        else:
            logger.error("Arabic merge failed!")

    logger.info("\nMerge process completed!")
