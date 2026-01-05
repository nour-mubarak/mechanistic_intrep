#!/usr/bin/env python3
"""
Merge activation checkpoints into smaller sample files (250 samples per language)
"""

import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_sample_checkpoints(checkpoint_dir, num_chunks=5, output_suffix='small'):
    """Merge first N chunks into a smaller sample file."""
    checkpoint_dir = Path(checkpoint_dir)

    for language in ['english', 'arabic']:
        logger.info(f"Merging {language} checkpoints (first {num_chunks} chunks)...")

        merged_data = None

        for chunk_idx in range(num_chunks):
            chunk_path = checkpoint_dir / f'activations_{language}_chunk_{chunk_idx}.pt'

            if not chunk_path.exists():
                logger.warning(f"Chunk {chunk_idx} not found, skipping")
                continue

            logger.info(f"  Loading chunk {chunk_idx}...")
            chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)

            if merged_data is None:
                merged_data = {
                    'activations': {},
                    'genders': chunk_data['genders'],
                    'image_ids': chunk_data['image_ids'],
                    'num_samples': 0
                }
                # Initialize activation storage
                for layer in chunk_data['activations'].keys():
                    merged_data['activations'][layer] = []

            # Append activations
            for layer in chunk_data['activations'].keys():
                merged_data['activations'][layer].append(chunk_data['activations'][layer])

            # Extend metadata
            if chunk_idx > 0:
                merged_data['genders'].extend(chunk_data['genders'])
                merged_data['image_ids'].extend(chunk_data['image_ids'])

            merged_data['num_samples'] += chunk_data.get('num_samples', len(chunk_data['genders']))

            del chunk_data

        # Concatenate all activations
        logger.info(f"Concatenating {len(merged_data['activations'][list(merged_data['activations'].keys())[0]])} chunks...")
        for layer in merged_data['activations'].keys():
            merged_data['activations'][layer] = torch.cat(merged_data['activations'][layer], dim=0)

        # Save
        output_path = checkpoint_dir / f'activations_{language}_sample_{output_suffix}.pt'
        logger.info(f"Saving to {output_path}...")
        logger.info(f"  Samples: {merged_data['num_samples']}")
        logger.info(f"  Layers: {list(merged_data['activations'].keys())}")
        for layer, acts in merged_data['activations'].items():
            logger.info(f"    Layer {layer}: {acts.shape}")

        torch.save(merged_data, output_path)
        logger.info(f"Saved: {output_path}")

if __name__ == '__main__':
    merge_sample_checkpoints('checkpoints', num_chunks=5, output_suffix='small')
    logger.info("Done!")
