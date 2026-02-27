#!/usr/bin/env python3
"""
49_extract_full_40k_activations.py — Extract activations from ALL ~40K samples
==============================================================================

Previous extraction: 10,000 samples per language
This script: Extract from ALL available samples (40,455 pairs)

For PaLiGemma-3B, extracts from specified layers of the language model decoder.
Saves as chunked .pt files to avoid OOM.

Usage:
  python scripts/improved/49_extract_full_40k_activations.py \
      --language english --layers 9 17 --batch_size 4 --chunk_size 1000
"""

import argparse
import logging
import sys
import gc
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')


def load_dataset():
    """Load the full image-caption dataset."""
    import pandas as pd
    from PIL import Image
    
    # Try samples.csv first (processed), then captions.csv (raw)
    samples_path = BASE_DIR / 'data/processed/samples.csv'
    captions_path = BASE_DIR / 'data/raw/captions.csv'
    images_dir = BASE_DIR / 'data/raw/images'
    
    if samples_path.exists():
        df = pd.read_csv(samples_path)
        logger.info(f"Loaded {len(df)} samples from {samples_path}")
    elif captions_path.exists():
        df = pd.read_csv(captions_path)
        logger.info(f"Loaded {len(df)} samples from {captions_path}")
    else:
        raise FileNotFoundError("No dataset file found")
    
    # Check columns
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"First row: {df.iloc[0].to_dict()}")
    
    return df, images_dir


def load_model(device='cuda'):
    """Load PaLiGemma-3B for activation extraction."""
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    
    model_id = "google/paligemma-3b-mix-224"
    logger.info(f"Loading {model_id}...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # float32 for stability
        device_map=device
    )
    model.eval()
    
    logger.info(f"Model loaded. Hidden size: {model.config.text_config.hidden_size}")
    return model, processor


def extract_activations_batch(model, processor, images, texts, layers, device='cuda'):
    """Extract activations from specified layers for a batch.
    
    Returns dict: {layer_idx: tensor of shape (batch_size, d_model)} (mean-pooled)
    """
    from PIL import Image
    
    # Register hooks
    hooks = {}
    activations = {}
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Mean pool across sequence dimension
            activations[layer_idx] = hidden.mean(dim=1).detach().cpu()
        return hook_fn
    
    handles = []
    for layer_idx in layers:
        layer_module = model.language_model.layers[layer_idx]
        handle = layer_module.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)
    
    try:
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in handles:
            h.remove()
    
    return activations


def main():
    parser = argparse.ArgumentParser(description='Extract full 40K activations')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'arabic'])
    parser.add_argument('--layers', type=int, nargs='+', default=[9, 17])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Save checkpoint every N samples')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to extract (None = all)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = str(BASE_DIR / 'checkpoints' / 'full_40k_activations')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"Full 40K Activation Extraction")
    logger.info(f"Language: {args.language}, Layers: {args.layers}")
    logger.info("=" * 60)
    
    # Load dataset
    df, images_dir = load_dataset()
    
    # Determine caption column
    if args.language == 'english':
        caption_col = 'en_caption' if 'en_caption' in df.columns else 'english_prompt'
    else:
        caption_col = 'ar_caption' if 'ar_caption' in df.columns else 'arabic_prompt'
    
    image_col = 'image' if 'image' in df.columns else df.columns[0]
    
    n_total = len(df) if args.max_samples is None else min(args.max_samples, len(df))
    logger.info(f"Will extract from {n_total} samples")
    
    # Load model
    model, processor = load_model(args.device)
    
    from PIL import Image
    
    # Process in chunks
    chunk_idx = 0
    all_chunk_paths = []
    
    for chunk_start in range(0, n_total, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, n_total)
        logger.info(f"\n--- Chunk {chunk_idx}: samples {chunk_start}-{chunk_end} ---")
        
        chunk_activations = {layer: [] for layer in args.layers}
        chunk_df = df.iloc[chunk_start:chunk_end]
        
        # Process in batches within chunk
        for batch_start in tqdm(range(0, len(chunk_df), args.batch_size),
                                desc=f"Chunk {chunk_idx}"):
            batch_end = min(batch_start + args.batch_size, len(chunk_df))
            batch_df = chunk_df.iloc[batch_start:batch_end]
            
            # Load images and texts
            images = []
            texts = []
            valid_indices = []
            
            for i, (_, row) in enumerate(batch_df.iterrows()):
                img_name = row[image_col]
                img_path = images_dir / img_name
                
                if not img_path.exists():
                    # Try with .jpg extension
                    img_path = images_dir / f"{img_name}.jpg"
                
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert('RGB')
                        images.append(img)
                        texts.append(str(row[caption_col]))
                        valid_indices.append(i)
                    except Exception as e:
                        logger.warning(f"Error loading {img_path}: {e}")
                        continue
            
            if len(images) == 0:
                continue
            
            # Extract
            try:
                batch_acts = extract_activations_batch(
                    model, processor, images, texts, args.layers, args.device
                )
                
                for layer_idx in args.layers:
                    if layer_idx in batch_acts:
                        chunk_activations[layer_idx].append(batch_acts[layer_idx])
            except Exception as e:
                logger.error(f"Error in batch: {e}")
                continue
            
            # Clean up
            del images, texts, batch_acts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save chunk
        chunk_data = {}
        for layer_idx in args.layers:
            if chunk_activations[layer_idx]:
                chunk_data[layer_idx] = torch.cat(chunk_activations[layer_idx], dim=0)
                logger.info(f"  Layer {layer_idx}: {chunk_data[layer_idx].shape}")
        
        chunk_path = output_dir / f'activations_{args.language}_chunk_{chunk_idx:04d}.pt'
        torch.save({
            'activations': chunk_data,
            'chunk_index': chunk_idx,
            'sample_range': (chunk_start, chunk_end),
            'language': args.language,
            'layers': args.layers,
            'n_samples': chunk_end - chunk_start,
        }, chunk_path)
        all_chunk_paths.append(str(chunk_path))
        logger.info(f"  Saved chunk to {chunk_path}")
        
        chunk_idx += 1
        del chunk_activations, chunk_data
        gc.collect()
    
    # Save metadata
    metadata = {
        'model': 'google/paligemma-3b-mix-224',
        'language': args.language,
        'layers': args.layers,
        'n_total_samples': n_total,
        'n_chunks': chunk_idx,
        'chunk_paths': all_chunk_paths,
        'extraction_date': datetime.now().isoformat(),
        'hidden_size': 2048,
    }
    
    import json
    with open(output_dir / f'metadata_{args.language}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nExtraction complete! {n_total} samples in {chunk_idx} chunks")
    logger.info(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
