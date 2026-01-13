#!/usr/bin/env python3
"""
Script 22: Extract English Activations with Memory-Efficient Streaming Merge
=============================================================================

Re-extracts English activations with improved memory handling.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from tqdm import tqdm
import gc
from datetime import datetime
from typing import List, Optional
import numpy as np
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict):
    """Load the vision-language model and processor."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    model_name = config['model']['name']
    
    logger.info(f"Loading model: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Get number of layers
    if hasattr(model, 'language_model'):
        if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            num_layers = len(model.language_model.model.layers)
        elif hasattr(model.language_model, 'layers'):
            num_layers = len(model.language_model.layers)
        else:
            num_layers = 18
    else:
        num_layers = 18
    
    logger.info(f"Model loaded with {num_layers} layers")
    return model, processor, num_layers


class SimpleDataset:
    """Simple dataset for loading images and prompts."""
    
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        
        # Load captions CSV
        csv_path = self.data_dir / "captions.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
        else:
            # Try to find images directly
            image_files = list(self.images_dir.glob("*.jpg")) + \
                         list(self.images_dir.glob("*.png")) + \
                         list(self.images_dir.glob("*.jpeg"))
            self.df = pd.DataFrame({'image': [f.name for f in image_files]})
        
        if max_samples:
            self.df = self.df.head(max_samples)
        
        logger.info(f"Loaded {len(self.df)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image filename
        image_name = row.get('image', row.get('image_id', f'{idx}.jpg'))
        image_path = self.images_dir / image_name
        
        # Handle missing images
        if not image_path.exists():
            # Try without extension
            for ext in ['.jpg', '.png', '.jpeg']:
                alt_path = self.images_dir / f"{Path(image_name).stem}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            return None
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            return None
        
        # Get gender from row if available
        gender = row.get('ground_truth_gender', row.get('gender', 'unknown'))
        if pd.isna(gender):
            gender = 'unknown'
        
        # English prompt
        english_prompt = row.get('en_caption', row.get('english_caption', 
                                  "Describe the person in this image."))
        
        return {
            'image': image,
            'image_id': str(image_name),
            'english_prompt': english_prompt,
            'gender': gender
        }


def extract_layer_activation(model, processor, sample, layer_idx: int, device: str):
    """Extract activation from a specific layer."""
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()
        return hook
    
    # Register hook
    if hasattr(model, 'language_model'):
        if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            layer = model.language_model.model.layers[layer_idx]
        elif hasattr(model.language_model, 'layers'):
            layer = model.language_model.layers[layer_idx]
        else:
            raise ValueError("Cannot find layers in model")
    else:
        layer = model.model.layers[layer_idx]
    
    hook_handle = layer.register_forward_hook(get_activation(f'layer_{layer_idx}'))
    
    try:
        # Prepare inputs
        inputs = processor(
            images=sample['image'],
            text=sample['english_prompt'],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)
        
        # Get mean activation across sequence
        activation = activations[f'layer_{layer_idx}']
        mean_activation = activation.mean(dim=1).squeeze(0)  # [hidden_size]
        
        return mean_activation.numpy().astype(np.float16)
        
    finally:
        hook_handle.remove()


def streaming_merge_chunks(chunk_dir: Path, layer_idx: int, output_path: Path):
    """
    Merge chunks using memory-efficient streaming approach.
    """
    chunk_files = sorted(chunk_dir.glob(f'layer_{layer_idx}_english_chunk_*.pt'))
    
    if not chunk_files:
        logger.warning(f"No chunks found for layer {layer_idx}")
        return None
    
    logger.info(f"Found {len(chunk_files)} chunks for layer {layer_idx}")
    
    # First pass: count total samples and get metadata
    total_samples = 0
    hidden_size = None
    all_image_ids = []
    all_genders = []
    
    for chunk_path in tqdm(chunk_files, desc="Counting samples"):
        chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
        chunk_acts = chunk['activations']
        if hidden_size is None:
            hidden_size = chunk_acts.shape[1]
        total_samples += chunk_acts.shape[0]
        all_image_ids.extend(chunk['image_ids'])
        all_genders.extend(chunk['genders'])
        del chunk
        gc.collect()
    
    logger.info(f"Total samples: {total_samples}, Hidden size: {hidden_size}")
    
    # Use temporary memmap file for efficient streaming
    memmap_path = output_path.parent / f'temp_layer_{layer_idx}_memmap.npy'
    
    logger.info(f"Creating memory-mapped array at {memmap_path}")
    memmap_array = np.memmap(
        memmap_path, 
        dtype='float16',
        mode='w+',
        shape=(total_samples, hidden_size)
    )
    
    # Second pass: write chunks directly to memmap
    current_idx = 0
    for chunk_path in tqdm(chunk_files, desc="Streaming merge"):
        chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
        chunk_acts = chunk['activations'].numpy().astype(np.float16)
        
        n_samples = chunk_acts.shape[0]
        memmap_array[current_idx:current_idx + n_samples] = chunk_acts
        current_idx += n_samples
        
        del chunk, chunk_acts
        gc.collect()
    
    memmap_array.flush()
    
    logger.info("Converting to torch tensor and saving...")
    
    # Convert to torch tensor in chunks
    final_activations = torch.zeros(total_samples, hidden_size, dtype=torch.float32)
    
    batch_size = 1000
    for i in tqdm(range(0, total_samples, batch_size), desc="Converting"):
        end_idx = min(i + batch_size, total_samples)
        final_activations[i:end_idx] = torch.from_numpy(
            memmap_array[i:end_idx].astype(np.float32)
        )
    
    # Save final checkpoint
    checkpoint = {
        'activations': final_activations,
        'image_ids': all_image_ids,
        'genders': all_genders,
        'layer': layer_idx,
        'language': 'english',
        'num_samples': total_samples,
        'hidden_size': hidden_size,
        'extraction_date': datetime.now().isoformat(),
    }
    
    logger.info(f"Saving to {output_path}")
    torch.save(checkpoint, output_path)
    
    # Cleanup
    del memmap_array, final_activations
    gc.collect()
    
    if memmap_path.exists():
        memmap_path.unlink()
    
    return output_path


def extract_english_activations(
    config: dict,
    layer_idx: int,
    output_dir: Path,
    checkpoint_interval: int = 100,
    resume: bool = True,
    device: str = 'cuda'
):
    """Extract English activations for a single layer with streaming merge."""
    
    output_dir = Path(output_dir)
    layer_checkpoints_dir = output_dir / 'layer_checkpoints'
    layer_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    final_output = layer_checkpoints_dir / f'layer_{layer_idx}_english.pt'
    
    # Check if already extracted
    if final_output.exists():
        logger.info(f"Layer {layer_idx} English activations already exist at {final_output}")
        return final_output
    
    # Check for existing chunks
    existing_chunks = list(layer_checkpoints_dir.glob(f'layer_{layer_idx}_english_chunk_*.pt'))
    
    if existing_chunks and resume:
        chunk_indices = [int(p.stem.split('_')[-1]) for p in existing_chunks]
        last_chunk = max(chunk_indices)
        resume_sample_idx = (last_chunk + 1) * checkpoint_interval
        logger.info(f"Resuming from chunk {last_chunk + 1}, sample {resume_sample_idx}")
    else:
        resume_sample_idx = 0
        for chunk_path in existing_chunks:
            chunk_path.unlink()
    
    # Load model
    model, processor, num_layers = load_model(config)
    
    # Load dataset
    logger.info("Loading dataset...")
    data_dir = config['paths']['raw_data']
    max_samples = config['data'].get('num_samples', None)
    dataset = SimpleDataset(data_dir, max_samples=max_samples)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Check if we need to extract
    if resume_sample_idx >= len(dataset):
        logger.info("All samples already extracted, proceeding to merge...")
    else:
        # Extract activations
        logger.info(f"Extracting layer {layer_idx} English activations...")
        
        chunk_activations = []
        chunk_image_ids = []
        chunk_genders = []
        processed_in_chunk = 0
        
        for sample_idx in tqdm(range(resume_sample_idx, len(dataset)), desc=f"Layer {layer_idx} English"):
            try:
                sample = dataset[sample_idx]
                
                if sample is None:
                    continue
                
                activation = extract_layer_activation(
                    model, processor, sample, layer_idx, device
                )
                
                chunk_activations.append(activation)
                chunk_image_ids.append(sample.get('image_id', str(sample_idx)))
                chunk_genders.append(sample.get('gender', 'unknown'))
                processed_in_chunk += 1
                
                # Save chunk periodically
                if processed_in_chunk >= checkpoint_interval:
                    chunk_idx = len(list(layer_checkpoints_dir.glob(f'layer_{layer_idx}_english_chunk_*.pt')))
                    chunk_path = layer_checkpoints_dir / f'layer_{layer_idx}_english_chunk_{chunk_idx:04d}.pt'
                    
                    chunk_data = {
                        'activations': torch.from_numpy(np.stack(chunk_activations)),
                        'image_ids': chunk_image_ids,
                        'genders': chunk_genders,
                    }
                    torch.save(chunk_data, chunk_path)
                    logger.info(f"Saved chunk {chunk_idx}")
                    
                    chunk_activations = []
                    chunk_image_ids = []
                    chunk_genders = []
                    processed_in_chunk = 0
                    gc.collect()
                    
            except Exception as e:
                logger.warning(f"Error processing sample {sample_idx}: {e}")
                continue
        
        # Save remaining samples
        if chunk_activations:
            chunk_idx = len(list(layer_checkpoints_dir.glob(f'layer_{layer_idx}_english_chunk_*.pt')))
            chunk_path = layer_checkpoints_dir / f'layer_{layer_idx}_english_chunk_{chunk_idx:04d}.pt'
            
            chunk_data = {
                'activations': torch.from_numpy(np.stack(chunk_activations)),
                'image_ids': chunk_image_ids,
                'genders': chunk_genders,
            }
            torch.save(chunk_data, chunk_path)
            logger.info(f"Saved final chunk {chunk_idx}")
        
        # Clear model from memory
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    
    # Streaming merge
    logger.info("Starting streaming merge...")
    streaming_merge_chunks(layer_checkpoints_dir, layer_idx, final_output)
    
    # Clean up chunk files
    if final_output.exists():
        logger.info("Cleaning up chunk files...")
        for chunk_path in layer_checkpoints_dir.glob(f'layer_{layer_idx}_english_chunk_*.pt'):
            chunk_path.unlink()
    
    return final_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--output-dir', default='checkpoints/full_layers_ncc')
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    extract_english_activations(
        config=config,
        layer_idx=args.layer,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        device=args.device
    )


if __name__ == '__main__':
    main()
