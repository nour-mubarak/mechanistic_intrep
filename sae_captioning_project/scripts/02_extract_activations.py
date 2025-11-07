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
    else:
        # Try to infer from first layer
        try:
            return model.model.layers[0].mlp.gate_proj.in_features
        except:
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
    
    # Prepare inputs
    inputs = processor(
        images=images,
        text=prompts,
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
        layer = model.model.layers[layer_idx]
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
    for language in ['english', 'arabic']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Extracting activations for {language.upper()}")
        logger.info(f"{'='*50}")
        
        all_activations = {layer: [] for layer in layers}
        all_genders = []
        all_image_ids = []
        
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
                        all_activations[layer].append(batch_activations[layer])
                
                all_genders.extend(genders)
                all_image_ids.extend(image_ids)
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
            
            # Clear GPU memory periodically
            if i % (batch_size * 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Stack all activations
        stacked_activations = {}
        for layer in layers:
            if all_activations[layer]:
                stacked_activations[layer] = torch.cat(all_activations[layer], dim=0)
                logger.info(f"Layer {layer}: {stacked_activations[layer].shape}")
        
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
        
        # Clear memory before next language
        del all_activations
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("\nActivation extraction complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
