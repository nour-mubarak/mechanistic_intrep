#!/usr/bin/env python3
"""
Extract activations for Arabic only (English already done)
Simplified version that skips the problematic merge step
"""
import sys
import gc
import torch
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config):
    """Load the vision-language model."""
    model_name = config['model']['name']
    logger.info(f"Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=config['model'].get('device_map', 'auto'),
        torch_dtype=getattr(torch, config['model'].get('dtype', 'float16'))
    )

    logger.info(f"Model loaded with dtype {model.dtype}")
    return model, processor

def get_model_hidden_size(model):
    """Get the hidden size of the model."""
    if hasattr(model, 'language_model'):
        return model.language_model.config.hidden_size
    elif hasattr(model, 'model'):
        return model.model.config.hidden_size
    elif hasattr(model, 'config'):
        return model.config.hidden_size
    else:
        raise ValueError("Could not determine model hidden size")

def load_dataset(data_dir, csv_path, max_samples=None):
    """Load dataset from CSV."""
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    data = []
    for _, row in df.iterrows():
        image_filename = row['image']
        image_path = Path(data_dir) / 'images' / image_filename
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        # Extract image ID from filename (remove .jpg extension)
        image_id = image_filename.replace('.jpg', '')

        data.append({
            'image': Image.open(image_path).convert('RGB'),
            'image_id': image_id,
            'english_caption': row['arabic_prompt'],  # Use the prompt, not caption
            'arabic_caption': row['arabic_prompt'],
            'gender': row['ground_truth_gender']
        })

    return data

def extract_activations_single(model, processor, image, prompt, layers, device):
    """Extract activations for a single image."""
    from PIL import Image

    # Format prompt with image placeholder for Gemma-3
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Process inputs
    inputs = processor(
        images=image,
        text=formatted_prompt,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Hook to capture activations
    activations = {layer: [] for layer in layers}

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            activations[layer_idx].append(act.detach().cpu())
        return hook

    # Register hooks
    hooks = []
    for layer_idx in layers:
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            layer = model.language_model.layers[layer_idx]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[layer_idx]
        else:
            raise ValueError(f"Could not access layer {layer_idx}")
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
    # Load config
    config = load_config('configs/config.yaml')

    # Setup paths
    processed_dir = Path(config['paths']['processed_data'])
    checkpoints_dir = Path(config['paths']['checkpoints'])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model, processor = load_model(config)
    device = config['model'].get('device', 'cuda')

    # Get model info
    hidden_size = get_model_hidden_size(model)
    logger.info(f"Model hidden size: {hidden_size}")

    # Determine layers
    layers = config['layers']['extraction']
    logger.info(f"Extracting layers: {layers}")

    # Load dataset
    csv_path = processed_dir / 'samples.csv'
    dataset = load_dataset(
        data_dir=processed_dir,
        csv_path=csv_path,
        max_samples=config['data'].get('num_samples')
    )

    logger.info(f"Dataset size: {len(dataset)}")

    # Extract Arabic only
    language = 'arabic'
    logger.info(f"\nExtracting activations for ARABIC")
    logger.info("="*50)

    # Process in chunks to save checkpoints
    chunk_size = 50
    num_chunks = (len(dataset) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(dataset))

        chunk_activations = {layer: [] for layer in layers}
        chunk_genders = []
        chunk_image_ids = []

        # Process this chunk
        for i in tqdm(range(start_idx, end_idx), desc=f"Chunk {chunk_idx}"):
            sample = dataset[i]

            # Extract activations
            acts = extract_activations_single(
                model=model,
                processor=processor,
                image=sample['image'],
                prompt=sample[f'{language}_caption'],
                layers=layers,
                device=device
            )

            # Store
            for layer in layers:
                chunk_activations[layer].append(acts[layer])
            chunk_genders.append(sample['gender'])
            chunk_image_ids.append(sample['image_id'])

            # Clear memory periodically
            if (i - start_idx) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Stack chunk activations
        stacked_chunk = {}
        for layer in layers:
            if chunk_activations[layer]:
                stacked_chunk[layer] = torch.cat(chunk_activations[layer], dim=0)

        # Save checkpoint
        checkpoint_path = checkpoints_dir / f'activations_{language}_chunk_{chunk_idx}.pt'
        torch.save({
            'activations': stacked_chunk,
            'genders': chunk_genders,
            'image_ids': chunk_image_ids,
            'hidden_size': hidden_size,
            'model': config['model']['name'],
        }, checkpoint_path)

        logger.info(f"Saved checkpoint {chunk_idx} with {len(chunk_genders)} samples")

        # Clear memory
        del chunk_activations, stacked_chunk
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"\nArabic extraction completed! Created {num_chunks} checkpoint files.")
    logger.info("Run merge_checkpoints.py to merge them into final file.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
