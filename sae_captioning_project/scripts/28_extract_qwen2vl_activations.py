#!/usr/bin/env python3
"""
Qwen2-VL-7B Activation Extraction
==================================

Extract activations from Qwen2-VL-7B-Instruct for cross-lingual gender bias analysis.
This script is adapted from the PaLiGemma extraction pipeline.

Model: Qwen/Qwen2-VL-7B-Instruct
- Hidden size: 3584
- Layers: 28
- Context: 32k tokens
- Vision: Native multimodal

Usage:
    python scripts/28_extract_qwen2vl_activations.py --language arabic --layers 0,4,8,12,16,20,24,27
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import gc
import json
from datetime import datetime
from typing import Dict, List, Optional

# Qwen2-VL specific imports
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info  # May need to install


def load_qwen2vl_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load Qwen2-VL model and processor."""
    print("Loading Qwen2-VL-7B-Instruct...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = Qwen2VLProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True
    )
    
    model.eval()
    print(f"Model loaded on {device} with dtype {dtype}")
    
    return model, processor


class Qwen2VLActivationHook:
    """Hook for extracting activations from Qwen2-VL layers."""
    
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []
        
    def _get_hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Store activation
            self.activations[layer_idx] = output.detach().cpu()
        return hook
    
    def register_hooks(self):
        """Register hooks on specified layers."""
        for layer_idx in self.layers:
            # Qwen2-VL uses model.model.layers[i]
            try:
                layer = self.model.model.layers[layer_idx]
                hook = layer.register_forward_hook(self._get_hook_fn(layer_idx))
                self.hooks.append(hook)
                print(f"  Registered hook for layer {layer_idx}")
            except Exception as e:
                print(f"  Failed to register hook for layer {layer_idx}: {e}")
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
        gc.collect()


def extract_gender_from_caption(caption: str, language: str) -> Optional[str]:
    """Extract gender from caption text."""
    caption_lower = caption.lower()
    
    if language == "arabic":
        # Arabic gender indicators
        male_indicators = ["رجل", "ولد", "صبي", "شاب", "أب", "جد", "عم", "خال", "ابن", "زوج"]
        female_indicators = ["امرأة", "بنت", "فتاة", "أم", "جدة", "عمة", "خالة", "ابنة", "زوجة", "سيدة"]
    else:
        # English gender indicators
        male_indicators = ["man", "boy", "male", "father", "son", "husband", "grandfather", "uncle", "brother", "he", "his"]
        female_indicators = ["woman", "girl", "female", "mother", "daughter", "wife", "grandmother", "aunt", "sister", "she", "her"]
    
    has_male = any(ind in caption_lower for ind in male_indicators)
    has_female = any(ind in caption_lower for ind in female_indicators)
    
    if has_male and not has_female:
        return "male"
    elif has_female and not has_male:
        return "female"
    return None


def prepare_qwen2vl_input(processor, image_path: str, caption: str, language: str):
    """Prepare input for Qwen2-VL model."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Format for Qwen2-VL
    if language == "arabic":
        prompt = f"<|im_start|>user\n<image>\nوصف الصورة: {caption}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n<image>\nDescribe the image: {caption}<|im_end|>\n<|im_start|>assistant\n"
    
    # Process inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": caption}
            ]
        }
    ]
    
    # Use processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    return inputs


def extract_activations_qwen2vl(
    model,
    processor,
    hook: Qwen2VLActivationHook,
    samples_df: pd.DataFrame,
    language: str,
    images_dir: Path,
    output_dir: Path,
    layers: List[int],
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    device: str = "cuda"
):
    """Extract activations from Qwen2-VL for all samples."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter samples with valid gender
    caption_col = f"{language}_caption" if f"{language}_caption" in samples_df.columns else "caption"
    
    valid_samples = []
    for idx, row in samples_df.iterrows():
        caption = row.get(caption_col, row.get('caption', ''))
        gender = extract_gender_from_caption(caption, language)
        if gender:
            valid_samples.append({
                'idx': idx,
                'image_id': row.get('image_id', row.get('filename', f'img_{idx}')),
                'caption': caption,
                'gender': gender
            })
    
    if max_samples:
        valid_samples = valid_samples[:max_samples]
    
    print(f"Processing {len(valid_samples)} {language} samples...")
    
    # Initialize storage per layer
    layer_activations = {layer: [] for layer in layers}
    genders = []
    image_ids = []
    
    # Process samples
    for sample in tqdm(valid_samples, desc=f"Extracting {language}"):
        try:
            # Find image
            image_id = sample['image_id']
            image_path = None
            
            for ext in ['.jpg', '.jpeg', '.png', '']:
                potential_path = images_dir / f"{image_id}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if not image_path:
                # Try COCO format
                image_path = images_dir / f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
            
            if not image_path.exists():
                continue
            
            # Prepare input
            inputs = prepare_qwen2vl_input(processor, str(image_path), sample['caption'], language)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Clear previous activations
            hook.clear_activations()
            
            # Forward pass
            with torch.no_grad():
                _ = model(**inputs)
            
            # Store activations (mean over sequence length)
            for layer in layers:
                if layer in hook.activations:
                    act = hook.activations[layer]
                    if len(act.shape) == 3:
                        act = act.mean(dim=1)  # [batch, seq, hidden] -> [batch, hidden]
                    layer_activations[layer].append(act.squeeze(0))
            
            genders.append(sample['gender'])
            image_ids.append(image_id)
            
            # Clear GPU memory periodically
            if len(genders) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing sample {sample['image_id']}: {e}")
            continue
    
    # Save activations per layer
    print(f"\nSaving activations for {len(genders)} samples...")
    
    metadata = {
        'genders': genders,
        'image_ids': image_ids,
        'language': language,
        'model': 'Qwen/Qwen2-VL-7B-Instruct',
        'n_samples': len(genders),
        'timestamp': datetime.now().isoformat()
    }
    
    for layer in layers:
        if layer_activations[layer]:
            acts_tensor = torch.stack(layer_activations[layer])
            
            save_data = {
                'activations': acts_tensor,
                'genders': genders,
                'image_ids': image_ids,
                'metadata': metadata
            }
            
            save_path = output_dir / f"qwen2vl_layer_{layer}_{language}.pt"
            torch.save(save_data, save_path)
            print(f"  Saved layer {layer}: {acts_tensor.shape} -> {save_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen2-VL activations")
    parser.add_argument("--language", type=str, required=True, choices=["arabic", "english"])
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,27")
    parser.add_argument("--data_file", type=str, default="data/processed/samples.csv")
    parser.add_argument("--images_dir", type=str, default="data/raw/images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen2vl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    
    print("="*60)
    print(f"Qwen2-VL Activation Extraction - {args.language.upper()}")
    print("="*60)
    print(f"Layers: {layers}")
    print(f"Output: {args.output_dir}")
    
    # Load model
    model, processor = load_qwen2vl_model(args.device)
    
    # Setup hooks
    hook = Qwen2VLActivationHook(model, layers)
    hook.register_hooks()
    
    # Load data
    samples_df = pd.read_csv(args.data_file)
    print(f"Loaded {len(samples_df)} samples")
    
    # Extract activations
    metadata = extract_activations_qwen2vl(
        model=model,
        processor=processor,
        hook=hook,
        samples_df=samples_df,
        language=args.language,
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir) / "layer_checkpoints",
        layers=layers,
        max_samples=args.max_samples,
        device=args.device
    )
    
    # Cleanup
    hook.remove_hooks()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Extraction complete!")
    print(f"Samples processed: {metadata['n_samples']}")
    print("="*60)


if __name__ == "__main__":
    main()
