#!/usr/bin/env python3
"""
LLaVA-1.5-7B Activation Extraction
===================================

Extract activations from LLaVA-1.5-7B-hf for cross-lingual gender bias analysis.

Model: llava-hf/llava-1.5-7b-hf
- Base LLM: Vicuna-7B-v1.5 (Llama 2 architecture)
- Vision Encoder: CLIP ViT-L/14-336px
- Hidden Size: 4096
- Layers: 32
- Arabic Support: Via byte-fallback tokenization (NOT trained on Arabic)

IMPORTANT: LLaVA was NOT trained on Arabic text. However:
- The tokenizer uses byte-fallback for unknown characters
- Arabic text is tokenized as UTF-8 bytes
- Activations are valid for mechanistic analysis
- This provides a "zero-shot bilingual" comparison condition

Usage:
    python scripts/33_llava_extract_activations.py --language arabic --layers 0,8,16,24,31
    python scripts/33_llava_extract_activations.py --language english --layers 0,8,16,24,31
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
import wandb

from transformers import LlavaForConditionalGeneration, AutoProcessor


def load_llava_model(device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Load LLaVA model and processor."""
    print("=" * 60)
    print("Loading LLaVA-1.5-7B-hf...")
    print("=" * 60)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    model.eval()
    
    # Get model info
    # LLaVA structure: model.language_model.layers[i] (LlamaModel has layers directly)
    num_layers = len(model.language_model.layers)
    hidden_size = model.config.text_config.hidden_size
    
    print(f"Model loaded on {device} with dtype {dtype}")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of LLM layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Tokenizer vocab size: {processor.tokenizer.vocab_size}")
    print("=" * 60)
    
    return model, processor


class LLaVAActivationHook:
    """Hook for extracting activations from LLaVA language model layers."""
    
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []
        
    def _get_hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Store activation (detach and move to CPU)
            self.activations[layer_idx] = output.detach().cpu()
        return hook
    
    def register_hooks(self):
        """Register hooks on specified layers."""
        print(f"Registering hooks for layers: {self.layers}")
        for layer_idx in self.layers:
            try:
                # LLaVA: model.language_model.layers[i]
                layer = self.model.language_model.layers[layer_idx]
                hook = layer.register_forward_hook(self._get_hook_fn(layer_idx))
                self.hooks.append(hook)
                print(f"  ✓ Registered hook for layer {layer_idx}")
            except Exception as e:
                print(f"  ✗ Failed to register hook for layer {layer_idx}: {e}")
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_gender_from_caption(caption: str, language: str) -> Optional[str]:
    """Extract gender from caption text."""
    caption_lower = caption.lower()
    
    if language == "arabic":
        # Arabic gender indicators
        male_indicators = ["رجل", "ولد", "صبي", "شاب", "أب", "جد", "عم", "خال", "ابن", "زوج", "ذكر"]
        female_indicators = ["امرأة", "بنت", "فتاة", "أم", "جدة", "عمة", "خالة", "ابنة", "زوجة", "سيدة", "أنثى"]
    else:
        # English gender indicators
        male_indicators = ["man", "boy", "male", "father", "son", "husband", "grandfather", 
                          "uncle", "brother", "he", "his", "gentleman", "guy"]
        female_indicators = ["woman", "girl", "female", "mother", "daughter", "wife", 
                            "grandmother", "aunt", "sister", "she", "her", "lady"]
    
    has_male = any(ind in caption_lower for ind in male_indicators)
    has_female = any(ind in caption_lower for ind in female_indicators)
    
    if has_male and not has_female:
        return "male"
    elif has_female and not has_male:
        return "female"
    return None


def prepare_llava_input(processor, image_path: str, caption: str, language: str, device: str = "cuda"):
    """Prepare input for LLaVA model."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Format conversation for LLaVA
    if language == "arabic":
        # Arabic prompt with caption context
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"وصف الصورة: {caption}"}
                ]
            }
        ]
    else:
        # English prompt with caption context
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Image description: {caption}"}
                ]
            }
        ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    return inputs


def extract_activations_for_layer(
    model,
    processor,
    samples_df: pd.DataFrame,
    language: str,
    images_dir: Path,
    layer: int,
    output_dir: Path,
    max_samples: Optional[int] = None,
    device: str = "cuda"
):
    """Extract activations for a single layer."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine caption column
    if language == "arabic":
        caption_col_options = ["ar_caption", "arabic_caption", "caption_ar", "caption"]
    else:
        caption_col_options = ["en_caption", "english_caption", "caption_en", "caption"]
    
    caption_col = None
    for col in caption_col_options:
        if col in samples_df.columns:
            caption_col = col
            break
    
    if caption_col is None:
        print(f"ERROR: No caption column found for {language}")
        print(f"Available columns: {list(samples_df.columns)}")
        return
    
    # Determine image column
    image_col = None
    for col in ["image", "image_id", "filename", "image_path"]:
        if col in samples_df.columns:
            image_col = col
            break
    
    if image_col is None:
        print(f"ERROR: No image column found")
        return
    
    print(f"\nExtracting layer {layer} for {language}")
    print(f"  Caption column: {caption_col}")
    print(f"  Image column: {image_col}")
    
    # Filter for valid samples
    valid_samples = []
    for idx, row in samples_df.iterrows():
        gender = extract_gender_from_caption(str(row[caption_col]), language)
        if gender is not None:
            img_path = images_dir / row[image_col]
            if img_path.exists():
                valid_samples.append({
                    'idx': idx,
                    'image_path': str(img_path),
                    'caption': row[caption_col],
                    'gender': gender
                })
    
    if max_samples:
        valid_samples = valid_samples[:max_samples]
    
    print(f"  Valid samples: {len(valid_samples)}")
    
    # Setup hook for this layer only
    hook_handler = LLaVAActivationHook(model, [layer])
    hook_handler.register_hooks()
    
    # Extract activations
    all_activations = []
    all_genders = []
    
    try:
        for sample in tqdm(valid_samples, desc=f"Layer {layer}"):
            try:
                # Prepare input
                inputs = prepare_llava_input(
                    processor, 
                    sample['image_path'], 
                    sample['caption'], 
                    language,
                    device
                )
                
                # Forward pass
                with torch.no_grad():
                    _ = model(**inputs)
                
                # Get activation for this layer
                if layer in hook_handler.activations:
                    act = hook_handler.activations[layer]
                    # Take mean across sequence dimension (pooling)
                    act_pooled = act.mean(dim=1).squeeze(0)
                    all_activations.append(act_pooled)
                    all_genders.append(sample['gender'])
                
                # Clear for next sample
                hook_handler.clear_activations()
                
            except Exception as e:
                print(f"  Error processing sample: {e}")
                continue
    
    finally:
        hook_handler.remove_hooks()
    
    if len(all_activations) > 0:
        # Stack activations
        activations_tensor = torch.stack(all_activations)
        
        # Save checkpoint
        output_path = output_dir / f"llava_layer_{layer}_{language}.pt"
        torch.save({
            'activations': activations_tensor,
            'genders': all_genders,
            'layer': layer,
            'language': language,
            'model': 'llava-hf/llava-1.5-7b-hf',
            'num_samples': len(all_genders),
            'hidden_size': activations_tensor.shape[1],
            'timestamp': datetime.now().isoformat()
        }, output_path)
        
        print(f"  Saved: {output_path}")
        print(f"  Shape: {activations_tensor.shape}")
        print(f"  Genders: {sum(1 for g in all_genders if g == 'male')} male, "
              f"{sum(1 for g in all_genders if g == 'female')} female")
        
        return activations_tensor.shape[0]
    else:
        print(f"  No activations extracted for layer {layer}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Extract LLaVA activations")
    parser.add_argument("--language", type=str, required=True, choices=["arabic", "english"])
    parser.add_argument("--layers", type=str, default="0,8,16,24,31", 
                       help="Comma-separated layer indices")
    parser.add_argument("--data_file", type=str, default="data/processed/captions_with_gender.csv")
    parser.add_argument("--images_dir", type=str, default="data/raw/images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llava/layer_checkpoints")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llava-sae-analysis")
    args = parser.parse_args()
    
    # Parse layers
    layers = [int(l.strip()) for l in args.layers.split(",")]
    
    print(f"\n{'='*60}")
    print("LLaVA-1.5-7B Activation Extraction")
    print(f"{'='*60}")
    print(f"Language: {args.language}")
    print(f"Layers: {layers}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples or 'all'}")
    
    # Note about Arabic support
    if args.language == "arabic":
        print("\n" + "="*60)
        print("IMPORTANT: LLaVA was NOT trained on Arabic!")
        print("Arabic is processed via byte-fallback tokenization.")
        print("This provides a 'zero-shot bilingual' comparison condition.")
        print("="*60 + "\n")
    
    # Initialize W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"llava_extract_{args.language}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": "llava-hf/llava-1.5-7b-hf",
                "language": args.language,
                "layers": layers,
                "max_samples": args.max_samples,
                "stage": "activation_extraction"
            }
        )
    
    # Load model
    model, processor = load_llava_model(args.device)
    
    # Load data
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return
    
    samples_df = pd.read_csv(data_path)
    print(f"\nLoaded {len(samples_df)} samples from {data_path}")
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    # Extract for each layer
    total_samples = 0
    for layer in layers:
        n_samples = extract_activations_for_layer(
            model=model,
            processor=processor,
            samples_df=samples_df,
            language=args.language,
            images_dir=images_dir,
            layer=layer,
            output_dir=output_dir,
            max_samples=args.max_samples,
            device=args.device
        )
        total_samples += n_samples if n_samples else 0
        
        # Log to W&B
        if args.wandb and n_samples:
            wandb.log({
                f"extraction/layer_{layer}_samples": n_samples,
                "extraction/layer": layer
            })
        
        # Clear GPU memory between layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total samples extracted: {total_samples}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    if args.wandb:
        wandb.log({"extraction/total_samples": total_samples})
        wandb.finish()


if __name__ == "__main__":
    main()
