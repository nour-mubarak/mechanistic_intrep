#!/usr/bin/env python3
"""
Llama 3.2 Vision (11B) Activation Extraction
=============================================

Extract activations from meta-llama/Llama-3.2-11B-Vision-Instruct for cross-lingual
gender bias analysis.

Model: meta-llama/Llama-3.2-11B-Vision-Instruct
- Base LLM: Llama 3.2 (11B parameters)
- Vision Encoder: Integrated multimodal architecture
- Hidden Size: 4096
- Layers: 40
- Arabic Support: Native multilingual training (better than LLaVA)

Llama 3.2 Vision was trained on multilingual data including Arabic, making it
a strong comparison point for cross-lingual analysis.

Usage:
    python scripts/38_llama32vision_extract_activations.py --language arabic --layers 0,8,16,24,32,39
    python scripts/38_llama32vision_extract_activations.py --language english --layers 0,8,16,24,32,39
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

from transformers import MllamaForConditionalGeneration, AutoProcessor


# Model configuration
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_NAME = "Llama-3.2-Vision-11B"
HIDDEN_SIZE = 4096
NUM_LAYERS = 40
SAE_EXPANSION = 8
SAE_FEATURES = HIDDEN_SIZE * SAE_EXPANSION  # 32768


def load_llama32_vision_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load Llama 3.2 Vision model and processor."""
    print("=" * 60)
    print(f"Loading {MODEL_NAME}...")
    print("=" * 60)

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model.eval()

    # Get model info - Llama 3.2 Vision: model.language_model returns MllamaTextModel which has .layers directly
    try:
        num_layers = len(model.language_model.layers)
    except:
        num_layers = NUM_LAYERS

    try:
        hidden_size = model.config.text_config.hidden_size
    except:
        hidden_size = HIDDEN_SIZE

    print(f"Model loaded on {device} with dtype {dtype}")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of LLM layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Tokenizer vocab size: {processor.tokenizer.vocab_size}")
    print("=" * 60)

    return model, processor


class Llama32VisionActivationHook:
    """Hook for extracting activations from Llama 3.2 Vision language model layers."""

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
                # Llama 3.2 Vision: model.language_model returns MllamaTextModel which has .layers directly
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


def prepare_llama32_input(processor, image_path: str, caption: str, language: str, device: str = "cuda"):
    """Prepare input for Llama 3.2 Vision model."""

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Format conversation for Llama 3.2 Vision
    if language == "arabic":
        prompt = f"<|image|><|begin_of_text|>صف الشخص في هذه الصورة: {caption}"
    else:
        prompt = f"<|image|><|begin_of_text|>Describe the person in this image: {caption}"

    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    return inputs


def extract_activations_batch(
    model,
    processor,
    hook: Llama32VisionActivationHook,
    data: pd.DataFrame,
    language: str,
    images_dir: Path,
    device: str = "cuda"
) -> Dict[int, List[np.ndarray]]:
    """Extract activations for a batch of samples."""

    all_activations = {layer: [] for layer in hook.layers}
    all_genders = []
    all_indices = []

    caption_col = "ar_caption" if language == "arabic" else "en_caption"

    for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Extracting {language}"):
        try:
            # Get image path
            image_path = images_dir / row["image"]
            if not image_path.exists():
                continue

            # Get caption
            caption = row[caption_col]
            if pd.isna(caption) or not caption.strip():
                continue

            # Extract gender
            gender = extract_gender_from_caption(caption, language)
            if gender is None:
                continue

            # Clear previous activations
            hook.clear_activations()

            # Prepare inputs
            inputs = prepare_llama32_input(processor, str(image_path), caption, language, device)

            # Forward pass (no gradient)
            with torch.no_grad():
                _ = model(**inputs)

            # Store activations (mean pool over sequence)
            for layer_idx in hook.layers:
                if layer_idx in hook.activations:
                    act = hook.activations[layer_idx]
                    # Mean pool over sequence dimension, convert bfloat16->float32 for numpy
                    act_mean = act.float().mean(dim=1).squeeze().numpy()
                    all_activations[layer_idx].append(act_mean)

            all_genders.append(gender)
            all_indices.append(idx)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Convert to numpy arrays
    for layer_idx in all_activations:
        if all_activations[layer_idx]:
            all_activations[layer_idx] = np.stack(all_activations[layer_idx])
        else:
            all_activations[layer_idx] = np.array([])

    return all_activations, all_genders, all_indices


def save_checkpoint(
    activations: Dict[int, np.ndarray],
    genders: List[str],
    indices: List[int],
    output_dir: Path,
    language: str,
    checkpoint_num: int
):
    """Save extraction checkpoint."""
    checkpoint_dir = output_dir / "layer_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save activations for each layer
    for layer_idx, acts in activations.items():
        if len(acts) > 0:
            filename = checkpoint_dir / f"llama32vision_{language}_layer{layer_idx}_checkpoint{checkpoint_num}.npz"
            np.savez_compressed(
                filename,
                activations=acts,
                genders=np.array(genders),
                indices=np.array(indices)
            )
            print(f"  Saved {filename.name}: {acts.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract Llama 3.2 Vision activations")
    parser.add_argument("--language", type=str, required=True, choices=["arabic", "english"])
    parser.add_argument("--layers", type=str, default="0,5,10,15,20,25,30,35,39",
                        help="Comma-separated list of layers to extract")
    parser.add_argument("--data_file", type=str, default="data/processed/samples.csv")
    parser.add_argument("--images_dir", type=str, default="data/raw/images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llama32vision")
    parser.add_argument("--batch_size", type=int, default=50, help="Samples per checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true", default=True, help="Log to W&B (enabled by default)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llama32vision-sae-analysis")
    parser.add_argument("--wandb_entity", type=str, default="nourmubarak")
    args = parser.parse_args()

    # Handle wandb flag
    if args.no_wandb:
        args.wandb = False

    # Parse layers
    layers = [int(l.strip()) for l in args.layers.split(",")]

    # Initialize W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"llama32vision_extract_{args.language}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": MODEL_NAME,
                "model_id": MODEL_ID,
                "language": args.language,
                "layers": layers,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "data_file": args.data_file,
                "max_samples": args.max_samples,
                "device": args.device
            },
            tags=["extraction", args.language, "llama32vision"]
        )
        print(f"W&B initialized: {wandb.run.url}")

    print("=" * 60)
    print(f"Llama 3.2 Vision Activation Extraction")
    print(f"Language: {args.language}")
    print(f"Layers: {layers}")
    print("=" * 60)

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    # Load data
    print(f"\nLoading data from {args.data_file}...")
    data = pd.read_csv(args.data_file)
    print(f"Loaded {len(data)} samples")

    if args.max_samples:
        data = data.head(args.max_samples)
        print(f"Limited to {len(data)} samples")

    # Load model
    model, processor = load_llama32_vision_model(device=args.device)

    # Setup hooks
    hook = Llama32VisionActivationHook(model, layers)
    hook.register_hooks()

    # Extract activations
    print(f"\nExtracting activations for {args.language}...")
    activations, genders, indices = extract_activations_batch(
        model, processor, hook, data, args.language, images_dir, args.device
    )

    # Save final checkpoint
    print(f"\nSaving final checkpoint...")
    save_checkpoint(activations, genders, indices, output_dir, args.language, 0)

    # Cleanup
    hook.remove_hooks()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    # Log to W&B
    if args.wandb:
        wandb.log({
            "total_samples": len(genders),
            "male_samples": genders.count("male"),
            "female_samples": genders.count("female"),
            "layers_extracted": len(layers)
        })
        wandb.finish()

    print(f"\n{'=' * 60}")
    print(f"Extraction Complete!")
    print(f"Total samples: {len(genders)}")
    print(f"Male: {genders.count('male')}, Female: {genders.count('female')}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
