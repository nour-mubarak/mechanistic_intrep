#!/usr/bin/env python3
"""
Caption Generation with SAE Feature Ablation (Real Intervention Experiment)

This script performs actual causal intervention by:
1. Loading a VLM and trained SAE
2. Generating captions for test images (baseline)
3. Ablating top-k gender-associated SAE features during generation
4. Measuring changes in gendered language

Design:
- Model: PaLiGemma-3B (simplest, fastest)
- Layer: 9 (middle layer, good gender encoding)
- k: 50, 100 (top gender features)
- Metrics: Gender term frequency, profession-gender skew
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import re
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Gender terms for detection
ENGLISH_GENDER_TERMS = {
    'male': ['man', 'men', 'boy', 'boys', 'he', 'him', 'his', 'male', 'gentleman', 
             'father', 'son', 'brother', 'husband', 'guy', 'guys'],
    'female': ['woman', 'women', 'girl', 'girls', 'she', 'her', 'hers', 'female', 
               'lady', 'ladies', 'mother', 'daughter', 'sister', 'wife']
}

ARABIC_GENDER_TERMS = {
    'male': ['رجل', 'صبي', 'ولد', 'هو', 'أب', 'ابن', 'أخ', 'زوج', 'شاب', 'سيد'],
    'female': ['امرأة', 'فتاة', 'بنت', 'هي', 'أم', 'ابنة', 'أخت', 'زوجة', 'سيدة']
}

# Professions for stereotype analysis
PROFESSIONS = {
    'male_stereotyped': ['engineer', 'doctor', 'pilot', 'mechanic', 'soldier', 'firefighter'],
    'female_stereotyped': ['nurse', 'teacher', 'secretary', 'maid', 'nanny', 'receptionist'],
    'neutral': ['person', 'worker', 'employee', 'professional', 'student', 'athlete']
}


def count_gender_terms(text, language='english'):
    """Count gender terms in caption."""
    text_lower = text.lower()
    terms = ENGLISH_GENDER_TERMS if language == 'english' else ARABIC_GENDER_TERMS
    
    male_count = sum(1 for term in terms['male'] if term in text_lower)
    female_count = sum(1 for term in terms['female'] if term in text_lower)
    
    return {'male': male_count, 'female': female_count, 'total': male_count + female_count}


def has_gender_term(text, language='english'):
    """Check if caption contains any gender term."""
    counts = count_gender_terms(text, language)
    return counts['total'] > 0


class SAEHook:
    """Hook to intercept and modify activations through SAE."""
    
    def __init__(self, sae, ablate_features=None, ablation_value=0.0):
        self.sae = sae
        self.ablate_features = ablate_features if ablate_features is not None else []
        self.ablation_value = ablation_value
        self.original_acts = None
        self.modified_acts = None
        
    def __call__(self, module, input, output):
        """Forward hook that passes activations through SAE with optional ablation."""
        # Get hidden states (handle different output formats)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        self.original_acts = hidden_states.clone()
        
        # Get shape info
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Flatten for SAE
        flat_acts = hidden_states.view(-1, hidden_dim)
        
        # Encode through SAE
        with torch.no_grad():
            # SAE forward: x -> encoder -> relu -> decoder
            pre_acts = flat_acts @ self.sae.encoder.weight.T + self.sae.encoder.bias
            sae_acts = torch.relu(pre_acts)
            
            # Ablate specified features
            if len(self.ablate_features) > 0:
                sae_acts[:, self.ablate_features] = self.ablation_value
            
            # Decode back
            reconstructed = sae_acts @ self.sae.decoder.weight.T + self.sae.decoder.bias
        
        # Reshape back
        modified = reconstructed.view(batch_size, seq_len, hidden_dim)
        self.modified_acts = modified
        
        # Return modified output in same format
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


def load_paligemma_and_sae(layer=9, device='cuda'):
    """Load PaLiGemma model and trained SAE."""
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    
    print("Loading PaLiGemma-3B...")
    model_id = "google/paligemma-3b-mix-224"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    
    # Load SAE - use English SAE as default
    sae_path = Path(f'/home2/jmsk62/mechanistic_intrep/sae_captioning_project/checkpoints/saes/sae_english_layer_{layer}.pt')
    
    print(f"Loading SAE from {sae_path}...")
    sae_checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    
    # Get dimensions from checkpoint
    d_model = sae_checkpoint['d_model']
    n_features = sae_checkpoint['d_hidden']
    
    print(f"SAE dimensions: d_model={d_model}, n_features={n_features}")
    
    # Create SAE module
    class SAE(torch.nn.Module):
        def __init__(self, d_model, n_features):
            super().__init__()
            self.encoder = torch.nn.Linear(d_model, n_features)
            self.decoder = torch.nn.Linear(n_features, d_model)
    
    sae = SAE(d_model, n_features).to(device).to(torch.bfloat16)
    sae.load_state_dict(sae_checkpoint['model_state_dict'])
    sae.eval()
    
    return model, processor, sae


def load_top_gender_features(layer=9, k=100):
    """Load top-k gender-associated features from analysis results."""
    # Try to load from feature interpretation results
    results_path = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/feature_interpretation/feature_interpretation_results.json')
    
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        
        # Find the layer
        for layer_data in data:
            if layer_data['layer'] == layer:
                male_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['male_associated'][:k//2]]
                female_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['female_associated'][:k//2]]
                return male_features + female_features
    
    # Fallback: load from cross-lingual results
    cross_path = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/proper_cross_lingual/cross_lingual_results.json')
    if cross_path.exists():
        with open(cross_path) as f:
            data = json.load(f)
        
        layer_key = str(layer)
        if layer_key in data:
            layer_data = data[layer_key]
            # Get top features from effect size
            if 'arabic' in layer_data and 'top_features' in layer_data['arabic']:
                features = layer_data['arabic']['top_features']
                if isinstance(features, dict):
                    male = features.get('male_associated', [])[:k//2]
                    female = features.get('female_associated', [])[:k//2]
                    return [f['feature_id'] if isinstance(f, dict) else f for f in male + female]
    
    print(f"Warning: Could not load top features for layer {layer}, using random features")
    return list(range(k))


def load_test_images(n_images=100):
    """Load test images for caption generation."""
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Load from local dataset
    data_dirs = [
        Path('/home2/jmsk62/project/mechanistic_intrep/sae_captioning_project/data/raw/images'),
        Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/data/raw/images'),
        Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/data/processed/images'),
    ]
    
    images = []
    image_paths = []
    
    # Try to find images
    for img_dir in data_dirs:
        if img_dir.exists():
            img_files = list(img_dir.glob('*.jpg'))[:n_images]
            for img_path in img_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    image_paths.append(str(img_path))
                except Exception as e:
                    continue
            if len(images) >= n_images:
                break
    
    if len(images) == 0:
        print("Warning: No local images found, using placeholder")
        # Create simple test images
        for i in range(min(10, n_images)):
            img = Image.new('RGB', (224, 224), color=(100 + i*10, 100 + i*10, 100 + i*10))
            images.append(img)
            image_paths.append(f"placeholder_{i}")
    
    return images[:n_images], image_paths[:n_images]


def generate_caption(model, processor, image, prompt="Describe this image:", 
                    max_new_tokens=50, device='cuda'):
    """Generate caption for a single image."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
    
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from caption
    if prompt in caption:
        caption = caption.replace(prompt, '').strip()
    
    return caption


def run_intervention_experiment(
    model, processor, sae, images, image_paths,
    layer=9, k_values=[50, 100], device='cuda'
):
    """Run the full intervention experiment."""
    results = {
        'config': {
            'model': 'PaLiGemma-3B',
            'layer': layer,
            'k_values': k_values,
            'n_images': len(images),
            'timestamp': datetime.now().isoformat()
        },
        'baseline': {
            'captions': [],
            'gender_stats': {'with_gender': 0, 'total': 0, 'male_terms': 0, 'female_terms': 0}
        },
        'ablations': {}
    }
    
    # Get the target layer for hooking
    # PaLiGemma uses language_model.layers[layer] directly
    target_layer = model.language_model.layers[layer]
    
    print(f"\n=== Generating Baseline Captions ===")
    for i, (image, path) in enumerate(tqdm(zip(images, image_paths), total=len(images))):
        caption = generate_caption(model, processor, image, device=device)
        results['baseline']['captions'].append({
            'image_path': path,
            'caption': caption
        })
        
        # Count gender terms
        if has_gender_term(caption):
            results['baseline']['gender_stats']['with_gender'] += 1
        counts = count_gender_terms(caption)
        results['baseline']['gender_stats']['male_terms'] += counts['male']
        results['baseline']['gender_stats']['female_terms'] += counts['female']
        results['baseline']['gender_stats']['total'] += 1
    
    baseline_gender_pct = results['baseline']['gender_stats']['with_gender'] / results['baseline']['gender_stats']['total'] * 100
    print(f"Baseline: {baseline_gender_pct:.1f}% captions with gender terms")
    
    # Run ablations for each k
    for k in k_values:
        print(f"\n=== Ablating Top {k} Gender Features ===")
        
        # Load top features
        ablate_features = load_top_gender_features(layer, k)[:k]
        print(f"Ablating features: {ablate_features[:10]}... ({len(ablate_features)} total)")
        
        results['ablations'][f'k={k}'] = {
            'captions': [],
            'ablated_features': ablate_features,
            'gender_stats': {'with_gender': 0, 'total': 0, 'male_terms': 0, 'female_terms': 0}
        }
        
        # Register hook for ablation
        hook = SAEHook(sae, ablate_features=ablate_features)
        hook_handle = target_layer.register_forward_hook(hook)
        
        try:
            for i, (image, path) in enumerate(tqdm(zip(images, image_paths), total=len(images))):
                caption = generate_caption(model, processor, image, device=device)
                results['ablations'][f'k={k}']['captions'].append({
                    'image_path': path,
                    'caption': caption
                })
                
                # Count gender terms
                if has_gender_term(caption):
                    results['ablations'][f'k={k}']['gender_stats']['with_gender'] += 1
                counts = count_gender_terms(caption)
                results['ablations'][f'k={k}']['gender_stats']['male_terms'] += counts['male']
                results['ablations'][f'k={k}']['gender_stats']['female_terms'] += counts['female']
                results['ablations'][f'k={k}']['gender_stats']['total'] += 1
        finally:
            hook_handle.remove()
        
        ablated_gender_pct = results['ablations'][f'k={k}']['gender_stats']['with_gender'] / results['ablations'][f'k={k}']['gender_stats']['total'] * 100
        print(f"After ablation (k={k}): {ablated_gender_pct:.1f}% captions with gender terms")
        print(f"Change: {ablated_gender_pct - baseline_gender_pct:+.1f}%")
    
    return results


def compute_summary_metrics(results):
    """Compute summary metrics from results."""
    summary = {
        'baseline': {
            'gender_term_pct': results['baseline']['gender_stats']['with_gender'] / results['baseline']['gender_stats']['total'] * 100,
            'male_terms_per_caption': results['baseline']['gender_stats']['male_terms'] / results['baseline']['gender_stats']['total'],
            'female_terms_per_caption': results['baseline']['gender_stats']['female_terms'] / results['baseline']['gender_stats']['total'],
        },
        'ablations': {}
    }
    
    for k_key, abl_data in results['ablations'].items():
        total = abl_data['gender_stats']['total']
        summary['ablations'][k_key] = {
            'gender_term_pct': abl_data['gender_stats']['with_gender'] / total * 100,
            'male_terms_per_caption': abl_data['gender_stats']['male_terms'] / total,
            'female_terms_per_caption': abl_data['gender_stats']['female_terms'] / total,
            'change_from_baseline': (abl_data['gender_stats']['with_gender'] / total * 100) - summary['baseline']['gender_term_pct']
        }
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Caption Generation Intervention Experiment')
    parser.add_argument('--layer', type=int, default=9, help='Layer to intervene on')
    parser.add_argument('--n_images', type=int, default=100, help='Number of test images')
    parser.add_argument('--k_values', type=int, nargs='+', default=[50, 100], help='k values for ablation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, 
                       default='/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/intervention_experiment',
                       help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Caption Generation Intervention Experiment")
    print("=" * 60)
    print(f"Model: PaLiGemma-3B")
    print(f"Layer: {args.layer}")
    print(f"k values: {args.k_values}")
    print(f"N images: {args.n_images}")
    print("=" * 60)
    
    # Load model and SAE
    model, processor, sae = load_paligemma_and_sae(layer=args.layer, device=args.device)
    
    # Load test images
    print("\nLoading test images...")
    images, image_paths = load_test_images(args.n_images)
    print(f"Loaded {len(images)} images")
    
    # Run experiment
    results = run_intervention_experiment(
        model, processor, sae, images, image_paths,
        layer=args.layer, k_values=args.k_values, device=args.device
    )
    
    # Compute summary
    summary = compute_summary_metrics(results)
    results['summary'] = summary
    
    # Save results
    results_path = output_dir / 'intervention_results.json'
    
    # Convert non-serializable items
    results_serializable = json.loads(json.dumps(results, default=str))
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nBaseline:")
    print(f"  Gender term %: {summary['baseline']['gender_term_pct']:.1f}%")
    print(f"  Male terms/caption: {summary['baseline']['male_terms_per_caption']:.2f}")
    print(f"  Female terms/caption: {summary['baseline']['female_terms_per_caption']:.2f}")
    
    for k_key, abl_summary in summary['ablations'].items():
        print(f"\n{k_key} ablation:")
        print(f"  Gender term %: {abl_summary['gender_term_pct']:.1f}%")
        print(f"  Change from baseline: {abl_summary['change_from_baseline']:+.1f}%")
    
    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
