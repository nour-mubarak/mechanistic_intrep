#!/usr/bin/env python3
"""
Random Ablation Control Experiment

This script runs a random-feature ablation control to prove that the
targeted gender ablation is causally specific, not just any ablation.

Compares:
1. Targeted ablation (100 gender-associated features) -> ~30% reduction
2. Random ablation (100 random features) -> expected minimal change

If random ablation shows much smaller reduction, this proves the causal
claim is specific to gender-encoding features.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Gender terms for detection
ENGLISH_GENDER_TERMS = {
    'male': ['man', 'men', 'boy', 'boys', 'he', 'him', 'his', 'male', 'gentleman', 
             'father', 'son', 'brother', 'husband', 'guy', 'guys'],
    'female': ['woman', 'women', 'girl', 'girls', 'she', 'her', 'hers', 'female', 
               'lady', 'ladies', 'mother', 'daughter', 'sister', 'wife']
}


def count_gender_terms(text):
    """Count gender terms in caption."""
    text_lower = text.lower()
    terms = ENGLISH_GENDER_TERMS
    
    male_count = sum(text_lower.count(' ' + term + ' ') + text_lower.count(' ' + term + '.') + 
                     text_lower.count(' ' + term + ',') + text_lower.startswith(term + ' ')
                     for term in terms['male'])
    female_count = sum(text_lower.count(' ' + term + ' ') + text_lower.count(' ' + term + '.') + 
                       text_lower.count(' ' + term + ',') + text_lower.startswith(term + ' ')
                       for term in terms['female'])
    
    # Simpler count - just check if term appears
    male_count = sum(1 for term in terms['male'] if term in text_lower)
    female_count = sum(1 for term in terms['female'] if term in text_lower)
    
    return {'male': male_count, 'female': female_count, 'total': male_count + female_count}


def has_gender_term(text):
    """Check if caption contains any gender term."""
    counts = count_gender_terms(text)
    return counts['total'] > 0


class SAEHook:
    """Hook to intercept and modify activations through SAE."""
    
    def __init__(self, sae, ablate_features=None, ablation_value=0.0):
        self.sae = sae
        self.ablate_features = ablate_features if ablate_features is not None else []
        self.ablation_value = ablation_value
        
    def __call__(self, module, input, output):
        """Forward hook that passes activations through SAE with optional ablation."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_acts = hidden_states.view(-1, hidden_dim)
        
        with torch.no_grad():
            pre_acts = flat_acts @ self.sae.encoder.weight.T + self.sae.encoder.bias
            sae_acts = torch.relu(pre_acts)
            
            if len(self.ablate_features) > 0:
                sae_acts[:, self.ablate_features] = self.ablation_value
            
            reconstructed = sae_acts @ self.sae.decoder.weight.T + self.sae.decoder.bias
        
        modified = reconstructed.view(batch_size, seq_len, hidden_dim)
        
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
    
    sae_path = Path(f'/home2/jmsk62/mechanistic_intrep/sae_captioning_project/checkpoints/saes/sae_english_layer_{layer}.pt')
    
    print(f"Loading SAE from {sae_path}...")
    sae_checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    
    d_model = sae_checkpoint['d_model']
    n_features = sae_checkpoint['d_hidden']
    
    print(f"SAE dimensions: d_model={d_model}, n_features={n_features}")
    
    class SAE(torch.nn.Module):
        def __init__(self, d_model, n_features):
            super().__init__()
            self.encoder = torch.nn.Linear(d_model, n_features)
            self.decoder = torch.nn.Linear(n_features, d_model)
    
    sae = SAE(d_model, n_features).to(device).to(torch.bfloat16)
    sae.load_state_dict(sae_checkpoint['model_state_dict'])
    sae.eval()
    
    return model, processor, sae, n_features


def load_top_gender_features(layer=9, k=100):
    """Load top-k gender-associated features from analysis results."""
    results_path = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/feature_interpretation/feature_interpretation_results.json')
    
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        
        for layer_data in data:
            if layer_data['layer'] == layer:
                male_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['male_associated'][:k//2]]
                female_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['female_associated'][:k//2]]
                return male_features + female_features
    
    print(f"Warning: Could not load top features for layer {layer}")
    return list(range(k))


def load_test_images(n_images=100):
    """Load test images for caption generation."""
    from PIL import Image
    
    data_dirs = [
        Path('/home2/jmsk62/project/mechanistic_intrep/sae_captioning_project/data/raw/images'),
        Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/data/raw/images'),
    ]
    
    images = []
    image_paths = []
    
    for img_dir in data_dirs:
        if img_dir.exists():
            img_files = list(img_dir.glob('*.jpg'))[:n_images]
            for img_path in img_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    image_paths.append(str(img_path))
                except:
                    continue
            if len(images) >= n_images:
                break
    
    return images[:n_images], image_paths[:n_images]


def generate_caption(model, processor, image, device='cuda'):
    """Generate caption for a single image."""
    inputs = processor(images=image, text="Caption:", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1
        )
    
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    if "Caption:" in caption:
        caption = caption.replace("Caption:", '').strip()
    
    return caption


def run_ablation_batch(model, processor, sae, images, image_paths, 
                       ablate_features, target_layer, device='cuda'):
    """Run ablation with given features and return stats."""
    stats = {'captions': [], 'with_gender': 0, 'male_terms': 0, 'female_terms': 0, 'total': 0}
    
    hook = SAEHook(sae, ablate_features=ablate_features)
    hook_handle = target_layer.register_forward_hook(hook)
    
    try:
        for image, path in tqdm(zip(images, image_paths), total=len(images)):
            caption = generate_caption(model, processor, image, device=device)
            stats['captions'].append({'image_path': path, 'caption': caption})
            
            if has_gender_term(caption):
                stats['with_gender'] += 1
            counts = count_gender_terms(caption)
            stats['male_terms'] += counts['male']
            stats['female_terms'] += counts['female']
            stats['total'] += 1
    finally:
        hook_handle.remove()
    
    return stats


def main():
    print("=" * 60)
    print("Random Ablation Control Experiment")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layer = 9
    k = 100  # Same number of features
    n_images = 100
    n_random_runs = 3  # Multiple random runs for robustness
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load model
    model, processor, sae, n_features = load_paligemma_and_sae(layer, device)
    target_layer = model.language_model.layers[layer]
    
    # Load images
    print(f"\nLoading test images...")
    images, image_paths = load_test_images(n_images)
    print(f"Loaded {len(images)} images")
    
    # Load targeted gender features
    targeted_features = load_top_gender_features(layer, k)[:k]
    print(f"\nTargeted gender features: {len(targeted_features)} features")
    print(f"Features: {targeted_features[:10]}...")
    
    results = {
        'config': {
            'model': 'PaLiGemma-3B',
            'layer': layer,
            'k': k,
            'n_images': n_images,
            'n_random_runs': n_random_runs,
            'n_features_total': n_features,
            'timestamp': datetime.now().isoformat()
        },
        'baseline': None,
        'targeted_ablation': None,
        'random_ablations': []
    }
    
    # 1. BASELINE (no ablation)
    print("\n" + "=" * 40)
    print("BASELINE (no ablation)")
    print("=" * 40)
    
    baseline_stats = {'captions': [], 'with_gender': 0, 'male_terms': 0, 'female_terms': 0, 'total': 0}
    for image, path in tqdm(zip(images, image_paths), total=len(images)):
        caption = generate_caption(model, processor, image, device=device)
        baseline_stats['captions'].append({'image_path': path, 'caption': caption})
        
        if has_gender_term(caption):
            baseline_stats['with_gender'] += 1
        counts = count_gender_terms(caption)
        baseline_stats['male_terms'] += counts['male']
        baseline_stats['female_terms'] += counts['female']
        baseline_stats['total'] += 1
    
    baseline_total = baseline_stats['male_terms'] + baseline_stats['female_terms']
    results['baseline'] = {
        'gender_term_count': baseline_total,
        'male_terms': baseline_stats['male_terms'],
        'female_terms': baseline_stats['female_terms'],
        'captions_with_gender': baseline_stats['with_gender'],
        'captions': baseline_stats['captions']
    }
    print(f"Baseline gender terms: {baseline_total}")
    
    # 2. TARGETED ABLATION (gender features)
    print("\n" + "=" * 40)
    print(f"TARGETED ABLATION ({k} gender features)")
    print("=" * 40)
    
    targeted_stats = run_ablation_batch(model, processor, sae, images, image_paths,
                                        targeted_features, target_layer, device)
    
    targeted_total = targeted_stats['male_terms'] + targeted_stats['female_terms']
    targeted_change = (targeted_total - baseline_total) / baseline_total * 100
    
    results['targeted_ablation'] = {
        'gender_term_count': targeted_total,
        'male_terms': targeted_stats['male_terms'],
        'female_terms': targeted_stats['female_terms'],
        'captions_with_gender': targeted_stats['with_gender'],
        'change_from_baseline': targeted_change,
        'features': targeted_features,
        'captions': targeted_stats['captions']
    }
    print(f"Targeted ablation gender terms: {targeted_total} ({targeted_change:+.1f}%)")
    
    # 3. RANDOM ABLATION (control)
    print("\n" + "=" * 40)
    print(f"RANDOM ABLATION CONTROL ({n_random_runs} runs, {k} random features each)")
    print("=" * 40)
    
    # Exclude gender features from random selection
    all_features = set(range(n_features))
    available_features = list(all_features - set(targeted_features))
    
    random_changes = []
    for run_idx in range(n_random_runs):
        print(f"\n--- Random run {run_idx + 1}/{n_random_runs} ---")
        
        # Select k random features (excluding gender features)
        random_features = random.sample(available_features, k)
        print(f"Random features: {random_features[:10]}...")
        
        random_stats = run_ablation_batch(model, processor, sae, images, image_paths,
                                          random_features, target_layer, device)
        
        random_total = random_stats['male_terms'] + random_stats['female_terms']
        random_change = (random_total - baseline_total) / baseline_total * 100
        random_changes.append(random_change)
        
        results['random_ablations'].append({
            'run_index': run_idx,
            'gender_term_count': random_total,
            'male_terms': random_stats['male_terms'],
            'female_terms': random_stats['female_terms'],
            'captions_with_gender': random_stats['with_gender'],
            'change_from_baseline': random_change,
            'features': random_features[:20],  # Save only first 20 for space
            'captions': random_stats['captions']
        })
        print(f"Random ablation gender terms: {random_total} ({random_change:+.1f}%)")
    
    # Compute summary
    avg_random_change = np.mean(random_changes)
    std_random_change = np.std(random_changes)
    
    results['summary'] = {
        'baseline_gender_terms': baseline_total,
        'targeted_change_pct': targeted_change,
        'random_change_mean_pct': avg_random_change,
        'random_change_std_pct': std_random_change,
        'effect_specificity': targeted_change - avg_random_change
    }
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<25} {'Gender Terms':>15} {'Change':>15}")
    print("-" * 55)
    print(f"{'Baseline':<25} {baseline_total:>15} {'-':>15}")
    print(f"{'Targeted ablation':<25} {targeted_total:>15} {targeted_change:>+14.1f}%")
    print(f"{'Random ablation (mean)':<25} {int(baseline_total * (1 + avg_random_change/100)):>15} {avg_random_change:>+14.1f}% (±{std_random_change:.1f}%)")
    print("-" * 55)
    print(f"\n✓ Targeted ablation effect: {targeted_change:.1f}%")
    print(f"✓ Random ablation effect: {avg_random_change:.1f}% (±{std_random_change:.1f}%)")
    print(f"✓ Effect specificity: {targeted_change - avg_random_change:.1f}% more reduction with targeted features")
    
    if abs(targeted_change) > abs(avg_random_change) + 2 * std_random_change:
        print(f"\n🎯 CAUSAL CLAIM VALIDATED: Targeted ablation has significantly larger effect")
    else:
        print(f"\n⚠️ Effect difference may not be significant")
    
    # Save results
    output_dir = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/intervention_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'random_ablation_control_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
