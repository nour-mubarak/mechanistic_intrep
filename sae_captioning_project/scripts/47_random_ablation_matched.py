#!/usr/bin/env python3
"""
Random Ablation Control - Using Original Baseline

This script:
1. Loads the ORIGINAL baseline captions from intervention_results.json
2. Runs ONLY random ablation (no new baseline generation)
3. Compares random ablation to the original targeted ablation results

This ensures apples-to-apples comparison.
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

ENGLISH_GENDER_TERMS = {
    'male': ['man', 'men', 'boy', 'boys', 'he', 'him', 'his', 'male', 'gentleman', 
             'father', 'son', 'brother', 'husband', 'guy', 'guys'],
    'female': ['woman', 'women', 'girl', 'girls', 'she', 'her', 'hers', 'female', 
               'lady', 'ladies', 'mother', 'daughter', 'sister', 'wife']
}


def count_gender_terms_matching_original(captions):
    """
    Count gender terms matching the ORIGINAL analysis method.
    Returns dict with count per term (how many captions contain each term).
    """
    term_counts = {}
    all_terms = ENGLISH_GENDER_TERMS['male'] + ENGLISH_GENDER_TERMS['female']
    
    for term in all_terms:
        count = sum(1 for c in captions if term in c['caption'].lower())
        if count > 0:
            term_counts[term] = count
    
    return term_counts


def count_total_terms(captions):
    """Sum of all term-caption pairs."""
    counts = count_gender_terms_matching_original(captions)
    return sum(counts.values())


class SAEHook:
    def __init__(self, sae, ablate_features=None):
        self.sae = sae
        self.ablate_features = ablate_features if ablate_features is not None else []
        
    def __call__(self, module, input, output):
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
                sae_acts[:, self.ablate_features] = 0.0
            
            reconstructed = sae_acts @ self.sae.decoder.weight.T + self.sae.decoder.bias
        
        modified = reconstructed.view(batch_size, seq_len, hidden_dim)
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


def load_paligemma_and_sae(layer=9, device='cuda'):
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
    results_path = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/feature_interpretation/feature_interpretation_results.json')
    
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        
        for layer_data in data:
            if layer_data['layer'] == layer:
                male_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['male_associated'][:k//2]]
                female_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['female_associated'][:k//2]]
                return male_features + female_features
    
    return list(range(k))


def generate_caption(model, processor, image, device='cuda'):
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


def main():
    print("=" * 60)
    print("Random Ablation Control (Using Original Baseline)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layer = 9
    k = 100
    n_random_runs = 3
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load ORIGINAL results
    orig_path = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/intervention_experiment/intervention_results.json')
    print(f"\nLoading original results from {orig_path}...")
    with open(orig_path) as f:
        orig_results = json.load(f)
    
    # Get original baseline and targeted counts
    orig_baseline_captions = orig_results['baseline']['captions']
    
    # Find the k=100 or k=200 ablation (they should be same since we only have 100 features)
    orig_targeted_captions = None
    for key in ['k=100', 'k=200', 'k=50']:
        if key in orig_results['ablations']:
            orig_targeted_captions = orig_results['ablations'][key]['captions']
            print(f"Using original {key} ablation as targeted reference")
            break
    
    if orig_targeted_captions is None:
        print("ERROR: Could not find targeted ablation in original results")
        return
    
    # Count using original method
    baseline_total = count_total_terms(orig_baseline_captions)
    targeted_total = count_total_terms(orig_targeted_captions)
    targeted_change = (targeted_total - baseline_total) / baseline_total * 100
    
    print(f"\n=== ORIGINAL RESULTS ===")
    print(f"Baseline: {baseline_total} term-caption pairs")
    print(f"Targeted ablation: {targeted_total} ({targeted_change:+.1f}%)")
    
    # Load model and SAE for random ablation
    model, processor, sae, n_features = load_paligemma_and_sae(layer, device)
    target_layer = model.language_model.layers[layer]
    
    # Load images matching the original experiment
    from PIL import Image
    images = []
    image_paths = []
    for cap_data in orig_baseline_captions:
        img_path = cap_data['image_path']
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    
    print(f"\nLoaded {len(images)} images matching original experiment")
    
    # Get targeted features (to exclude from random selection)
    targeted_features = load_top_gender_features(layer, k)[:k]
    available_features = list(set(range(n_features)) - set(targeted_features))
    
    # Run random ablations
    print(f"\n=== RANDOM ABLATION CONTROL ({n_random_runs} runs) ===")
    
    random_results = []
    
    for run_idx in range(n_random_runs):
        print(f"\n--- Random run {run_idx + 1}/{n_random_runs} ---")
        
        # Select random features
        random_features = random.sample(available_features, k)
        print(f"Random features: {random_features[:5]}... ({len(random_features)} total)")
        
        # Generate captions with random ablation
        hook = SAEHook(sae, ablate_features=random_features)
        hook_handle = target_layer.register_forward_hook(hook)
        
        random_captions = []
        try:
            for image, path in tqdm(zip(images, image_paths), total=len(images)):
                caption = generate_caption(model, processor, image, device=device)
                random_captions.append({'image_path': path, 'caption': caption})
        finally:
            hook_handle.remove()
        
        random_total = count_total_terms(random_captions)
        random_change = (random_total - baseline_total) / baseline_total * 100
        
        print(f"Random ablation: {random_total} ({random_change:+.1f}%)")
        
        random_results.append({
            'run_index': run_idx,
            'total_terms': random_total,
            'change_pct': random_change,
            'features': random_features[:20],
            'captions': random_captions
        })
    
    # Compute summary
    avg_random_change = np.mean([r['change_pct'] for r in random_results])
    std_random_change = np.std([r['change_pct'] for r in random_results])
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (Same Baseline)")
    print("=" * 60)
    print(f"{'Condition':<25} {'Terms':>10} {'Change':>15}")
    print("-" * 50)
    print(f"{'Baseline (original)':<25} {baseline_total:>10} {'-':>15}")
    print(f"{'Targeted (original)':<25} {targeted_total:>10} {targeted_change:>+14.1f}%")
    print(f"{'Random (mean of 3)':<25} {int(np.mean([r['total_terms'] for r in random_results])):>10} {avg_random_change:>+14.1f}% (±{std_random_change:.1f}%)")
    print("-" * 50)
    
    effect_diff = targeted_change - avg_random_change
    print(f"\n✓ Targeted ablation: {targeted_change:.1f}%")
    print(f"✓ Random ablation: {avg_random_change:.1f}% (±{std_random_change:.1f}%)")
    print(f"✓ Effect specificity: {effect_diff:.1f}% more reduction with targeted features")
    
    if abs(targeted_change) > abs(avg_random_change) + 2 * std_random_change:
        print(f"\n🎯 CAUSAL CLAIM VALIDATED: Targeted >> Random")
    
    # Save results
    output = {
        'config': {
            'description': 'Random ablation control using original baseline',
            'original_results_file': str(orig_path),
            'layer': layer,
            'k': k,
            'n_random_runs': n_random_runs,
            'timestamp': datetime.now().isoformat()
        },
        'original_baseline': {
            'total_terms': baseline_total,
            'term_counts': count_gender_terms_matching_original(orig_baseline_captions)
        },
        'original_targeted': {
            'total_terms': targeted_total,
            'change_pct': targeted_change,
            'term_counts': count_gender_terms_matching_original(orig_targeted_captions)
        },
        'random_ablations': random_results,
        'summary': {
            'baseline_terms': baseline_total,
            'targeted_change_pct': targeted_change,
            'random_change_mean_pct': avg_random_change,
            'random_change_std_pct': std_random_change,
            'effect_specificity': effect_diff
        }
    }
    
    output_path = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/intervention_experiment/random_ablation_matched_baseline.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
