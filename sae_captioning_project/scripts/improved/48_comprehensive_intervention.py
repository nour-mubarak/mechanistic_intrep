#!/usr/bin/env python3
"""
48_comprehensive_intervention.py — Addresses ALL reviewer feedback
===================================================================

Improvements over scripts 45-47:
  1. 500+ images (up from 100)     — --n_images 500
  2. Multi-model support            — --model paligemma/qwen2vl/llama32vision  
  3. Multi-layer ablation           — --layers 9 17 (or 9 17 combined)
  4. 25 random runs (up from 3)     — --n_random_runs 25
  5. Length normalization            — gender_terms / total_tokens reported
  6. Per-image paired statistics     — bootstrap CI for Δ_targeted vs Δ_random
  7. Non-binary term tracking       — logged alongside binary counts

Usage:
  # PaLiGemma, Layer 9, 500 images, 25 random runs
  python scripts/improved/48_comprehensive_intervention.py \
      --model paligemma --layers 9 --n_images 500 --n_random_runs 25

  # PaLiGemma, Multi-layer (L9 + L17)  
  python scripts/improved/48_comprehensive_intervention.py \
      --model paligemma --layers 9 17 --n_images 500 --n_random_runs 25

  # Qwen2-VL, Layer 12 (equivalent middle layer for 28-layer model)
  python scripts/improved/48_comprehensive_intervention.py \
      --model qwen2vl --layers 12 --n_images 500 --n_random_runs 25

  # Llama-3.2-Vision, Layer 20 (equivalent middle layer for 40-layer model)
  python scripts/improved/48_comprehensive_intervention.py \
      --model llama32vision --layers 20 --n_images 500 --n_random_runs 25
"""

import os
import sys
import json
import torch
import numpy as np
import random
import argparse
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from shared_utils import (
    count_gender_terms, count_gender_terms_detailed, count_nonbinary_terms,
    has_gender_term, compute_per_image_deltas, bootstrap_ci,
    compute_paired_statistics, compute_aggregate_stats,
    SAEHook, load_model_and_sae, load_top_gender_features,
    load_test_images, save_results
)


def run_baseline(generate_fn, images, image_paths):
    """Generate baseline captions (no ablation)."""
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE (no ablation)")
    print("=" * 60)
    
    captions = []
    for image, path in tqdm(zip(images, image_paths), total=len(images), desc="Baseline"):
        caption = generate_fn(image)
        captions.append({'image_path': path, 'caption': caption})
    
    stats = compute_aggregate_stats(captions)
    print(f"  Baseline: {stats['total_gender_terms']} gender terms in {stats['n_images']} images")
    print(f"  Gender rate: {stats['gender_rate']:.4f} (terms/token)")
    print(f"  Nonbinary terms: {stats['nonbinary_terms']}")
    
    return captions, stats


def run_targeted_ablation(generate_fn, sae, target_layer, images, image_paths,
                          ablate_features, label="targeted"):
    """Run ablation with specified features."""
    print(f"\n--- {label}: ablating {len(ablate_features)} features ---")
    
    hook = SAEHook(sae, ablate_features=ablate_features)
    hook_handle = target_layer.register_forward_hook(hook)
    
    captions = []
    try:
        for image, path in tqdm(zip(images, image_paths), total=len(images), desc=label):
            caption = generate_fn(image)
            captions.append({'image_path': path, 'caption': caption})
    finally:
        hook_handle.remove()
    
    stats = compute_aggregate_stats(captions)
    print(f"  {label}: {stats['total_gender_terms']} gender terms, rate={stats['gender_rate']:.4f}")
    
    return captions, stats


def run_multi_layer_ablation(model, generate_fn, sae_dict, layer_dict, images, image_paths,
                             features_dict, label="multi-layer"):
    """Run ablation across multiple layers simultaneously.
    
    Args:
        sae_dict: {layer: sae_module}
        layer_dict: {layer: target_layer_module}
        features_dict: {layer: [feature_ids]}
    """
    print(f"\n--- {label}: ablating across {len(sae_dict)} layers ---")
    
    hooks = []
    for layer_idx in sae_dict:
        sae = sae_dict[layer_idx]
        target_layer = layer_dict[layer_idx]
        features = features_dict[layer_idx]
        hook = SAEHook(sae, ablate_features=features)
        handle = target_layer.register_forward_hook(hook)
        hooks.append(handle)
        print(f"  Layer {layer_idx}: ablating {len(features)} features")
    
    captions = []
    try:
        for image, path in tqdm(zip(images, image_paths), total=len(images), desc=label):
            caption = generate_fn(image)
            captions.append({'image_path': path, 'caption': caption})
    finally:
        for h in hooks:
            h.remove()
    
    stats = compute_aggregate_stats(captions)
    print(f"  {label}: {stats['total_gender_terms']} gender terms, rate={stats['gender_rate']:.4f}")
    
    return captions, stats


def run_random_ablation_runs(generate_fn, sae, target_layer, images, image_paths,
                             n_features, targeted_features, k, n_runs=25):
    """Run n_runs of random ablation (excluding targeted features)."""
    print(f"\n" + "=" * 60)
    print(f"PHASE 3: RANDOM ABLATION CONTROL ({n_runs} runs, k={k})")
    print("=" * 60)
    
    available_features = list(set(range(n_features)) - set(targeted_features))
    
    all_random_captions = []
    all_random_stats = []
    random_changes = []
    
    for run_idx in range(n_runs):
        # Different random seed per run, but reproducible
        rng = random.Random(42 + run_idx * 1000)
        random_features = rng.sample(available_features, k)
        
        hook = SAEHook(sae, ablate_features=random_features)
        hook_handle = target_layer.register_forward_hook(hook)
        
        captions = []
        try:
            for image, path in tqdm(zip(images, image_paths), total=len(images),
                                     desc=f"Random {run_idx+1}/{n_runs}"):
                caption = generate_fn(image)
                captions.append({'image_path': path, 'caption': caption})
        finally:
            hook_handle.remove()
        
        stats = compute_aggregate_stats(captions)
        all_random_captions.append(captions)
        all_random_stats.append(stats)
        
        print(f"  Run {run_idx+1}: {stats['total_gender_terms']} gender terms, "
              f"rate={stats['gender_rate']:.4f}")
    
    return all_random_captions, all_random_stats


def compute_summary(baseline_stats, targeted_stats, random_stats_list,
                    baseline_caps, targeted_caps, random_caps_list):
    """Compute comprehensive summary with all improvements."""
    
    baseline_total = baseline_stats['total_gender_terms']
    targeted_total = targeted_stats['total_gender_terms']
    targeted_change = (targeted_total - baseline_total) / max(baseline_total, 1) * 100
    
    random_totals = [s['total_gender_terms'] for s in random_stats_list]
    random_changes = [(t - baseline_total) / max(baseline_total, 1) * 100 for t in random_totals]
    
    # Length-normalized rates
    baseline_rate = baseline_stats['gender_rate']
    targeted_rate = targeted_stats['gender_rate']
    random_rates = [s['gender_rate'] for s in random_stats_list]
    
    # Per-image paired statistics (the key improvement!)
    paired_stats = compute_paired_statistics(
        baseline_caps, targeted_caps, random_caps_list
    )
    
    summary = {
        'n_images': baseline_stats['n_images'],
        
        # Raw counts
        'baseline_gender_terms': baseline_total,
        'targeted_gender_terms': targeted_total,
        'targeted_change_pct': targeted_change,
        'random_change_mean_pct': float(np.mean(random_changes)),
        'random_change_std_pct': float(np.std(random_changes)),
        'random_changes_all': random_changes,
        'effect_specificity_pct': targeted_change - float(np.mean(random_changes)),
        'ratio_targeted_vs_random': abs(targeted_change) / max(abs(float(np.mean(random_changes))), 0.001),
        
        # Length-normalized rates (NEW)
        'baseline_gender_rate': baseline_rate,
        'targeted_gender_rate': targeted_rate,
        'targeted_rate_change_pct': (targeted_rate - baseline_rate) / max(baseline_rate, 1e-6) * 100,
        'random_rate_mean': float(np.mean(random_rates)),
        'random_rate_std': float(np.std(random_rates)),
        
        # Per-image paired statistics (NEW)
        'paired_statistics': paired_stats,
        
        # Non-binary tracking (NEW)
        'baseline_nonbinary_terms': baseline_stats['nonbinary_terms'],
        'targeted_nonbinary_terms': targeted_stats['nonbinary_terms'],
        
        # Per-term breakdown
        'baseline_per_term': baseline_stats['per_term_counts'],
        'targeted_per_term': targeted_stats['per_term_counts'],
    }
    
    return summary


def print_final_summary(summary, config):
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nExperiment: {config['model']} | Layers: {config['layers']} | "
          f"k={config['k']} | {summary['n_images']} images | {config['n_random_runs']} random runs")
    
    print(f"\n{'Metric':<40} {'Value':>20}")
    print("-" * 60)
    print(f"{'Baseline gender terms':<40} {summary['baseline_gender_terms']:>20}")
    print(f"{'Targeted ablation terms':<40} {summary['targeted_gender_terms']:>20}")
    print(f"{'Targeted change':<40} {summary['targeted_change_pct']:>+19.1f}%")
    print(f"{'Random ablation change (mean±SD)':<40} {summary['random_change_mean_pct']:>+10.1f}% ± {summary['random_change_std_pct']:.1f}%")
    print(f"{'Effect specificity':<40} {summary['effect_specificity_pct']:>+19.1f}%")
    print(f"{'Targeted/Random ratio':<40} {summary['ratio_targeted_vs_random']:>19.1f}×")
    
    print(f"\n--- Length-Normalized Rates (NEW) ---")
    print(f"{'Baseline rate (terms/token)':<40} {summary['baseline_gender_rate']:>20.4f}")
    print(f"{'Targeted rate (terms/token)':<40} {summary['targeted_gender_rate']:>20.4f}")
    print(f"{'Rate change':<40} {summary['targeted_rate_change_pct']:>+19.1f}%")
    print(f"{'Random rate (mean±SD)':<40} {summary['random_rate_mean']:>10.4f} ± {summary['random_rate_std']:.4f}")
    
    ps = summary['paired_statistics']
    print(f"\n--- Per-Image Paired Statistics (NEW) ---")
    print(f"{'Targeted Δ [95% CI]':<40} {ps['targeted_delta']['mean']:>+6.3f} [{ps['targeted_delta']['ci_95'][0]:+.3f}, {ps['targeted_delta']['ci_95'][1]:+.3f}]")
    print(f"{'Random Δ [95% CI]':<40} {ps['random_delta']['mean']:>+6.3f} [{ps['random_delta']['ci_95'][0]:+.3f}, {ps['random_delta']['ci_95'][1]:+.3f}]")
    print(f"{'Difference Δ [95% CI]':<40} {ps['difference_targeted_minus_random']['mean']:>+6.3f} [{ps['difference_targeted_minus_random']['ci_95'][0]:+.3f}, {ps['difference_targeted_minus_random']['ci_95'][1]:+.3f}]")
    print(f"{'Wilcoxon p-value':<40} {ps['wilcoxon_test']['p_value']:>20.6f}")
    
    print(f"\n--- Non-Binary Term Tracking (NEW) ---")
    print(f"{'Baseline non-binary terms':<40} {summary['baseline_nonbinary_terms']:>20}")
    print(f"{'Targeted non-binary terms':<40} {summary['targeted_nonbinary_terms']:>20}")
    
    # Validation
    print(f"\n--- Validation ---")
    n_runs = len(summary.get('random_changes_all', []))
    if abs(summary['targeted_change_pct']) > abs(summary['random_change_mean_pct']) + 2 * summary['random_change_std_pct']:
        print(f"🎯 CAUSAL CLAIM VALIDATED: Targeted >> Random (outside 2σ, {n_runs} runs)")
    else:
        print(f"⚠️  Effect may not be significant at 2σ threshold")
    
    if ps['wilcoxon_test']['p_value'] < 0.05:
        print(f"🎯 PAIRED TEST SIGNIFICANT: Wilcoxon p={ps['wilcoxon_test']['p_value']:.2e}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Intervention Experiment (All Reviewer Feedback)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard PaLiGemma experiment (replicates + improves original)
  python 48_comprehensive_intervention.py --model paligemma --layers 9 --n_images 500 --n_random_runs 25

  # Multi-layer ablation
  python 48_comprehensive_intervention.py --model paligemma --layers 9 17 --n_images 500 --n_random_runs 25 --multi_layer

  # Cross-model replication on Qwen2-VL
  python 48_comprehensive_intervention.py --model qwen2vl --layers 12 --n_images 500 --n_random_runs 25
        """
    )
    parser.add_argument('--model', type=str, default='paligemma',
                       choices=['paligemma', 'qwen2vl', 'llama32vision'],
                       help='Model to use for intervention')
    parser.add_argument('--layers', type=int, nargs='+', default=[9],
                       help='Layer(s) to intervene on')
    parser.add_argument('--multi_layer', action='store_true',
                       help='If multiple layers specified, ablate all simultaneously')
    parser.add_argument('--k', type=int, default=100,
                       help='Number of features to ablate')
    parser.add_argument('--n_images', type=int, default=500,
                       help='Number of test images (up from 100)')
    parser.add_argument('--n_random_runs', type=int, default=25,
                       help='Number of random ablation runs (up from 3)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results/improved_intervention/)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Output directory
    if args.output_dir is None:
        base = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/improved_intervention')
        layers_str = '_'.join(map(str, args.layers))
        args.output_dir = str(base / f'{args.model}_L{layers_str}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model': args.model,
        'layers': args.layers,
        'multi_layer': args.multi_layer,
        'k': args.k,
        'n_images': args.n_images,
        'n_random_runs': args.n_random_runs,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'improvements': [
            f'{args.n_images} images (up from 100)',
            f'{args.n_random_runs} random runs (up from 3)',
            'Length-normalized gender rate',
            'Per-image paired statistics with bootstrap CI',
            'Wilcoxon signed-rank test',
            'Non-binary term tracking',
            f'Model: {args.model}',
            f'Layers: {args.layers}',
        ]
    }
    
    print("=" * 70)
    print("COMPREHENSIVE INTERVENTION EXPERIMENT")
    print("All Reviewer Feedback Addressed")
    print("=" * 70)
    for imp in config['improvements']:
        print(f"  ✓ {imp}")
    print("=" * 70)
    
    # Load images
    images, image_paths = load_test_images(args.n_images, seed=args.seed)
    
    # Primary layer for single-layer experiments
    primary_layer = args.layers[0]
    
    # Load model + SAE for primary layer
    model, processor, sae, n_features, target_layer, generate_fn = load_model_and_sae(
        args.model, primary_layer, args.device
    )
    
    # Load targeted features
    targeted_features = load_top_gender_features(primary_layer, args.k, args.model)[:args.k]
    print(f"\nTargeted features for layer {primary_layer}: {len(targeted_features)} features")
    print(f"Feature examples: {targeted_features[:10]}...")
    
    # ---- PHASE 1: BASELINE ----
    baseline_caps, baseline_stats = run_baseline(generate_fn, images, image_paths)
    
    # ---- PHASE 2: TARGETED ABLATION ----
    print("\n" + "=" * 60)
    print("PHASE 2: TARGETED ABLATION")
    print("=" * 60)
    
    results = {
        'config': config,
        'baseline': {'captions': baseline_caps, 'stats': baseline_stats},
        'ablations': {}
    }
    
    # Single-layer ablation for each specified layer
    for layer_idx in args.layers:
        if layer_idx == primary_layer:
            targ_caps, targ_stats = run_targeted_ablation(
                generate_fn, sae, target_layer, images, image_paths,
                targeted_features, label=f"targeted_L{layer_idx}"
            )
        else:
            # Load SAE for this layer
            print(f"\nLoading SAE for layer {layer_idx}...")
            _, _, sae_other, n_feat_other, target_other, _ = load_model_and_sae(
                args.model, layer_idx, args.device
            )
            feat_other = load_top_gender_features(layer_idx, args.k, args.model)[:args.k]
            targ_caps, targ_stats = run_targeted_ablation(
                generate_fn, sae_other, target_other, images, image_paths,
                feat_other, label=f"targeted_L{layer_idx}"
            )
            del sae_other
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        results['ablations'][f'targeted_L{layer_idx}'] = {
            'captions': targ_caps, 'stats': targ_stats,
            'features': targeted_features if layer_idx == primary_layer else feat_other,
            'layer': layer_idx
        }
    
    # Multi-layer simultaneous ablation (if requested and multiple layers)
    if args.multi_layer and len(args.layers) > 1:
        print("\n--- MULTI-LAYER SIMULTANEOUS ABLATION ---")
        sae_dict = {}
        layer_dict = {}
        features_dict = {}
        
        for layer_idx in args.layers:
            if layer_idx == primary_layer:
                sae_dict[layer_idx] = sae
                layer_dict[layer_idx] = target_layer
                features_dict[layer_idx] = targeted_features
            else:
                _, _, sae_ml, _, target_ml, _ = load_model_and_sae(
                    args.model, layer_idx, args.device
                )
                feat_ml = load_top_gender_features(layer_idx, args.k, args.model)[:args.k]
                sae_dict[layer_idx] = sae_ml
                layer_dict[layer_idx] = target_ml
                features_dict[layer_idx] = feat_ml
        
        ml_caps, ml_stats = run_multi_layer_ablation(
            model, generate_fn, sae_dict, layer_dict, images, image_paths,
            features_dict, label="multi-layer"
        )
        layers_key = '+'.join(map(str, args.layers))
        results['ablations'][f'targeted_L{layers_key}_combined'] = {
            'captions': ml_caps, 'stats': ml_stats,
            'features_per_layer': {str(k): v for k, v in features_dict.items()},
            'layers': args.layers
        }
    
    # ---- PHASE 3: RANDOM ABLATION ----
    random_caps_list, random_stats_list = run_random_ablation_runs(
        generate_fn, sae, target_layer, images, image_paths,
        n_features, targeted_features, args.k, n_runs=args.n_random_runs
    )
    
    results['random_ablations'] = {
        'n_runs': args.n_random_runs,
        'per_run_stats': random_stats_list,
        # Don't save all captions for 25 runs to avoid huge files
        # Save only first 3 and last run captions for verification
        'sample_captions': {
            'run_0': random_caps_list[0] if len(random_caps_list) > 0 else [],
            'run_1': random_caps_list[1] if len(random_caps_list) > 1 else [],
            'run_last': random_caps_list[-1] if len(random_caps_list) > 0 else [],
        }
    }
    
    # ---- PHASE 4: COMPUTE SUMMARY ----
    print("\n" + "=" * 60)
    print("PHASE 4: COMPUTING COMPREHENSIVE STATISTICS")
    print("=" * 60)
    
    # Use primary layer targeted ablation for summary
    primary_targ_key = f'targeted_L{primary_layer}'
    primary_targ_caps = results['ablations'][primary_targ_key]['captions']
    primary_targ_stats = results['ablations'][primary_targ_key]['stats']
    
    summary = compute_summary(
        baseline_stats, primary_targ_stats, random_stats_list,
        baseline_caps, primary_targ_caps, random_caps_list
    )
    results['summary'] = summary
    
    # Additional summaries for other layers
    for key, abl_data in results['ablations'].items():
        if key != primary_targ_key:
            other_summary = compute_summary(
                baseline_stats, abl_data['stats'], random_stats_list,
                baseline_caps, abl_data['captions'], random_caps_list
            )
            results[f'summary_{key}'] = other_summary
    
    # ---- PHASE 5: SAVE & PRINT ----
    save_results(results, output_dir / 'comprehensive_results.json')
    
    # Save a lightweight summary (no captions, for quick inspection)
    lightweight = {
        'config': config,
        'summary': summary,
    }
    for key in results:
        if key.startswith('summary_'):
            lightweight[key] = results[key]
    save_results(lightweight, output_dir / 'summary_only.json')
    
    print_final_summary(summary, config)
    
    print(f"\nAll results saved to: {output_dir}/")
    print("Files: comprehensive_results.json, summary_only.json")


if __name__ == '__main__':
    main()
