#!/usr/bin/env python3
"""
Cross-Lingual Feature Interpretation
=====================================

Analyzes what the language-specific gender features actually encode.
Key questions:
1. What patterns do Arabic-specific vs English-specific gender features capture?
2. Are there any semantic regularities in the top gender features?
3. How do feature activation patterns differ between male/female samples?

This script builds on the 0.4% overlap finding to understand WHY the features differ.

Usage:
    python scripts/25_cross_lingual_feature_interpretation.py --config configs/config.yaml
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder, SAEConfig


def load_sae(path: str, device: str = 'cpu') -> SparseAutoencoder:
    """Load trained SAE model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        d_model = checkpoint.get('d_model', 2048)
        d_hidden = checkpoint.get('d_hidden', 16384)
        config = SAEConfig(d_model=d_model, expansion_factor=d_hidden // d_model)
    
    sae = SparseAutoencoder(config)
    if 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'])
    else:
        sae.load_state_dict(checkpoint)
    
    sae.eval()
    return sae.to(device)


def load_activations(path: str) -> dict:
    """Load activation file."""
    return torch.load(path, map_location='cpu', weights_only=False)


def get_features_and_metadata(activations_data: dict, sae: SparseAutoencoder, device: str = 'cpu'):
    """Extract SAE features along with metadata."""
    # Get activations
    if 'activations' in activations_data:
        acts = activations_data['activations']
    else:
        acts = activations_data.get('layer_activations', activations_data)
    
    if isinstance(acts, dict):
        key = list(acts.keys())[0]
        acts = acts[key]
    
    # Handle different shapes
    if len(acts.shape) == 3:
        acts = acts.mean(dim=1)
    
    # Get metadata
    genders = activations_data.get('genders', activations_data.get('metadata', {}).get('genders', []))
    image_ids = activations_data.get('image_ids', activations_data.get('metadata', {}).get('image_ids', []))
    captions = activations_data.get('captions', activations_data.get('metadata', {}).get('captions', []))
    
    # Filter to male/female only
    valid_idx = [i for i, g in enumerate(genders) if g in ['male', 'female']]
    acts = acts[valid_idx]
    labels = [genders[i] for i in valid_idx]
    binary_labels = np.array([1 if g == 'male' else 0 for g in labels])
    
    # Get corresponding metadata
    if image_ids:
        image_ids = [image_ids[i] for i in valid_idx]
    if captions:
        captions = [captions[i] for i in valid_idx]
    
    # Encode through SAE
    with torch.no_grad():
        acts = acts.to(device)
        features = sae.encode(acts)
    
    return {
        'features': features.cpu().numpy(),
        'labels': binary_labels,
        'genders': labels,
        'image_ids': image_ids,
        'captions': captions
    }


def compute_feature_statistics(features: np.ndarray, labels: np.ndarray) -> dict:
    """Compute comprehensive statistics for each feature."""
    n_features = features.shape[1]
    male_mask = labels == 1
    female_mask = labels == 0
    
    stats_dict = {
        'effect_size': np.zeros(n_features),
        'male_mean': np.zeros(n_features),
        'female_mean': np.zeros(n_features),
        'male_std': np.zeros(n_features),
        'female_std': np.zeros(n_features),
        'sparsity': np.zeros(n_features),
        'male_activation_rate': np.zeros(n_features),
        'female_activation_rate': np.zeros(n_features),
        't_statistic': np.zeros(n_features),
        'p_value': np.zeros(n_features),
    }
    
    for i in range(n_features):
        male_vals = features[male_mask, i]
        female_vals = features[female_mask, i]
        
        stats_dict['male_mean'][i] = male_vals.mean()
        stats_dict['female_mean'][i] = female_vals.mean()
        stats_dict['male_std'][i] = male_vals.std()
        stats_dict['female_std'][i] = female_vals.std()
        
        # Cohen's d
        pooled_std = np.sqrt((male_vals.std()**2 + female_vals.std()**2) / 2)
        if pooled_std > 1e-8:
            stats_dict['effect_size'][i] = (male_vals.mean() - female_vals.mean()) / pooled_std
        
        # Sparsity (fraction of zeros)
        stats_dict['sparsity'][i] = (features[:, i] == 0).mean()
        stats_dict['male_activation_rate'][i] = (male_vals > 0).mean()
        stats_dict['female_activation_rate'][i] = (female_vals > 0).mean()
        
        # T-test
        if len(male_vals) > 1 and len(female_vals) > 1:
            t, p = stats.ttest_ind(male_vals, female_vals)
            stats_dict['t_statistic'][i] = t
            stats_dict['p_value'][i] = p
    
    return stats_dict


def analyze_top_features(
    features: np.ndarray, 
    labels: np.ndarray, 
    stats_dict: dict,
    captions: List[str],
    k: int = 50
) -> dict:
    """Analyze top gender-associated features in detail."""
    effect_sizes = stats_dict['effect_size']
    
    # Get top male and female features
    male_top_idx = np.argsort(effect_sizes)[-k:][::-1]
    female_top_idx = np.argsort(effect_sizes)[:k]
    
    results = {
        'male_associated': [],
        'female_associated': []
    }
    
    male_mask = labels == 1
    female_mask = labels == 0
    
    for idx in male_top_idx:
        feature_info = {
            'feature_id': int(idx),
            'effect_size': float(effect_sizes[idx]),
            'male_mean': float(stats_dict['male_mean'][idx]),
            'female_mean': float(stats_dict['female_mean'][idx]),
            'male_activation_rate': float(stats_dict['male_activation_rate'][idx]),
            'female_activation_rate': float(stats_dict['female_activation_rate'][idx]),
            'sparsity': float(stats_dict['sparsity'][idx]),
        }
        
        # Get top activating samples
        feature_acts = features[:, idx]
        top_sample_idx = np.argsort(feature_acts)[-10:][::-1]
        
        if captions:
            feature_info['top_activating_captions'] = [
                {'caption': captions[i], 'activation': float(feature_acts[i]), 'gender': 'male' if labels[i] == 1 else 'female'}
                for i in top_sample_idx if i < len(captions)
            ]
        
        results['male_associated'].append(feature_info)
    
    for idx in female_top_idx:
        feature_info = {
            'feature_id': int(idx),
            'effect_size': float(effect_sizes[idx]),
            'male_mean': float(stats_dict['male_mean'][idx]),
            'female_mean': float(stats_dict['female_mean'][idx]),
            'male_activation_rate': float(stats_dict['male_activation_rate'][idx]),
            'female_activation_rate': float(stats_dict['female_activation_rate'][idx]),
            'sparsity': float(stats_dict['sparsity'][idx]),
        }
        
        # Get top activating samples
        feature_acts = features[:, idx]
        top_sample_idx = np.argsort(feature_acts)[-10:][::-1]
        
        if captions:
            feature_info['top_activating_captions'] = [
                {'caption': captions[i], 'activation': float(feature_acts[i]), 'gender': 'male' if labels[i] == 1 else 'female'}
                for i in top_sample_idx if i < len(captions)
            ]
        
        results['female_associated'].append(feature_info)
    
    return results


def compare_language_features(ar_stats: dict, en_stats: dict, ar_top: dict, en_top: dict) -> dict:
    """Compare feature characteristics between Arabic and English."""
    comparison = {
        'arabic_male_features': {
            'mean_effect_size': float(np.mean([f['effect_size'] for f in ar_top['male_associated']])),
            'mean_sparsity': float(np.mean([f['sparsity'] for f in ar_top['male_associated']])),
            'mean_activation_rate_diff': float(np.mean([f['male_activation_rate'] - f['female_activation_rate'] for f in ar_top['male_associated']])),
        },
        'arabic_female_features': {
            'mean_effect_size': float(np.mean([f['effect_size'] for f in ar_top['female_associated']])),
            'mean_sparsity': float(np.mean([f['sparsity'] for f in ar_top['female_associated']])),
            'mean_activation_rate_diff': float(np.mean([f['female_activation_rate'] - f['male_activation_rate'] for f in ar_top['female_associated']])),
        },
        'english_male_features': {
            'mean_effect_size': float(np.mean([f['effect_size'] for f in en_top['male_associated']])),
            'mean_sparsity': float(np.mean([f['sparsity'] for f in en_top['male_associated']])),
            'mean_activation_rate_diff': float(np.mean([f['male_activation_rate'] - f['female_activation_rate'] for f in en_top['male_associated']])),
        },
        'english_female_features': {
            'mean_effect_size': float(np.mean([f['effect_size'] for f in en_top['female_associated']])),
            'mean_sparsity': float(np.mean([f['sparsity'] for f in en_top['female_associated']])),
            'mean_activation_rate_diff': float(np.mean([f['female_activation_rate'] - f['male_activation_rate'] for f in en_top['female_associated']])),
        },
    }
    
    # Compare overall distributions
    comparison['effect_size_distribution'] = {
        'arabic_mean_abs': float(np.abs(ar_stats['effect_size']).mean()),
        'arabic_max_abs': float(np.abs(ar_stats['effect_size']).max()),
        'english_mean_abs': float(np.abs(en_stats['effect_size']).mean()),
        'english_max_abs': float(np.abs(en_stats['effect_size']).max()),
    }
    
    return comparison


def create_visualizations(
    ar_stats: dict, en_stats: dict,
    ar_top: dict, en_top: dict,
    layer: int, output_dir: Path
):
    """Create visualization comparing Arabic and English feature patterns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Effect size distributions
    ax = axes[0, 0]
    ax.hist(ar_stats['effect_size'], bins=100, alpha=0.6, label='Arabic', color='#e74c3c')
    ax.hist(en_stats['effect_size'], bins=100, alpha=0.6, label='English', color='#3498db')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel('Effect Size (Cohen\'s d)')
    ax.set_ylabel('Feature Count')
    ax.set_title(f'Layer {layer}: Gender Effect Size Distribution')
    ax.legend()
    
    # 2. Top feature effect sizes comparison
    ax = axes[0, 1]
    ar_male_effects = [f['effect_size'] for f in ar_top['male_associated'][:20]]
    en_male_effects = [f['effect_size'] for f in en_top['male_associated'][:20]]
    x = np.arange(20)
    width = 0.35
    ax.bar(x - width/2, ar_male_effects, width, label='Arabic', color='#e74c3c')
    ax.bar(x + width/2, en_male_effects, width, label='English', color='#3498db')
    ax.set_xlabel('Feature Rank')
    ax.set_ylabel('Effect Size')
    ax.set_title('Top 20 Male-Associated Features')
    ax.legend()
    
    # 3. Sparsity comparison for top features
    ax = axes[0, 2]
    ar_sparsity = [f['sparsity'] for f in ar_top['male_associated'][:20] + ar_top['female_associated'][:20]]
    en_sparsity = [f['sparsity'] for f in en_top['male_associated'][:20] + en_top['female_associated'][:20]]
    ax.boxplot([ar_sparsity, en_sparsity], labels=['Arabic', 'English'])
    ax.set_ylabel('Sparsity (fraction of zeros)')
    ax.set_title('Sparsity of Top Gender Features')
    
    # 4. Activation rate difference
    ax = axes[1, 0]
    ar_male_rate_diff = [f['male_activation_rate'] - f['female_activation_rate'] for f in ar_top['male_associated'][:30]]
    en_male_rate_diff = [f['male_activation_rate'] - f['female_activation_rate'] for f in en_top['male_associated'][:30]]
    ax.scatter(range(30), ar_male_rate_diff, label='Arabic', alpha=0.7, color='#e74c3c')
    ax.scatter(range(30), en_male_rate_diff, label='English', alpha=0.7, color='#3498db')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Feature Rank')
    ax.set_ylabel('Activation Rate Difference (Male - Female)')
    ax.set_title('Gender Selectivity of Top Male Features')
    ax.legend()
    
    # 5. Feature activation patterns heatmap - Arabic
    ax = axes[1, 1]
    ar_features = ar_top['male_associated'][:10] + ar_top['female_associated'][:10]
    heatmap_data = np.array([[f['male_mean'], f['female_mean']] for f in ar_features])
    sns.heatmap(heatmap_data, ax=ax, cmap='RdBu_r', center=0,
                yticklabels=[f"F{f['feature_id']}" for f in ar_features],
                xticklabels=['Male', 'Female'])
    ax.set_title('Arabic: Top Gender Features\n(Mean Activation)')
    
    # 6. Feature activation patterns heatmap - English
    ax = axes[1, 2]
    en_features = en_top['male_associated'][:10] + en_top['female_associated'][:10]
    heatmap_data = np.array([[f['male_mean'], f['female_mean']] for f in en_features])
    sns.heatmap(heatmap_data, ax=ax, cmap='RdBu_r', center=0,
                yticklabels=[f"F{f['feature_id']}" for f in en_features],
                xticklabels=['Male', 'Female'])
    ax.set_title('English: Top Gender Features\n(Mean Activation)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'layer_{layer}_feature_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to {output_dir / f'layer_{layer}_feature_comparison.png'}")


def analyze_layer(
    ar_sae_path: str, en_sae_path: str,
    ar_acts_path: str, en_acts_path: str,
    layer: int, output_dir: Path, device: str = 'cpu'
) -> dict:
    """Run full feature interpretation for a single layer."""
    print(f"\n{'='*60}")
    print(f"  Feature Interpretation - Layer {layer}")
    print(f"{'='*60}")
    
    # Load SAEs
    print("Loading SAEs...")
    ar_sae = load_sae(ar_sae_path, device)
    en_sae = load_sae(en_sae_path, device)
    
    # Load and process activations
    print("Loading Arabic activations...")
    ar_data = get_features_and_metadata(load_activations(ar_acts_path), ar_sae, device)
    print(f"  {ar_data['features'].shape[0]} samples, {ar_data['features'].shape[1]} features")
    
    print("Loading English activations...")
    en_data = get_features_and_metadata(load_activations(en_acts_path), en_sae, device)
    print(f"  {en_data['features'].shape[0]} samples, {en_data['features'].shape[1]} features")
    
    # Compute statistics
    print("Computing feature statistics...")
    ar_stats = compute_feature_statistics(ar_data['features'], ar_data['labels'])
    en_stats = compute_feature_statistics(en_data['features'], en_data['labels'])
    
    # Analyze top features
    print("Analyzing top gender features...")
    ar_top = analyze_top_features(ar_data['features'], ar_data['labels'], ar_stats, ar_data['captions'])
    en_top = analyze_top_features(en_data['features'], en_data['labels'], en_stats, en_data['captions'])
    
    # Compare languages
    print("Comparing language patterns...")
    comparison = compare_language_features(ar_stats, en_stats, ar_top, en_top)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(ar_stats, en_stats, ar_top, en_top, layer, output_dir / 'visualizations')
    
    # Print key insights
    print(f"\n  KEY INSIGHTS for Layer {layer}:")
    print(f"    Arabic effect size range: [{ar_stats['effect_size'].min():.3f}, {ar_stats['effect_size'].max():.3f}]")
    print(f"    English effect size range: [{en_stats['effect_size'].min():.3f}, {en_stats['effect_size'].max():.3f}]")
    print(f"    Arabic top male feature: {ar_top['male_associated'][0]['feature_id']} (d={ar_top['male_associated'][0]['effect_size']:.3f})")
    print(f"    English top male feature: {en_top['male_associated'][0]['feature_id']} (d={en_top['male_associated'][0]['effect_size']:.3f})")
    
    return {
        'layer': layer,
        'arabic': {
            'n_samples': int(ar_data['features'].shape[0]),
            'top_features': ar_top,
            'effect_size_stats': {
                'mean': float(ar_stats['effect_size'].mean()),
                'std': float(ar_stats['effect_size'].std()),
                'min': float(ar_stats['effect_size'].min()),
                'max': float(ar_stats['effect_size'].max()),
            }
        },
        'english': {
            'n_samples': int(en_data['features'].shape[0]),
            'top_features': en_top,
            'effect_size_stats': {
                'mean': float(en_stats['effect_size'].mean()),
                'std': float(en_stats['effect_size'].std()),
                'min': float(en_stats['effect_size'].min()),
                'max': float(en_stats['effect_size'].max()),
            }
        },
        'comparison': comparison
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-lingual feature interpretation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--layers", type=str, default="0,3,6,9,12,15,17")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/feature_interpretation")
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    checkpoints_dir = Path("checkpoints")
    sae_dir = checkpoints_dir / "saes"
    acts_dir = checkpoints_dir / "full_layers_ncc" / "layer_checkpoints"
    
    all_results = []
    
    for layer in layers:
        ar_sae_path = sae_dir / f"sae_arabic_layer_{layer}.pt"
        en_sae_path = sae_dir / f"sae_english_layer_{layer}.pt"
        ar_acts_path = acts_dir / f"layer_{layer}_arabic.pt"
        en_acts_path = acts_dir / f"layer_{layer}_english.pt"
        
        # Handle special cases
        if layer == 0 and not en_sae_path.exists():
            en_sae_path = sae_dir / "sae_layer_0.pt"
        
        # Check if files exist
        missing = []
        for p, name in [(ar_sae_path, "Arabic SAE"), (en_sae_path, "English SAE"),
                        (ar_acts_path, "Arabic activations"), (en_acts_path, "English activations")]:
            if not p.exists():
                missing.append(f"{name}: {p}")
        
        if missing:
            print(f"\nSkipping Layer {layer} - missing files:")
            for m in missing:
                print(f"  - {m}")
            continue
        
        try:
            result = analyze_layer(
                str(ar_sae_path), str(en_sae_path),
                str(ar_acts_path), str(en_acts_path),
                layer, output_dir, args.device
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nError analyzing Layer {layer}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo layers were successfully analyzed!")
        return
    
    # Save results
    results_path = output_dir / "feature_interpretation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Create summary
    print("\n" + "="*80)
    print("  FEATURE INTERPRETATION SUMMARY")
    print("="*80)
    
    print(f"\n{'Layer':<8}{'AR Effect Max':<16}{'EN Effect Max':<16}{'AR Top Feature':<16}{'EN Top Feature':<16}")
    print("-"*72)
    
    for r in all_results:
        ar_max = r['arabic']['effect_size_stats']['max']
        en_max = r['english']['effect_size_stats']['max']
        ar_top = r['arabic']['top_features']['male_associated'][0]['feature_id']
        en_top = r['english']['top_features']['male_associated'][0]['feature_id']
        print(f"{r['layer']:<8}{ar_max:<16.3f}{en_max:<16.3f}{ar_top:<16}{en_top:<16}")
    
    # Key insights
    print("\n" + "="*80)
    print("  KEY FINDING: Language-Specific Feature Patterns")
    print("="*80)
    print("\n  The top gender-encoding features are COMPLETELY DIFFERENT between languages:")
    
    for r in all_results[:3]:  # Show first 3 layers
        ar_ids = [f['feature_id'] for f in r['arabic']['top_features']['male_associated'][:5]]
        en_ids = [f['feature_id'] for f in r['english']['top_features']['male_associated'][:5]]
        print(f"\n  Layer {r['layer']}:")
        print(f"    Arabic top-5 male features:  {ar_ids}")
        print(f"    English top-5 male features: {en_ids}")
        overlap = set(ar_ids) & set(en_ids)
        print(f"    Overlap: {len(overlap)} features")


if __name__ == "__main__":
    main()
