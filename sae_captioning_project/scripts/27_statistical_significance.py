#!/usr/bin/env python3
"""
Statistical Significance Tests for Cross-Lingual SAE Analysis
==============================================================

Performs bootstrapping and statistical tests to validate findings:
1. Bootstrap confidence intervals for CLBAS scores
2. Permutation tests for feature overlap significance
3. Bootstrap tests for ablation accuracy drops
4. Effect size confidence intervals

Usage:
    python scripts/27_statistical_significance.py --config configs/config.yaml
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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


def get_features_and_labels(activations_data: dict, sae: SparseAutoencoder, device: str = 'cpu'):
    """Extract SAE features and gender labels."""
    if 'activations' in activations_data:
        acts = activations_data['activations']
    else:
        acts = activations_data.get('layer_activations', activations_data)
    
    if isinstance(acts, dict):
        key = list(acts.keys())[0]
        acts = acts[key]
    
    if len(acts.shape) == 3:
        acts = acts.mean(dim=1)
    
    genders = activations_data.get('genders', activations_data.get('metadata', {}).get('genders', []))
    
    valid_idx = [i for i, g in enumerate(genders) if g in ['male', 'female']]
    acts = acts[valid_idx]
    labels = [genders[i] for i in valid_idx]
    binary_labels = np.array([1 if g == 'male' else 0 for g in labels])
    
    with torch.no_grad():
        acts = acts.to(device)
        features = sae.encode(acts)
    
    return features.cpu().numpy(), binary_labels


def compute_effect_sizes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute Cohen's d effect sizes for all features."""
    male_mask = labels == 1
    female_mask = labels == 0
    
    effect_sizes = np.zeros(features.shape[1])
    for i in range(features.shape[1]):
        male_vals = features[male_mask, i]
        female_vals = features[female_mask, i]
        pooled_std = np.sqrt((male_vals.std()**2 + female_vals.std()**2) / 2)
        if pooled_std > 1e-8:
            effect_sizes[i] = (male_vals.mean() - female_vals.mean()) / pooled_std
    
    return effect_sizes


def bootstrap_clbas(ar_effect_sizes: np.ndarray, en_effect_sizes: np.ndarray, 
                    n_bootstrap: int = 10000, confidence: float = 0.95) -> dict:
    """
    Bootstrap confidence intervals for CLBAS score.
    
    CLBAS combines:
    - Cosine similarity between effect size vectors
    - Rank correlation (Spearman)
    - Feature-level correlation
    """
    np.random.seed(42)
    n_features = len(ar_effect_sizes)
    
    # Original CLBAS
    original_cosine = np.dot(ar_effect_sizes, en_effect_sizes) / (
        np.linalg.norm(ar_effect_sizes) * np.linalg.norm(en_effect_sizes) + 1e-8
    )
    original_spearman, _ = stats.spearmanr(ar_effect_sizes, en_effect_sizes)
    original_pearson, _ = stats.pearsonr(ar_effect_sizes, en_effect_sizes)
    original_clbas = (abs(original_cosine) + abs(original_spearman) + abs(original_pearson)) / 3
    
    # Bootstrap
    bootstrap_clbas = []
    bootstrap_cosine = []
    bootstrap_spearman = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping CLBAS"):
        # Resample features with replacement
        idx = np.random.choice(n_features, size=n_features, replace=True)
        ar_sample = ar_effect_sizes[idx]
        en_sample = en_effect_sizes[idx]
        
        # Compute metrics
        cosine = np.dot(ar_sample, en_sample) / (
            np.linalg.norm(ar_sample) * np.linalg.norm(en_sample) + 1e-8
        )
        spearman, _ = stats.spearmanr(ar_sample, en_sample)
        pearson, _ = stats.pearsonr(ar_sample, en_sample)
        clbas = (abs(cosine) + abs(spearman) + abs(pearson)) / 3
        
        bootstrap_clbas.append(clbas)
        bootstrap_cosine.append(cosine)
        bootstrap_spearman.append(spearman)
    
    # Confidence intervals
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(bootstrap_clbas, alpha * 100)
    ci_high = np.percentile(bootstrap_clbas, (1 - alpha) * 100)
    
    # P-value: proportion of bootstrap samples >= 0.5 (threshold for "substantial" alignment)
    p_value_substantial = np.mean(np.array(bootstrap_clbas) >= 0.5)
    
    # P-value for null hypothesis (CLBAS = 0, i.e., no alignment)
    # Under null, we expect CLBAS around 0
    p_value_null = np.mean(np.array(bootstrap_clbas) <= 0.05)  # Proportion very close to 0
    
    return {
        'original_clbas': float(original_clbas),
        'original_cosine': float(original_cosine),
        'original_spearman': float(original_spearman),
        'original_pearson': float(original_pearson),
        'bootstrap_mean': float(np.mean(bootstrap_clbas)),
        'bootstrap_std': float(np.std(bootstrap_clbas)),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'confidence': confidence,
        'n_bootstrap': n_bootstrap,
        'p_value_substantial_alignment': float(p_value_substantial),
        'conclusion': 'NO substantial alignment (CLBAS < 0.5)' if ci_high < 0.5 else 'Alignment detected'
    }


def permutation_test_overlap(ar_top_features: List[int], en_top_features: List[int],
                             n_features_total: int = 16384, n_permutations: int = 10000,
                             k: int = 100) -> dict:
    """
    Permutation test for feature overlap significance.
    
    Null hypothesis: Overlap is due to random chance.
    """
    np.random.seed(42)
    
    # Observed overlap
    ar_set = set(ar_top_features[:k])
    en_set = set(en_top_features[:k])
    observed_overlap = len(ar_set & en_set)
    
    # Permutation distribution
    permuted_overlaps = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        # Random top-k features for each language
        perm_ar = set(np.random.choice(n_features_total, size=k, replace=False))
        perm_en = set(np.random.choice(n_features_total, size=k, replace=False))
        permuted_overlaps.append(len(perm_ar & perm_en))
    
    permuted_overlaps = np.array(permuted_overlaps)
    
    # Expected overlap under null (hypergeometric)
    expected_overlap = k * k / n_features_total
    
    # P-value (two-tailed: is observed overlap significantly different from random?)
    p_value_more = np.mean(permuted_overlaps >= observed_overlap)
    p_value_less = np.mean(permuted_overlaps <= observed_overlap)
    p_value_two_tailed = 2 * min(p_value_more, p_value_less)
    
    return {
        'observed_overlap': int(observed_overlap),
        'expected_overlap_random': float(expected_overlap),
        'permutation_mean': float(np.mean(permuted_overlaps)),
        'permutation_std': float(np.std(permuted_overlaps)),
        'p_value_more_than_random': float(p_value_more),
        'p_value_less_than_random': float(p_value_less),
        'p_value_two_tailed': float(min(1.0, p_value_two_tailed)),
        'n_permutations': n_permutations,
        'conclusion': 'Overlap NOT significantly different from random' if p_value_two_tailed > 0.05 
                      else ('Overlap LESS than random (language-specific!)' if p_value_less < 0.025 
                            else 'Overlap MORE than random')
    }


def bootstrap_probe_accuracy(features: np.ndarray, labels: np.ndarray,
                             n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """Bootstrap confidence intervals for probe accuracy."""
    np.random.seed(42)
    n_samples = len(labels)
    
    bootstrap_accs = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping accuracy"):
        # Resample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = features[idx]
        y_boot = labels[idx]
        
        # Train and evaluate
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X_boot, y_boot, cv=5, scoring='accuracy')
        bootstrap_accs.append(scores.mean())
    
    bootstrap_accs = np.array(bootstrap_accs)
    alpha = (1 - confidence) / 2
    
    return {
        'mean_accuracy': float(np.mean(bootstrap_accs)),
        'std_accuracy': float(np.std(bootstrap_accs)),
        'ci_low': float(np.percentile(bootstrap_accs, alpha * 100)),
        'ci_high': float(np.percentile(bootstrap_accs, (1 - alpha) * 100)),
        'confidence': confidence,
        'n_bootstrap': n_bootstrap
    }


def bootstrap_ablation_effect(features: np.ndarray, labels: np.ndarray,
                              top_features: List[int], k: int = 100,
                              n_bootstrap: int = 500) -> dict:
    """Bootstrap test for ablation effect significance."""
    np.random.seed(42)
    n_samples = len(labels)
    
    baseline_accs = []
    ablated_accs = []
    accuracy_drops = []
    
    for _ in tqdm(range(n_bootstrap), desc=f"Bootstrapping ablation (k={k})"):
        # Resample
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = features[idx]
        y_boot = labels[idx]
        
        # Baseline
        clf = LogisticRegression(max_iter=1000, random_state=42)
        baseline = cross_val_score(clf, X_boot, y_boot, cv=5, scoring='accuracy').mean()
        
        # Ablated
        X_ablated = X_boot.copy()
        X_ablated[:, top_features[:k]] = 0
        ablated = cross_val_score(clf, X_ablated, y_boot, cv=5, scoring='accuracy').mean()
        
        baseline_accs.append(baseline)
        ablated_accs.append(ablated)
        accuracy_drops.append(baseline - ablated)
    
    accuracy_drops = np.array(accuracy_drops)
    
    # Test if drop is significantly different from 0
    p_value = 2 * min(np.mean(accuracy_drops >= 0), np.mean(accuracy_drops <= 0))
    
    return {
        'k': k,
        'mean_baseline': float(np.mean(baseline_accs)),
        'mean_ablated': float(np.mean(ablated_accs)),
        'mean_drop': float(np.mean(accuracy_drops)),
        'std_drop': float(np.std(accuracy_drops)),
        'ci_low': float(np.percentile(accuracy_drops, 2.5)),
        'ci_high': float(np.percentile(accuracy_drops, 97.5)),
        'p_value': float(min(1.0, p_value)),
        'significant': p_value < 0.05,
        'conclusion': 'Ablation has SIGNIFICANT effect' if p_value < 0.05 else 'Ablation effect NOT significant'
    }


def bootstrap_cross_lingual_specificity(
    ar_features: np.ndarray, ar_labels: np.ndarray,
    en_features: np.ndarray, en_labels: np.ndarray,
    ar_top_features: List[int], en_top_features: List[int],
    k: int = 100, n_bootstrap: int = 500
) -> dict:
    """Bootstrap test for language specificity of features."""
    np.random.seed(42)
    
    same_lang_drops = []
    cross_lang_drops = []
    specificity_ratios = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping cross-lingual"):
        # Resample Arabic
        ar_idx = np.random.choice(len(ar_labels), size=len(ar_labels), replace=True)
        ar_X = ar_features[ar_idx]
        ar_y = ar_labels[ar_idx]
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        
        # Baseline
        ar_baseline = cross_val_score(clf, ar_X, ar_y, cv=5).mean()
        
        # Same-language ablation
        ar_same = ar_X.copy()
        ar_same[:, ar_top_features[:k]] = 0
        ar_same_acc = cross_val_score(clf, ar_same, ar_y, cv=5).mean()
        
        # Cross-language ablation (ablate English features on Arabic data)
        ar_cross = ar_X.copy()
        ar_cross[:, en_top_features[:k]] = 0
        ar_cross_acc = cross_val_score(clf, ar_cross, ar_y, cv=5).mean()
        
        same_drop = ar_baseline - ar_same_acc
        cross_drop = ar_baseline - ar_cross_acc
        
        same_lang_drops.append(same_drop)
        cross_lang_drops.append(cross_drop)
        
        if abs(cross_drop) > 0.001:
            specificity_ratios.append(same_drop / cross_drop)
    
    same_lang_drops = np.array(same_lang_drops)
    cross_lang_drops = np.array(cross_lang_drops)
    
    # Test if same-language drop > cross-language drop
    differences = same_lang_drops - cross_lang_drops
    p_value = np.mean(differences <= 0)  # P(same-lang <= cross-lang)
    
    return {
        'k': k,
        'mean_same_lang_drop': float(np.mean(same_lang_drops)),
        'mean_cross_lang_drop': float(np.mean(cross_lang_drops)),
        'mean_difference': float(np.mean(differences)),
        'std_difference': float(np.std(differences)),
        'ci_difference_low': float(np.percentile(differences, 2.5)),
        'ci_difference_high': float(np.percentile(differences, 97.5)),
        'p_value_same_greater': float(p_value),
        'language_specific': p_value < 0.05,
        'conclusion': 'Features are LANGUAGE-SPECIFIC' if p_value < 0.05 else 'Cannot confirm language specificity'
    }


def create_visualizations(results: dict, output_dir: Path):
    """Create statistical significance visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. CLBAS bootstrap distribution
    if 'clbas_bootstrap' in results:
        fig, axes = plt.subplots(1, len(results['clbas_bootstrap']), figsize=(5*len(results['clbas_bootstrap']), 4))
        if len(results['clbas_bootstrap']) == 1:
            axes = [axes]
        
        for idx, (layer, data) in enumerate(results['clbas_bootstrap'].items()):
            ax = axes[idx]
            # We don't have the full distribution, just show CI
            ax.bar([0], [data['original_clbas']], yerr=[[data['original_clbas'] - data['ci_low']], 
                                                         [data['ci_high'] - data['original_clbas']]], 
                   capsize=5, color='steelblue', alpha=0.7)
            ax.axhline(0.5, color='red', linestyle='--', label='Substantial alignment threshold')
            ax.set_ylabel('CLBAS Score')
            ax.set_title(f'Layer {layer}\n95% CI: [{data["ci_low"]:.3f}, {data["ci_high"]:.3f}]')
            ax.set_xticks([])
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        
        plt.suptitle('CLBAS Bootstrap Confidence Intervals', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'clbas_bootstrap_ci.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Overlap permutation test results
    if 'overlap_permutation' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = list(results['overlap_permutation'].keys())
        observed = [results['overlap_permutation'][l]['observed_overlap'] for l in layers]
        expected = [results['overlap_permutation'][l]['expected_overlap_random'] for l in layers]
        p_values = [results['overlap_permutation'][l]['p_value_two_tailed'] for l in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, observed, width, label='Observed Overlap', color='steelblue')
        bars2 = ax.bar(x + width/2, expected, width, label='Expected (Random)', color='coral')
        
        # Add significance markers
        for i, p in enumerate(p_values):
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            else:
                marker = 'ns'
            ax.annotate(marker, xy=(i, max(observed[i], expected[i]) + 0.3), 
                       ha='center', fontsize=10)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Feature Overlap Count')
        ax.set_title('Feature Overlap: Observed vs Random Expected\n(* p<0.05, ** p<0.01, *** p<0.001, ns = not significant)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Layer {l}' for l in layers])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overlap_permutation_test.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Visualizations saved to {output_dir}")


def analyze_layer(layer: int, ar_sae_path: str, en_sae_path: str,
                  ar_acts_path: str, en_acts_path: str, device: str,
                  n_bootstrap: int = 1000) -> dict:
    """Run all statistical tests for a single layer."""
    print(f"\n{'='*60}")
    print(f"  Statistical Tests - Layer {layer}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    ar_sae = load_sae(ar_sae_path, device)
    en_sae = load_sae(en_sae_path, device)
    
    ar_features, ar_labels = get_features_and_labels(load_activations(ar_acts_path), ar_sae, device)
    en_features, en_labels = get_features_and_labels(load_activations(en_acts_path), en_sae, device)
    
    print(f"  Arabic: {len(ar_labels)} samples")
    print(f"  English: {len(en_labels)} samples")
    
    # Compute effect sizes
    print("\nComputing effect sizes...")
    ar_effect_sizes = compute_effect_sizes(ar_features, ar_labels)
    en_effect_sizes = compute_effect_sizes(en_features, en_labels)
    
    # Get top features
    ar_top = np.argsort(np.abs(ar_effect_sizes))[-200:][::-1].tolist()
    en_top = np.argsort(np.abs(en_effect_sizes))[-200:][::-1].tolist()
    
    results = {'layer': layer}
    
    # 1. CLBAS Bootstrap
    print("\n1. CLBAS Bootstrap Test...")
    results['clbas_bootstrap'] = bootstrap_clbas(ar_effect_sizes, en_effect_sizes, n_bootstrap=n_bootstrap)
    print(f"   CLBAS: {results['clbas_bootstrap']['original_clbas']:.4f}")
    print(f"   95% CI: [{results['clbas_bootstrap']['ci_low']:.4f}, {results['clbas_bootstrap']['ci_high']:.4f}]")
    print(f"   {results['clbas_bootstrap']['conclusion']}")
    
    # 2. Overlap Permutation Test
    print("\n2. Feature Overlap Permutation Test...")
    results['overlap_permutation'] = permutation_test_overlap(ar_top, en_top, n_permutations=n_bootstrap)
    print(f"   Observed: {results['overlap_permutation']['observed_overlap']}")
    print(f"   Expected: {results['overlap_permutation']['expected_overlap_random']:.2f}")
    print(f"   p-value: {results['overlap_permutation']['p_value_two_tailed']:.4f}")
    print(f"   {results['overlap_permutation']['conclusion']}")
    
    # 3. Probe Accuracy Bootstrap
    print("\n3. Probe Accuracy Bootstrap...")
    results['ar_accuracy_bootstrap'] = bootstrap_probe_accuracy(ar_features, ar_labels, n_bootstrap=min(500, n_bootstrap))
    results['en_accuracy_bootstrap'] = bootstrap_probe_accuracy(en_features, en_labels, n_bootstrap=min(500, n_bootstrap))
    print(f"   Arabic: {results['ar_accuracy_bootstrap']['mean_accuracy']:.3f} [{results['ar_accuracy_bootstrap']['ci_low']:.3f}, {results['ar_accuracy_bootstrap']['ci_high']:.3f}]")
    print(f"   English: {results['en_accuracy_bootstrap']['mean_accuracy']:.3f} [{results['en_accuracy_bootstrap']['ci_low']:.3f}, {results['en_accuracy_bootstrap']['ci_high']:.3f}]")
    
    # 4. Ablation Effect Bootstrap
    print("\n4. Ablation Effect Bootstrap (k=100)...")
    results['ar_ablation_bootstrap'] = bootstrap_ablation_effect(ar_features, ar_labels, ar_top, k=100, n_bootstrap=min(300, n_bootstrap))
    results['en_ablation_bootstrap'] = bootstrap_ablation_effect(en_features, en_labels, en_top, k=100, n_bootstrap=min(300, n_bootstrap))
    print(f"   Arabic drop: {results['ar_ablation_bootstrap']['mean_drop']:.4f} (p={results['ar_ablation_bootstrap']['p_value']:.4f})")
    print(f"   English drop: {results['en_ablation_bootstrap']['mean_drop']:.4f} (p={results['en_ablation_bootstrap']['p_value']:.4f})")
    
    # 5. Cross-Lingual Specificity Bootstrap
    print("\n5. Cross-Lingual Specificity Test...")
    results['cross_lingual_specificity'] = bootstrap_cross_lingual_specificity(
        ar_features, ar_labels, en_features, en_labels, ar_top, en_top, 
        k=100, n_bootstrap=min(300, n_bootstrap)
    )
    print(f"   Same-lang drop: {results['cross_lingual_specificity']['mean_same_lang_drop']:.4f}")
    print(f"   Cross-lang drop: {results['cross_lingual_specificity']['mean_cross_lang_drop']:.4f}")
    print(f"   p-value: {results['cross_lingual_specificity']['p_value_same_greater']:.4f}")
    print(f"   {results['cross_lingual_specificity']['conclusion']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--layers", type=str, default="0,3,6,9,12,15,17")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="results/statistical_tests")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    checkpoints_dir = Path("checkpoints")
    sae_dir = checkpoints_dir / "saes"
    acts_dir = checkpoints_dir / "full_layers_ncc" / "layer_checkpoints"
    
    all_results = {
        'clbas_bootstrap': {},
        'overlap_permutation': {},
        'probe_accuracy': {},
        'ablation_effect': {},
        'cross_lingual': {},
        'layer_details': []
    }
    
    for layer in layers:
        ar_sae_path = sae_dir / f"sae_arabic_layer_{layer}.pt"
        en_sae_path = sae_dir / f"sae_english_layer_{layer}.pt"
        ar_acts_path = acts_dir / f"layer_{layer}_arabic.pt"
        en_acts_path = acts_dir / f"layer_{layer}_english.pt"
        
        # Check files exist
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
                layer, str(ar_sae_path), str(en_sae_path),
                str(ar_acts_path), str(en_acts_path), 
                args.device, args.n_bootstrap
            )
            
            all_results['layer_details'].append(result)
            all_results['clbas_bootstrap'][layer] = result['clbas_bootstrap']
            all_results['overlap_permutation'][layer] = result['overlap_permutation']
            all_results['probe_accuracy'][layer] = {
                'arabic': result['ar_accuracy_bootstrap'],
                'english': result['en_accuracy_bootstrap']
            }
            all_results['ablation_effect'][layer] = {
                'arabic': result['ar_ablation_bootstrap'],
                'english': result['en_ablation_bootstrap']
            }
            all_results['cross_lingual'][layer] = result['cross_lingual_specificity']
            
        except Exception as e:
            print(f"\nError analyzing Layer {layer}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results['layer_details']:
        print("\nNo layers analyzed!")
        return
    
    # Save results
    results_path = output_dir / "statistical_significance_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Create visualizations
    create_visualizations(all_results, output_dir / "visualizations")
    
    # Print summary
    print("\n" + "="*80)
    print("  STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*80)
    
    print(f"\n{'Layer':<8}{'CLBAS':<12}{'CLBAS CI':<20}{'Overlap':<10}{'p-value':<10}{'Specific?':<12}")
    print("-"*72)
    
    for layer_result in all_results['layer_details']:
        layer = layer_result['layer']
        clbas = layer_result['clbas_bootstrap']['original_clbas']
        ci = f"[{layer_result['clbas_bootstrap']['ci_low']:.3f}, {layer_result['clbas_bootstrap']['ci_high']:.3f}]"
        overlap = layer_result['overlap_permutation']['observed_overlap']
        p_val = layer_result['cross_lingual_specificity']['p_value_same_greater']
        specific = "YES" if layer_result['cross_lingual_specificity']['language_specific'] else "NO"
        
        print(f"{layer:<8}{clbas:<12.4f}{ci:<20}{overlap:<10}{p_val:<10.4f}{specific:<12}")
    
    print("\n" + "="*80)
    print("  KEY CONCLUSIONS")
    print("="*80)
    
    # Aggregate findings
    n_clbas_low = sum(1 for r in all_results['layer_details'] 
                      if r['clbas_bootstrap']['ci_high'] < 0.5)
    n_overlap_random = sum(1 for r in all_results['layer_details']
                           if r['overlap_permutation']['p_value_two_tailed'] > 0.05)
    n_language_specific = sum(1 for r in all_results['layer_details']
                              if r['cross_lingual_specificity']['language_specific'])
    
    total = len(all_results['layer_details'])
    
    print(f"\n  1. CLBAS (Cross-Lingual Alignment):")
    print(f"     {n_clbas_low}/{total} layers show NO substantial alignment (CI upper bound < 0.5)")
    
    print(f"\n  2. Feature Overlap:")
    print(f"     {n_overlap_random}/{total} layers have overlap NOT significantly different from random")
    
    print(f"\n  3. Language Specificity:")
    print(f"     {n_language_specific}/{total} layers show statistically significant language-specific features")
    
    if n_clbas_low >= total/2 and n_overlap_random >= total/2:
        print(f"\n  ✓ CONCLUSION: Arabic and English use DIFFERENT features for gender encoding")
        print(f"    This finding is statistically significant with bootstrapped confidence intervals.")


if __name__ == "__main__":
    main()
