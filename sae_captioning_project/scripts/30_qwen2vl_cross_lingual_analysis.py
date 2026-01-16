#!/usr/bin/env python3
"""
Qwen2-VL Cross-Lingual Analysis
================================

Run the same cross-lingual overlap and SBI analysis on Qwen2-VL model
to compare with PaLiGemma findings.

Usage:
    python scripts/30_qwen2vl_cross_lingual_analysis.py --layers 0,4,8,12,16,20,24,27
"""

import torch
import numpy as np
import pandas as pd
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


class Qwen2VLSAE(torch.nn.Module):
    """Sparse Autoencoder for Qwen2-VL (d_model=3584)."""
    
    def __init__(self, d_model: int = 3584, expansion_factor: int = 8, l1_coefficient: float = 5e-4):
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.l1_coefficient = l1_coefficient
        
        self.encoder = torch.nn.Linear(d_model, self.d_hidden)
        self.decoder = torch.nn.Linear(self.d_hidden, d_model)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


def load_qwen2vl_sae(path: str, device: str = 'cpu') -> Qwen2VLSAE:
    """Load trained Qwen2-VL SAE."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    d_model = checkpoint.get('d_model', 3584)
    d_hidden = checkpoint.get('d_hidden', 28672)
    expansion_factor = d_hidden // d_model
    
    sae = Qwen2VLSAE(d_model=d_model, expansion_factor=expansion_factor)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    
    return sae.to(device)


def load_activations(path: str) -> tuple:
    """Load activation file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['activations'], data.get('genders', [])


def get_features_and_labels(activations: torch.Tensor, genders: list, sae: Qwen2VLSAE, device: str = 'cpu'):
    """Extract SAE features and labels."""
    valid_idx = [i for i, g in enumerate(genders) if g in ['male', 'female']]
    acts = activations[valid_idx]
    labels = [genders[i] for i in valid_idx]
    binary_labels = np.array([1 if g == 'male' else 0 for g in labels])
    
    with torch.no_grad():
        acts = acts.to(device)
        features = sae.encode(acts)
    
    return features.cpu().numpy(), binary_labels


def compute_effect_sizes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute Cohen's d effect sizes."""
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


def compute_clbas(ar_effect: np.ndarray, en_effect: np.ndarray) -> dict:
    """Compute Cross-Lingual Bias Alignment Score."""
    cosine = np.dot(ar_effect, en_effect) / (np.linalg.norm(ar_effect) * np.linalg.norm(en_effect) + 1e-8)
    spearman, _ = stats.spearmanr(ar_effect, en_effect)
    pearson, _ = stats.pearsonr(ar_effect, en_effect)
    
    clbas = (abs(cosine) + abs(spearman) + abs(pearson)) / 3
    
    return {
        'clbas_score': float(clbas),
        'cosine_similarity': float(cosine),
        'rank_correlation': float(spearman),
        'pearson_correlation': float(pearson)
    }


def compute_feature_overlap(ar_effect: np.ndarray, en_effect: np.ndarray, k: int = 100) -> dict:
    """Compute feature overlap between languages."""
    ar_top = set(np.argsort(np.abs(ar_effect))[-k:][::-1])
    en_top = set(np.argsort(np.abs(en_effect))[-k:][::-1])
    
    overlap = ar_top & en_top
    jaccard = len(overlap) / (2*k - len(overlap)) if (2*k - len(overlap)) > 0 else 0
    
    return {
        'overlap_count': len(overlap),
        'overlap_pct': len(overlap) / k * 100,
        'jaccard_index': jaccard,
        'ar_specific': len(ar_top - en_top),
        'en_specific': len(en_top - ar_top)
    }


def train_probe(features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Train gender probe."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=5, scoring='accuracy')
    return scores.mean(), scores.std()


def ablate_and_test(features: np.ndarray, labels: np.ndarray, indices: List[int]) -> float:
    """Ablate features and test probe accuracy."""
    ablated = features.copy()
    ablated[:, indices] = 0
    acc, _ = train_probe(ablated, labels)
    return acc


def analyze_layer(layer: int, checkpoints_dir: Path, device: str = 'cpu') -> dict:
    """Run full analysis for a single layer."""
    print(f"\n{'='*60}")
    print(f"  Qwen2-VL Analysis - Layer {layer}")
    print(f"{'='*60}")
    
    # Paths
    ar_sae_path = checkpoints_dir / "saes" / f"qwen2vl_sae_arabic_layer_{layer}.pt"
    en_sae_path = checkpoints_dir / "saes" / f"qwen2vl_sae_english_layer_{layer}.pt"
    ar_acts_path = checkpoints_dir / "layer_checkpoints" / f"qwen2vl_layer_{layer}_arabic.pt"
    en_acts_path = checkpoints_dir / "layer_checkpoints" / f"qwen2vl_layer_{layer}_english.pt"
    
    # Check files exist
    for p in [ar_sae_path, en_sae_path, ar_acts_path, en_acts_path]:
        if not p.exists():
            print(f"  Missing: {p}")
            return None
    
    # Load SAEs
    print("Loading SAEs...")
    ar_sae = load_qwen2vl_sae(str(ar_sae_path), device)
    en_sae = load_qwen2vl_sae(str(en_sae_path), device)
    
    # Load activations and get features
    print("Loading activations...")
    ar_acts, ar_genders = load_activations(str(ar_acts_path))
    en_acts, en_genders = load_activations(str(en_acts_path))
    
    ar_features, ar_labels = get_features_and_labels(ar_acts, ar_genders, ar_sae, device)
    en_features, en_labels = get_features_and_labels(en_acts, en_genders, en_sae, device)
    
    print(f"  Arabic: {len(ar_labels)} samples, {ar_features.shape[1]} features")
    print(f"  English: {len(en_labels)} samples, {en_features.shape[1]} features")
    
    # Compute effect sizes
    print("Computing effect sizes...")
    ar_effect = compute_effect_sizes(ar_features, ar_labels)
    en_effect = compute_effect_sizes(en_features, en_labels)
    
    # CLBAS
    print("Computing CLBAS...")
    clbas = compute_clbas(ar_effect, en_effect)
    print(f"  CLBAS: {clbas['clbas_score']:.4f}")
    
    # Feature overlap
    print("Computing overlap...")
    overlap = compute_feature_overlap(ar_effect, en_effect, k=100)
    print(f"  Overlap: {overlap['overlap_count']}/100 ({overlap['overlap_pct']:.1f}%)")
    
    # Probe accuracies
    print("Training probes...")
    ar_acc, ar_std = train_probe(ar_features, ar_labels)
    en_acc, en_std = train_probe(en_features, en_labels)
    print(f"  Arabic: {ar_acc:.3f} ± {ar_std:.3f}")
    print(f"  English: {en_acc:.3f} ± {en_std:.3f}")
    
    # Ablation tests (k=100)
    print("Running ablation tests...")
    ar_top = np.argsort(np.abs(ar_effect))[-100:][::-1].tolist()
    en_top = np.argsort(np.abs(en_effect))[-100:][::-1].tolist()
    
    ar_same_ablate = ablate_and_test(ar_features, ar_labels, ar_top)
    ar_cross_ablate = ablate_and_test(ar_features, ar_labels, en_top)
    en_same_ablate = ablate_and_test(en_features, en_labels, en_top)
    en_cross_ablate = ablate_and_test(en_features, en_labels, ar_top)
    
    print(f"  AR same-lang drop: {ar_acc - ar_same_ablate:.4f}")
    print(f"  AR cross-lang drop: {ar_acc - ar_cross_ablate:.4f}")
    print(f"  EN same-lang drop: {en_acc - en_same_ablate:.4f}")
    print(f"  EN cross-lang drop: {en_acc - en_cross_ablate:.4f}")
    
    return {
        'layer': layer,
        'model': 'Qwen/Qwen2-VL-7B-Instruct',
        'clbas': clbas,
        'overlap': overlap,
        'probe_accuracy': {
            'arabic': {'mean': float(ar_acc), 'std': float(ar_std)},
            'english': {'mean': float(en_acc), 'std': float(en_std)}
        },
        'ablation': {
            'arabic_same_lang_drop': float(ar_acc - ar_same_ablate),
            'arabic_cross_lang_drop': float(ar_acc - ar_cross_ablate),
            'english_same_lang_drop': float(en_acc - en_same_ablate),
            'english_cross_lang_drop': float(en_acc - en_cross_ablate)
        },
        'samples': {
            'arabic': len(ar_labels),
            'english': len(en_labels)
        },
        'effect_size_stats': {
            'arabic': {
                'max_abs': float(np.max(np.abs(ar_effect))),
                'mean_abs': float(np.mean(np.abs(ar_effect))),
                'significant': int(np.sum(np.abs(ar_effect) > 0.2))
            },
            'english': {
                'max_abs': float(np.max(np.abs(en_effect))),
                'mean_abs': float(np.mean(np.abs(en_effect))),
                'significant': int(np.sum(np.abs(en_effect) > 0.2))
            }
        }
    }


def create_comparison_visualizations(qwen_results: list, paligemma_results_path: str, output_dir: Path):
    """Create comparison visualizations between Qwen2-VL and PaLiGemma."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PaLiGemma results if available
    try:
        with open(paligemma_results_path, 'r') as f:
            paligemma_results = json.load(f)
    except:
        paligemma_results = None
        print("  PaLiGemma results not found, creating Qwen2-VL only plots")
    
    # Plot 1: CLBAS comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    qwen_layers = [r['layer'] for r in qwen_results]
    qwen_clbas = [r['clbas']['clbas_score'] for r in qwen_results]
    
    ax.plot(qwen_layers, qwen_clbas, 'o-', color='#e74c3c', label='Qwen2-VL-7B', linewidth=2, markersize=8)
    
    if paligemma_results:
        pali_layers = [r['layer'] for r in paligemma_results]
        pali_clbas = [r['clbas']['clbas_score'] for r in paligemma_results]
        ax.plot(pali_layers, pali_clbas, 's-', color='#3498db', label='PaLiGemma-3B', linewidth=2, markersize=8)
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Alignment threshold')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('CLBAS Score', fontsize=12)
    ax.set_title('Cross-Lingual Bias Alignment Score: Model Comparison', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 0.6)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clbas_model_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: Feature overlap comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    qwen_overlap = [r['overlap']['overlap_pct'] for r in qwen_results]
    
    ax.bar(np.array(qwen_layers) - 0.2, qwen_overlap, 0.4, label='Qwen2-VL-7B', color='#e74c3c', alpha=0.8)
    
    if paligemma_results:
        pali_overlap = [r['overlap']['summary']['mean_overlap_pct'] for r in paligemma_results]
        ax.bar(np.array(pali_layers) + 0.2, pali_overlap, 0.4, label='PaLiGemma-3B', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Feature Overlap (%)', fontsize=12)
    ax.set_title('Top-100 Gender Feature Overlap: Model Comparison', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overlap_model_comparison.png', dpi=150)
    plt.close()
    
    # Plot 3: Probe accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Arabic
    ax = axes[0]
    qwen_ar = [r['probe_accuracy']['arabic']['mean'] for r in qwen_results]
    ax.plot(qwen_layers, qwen_ar, 'o-', color='#e74c3c', label='Qwen2-VL-7B', linewidth=2)
    if paligemma_results:
        # Note: PaLiGemma results structure might differ
        ax.axhline(0.89, color='#3498db', linestyle='--', label='PaLiGemma-3B (avg)', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Probe Accuracy')
    ax.set_title('Arabic Gender Probe Accuracy')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    
    # English
    ax = axes[1]
    qwen_en = [r['probe_accuracy']['english']['mean'] for r in qwen_results]
    ax.plot(qwen_layers, qwen_en, 'o-', color='#e74c3c', label='Qwen2-VL-7B', linewidth=2)
    if paligemma_results:
        ax.axhline(0.85, color='#3498db', linestyle='--', label='PaLiGemma-3B (avg)', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Probe Accuracy')
    ax.set_title('English Gender Probe Accuracy')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_accuracy_comparison.png', dpi=150)
    plt.close()
    
    print(f"  Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL Cross-Lingual Analysis")
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,27")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/qwen2vl")
    parser.add_argument("--output_dir", type=str, default="results/qwen2vl_analysis")
    parser.add_argument("--paligemma_results", type=str, default="results/cross_lingual_overlap/cross_lingual_overlap_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Qwen2-VL Cross-Lingual Analysis")
    print("="*60)
    
    all_results = []
    
    for layer in layers:
        result = analyze_layer(layer, Path(args.checkpoints_dir), args.device)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\nNo layers analyzed successfully!")
        return
    
    # Save results
    results_path = output_dir / "qwen2vl_cross_lingual_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Create comparison visualizations
    create_comparison_visualizations(
        all_results, 
        args.paligemma_results,
        output_dir / "visualizations"
    )
    
    # Print summary
    print("\n" + "="*80)
    print("  QWEN2-VL CROSS-LINGUAL ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n{'Layer':<8}{'CLBAS':<10}{'Overlap%':<12}{'AR Acc':<10}{'EN Acc':<10}{'AR Drop':<10}{'EN Drop':<10}")
    print("-"*70)
    
    for r in all_results:
        layer = r['layer']
        clbas = r['clbas']['clbas_score']
        overlap = r['overlap']['overlap_pct']
        ar_acc = r['probe_accuracy']['arabic']['mean']
        en_acc = r['probe_accuracy']['english']['mean']
        ar_drop = r['ablation']['arabic_same_lang_drop']
        en_drop = r['ablation']['english_same_lang_drop']
        
        print(f"{layer:<8}{clbas:<10.4f}{overlap:<12.1f}{ar_acc:<10.3f}{en_acc:<10.3f}{ar_drop:<10.4f}{en_drop:<10.4f}")
    
    # Aggregate stats
    mean_clbas = np.mean([r['clbas']['clbas_score'] for r in all_results])
    mean_overlap = np.mean([r['overlap']['overlap_pct'] for r in all_results])
    
    print("\n" + "="*80)
    print("  KEY FINDINGS - QWEN2-VL-7B-INSTRUCT")
    print("="*80)
    print(f"\n  Mean CLBAS: {mean_clbas:.4f}")
    print(f"  Mean Overlap: {mean_overlap:.2f}%")
    
    if mean_clbas < 0.1:
        print("\n  ✓ FINDING: Near-zero cross-lingual alignment (similar to PaLiGemma)")
    else:
        print("\n  ~ FINDING: Some cross-lingual alignment detected")
    
    if mean_overlap < 5:
        print("  ✓ FINDING: Minimal feature overlap (language-specific features)")
    else:
        print("  ~ FINDING: Some feature overlap between languages")


if __name__ == "__main__":
    main()
