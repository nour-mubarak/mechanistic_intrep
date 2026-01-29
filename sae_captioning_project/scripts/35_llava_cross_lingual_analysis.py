#!/usr/bin/env python3
"""
LLaVA Cross-Lingual Analysis
=============================

Run cross-lingual overlap and bias analysis on LLaVA-1.5-7B model.
Compare with PaLiGemma and Qwen2-VL findings.

KEY RESEARCH QUESTION:
LLaVA was NOT trained on Arabic (uses byte-fallback tokenization).
This provides a "zero-shot bilingual" condition to compare against
models with native Arabic support (PaLiGemma, Qwen2-VL).

Usage:
    python scripts/35_llava_cross_lingual_analysis.py --layers 0,8,16,24,31
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
import wandb
warnings.filterwarnings('ignore')


class LLaVASAE(torch.nn.Module):
    """Sparse Autoencoder for LLaVA (d_model=4096)."""
    
    def __init__(self, d_model: int = 4096, expansion_factor: int = 8, l1_coefficient: float = 1e-4):
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


def load_llava_sae(path: str, device: str = 'cpu') -> LLaVASAE:
    """Load trained LLaVA SAE."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    d_model = checkpoint.get('d_model', 4096)
    d_hidden = checkpoint.get('d_hidden', 32768)
    expansion_factor = d_hidden // d_model
    
    sae = LLaVASAE(d_model=d_model, expansion_factor=expansion_factor)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    
    return sae.to(device)


def load_activations(path: str) -> tuple:
    """Load activation file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    activations = data['activations']
    if activations.dtype != torch.float32:
        activations = activations.float()
    return activations, data.get('genders', [])


def get_features_and_labels(activations: torch.Tensor, genders: list, sae: LLaVASAE, device: str = 'cpu'):
    """Extract SAE features and filter for valid gender labels."""
    valid_idx = [i for i, g in enumerate(genders) if g in ['male', 'female']]
    acts = activations[valid_idx]
    labels = [genders[i] for i in valid_idx]
    binary_labels = np.array([1 if g == 'male' else 0 for g in labels])
    
    with torch.no_grad():
        acts = acts.float().to(device)
        features = sae.encode(acts)
    
    return features.cpu().numpy(), binary_labels


def compute_effect_sizes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute Cohen's d effect sizes for each feature."""
    male_mask = labels == 1
    female_mask = labels == 0
    
    effect_sizes = np.zeros(features.shape[1])
    for i in range(features.shape[1]):
        male_vals = features[male_mask, i]
        female_vals = features[female_mask, i]
        
        # Pooled standard deviation
        n1, n2 = len(male_vals), len(female_vals)
        var1, var2 = male_vals.var(), female_vals.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if (n1+n2) > 2 else 1e-8
        
        if pooled_std > 1e-8:
            effect_sizes[i] = (male_vals.mean() - female_vals.mean()) / pooled_std
    
    return effect_sizes


def compute_clbas_metrics(ar_effect: np.ndarray, en_effect: np.ndarray) -> dict:
    """Compute Cross-Lingual Bias Alignment Score and related metrics."""
    
    # Filter out features with zero variance
    valid_mask = (np.abs(ar_effect) > 1e-8) | (np.abs(en_effect) > 1e-8)
    ar_valid = ar_effect[valid_mask]
    en_valid = en_effect[valid_mask]
    
    # Cosine similarity
    norm_ar = np.linalg.norm(ar_valid)
    norm_en = np.linalg.norm(en_valid)
    if norm_ar > 1e-8 and norm_en > 1e-8:
        cosine = np.dot(ar_valid, en_valid) / (norm_ar * norm_en)
    else:
        cosine = 0.0
    
    # Correlation metrics
    if len(ar_valid) > 2:
        spearman, spearman_p = stats.spearmanr(ar_valid, en_valid)
        pearson, pearson_p = stats.pearsonr(ar_valid, en_valid)
    else:
        spearman, spearman_p = 0.0, 1.0
        pearson, pearson_p = 0.0, 1.0
    
    # Combined CLBAS score
    clbas = (abs(cosine) + abs(spearman) + abs(pearson)) / 3
    
    return {
        'clbas_score': float(clbas),
        'cosine_similarity': float(cosine),
        'spearman_correlation': float(spearman),
        'spearman_pvalue': float(spearman_p),
        'pearson_correlation': float(pearson),
        'pearson_pvalue': float(pearson_p),
        'valid_features': int(valid_mask.sum())
    }


def compute_feature_overlap(ar_effect: np.ndarray, en_effect: np.ndarray, k: int = 100) -> dict:
    """Compute feature overlap between languages."""
    ar_top = set(np.argsort(np.abs(ar_effect))[-k:][::-1])
    en_top = set(np.argsort(np.abs(en_effect))[-k:][::-1])
    
    overlap = ar_top & en_top
    union = ar_top | en_top
    
    jaccard = len(overlap) / len(union) if len(union) > 0 else 0
    
    # Direction agreement for overlapping features
    overlap_indices = list(overlap)
    if len(overlap_indices) > 0:
        ar_signs = np.sign(ar_effect[overlap_indices])
        en_signs = np.sign(en_effect[overlap_indices])
        direction_agreement = (ar_signs == en_signs).mean()
    else:
        direction_agreement = 0.0
    
    return {
        'overlap_count': len(overlap),
        'overlap_percentage': len(overlap) / k * 100,
        'jaccard_index': float(jaccard),
        'ar_specific_count': len(ar_top - en_top),
        'en_specific_count': len(en_top - ar_top),
        'direction_agreement': float(direction_agreement)
    }


def train_gender_probe(features: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Train gender classification probe with cross-validation."""
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Accuracy scores
    acc_scores = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy')
    
    # F1 scores
    f1_scores = cross_val_score(clf, features, labels, cv=cv, scoring='f1')
    
    return acc_scores.mean(), acc_scores.std(), f1_scores.mean()


def run_ablation_analysis(
    features: np.ndarray, 
    labels: np.ndarray, 
    effect_sizes: np.ndarray,
    k_values: List[int] = [50, 100, 200]
) -> dict:
    """Run ablation analysis removing top-k gender features."""
    
    base_acc, _, _ = train_gender_probe(features, labels)
    
    results = {'baseline_accuracy': float(base_acc), 'ablations': {}}
    
    for k in k_values:
        # Get top-k features by effect size
        top_k_indices = np.argsort(np.abs(effect_sizes))[-k:][::-1].tolist()
        
        # Ablate (zero out) these features
        ablated_features = features.copy()
        ablated_features[:, top_k_indices] = 0
        
        # Test accuracy after ablation
        ablated_acc, _, _ = train_gender_probe(ablated_features, labels)
        
        results['ablations'][f'top_{k}'] = {
            'ablated_accuracy': float(ablated_acc),
            'accuracy_drop': float(base_acc - ablated_acc),
            'drop_percentage': float((base_acc - ablated_acc) / base_acc * 100)
        }
    
    return results


def cross_language_ablation(
    ar_features: np.ndarray, ar_labels: np.ndarray, ar_effect: np.ndarray,
    en_features: np.ndarray, en_labels: np.ndarray, en_effect: np.ndarray,
    k: int = 100
) -> dict:
    """Test if Arabic features affect English classification and vice versa."""
    
    ar_top = np.argsort(np.abs(ar_effect))[-k:][::-1].tolist()
    en_top = np.argsort(np.abs(en_effect))[-k:][::-1].tolist()
    
    # Baseline accuracies
    ar_base, _, _ = train_gender_probe(ar_features, ar_labels)
    en_base, _, _ = train_gender_probe(en_features, en_labels)
    
    # Same-language ablation
    ar_ablated_same = ar_features.copy()
    ar_ablated_same[:, ar_top] = 0
    ar_same_acc, _, _ = train_gender_probe(ar_ablated_same, ar_labels)
    
    en_ablated_same = en_features.copy()
    en_ablated_same[:, en_top] = 0
    en_same_acc, _, _ = train_gender_probe(en_ablated_same, en_labels)
    
    # Cross-language ablation
    ar_ablated_cross = ar_features.copy()
    ar_ablated_cross[:, en_top] = 0
    ar_cross_acc, _, _ = train_gender_probe(ar_ablated_cross, ar_labels)
    
    en_ablated_cross = en_features.copy()
    en_ablated_cross[:, ar_top] = 0
    en_cross_acc, _, _ = train_gender_probe(en_ablated_cross, en_labels)
    
    return {
        'k': k,
        'arabic': {
            'baseline': float(ar_base),
            'same_lang_ablation': float(ar_same_acc),
            'same_lang_drop': float(ar_base - ar_same_acc),
            'cross_lang_ablation': float(ar_cross_acc),
            'cross_lang_drop': float(ar_base - ar_cross_acc)
        },
        'english': {
            'baseline': float(en_base),
            'same_lang_ablation': float(en_same_acc),
            'same_lang_drop': float(en_base - en_same_acc),
            'cross_lang_ablation': float(en_cross_acc),
            'cross_lang_drop': float(en_base - en_cross_acc)
        }
    }


def analyze_layer(layer: int, checkpoints_dir: Path, results_dir: Path, device: str = 'cpu') -> Optional[dict]:
    """Run full cross-lingual analysis for a single layer."""
    
    print(f"\n{'='*70}")
    print(f"  LLaVA Cross-Lingual Analysis - Layer {layer}")
    print(f"{'='*70}")
    
    # Paths
    ar_sae_path = checkpoints_dir / "saes" / f"llava_sae_arabic_layer_{layer}.pt"
    en_sae_path = checkpoints_dir / "saes" / f"llava_sae_english_layer_{layer}.pt"
    ar_acts_path = checkpoints_dir / "layer_checkpoints" / f"llava_layer_{layer}_arabic.pt"
    en_acts_path = checkpoints_dir / "layer_checkpoints" / f"llava_layer_{layer}_english.pt"
    
    # Check files exist
    missing = []
    for p, name in [(ar_sae_path, "AR SAE"), (en_sae_path, "EN SAE"), 
                    (ar_acts_path, "AR Acts"), (en_acts_path, "EN Acts")]:
        if not p.exists():
            missing.append(f"{name}: {p}")
    
    if missing:
        print("  Missing files:")
        for m in missing:
            print(f"    - {m}")
        return None
    
    # Load SAEs
    print("\n  Loading SAEs...")
    ar_sae = load_llava_sae(str(ar_sae_path), device)
    en_sae = load_llava_sae(str(en_sae_path), device)
    
    # Load activations and extract features
    print("  Loading activations...")
    ar_acts, ar_genders = load_activations(str(ar_acts_path))
    en_acts, en_genders = load_activations(str(en_acts_path))
    
    print("  Extracting SAE features...")
    ar_features, ar_labels = get_features_and_labels(ar_acts, ar_genders, ar_sae, device)
    en_features, en_labels = get_features_and_labels(en_acts, en_genders, en_sae, device)
    
    print(f"    Arabic:  {len(ar_labels)} samples, {ar_features.shape[1]} features")
    print(f"    English: {len(en_labels)} samples, {en_features.shape[1]} features")
    
    # Compute effect sizes
    print("\n  Computing effect sizes...")
    ar_effect = compute_effect_sizes(ar_features, ar_labels)
    en_effect = compute_effect_sizes(en_features, en_labels)
    
    ar_sig = np.sum(np.abs(ar_effect) > 0.2)
    en_sig = np.sum(np.abs(en_effect) > 0.2)
    print(f"    Arabic significant features (|d| > 0.2): {ar_sig}")
    print(f"    English significant features (|d| > 0.2): {en_sig}")
    
    # CLBAS metrics
    print("\n  Computing CLBAS metrics...")
    clbas = compute_clbas_metrics(ar_effect, en_effect)
    print(f"    CLBAS Score: {clbas['clbas_score']:.4f}")
    print(f"    Cosine Similarity: {clbas['cosine_similarity']:.4f}")
    print(f"    Spearman Correlation: {clbas['spearman_correlation']:.4f}")
    
    # Feature overlap
    print("\n  Computing feature overlap...")
    overlap_100 = compute_feature_overlap(ar_effect, en_effect, k=100)
    overlap_50 = compute_feature_overlap(ar_effect, en_effect, k=50)
    print(f"    Top-100 overlap: {overlap_100['overlap_count']}/100 ({overlap_100['overlap_percentage']:.1f}%)")
    print(f"    Top-50 overlap: {overlap_50['overlap_count']}/50 ({overlap_50['overlap_percentage']:.1f}%)")
    
    # Probe accuracies
    print("\n  Training gender probes...")
    ar_acc, ar_std, ar_f1 = train_gender_probe(ar_features, ar_labels)
    en_acc, en_std, en_f1 = train_gender_probe(en_features, en_labels)
    print(f"    Arabic:  {ar_acc:.3f} ± {ar_std:.3f} (F1: {ar_f1:.3f})")
    print(f"    English: {en_acc:.3f} ± {en_std:.3f} (F1: {en_f1:.3f})")
    
    # Ablation analysis (SKIPPED for speed - 32k features too slow)
    print("\n  Skipping ablation analysis (too slow with 32k features)")
    ar_ablation = {'skipped': True}
    en_ablation = {'skipped': True}
    cross_ablation = {'skipped': True}
    
    # Compile results
    results = {
        'layer': layer,
        'model': 'llava-hf/llava-1.5-7b-hf',
        'arabic_support': 'byte_fallback',
        'timestamp': datetime.now().isoformat(),
        
        'samples': {
            'arabic': int(len(ar_labels)),
            'english': int(len(en_labels)),
            'arabic_male': int(np.sum(ar_labels == 1)),
            'arabic_female': int(np.sum(ar_labels == 0)),
            'english_male': int(np.sum(en_labels == 1)),
            'english_female': int(np.sum(en_labels == 0))
        },
        
        'clbas': clbas,
        
        'feature_overlap': {
            'top_100': overlap_100,
            'top_50': overlap_50
        },
        
        'probe_accuracy': {
            'arabic': {'mean': float(ar_acc), 'std': float(ar_std), 'f1': float(ar_f1)},
            'english': {'mean': float(en_acc), 'std': float(en_std), 'f1': float(en_f1)}
        },
        
        'ablation': {
            'arabic': ar_ablation,
            'english': en_ablation
        },
        
        'cross_language_ablation': cross_ablation,
        
        'effect_size_stats': {
            'arabic': {
                'max_abs': float(np.max(np.abs(ar_effect))),
                'mean_abs': float(np.mean(np.abs(ar_effect))),
                'significant_count': int(ar_sig)
            },
            'english': {
                'max_abs': float(np.max(np.abs(en_effect))),
                'mean_abs': float(np.mean(np.abs(en_effect))),
                'significant_count': int(en_sig)
            }
        }
    }
    
    return results


def load_comparison_results(results_dir: Path) -> dict:
    """Load results from PaLiGemma and Qwen2-VL for comparison."""
    comparison = {}
    
    # PaLiGemma results
    pali_path = results_dir.parent / "proper_cross_lingual" / "cross_lingual_results.json"
    if pali_path.exists():
        with open(pali_path, 'r') as f:
            comparison['paligemma'] = json.load(f)
        print(f"  Loaded PaLiGemma results from {pali_path}")
    
    # Qwen2-VL results
    qwen_path = results_dir.parent / "qwen2vl_analysis" / "cross_lingual_results.json"
    if qwen_path.exists():
        with open(qwen_path, 'r') as f:
            comparison['qwen2vl'] = json.load(f)
        print(f"  Loaded Qwen2-VL results from {qwen_path}")
    
    return comparison


def create_three_model_comparison(llava_results: list, comparison: dict, output_dir: Path):
    """Create visualization comparing all three models."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {
        'llava': '#9b59b6',      # Purple
        'paligemma': '#3498db',  # Blue
        'qwen2vl': '#e74c3c'     # Red
    }
    
    # Plot 1: CLBAS comparison
    ax = axes[0, 0]
    llava_layers = [r['layer'] for r in llava_results]
    llava_clbas = [r['clbas']['clbas_score'] for r in llava_results]
    
    ax.plot(llava_layers, llava_clbas, 'o-', color=colors['llava'], 
            label='LLaVA-1.5-7B (byte-fallback)', linewidth=2, markersize=8)
    
    if 'paligemma' in comparison and comparison['paligemma']:
        pali_data = comparison['paligemma']
        if isinstance(pali_data, dict) and 'layer_results' in pali_data:
            pali_layers = [r['layer'] for r in pali_data['layer_results']]
            pali_clbas = [r['clbas']['clbas_score'] for r in pali_data['layer_results']]
            ax.plot(pali_layers, pali_clbas, 's-', color=colors['paligemma'],
                   label='PaLiGemma-3B (native)', linewidth=2, markersize=8)
    
    if 'qwen2vl' in comparison and comparison['qwen2vl']:
        qwen_data = comparison['qwen2vl']
        if isinstance(qwen_data, dict) and 'layer_results' in qwen_data:
            qwen_layers = [r['layer'] for r in qwen_data['layer_results']]
            qwen_clbas = [r['clbas']['clbas_score'] for r in qwen_data['layer_results']]
            ax.plot(qwen_layers, qwen_clbas, '^-', color=colors['qwen2vl'],
                   label='Qwen2-VL-7B (native)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('CLBAS Score', fontsize=12)
    ax.set_title('Cross-Lingual Bias Alignment Score', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.6)
    
    # Plot 2: Cosine Similarity
    ax = axes[0, 1]
    llava_cosine = [r['clbas']['cosine_similarity'] for r in llava_results]
    ax.plot(llava_layers, llava_cosine, 'o-', color=colors['llava'],
            label='LLaVA-1.5-7B', linewidth=2, markersize=8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Cross-Lingual Feature Alignment (Cosine)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Feature Overlap
    ax = axes[1, 0]
    llava_overlap = [r['feature_overlap']['top_100']['overlap_percentage'] for r in llava_results]
    ax.bar(llava_layers, llava_overlap, color=colors['llava'], alpha=0.8, label='LLaVA-1.5-7B')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Overlap Percentage', fontsize=12)
    ax.set_title('Top-100 Feature Overlap (%)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Probe Accuracy Comparison
    ax = axes[1, 1]
    x = np.arange(len(llava_layers))
    width = 0.35
    
    ar_acc = [r['probe_accuracy']['arabic']['mean'] for r in llava_results]
    en_acc = [r['probe_accuracy']['english']['mean'] for r in llava_results]
    
    ax.bar(x - width/2, ar_acc, width, label='Arabic', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, en_acc, width, label='English', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probe Accuracy', fontsize=12)
    ax.set_title('Gender Probe Accuracy by Language', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(llava_layers)
    ax.legend(loc='best')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0.5, 1.0)
    
    plt.suptitle('LLaVA-1.5-7B Cross-Lingual Gender Bias Analysis\n(Byte-Fallback Arabic Support)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llava_cross_lingual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved visualization: {output_dir / 'llava_cross_lingual_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="LLaVA Cross-Lingual Analysis")
    parser.add_argument("--layers", type=str, default="0,8,16,24,31",
                       help="Comma-separated layer indices")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/llava")
    parser.add_argument("--output_dir", type=str, default="results/llava_analysis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llava-sae-analysis")
    args = parser.parse_args()
    
    # Parse layers
    layers = [int(l.strip()) for l in args.layers.split(",")]
    
    print("\n" + "="*70)
    print("LLaVA-1.5-7B Cross-Lingual Analysis")
    print("="*70)
    print(f"Layers: {layers}")
    print(f"Device: {args.device}")
    print("\nNOTE: LLaVA uses byte-fallback for Arabic (NOT trained on Arabic)")
    print("This provides a 'zero-shot bilingual' comparison condition.")
    print("="*70)
    
    # Initialize W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"llava_cross_lingual_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": "llava-hf/llava-1.5-7b-hf",
                "arabic_support": "byte_fallback",
                "layers": layers,
                "stage": "cross_lingual_analysis"
            }
        )
    
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each layer
    all_results = []
    for layer in layers:
        result = analyze_layer(layer, checkpoints_dir, output_dir, args.device)
        if result:
            all_results.append(result)
            
            # Log to W&B
            if args.wandb:
                wandb.log({
                    f"layer_{layer}/clbas": result['clbas']['clbas_score'],
                    f"layer_{layer}/cosine_sim": result['clbas']['cosine_similarity'],
                    f"layer_{layer}/overlap_pct": result['feature_overlap']['top_100']['overlap_percentage'],
                    f"layer_{layer}/ar_probe_acc": result['probe_accuracy']['arabic']['mean'],
                    f"layer_{layer}/en_probe_acc": result['probe_accuracy']['english']['mean']
                })
    
    if not all_results:
        print("\nNo layers analyzed. Check if activation and SAE files exist.")
        return
    
    # Load comparison results
    print("\n  Loading comparison results...")
    comparison = load_comparison_results(output_dir)
    
    # Create visualizations
    print("\n  Creating visualizations...")
    viz_dir = Path("visualizations/llava")
    create_three_model_comparison(all_results, comparison, viz_dir)
    
    # Save results
    final_results = {
        'model': 'llava-hf/llava-1.5-7b-hf',
        'arabic_support': 'byte_fallback',
        'analysis_date': datetime.now().isoformat(),
        'layer_results': all_results,
        'summary': {
            'mean_clbas': np.mean([r['clbas']['clbas_score'] for r in all_results]),
            'mean_cosine': np.mean([r['clbas']['cosine_similarity'] for r in all_results]),
            'mean_overlap': np.mean([r['feature_overlap']['top_100']['overlap_percentage'] for r in all_results]),
            'mean_ar_probe_acc': np.mean([r['probe_accuracy']['arabic']['mean'] for r in all_results]),
            'mean_en_probe_acc': np.mean([r['probe_accuracy']['english']['mean'] for r in all_results])
        }
    }
    
    results_path = output_dir / "cross_lingual_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  Mean CLBAS: {final_results['summary']['mean_clbas']:.4f}")
    print(f"  Mean Cosine Sim: {final_results['summary']['mean_cosine']:.4f}")
    print(f"  Mean Overlap: {final_results['summary']['mean_overlap']:.1f}%")
    print(f"  Mean AR Probe Acc: {final_results['summary']['mean_ar_probe_acc']:.3f}")
    print(f"  Mean EN Probe Acc: {final_results['summary']['mean_en_probe_acc']:.3f}")
    print(f"\nResults saved: {results_path}")
    print(f"Visualizations: {viz_dir}")
    
    if args.wandb:
        wandb.log({
            "summary/mean_clbas": final_results['summary']['mean_clbas'],
            "summary/mean_cosine": final_results['summary']['mean_cosine'],
            "summary/mean_overlap": final_results['summary']['mean_overlap'],
            "summary/mean_ar_probe_acc": final_results['summary']['mean_ar_probe_acc'],
            "summary/mean_en_probe_acc": final_results['summary']['mean_en_probe_acc']
        })
        wandb.finish()


if __name__ == "__main__":
    main()
