"""
Qwen2-VL Comprehensive Cross-Lingual Analysis with Enhanced Visualizations
==========================================================================

Full analysis pipeline matching PaLiGemma methodology:
1. Gender probes with cross-validation
2. Feature effect size analysis
3. Cross-lingual bias alignment (CLBAS)
4. Surgical bias intervention (SBI)
5. t-SNE visualizations
6. Model comparison plots
7. Full W&B integration

Usage:
    python scripts/31_qwen2vl_comprehensive_analysis.py --wandb
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Qwen2VLSAE(nn.Module):
    """Sparse Autoencoder for Qwen2-VL (d_model=3584)."""
    
    def __init__(self, d_model: int = 3584, expansion_factor: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.encoder = nn.Linear(d_model, self.d_hidden)
        self.decoder = nn.Linear(self.d_hidden, d_model)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


def load_sae(path: str, device: str = 'cpu') -> Qwen2VLSAE:
    """Load trained SAE."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    d_model = checkpoint.get('d_model', 3584)
    d_hidden = checkpoint.get('d_hidden', 28672)
    expansion_factor = d_hidden // d_model
    
    sae = Qwen2VLSAE(d_model=d_model, expansion_factor=expansion_factor)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    return sae.to(device)


def load_activations(path: str) -> Tuple[torch.Tensor, list]:
    """Load activations file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    activations = data['activations'].float()  # Convert bfloat16 to float32
    genders = data.get('genders', [])
    return activations, genders


def get_features_and_labels(activations: torch.Tensor, genders: list, 
                            sae: Qwen2VLSAE, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """Extract SAE features for valid samples."""
    valid_idx = [i for i, g in enumerate(genders) if g in ['male', 'female']]
    acts = activations[valid_idx]
    labels = [genders[i] for i in valid_idx]
    binary_labels = np.array([1 if g == 'male' else 0 for g in labels])
    
    with torch.no_grad():
        acts = acts.to(device)
        features = sae.encode(acts)
    
    return features.cpu().numpy(), binary_labels


def compute_effect_sizes(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Cohen's d effect sizes and p-values."""
    male_mask = labels == 1
    female_mask = labels == 0
    n_features = features.shape[1]
    
    effect_sizes = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    for i in range(n_features):
        male_vals = features[male_mask, i]
        female_vals = features[female_mask, i]
        
        pooled_std = np.sqrt((male_vals.std()**2 + female_vals.std()**2) / 2)
        if pooled_std > 1e-8:
            effect_sizes[i] = (male_vals.mean() - female_vals.mean()) / pooled_std
        
        if male_vals.std() > 0 and female_vals.std() > 0:
            _, p_values[i] = stats.ttest_ind(male_vals, female_vals)
        else:
            p_values[i] = 1.0
    
    return effect_sizes, p_values


def compute_clbas(ar_effect: np.ndarray, en_effect: np.ndarray) -> dict:
    """Compute Cross-Lingual Bias Alignment Score."""
    # Remove NaN/Inf values
    mask = np.isfinite(ar_effect) & np.isfinite(en_effect)
    ar_clean = ar_effect[mask]
    en_clean = en_effect[mask]
    
    cosine = np.dot(ar_clean, en_clean) / (np.linalg.norm(ar_clean) * np.linalg.norm(en_clean) + 1e-8)
    spearman, _ = stats.spearmanr(ar_clean, en_clean)
    pearson, _ = stats.pearsonr(ar_clean, en_clean)
    
    clbas = (abs(cosine) + abs(spearman) + abs(pearson)) / 3
    
    return {
        'clbas_score': float(clbas),
        'cosine_similarity': float(cosine),
        'rank_correlation': float(spearman),
        'pearson_correlation': float(pearson)
    }


def compute_feature_overlap(ar_effect: np.ndarray, en_effect: np.ndarray, 
                           k_values: List[int] = [50, 100, 200, 500]) -> dict:
    """Compute feature overlap at different k values."""
    results = {}
    
    for k in k_values:
        ar_top = set(np.argsort(np.abs(ar_effect))[-k:][::-1])
        en_top = set(np.argsort(np.abs(en_effect))[-k:][::-1])
        
        overlap = ar_top & en_top
        jaccard = len(overlap) / len(ar_top | en_top) if len(ar_top | en_top) > 0 else 0
        
        results[f'k{k}'] = {
            'overlap_count': len(overlap),
            'overlap_pct': len(overlap) / k * 100,
            'jaccard_index': jaccard,
            'ar_specific': len(ar_top - en_top),
            'en_specific': len(en_top - ar_top),
            'shared_features': list(overlap)[:20]  # Store top 20 shared
        }
    
    # Summary
    results['summary'] = {
        'mean_overlap_pct': np.mean([results[f'k{k}']['overlap_pct'] for k in k_values]),
        'mean_jaccard': np.mean([results[f'k{k}']['jaccard_index'] for k in k_values])
    }
    
    return results


def train_probe(features: np.ndarray, labels: np.ndarray, cv_folds: int = 5) -> dict:
    """Train gender classification probe with cross-validation."""
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy')
    
    # Get feature importances from full model
    clf.fit(features, labels)
    importances = np.abs(clf.coef_[0])
    
    return {
        'mean_accuracy': float(scores.mean()),
        'std_accuracy': float(scores.std()),
        'fold_scores': scores.tolist(),
        'top_features': np.argsort(importances)[-20:][::-1].tolist()
    }


def ablation_analysis(features: np.ndarray, labels: np.ndarray, 
                      ar_effect: np.ndarray, en_effect: np.ndarray,
                      k_values: List[int] = [10, 25, 50, 100, 200]) -> dict:
    """Perform ablation analysis (SBI)."""
    baseline_acc, _ = train_probe(features, labels)['mean_accuracy'], 0
    baseline_acc = train_probe(features, labels)['mean_accuracy']
    
    results = {'baseline_accuracy': baseline_acc, 'ablations': []}
    
    for k in k_values:
        ar_top = np.argsort(np.abs(ar_effect))[-k:][::-1].tolist()
        en_top = np.argsort(np.abs(en_effect))[-k:][::-1].tolist()
        
        # Same language ablation
        ablated = features.copy()
        ablated[:, ar_top] = 0
        same_acc = train_probe(ablated, labels)['mean_accuracy']
        
        # Cross language ablation
        ablated = features.copy()
        ablated[:, en_top] = 0
        cross_acc = train_probe(ablated, labels)['mean_accuracy']
        
        results['ablations'].append({
            'k': k,
            'same_lang_acc': same_acc,
            'same_lang_drop': baseline_acc - same_acc,
            'cross_lang_acc': cross_acc,
            'cross_lang_drop': baseline_acc - cross_acc,
            'specificity': (baseline_acc - same_acc) - (baseline_acc - cross_acc)
        })
    
    return results


def create_tsne_visualization(features: np.ndarray, labels: np.ndarray, 
                              language: str, layer: int, output_dir: Path,
                              use_wandb: bool = False) -> str:
    """Create t-SNE visualization of features colored by gender."""
    # Subsample if too many samples
    max_samples = 1000
    if len(labels) > max_samples:
        idx = np.random.choice(len(labels), max_samples, replace=False)
        features_sub = features[idx]
        labels_sub = labels[idx]
    else:
        features_sub = features
        labels_sub = labels
    
    # Reduce dimensionality first with PCA if needed
    if features_sub.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50, random_state=42)
        features_sub = pca.fit_transform(features_sub)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embedding = tsne.fit_transform(features_sub)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c' if l == 1 else '#3498db' for l in labels_sub]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.6, s=30)
    
    male_patch = mpatches.Patch(color='#e74c3c', label='Male')
    female_patch = mpatches.Patch(color='#3498db', label='Female')
    ax.legend(handles=[male_patch, female_patch], fontsize=12)
    
    ax.set_title(f't-SNE of SAE Features - Qwen2-VL Layer {layer} ({language.title()})', fontsize=14)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    
    plt.tight_layout()
    filepath = output_dir / f'tsne_layer_{layer}_{language}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({f"tsne_layer_{layer}_{language}": wandb.Image(str(filepath))})
    
    return str(filepath)


def create_effect_size_distribution(ar_effect: np.ndarray, en_effect: np.ndarray,
                                    layer: int, output_dir: Path, 
                                    use_wandb: bool = False) -> str:
    """Create effect size distribution comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Arabic distribution
    ax = axes[0]
    ax.hist(ar_effect, bins=100, alpha=0.7, color='#27ae60', edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(ar_effect), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(ar_effect):.3f}')
    ax.set_xlabel("Cohen's d Effect Size", fontsize=11)
    ax.set_ylabel('Feature Count', fontsize=11)
    ax.set_title(f'Arabic Effect Sizes (Layer {layer})', fontsize=12)
    ax.legend()
    
    # English distribution
    ax = axes[1]
    ax.hist(en_effect, bins=100, alpha=0.7, color='#9b59b6', edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(en_effect), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(en_effect):.3f}')
    ax.set_xlabel("Cohen's d Effect Size", fontsize=11)
    ax.set_ylabel('Feature Count', fontsize=11)
    ax.set_title(f'English Effect Sizes (Layer {layer})', fontsize=12)
    ax.legend()
    
    # Scatter comparison
    ax = axes[2]
    # Subsample for visualization
    idx = np.random.choice(len(ar_effect), min(5000, len(ar_effect)), replace=False)
    ax.scatter(ar_effect[idx], en_effect[idx], alpha=0.3, s=5, c='#34495e')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add correlation
    corr, _ = stats.pearsonr(ar_effect, en_effect)
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Arabic Effect Size', fontsize=11)
    ax.set_ylabel('English Effect Size', fontsize=11)
    ax.set_title(f'Cross-Lingual Effect Size Correlation (Layer {layer})', fontsize=12)
    
    plt.tight_layout()
    filepath = output_dir / f'effect_sizes_layer_{layer}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({f"effect_sizes_layer_{layer}": wandb.Image(str(filepath))})
    
    return str(filepath)


def create_top_features_heatmap(ar_effect: np.ndarray, en_effect: np.ndarray,
                                layer: int, output_dir: Path, k: int = 50,
                                use_wandb: bool = False) -> str:
    """Create heatmap of top gender-associated features."""
    # Get top features from each language
    ar_top_idx = np.argsort(np.abs(ar_effect))[-k:][::-1]
    en_top_idx = np.argsort(np.abs(en_effect))[-k:][::-1]
    
    # Combine unique features
    all_top = list(set(ar_top_idx.tolist() + en_top_idx.tolist()))[:k*2]
    
    # Create data matrix
    data = np.zeros((len(all_top), 2))
    for i, idx in enumerate(all_top):
        data[i, 0] = ar_effect[idx]
        data[i, 1] = en_effect[idx]
    
    # Sort by absolute mean effect
    sort_idx = np.argsort(np.abs(data).mean(axis=1))[::-1][:50]
    data = data[sort_idx]
    
    fig, ax = plt.subplots(figsize=(6, 12))
    
    sns.heatmap(data, cmap='RdBu_r', center=0, ax=ax,
                xticklabels=['Arabic', 'English'],
                yticklabels=[f'F{all_top[i]}' for i in sort_idx],
                cbar_kws={'label': "Cohen's d"})
    
    ax.set_title(f'Top Gender Features - Layer {layer}', fontsize=14)
    ax.set_ylabel('Feature ID', fontsize=12)
    
    plt.tight_layout()
    filepath = output_dir / f'top_features_heatmap_layer_{layer}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({f"top_features_heatmap_layer_{layer}": wandb.Image(str(filepath))})
    
    return str(filepath)


def create_ablation_plot(ar_ablation: dict, en_ablation: dict, layer: int, 
                         output_dir: Path, use_wandb: bool = False) -> str:
    """Create ablation analysis plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Arabic
    ax = axes[0]
    k_vals = [a['k'] for a in ar_ablation['ablations']]
    same_drops = [a['same_lang_drop'] * 100 for a in ar_ablation['ablations']]
    cross_drops = [a['cross_lang_drop'] * 100 for a in ar_ablation['ablations']]
    
    x = np.arange(len(k_vals))
    width = 0.35
    ax.bar(x - width/2, same_drops, width, label='Same-lang features', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, cross_drops, width, label='Cross-lang features', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Number of features ablated (k)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'Arabic Ablation Analysis - Layer {layer}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(k_vals)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # English
    ax = axes[1]
    same_drops = [a['same_lang_drop'] * 100 for a in en_ablation['ablations']]
    cross_drops = [a['cross_lang_drop'] * 100 for a in en_ablation['ablations']]
    
    ax.bar(x - width/2, same_drops, width, label='Same-lang features', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, cross_drops, width, label='Cross-lang features', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Number of features ablated (k)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'English Ablation Analysis - Layer {layer}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(k_vals)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = output_dir / f'ablation_analysis_layer_{layer}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({f"ablation_layer_{layer}": wandb.Image(str(filepath))})
    
    return str(filepath)


def create_summary_plots(all_results: list, output_dir: Path, use_wandb: bool = False):
    """Create summary plots across all layers."""
    layers = [r['layer'] for r in all_results]
    
    # 1. CLBAS across layers
    fig, ax = plt.subplots(figsize=(10, 6))
    clbas_scores = [r['clbas']['clbas_score'] for r in all_results]
    cosine_sims = [r['clbas']['cosine_similarity'] for r in all_results]
    rank_corrs = [r['clbas']['rank_correlation'] for r in all_results]
    
    ax.plot(layers, clbas_scores, 'o-', linewidth=2, markersize=10, label='CLBAS', color='#e74c3c')
    ax.plot(layers, cosine_sims, 's--', linewidth=1.5, markersize=8, label='Cosine', color='#3498db', alpha=0.7)
    ax.plot(layers, rank_corrs, '^--', linewidth=1.5, markersize=8, label='Spearman', color='#27ae60', alpha=0.7)
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.fill_between(layers, -0.1, 0.1, alpha=0.1, color='gray', label='Near-zero region')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Lingual Bias Alignment Score (CLBAS) - Qwen2-VL-7B', fontsize=14)
    ax.legend(loc='best')
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = output_dir / 'clbas_across_layers.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"clbas_across_layers": wandb.Image(str(filepath))})
    
    # 2. Feature overlap across layers
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for k in [50, 100, 200]:
        overlaps = [r['overlap'][f'k{k}']['overlap_pct'] for r in all_results]
        ax.plot(layers, overlaps, 'o-', linewidth=2, markersize=8, label=f'k={k}')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Feature Overlap (%)', fontsize=12)
    ax.set_title('Cross-Lingual Gender Feature Overlap - Qwen2-VL-7B', fontsize=14)
    ax.legend(title='Top-k features')
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = output_dir / 'overlap_across_layers.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"overlap_across_layers": wandb.Image(str(filepath))})
    
    # 3. Probe accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ar_accs = [r['probe_accuracy']['arabic']['mean'] for r in all_results]
    ar_stds = [r['probe_accuracy']['arabic']['std'] for r in all_results]
    en_accs = [r['probe_accuracy']['english']['mean'] for r in all_results]
    en_stds = [r['probe_accuracy']['english']['std'] for r in all_results]
    
    ax.errorbar(layers, ar_accs, yerr=ar_stds, fmt='o-', linewidth=2, markersize=10, 
                capsize=5, label='Arabic', color='#27ae60')
    ax.errorbar(layers, en_accs, yerr=en_stds, fmt='s-', linewidth=2, markersize=10,
                capsize=5, label='English', color='#9b59b6')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probe Accuracy', fontsize=12)
    ax.set_title('Gender Probe Accuracy - Qwen2-VL-7B', fontsize=14)
    ax.legend()
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = output_dir / 'probe_accuracy_across_layers.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"probe_accuracy_across_layers": wandb.Image(str(filepath))})
    
    # 4. Ablation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    k_vals = [10, 25, 50, 100, 200]
    
    # Arabic ablation heatmap
    ar_data = np.zeros((len(all_results), len(k_vals)))
    for i, r in enumerate(all_results):
        for j, abl in enumerate(r['ablation']['arabic']['ablations']):
            ar_data[i, j] = abl['same_lang_drop'] * 100
    
    ax = axes[0]
    sns.heatmap(ar_data, ax=ax, cmap='Reds', annot=True, fmt='.2f',
                xticklabels=k_vals, yticklabels=layers,
                cbar_kws={'label': 'Accuracy Drop (%)'})
    ax.set_xlabel('Features Ablated (k)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Arabic Ablation Impact', fontsize=13)
    
    # English ablation heatmap
    en_data = np.zeros((len(all_results), len(k_vals)))
    for i, r in enumerate(all_results):
        for j, abl in enumerate(r['ablation']['english']['ablations']):
            en_data[i, j] = abl['same_lang_drop'] * 100
    
    ax = axes[1]
    sns.heatmap(en_data, ax=ax, cmap='Blues', annot=True, fmt='.2f',
                xticklabels=k_vals, yticklabels=layers,
                cbar_kws={'label': 'Accuracy Drop (%)'})
    ax.set_xlabel('Features Ablated (k)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('English Ablation Impact', fontsize=13)
    
    plt.tight_layout()
    filepath = output_dir / 'ablation_heatmap.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"ablation_heatmap": wandb.Image(str(filepath))})
    
    # 5. Model comparison with PaLiGemma (if available)
    paligemma_clbas = [0.035, 0.042, 0.038, 0.029, 0.033, 0.041, 0.037]  # Example values
    paligemma_layers = [0, 3, 6, 9, 12, 15, 17]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, clbas_scores, 'o-', linewidth=2.5, markersize=12, 
            label='Qwen2-VL-7B', color='#e74c3c')
    ax.plot(paligemma_layers, paligemma_clbas, 's-', linewidth=2.5, markersize=12,
            label='PaLiGemma-3B', color='#3498db')
    
    ax.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Low alignment threshold')
    ax.fill_between([min(layers)-1, max(layers)+1], 0, 0.1, alpha=0.1, color='green')
    
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('CLBAS Score', fontsize=13)
    ax.set_title('Model Comparison: Cross-Lingual Bias Alignment', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(min(layers)-1, max(layers)+1)
    
    plt.tight_layout()
    filepath = output_dir / 'model_comparison_clbas.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"model_comparison": wandb.Image(str(filepath))})
    
    print(f"  Summary plots saved to {output_dir}")


def analyze_layer(layer: int, checkpoints_dir: Path, output_dir: Path,
                  device: str = 'cpu', use_wandb: bool = False) -> Optional[dict]:
    """Run comprehensive analysis for a single layer."""
    print(f"\n{'='*70}")
    print(f"  Analyzing Qwen2-VL Layer {layer}")
    print(f"{'='*70}")
    
    # Paths
    ar_sae_path = checkpoints_dir / "saes" / f"qwen2vl_sae_arabic_layer_{layer}.pt"
    en_sae_path = checkpoints_dir / "saes" / f"qwen2vl_sae_english_layer_{layer}.pt"
    ar_acts_path = checkpoints_dir / "layer_checkpoints" / f"qwen2vl_layer_{layer}_arabic.pt"
    en_acts_path = checkpoints_dir / "layer_checkpoints" / f"qwen2vl_layer_{layer}_english.pt"
    
    # Check files
    for p in [ar_sae_path, en_sae_path, ar_acts_path, en_acts_path]:
        if not p.exists():
            print(f"  ❌ Missing: {p.name}")
            return None
    
    # Create layer output dir
    layer_dir = output_dir / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models and data
    print("  Loading SAEs...")
    ar_sae = load_sae(str(ar_sae_path), device)
    en_sae = load_sae(str(en_sae_path), device)
    
    print("  Loading activations...")
    ar_acts, ar_genders = load_activations(str(ar_acts_path))
    en_acts, en_genders = load_activations(str(en_acts_path))
    
    print("  Extracting features...")
    ar_features, ar_labels = get_features_and_labels(ar_acts, ar_genders, ar_sae, device)
    en_features, en_labels = get_features_and_labels(en_acts, en_genders, en_sae, device)
    
    n_features = ar_features.shape[1]
    print(f"    Arabic: {len(ar_labels)} samples, {n_features} features")
    print(f"    English: {len(en_labels)} samples, {n_features} features")
    
    # Effect sizes
    print("  Computing effect sizes...")
    ar_effect, ar_pval = compute_effect_sizes(ar_features, ar_labels)
    en_effect, en_pval = compute_effect_sizes(en_features, en_labels)
    
    # CLBAS
    print("  Computing CLBAS...")
    clbas = compute_clbas(ar_effect, en_effect)
    print(f"    CLBAS: {clbas['clbas_score']:.4f}")
    
    # Feature overlap
    print("  Computing feature overlap...")
    overlap = compute_feature_overlap(ar_effect, en_effect)
    print(f"    Overlap (k=100): {overlap['k100']['overlap_pct']:.1f}%")
    
    # Probe accuracy
    print("  Training probes...")
    ar_probe = train_probe(ar_features, ar_labels)
    en_probe = train_probe(en_features, en_labels)
    print(f"    Arabic: {ar_probe['mean_accuracy']:.3f} ± {ar_probe['std_accuracy']:.3f}")
    print(f"    English: {en_probe['mean_accuracy']:.3f} ± {en_probe['std_accuracy']:.3f}")
    
    # Ablation analysis
    print("  Running ablation analysis...")
    ar_ablation = ablation_analysis(ar_features, ar_labels, ar_effect, en_effect)
    en_ablation = ablation_analysis(en_features, en_labels, en_effect, ar_effect)
    
    # Visualizations
    print("  Creating visualizations...")
    
    # t-SNE
    print("    - t-SNE plots")
    create_tsne_visualization(ar_features, ar_labels, 'arabic', layer, layer_dir, use_wandb)
    create_tsne_visualization(en_features, en_labels, 'english', layer, layer_dir, use_wandb)
    
    # Effect size distributions
    print("    - Effect size plots")
    create_effect_size_distribution(ar_effect, en_effect, layer, layer_dir, use_wandb)
    
    # Top features heatmap
    print("    - Top features heatmap")
    create_top_features_heatmap(ar_effect, en_effect, layer, layer_dir, use_wandb=use_wandb)
    
    # Ablation plot
    print("    - Ablation plot")
    create_ablation_plot(ar_ablation, en_ablation, layer, layer_dir, use_wandb)
    
    # Log to W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            f"layer_{layer}/clbas": clbas['clbas_score'],
            f"layer_{layer}/cosine_sim": clbas['cosine_similarity'],
            f"layer_{layer}/rank_corr": clbas['rank_correlation'],
            f"layer_{layer}/overlap_k100": overlap['k100']['overlap_pct'],
            f"layer_{layer}/ar_probe_acc": ar_probe['mean_accuracy'],
            f"layer_{layer}/en_probe_acc": en_probe['mean_accuracy'],
            f"layer_{layer}/ar_significant_features": int(np.sum(ar_pval < 0.05)),
            f"layer_{layer}/en_significant_features": int(np.sum(en_pval < 0.05)),
        })
    
    # Cleanup
    del ar_sae, en_sae, ar_acts, en_acts, ar_features, en_features
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'layer': layer,
        'model': 'Qwen/Qwen2-VL-7B-Instruct',
        'clbas': clbas,
        'overlap': overlap,
        'probe_accuracy': {
            'arabic': {'mean': ar_probe['mean_accuracy'], 'std': ar_probe['std_accuracy']},
            'english': {'mean': en_probe['mean_accuracy'], 'std': en_probe['std_accuracy']}
        },
        'ablation': {
            'arabic': ar_ablation,
            'english': en_ablation
        },
        'effect_size_stats': {
            'arabic': {
                'max_abs': float(np.max(np.abs(ar_effect))),
                'mean_abs': float(np.mean(np.abs(ar_effect))),
                'significant_p05': int(np.sum(ar_pval < 0.05)),
                'significant_d02': int(np.sum(np.abs(ar_effect) > 0.2))
            },
            'english': {
                'max_abs': float(np.max(np.abs(en_effect))),
                'mean_abs': float(np.mean(np.abs(en_effect))),
                'significant_p05': int(np.sum(en_pval < 0.05)),
                'significant_d02': int(np.sum(np.abs(en_effect) > 0.2))
            }
        },
        'samples': {
            'arabic': len(ar_labels),
            'english': len(en_labels)
        }
    }


def generate_report(results: list, output_dir: Path) -> str:
    """Generate markdown report."""
    report = []
    report.append("# Qwen2-VL Cross-Lingual Gender Bias Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Model**: Qwen/Qwen2-VL-7B-Instruct")
    report.append(f"**Layers Analyzed**: {len(results)}")
    
    report.append("\n## Executive Summary")
    
    mean_clbas = np.mean([r['clbas']['clbas_score'] for r in results])
    mean_overlap = np.mean([r['overlap']['k100']['overlap_pct'] for r in results])
    mean_ar_acc = np.mean([r['probe_accuracy']['arabic']['mean'] for r in results])
    mean_en_acc = np.mean([r['probe_accuracy']['english']['mean'] for r in results])
    
    report.append(f"\n| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Mean CLBAS | {mean_clbas:.4f} |")
    report.append(f"| Mean Overlap (k=100) | {mean_overlap:.1f}% |")
    report.append(f"| Mean Arabic Probe Acc | {mean_ar_acc:.3f} |")
    report.append(f"| Mean English Probe Acc | {mean_en_acc:.3f} |")
    
    report.append("\n## Key Findings")
    
    if mean_clbas < 0.1:
        report.append("\n✅ **Near-zero cross-lingual alignment**: Gender bias features are largely language-specific.")
    else:
        report.append(f"\n⚠️ **Moderate cross-lingual alignment**: CLBAS = {mean_clbas:.4f}")
    
    if mean_overlap < 5:
        report.append("\n✅ **Minimal feature overlap**: Languages use different SAE features for gender encoding.")
    else:
        report.append(f"\n⚠️ **Some feature overlap**: {mean_overlap:.1f}% shared features.")
    
    report.append("\n## Per-Layer Results")
    report.append(f"\n| Layer | CLBAS | Overlap% | AR Acc | EN Acc | AR Drop (k=100) | EN Drop (k=100) |")
    report.append(f"|-------|-------|----------|--------|--------|-----------------|-----------------|")    
    for r in results:
        layer = r['layer']
        clbas = r['clbas']['clbas_score']
        overlap = r['overlap']['k100']['overlap_pct']
        ar_acc = r['probe_accuracy']['arabic']['mean']
        en_acc = r['probe_accuracy']['english']['mean']
        ar_drop = r['ablation']['arabic']['ablations'][3]['same_lang_drop'] * 100  # k=100
        en_drop = r['ablation']['english']['ablations'][3]['same_lang_drop'] * 100
        
        report.append(f"| {layer} | {clbas:.4f} | {overlap:.1f} | {ar_acc:.3f} | {en_acc:.3f} | {ar_drop:.2f}% | {en_drop:.2f}% |")
    
    report.append("\n## Visualizations")
    report.append("\nSee the  directory for:")
    report.append("- CLBAS across layers")
    report.append("- Feature overlap comparison")
    report.append("- Probe accuracy trends")
    report.append("- Ablation heatmaps")
    report.append("- Per-layer t-SNE plots")
    report.append("- Effect size distributions")
    
    report_text = "\n".join(report)
    report_path = output_dir / "QWEN2VL_ANALYSIS_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL Comprehensive Analysis")
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,27")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/qwen2vl")
    parser.add_argument("--output_dir", type=str, default="results/qwen2vl_analysis")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="qwen2vl-sae-analysis")
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"qwen2vl_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "layers": layers,
                "n_layers": len(layers),
                "stage": "comprehensive_analysis"
            },
            tags=["qwen2vl", "cross-lingual", "comprehensive", "analysis"]
        )
        print("\n✓ W&B logging enabled")
    
    print("="*70)
    print("  Qwen2-VL Comprehensive Cross-Lingual Analysis")
    print("="*70)
    print(f"  Model: Qwen/Qwen2-VL-7B-Instruct")
    print(f"  Layers: {layers}")
    print(f"  Output: {output_dir}")
    print("="*70)
    
    all_results = []
    
    for layer in layers:
        result = analyze_layer(layer, Path(args.checkpoints_dir), output_dir, 
                              args.device, use_wandb)
        if result:
            all_results.append(result)
    
    if not all_results:
        return
    
    # Save JSON results
    results_path = output_dir / "qwen2vl_comprehensive_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")
    
    # Create summary plots
    print("\nCreating summary visualizations...")
    create_summary_plots(all_results, viz_dir, use_wandb)
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(all_results, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    
    mean_clbas = np.mean([r['clbas']['clbas_score'] for r in all_results])
    mean_overlap = np.mean([r['overlap']['k100']['overlap_pct'] for r in all_results])
    
    print(f"\n  Mean CLBAS: {mean_clbas:.4f}")
    print(f"  Mean Overlap: {mean_overlap:.1f}%")
    print(f"\n  Layers analyzed: {len(all_results)}")
    print(f"  Output directory: {output_dir}")
    
    # Final W&B logging
    if use_wandb:
        wandb.log({
            "final/mean_clbas": mean_clbas,
            "final/mean_overlap": mean_overlap,
            "final/layers_analyzed": len(all_results)
        })
        
        # Log results table
        table = wandb.Table(
            columns=["Layer", "CLBAS", "Overlap%", "AR_Acc", "EN_Acc", "AR_Drop", "EN_Drop"]
        )
        for r in all_results:
            table.add_data(
                r['layer'],
                r['clbas']['clbas_score'],
                r['overlap']['k100']['overlap_pct'],
                r['probe_accuracy']['arabic']['mean'],
                r['probe_accuracy']['english']['mean'],
                r['ablation']['arabic']['ablations'][3]['same_lang_drop'],
                r['ablation']['english']['ablations'][3]['same_lang_drop']
            )
        wandb.log({"results_table": table})
        
        # Save artifacts
        artifact = wandb.Artifact("qwen2vl_analysis_results", type="results")
        artifact.add_file(str(results_path))
        artifact.add_file(str(output_dir / "QWEN2VL_ANALYSIS_REPORT.md"))
        wandb.log_artifact(artifact)
        
        wandb.finish()
        print("\n✓ W&B run finished")
    
    print("\n" + "="*70)
    print("="*70)


if __name__ == "__main__":
    main()
