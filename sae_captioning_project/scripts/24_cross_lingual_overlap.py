#!/usr/bin/env python3
"""
Cross-Lingual Feature Overlap Analysis
=======================================

This script computes the KEY NOVEL METRIC for your paper:
- Feature overlap between Arabic and English gender features
- CLBAS (Cross-Lingual Bias Alignment Score)
- Language-specific vs shared gender features

Run this to complete your analysis for publication.

Usage:
    python scripts/24_cross_lingual_overlap.py --config configs/config.yaml
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import yaml
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder, SAEConfig


def load_sae(path: str, device: str = 'cpu') -> SparseAutoencoder:
    """Load trained SAE model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Build config from checkpoint fields
        d_model = checkpoint.get('d_model', 2048)
        d_hidden = checkpoint.get('d_hidden', 16384)
        config = SAEConfig(
            d_model=d_model,
            expansion_factor=d_hidden // d_model
        )

    # Create SAE
    sae = SparseAutoencoder(config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'])
    else:
        sae.load_state_dict(checkpoint)

    sae.eval()
    return sae.to(device)


def load_activations(path: str) -> dict:
    """Load activation file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data


def get_features_and_labels(activations_data: dict, sae: SparseAutoencoder, device: str = 'cpu'):
    """Extract SAE features and gender labels."""
    # Get activations
    if 'activations' in activations_data:
        acts = activations_data['activations']
    else:
        acts = activations_data.get('layer_activations', activations_data)

    if isinstance(acts, dict):
        # Take first key
        key = list(acts.keys())[0]
        acts = acts[key]

    # Handle different shapes
    if len(acts.shape) == 3:
        # (samples, seq, hidden) -> (samples, hidden) via mean
        acts = acts.mean(dim=1)

    # Get gender labels
    genders = activations_data.get('genders', activations_data.get('metadata', {}).get('genders', []))

    # Filter to male/female only
    valid_idx = [i for i, g in enumerate(genders) if g in ['male', 'female']]
    acts = acts[valid_idx]
    labels = [genders[i] for i in valid_idx]
    binary_labels = [1 if g == 'male' else 0 for g in labels]

    # Encode through SAE
    with torch.no_grad():
        acts = acts.to(device)
        features = sae.encode(acts)

    return features.cpu().numpy(), np.array(binary_labels)


def compute_gender_effect_sizes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute Cohen's d effect size for each feature."""
    male_mask = labels == 1
    female_mask = labels == 0

    male_features = features[male_mask]
    female_features = features[female_mask]

    effect_sizes = np.zeros(features.shape[1])

    for i in range(features.shape[1]):
        male_vals = male_features[:, i]
        female_vals = female_features[:, i]

        # Cohen's d
        pooled_std = np.sqrt((male_vals.std()**2 + female_vals.std()**2) / 2)
        if pooled_std > 1e-8:
            effect_sizes[i] = (male_vals.mean() - female_vals.mean()) / pooled_std
        else:
            effect_sizes[i] = 0

    return effect_sizes


def get_top_gender_features(effect_sizes: np.ndarray, k: int = 100) -> dict:
    """Get top-k gender-associated features."""
    # Top male-associated (positive effect size)
    male_top = np.argsort(effect_sizes)[-k:][::-1]

    # Top female-associated (negative effect size)
    female_top = np.argsort(effect_sizes)[:k]

    # Top overall (by absolute value)
    abs_top = np.argsort(np.abs(effect_sizes))[-k:][::-1]

    return {
        'male_associated': male_top.tolist(),
        'female_associated': female_top.tolist(),
        'top_overall': abs_top.tolist()
    }


def compute_feature_overlap(ar_top_features: dict, en_top_features: dict) -> dict:
    """
    Compute overlap between Arabic and English gender features.

    This is the KEY NOVEL METRIC for your paper!
    """
    results = {}

    for category in ['male_associated', 'female_associated', 'top_overall']:
        ar_set = set(ar_top_features[category])
        en_set = set(en_top_features[category])

        overlap = ar_set & en_set
        ar_specific = ar_set - en_set
        en_specific = en_set - ar_set
        union = ar_set | en_set

        results[category] = {
            'overlap_count': len(overlap),
            'overlap_pct': len(overlap) / len(ar_set) * 100,
            'ar_specific_count': len(ar_specific),
            'en_specific_count': len(en_specific),
            'jaccard_index': len(overlap) / len(union) if len(union) > 0 else 0,
            'overlap_features': list(overlap),
            'ar_specific_features': list(ar_specific)[:20],  # First 20
            'en_specific_features': list(en_specific)[:20]   # First 20
        }

    # Summary metrics
    results['summary'] = {
        'mean_overlap_pct': np.mean([results[c]['overlap_pct'] for c in ['male_associated', 'female_associated', 'top_overall']]),
        'mean_jaccard': np.mean([results[c]['jaccard_index'] for c in ['male_associated', 'female_associated', 'top_overall']])
    }

    return results


def compute_clbas(ar_features: np.ndarray, en_features: np.ndarray,
                  ar_labels: np.ndarray, en_labels: np.ndarray) -> dict:
    """
    Compute Cross-Lingual Bias Alignment Score (CLBAS).

    Measures if bias PATTERNS are similar despite using different FEATURES.

    Low CLBAS (0.0-0.3): Different bias patterns
    High CLBAS (0.7-1.0): Similar bias patterns
    """
    # Compute gender direction for each language
    ar_male = ar_features[ar_labels == 1].mean(axis=0)
    ar_female = ar_features[ar_labels == 0].mean(axis=0)
    ar_direction = ar_male - ar_female

    en_male = en_features[en_labels == 1].mean(axis=0)
    en_female = en_features[en_labels == 0].mean(axis=0)
    en_direction = en_male - en_female

    # Cosine similarity of gender directions
    cosine_sim = 1 - cosine(ar_direction, en_direction)

    # Correlation of effect sizes
    ar_effect_sizes = compute_gender_effect_sizes(ar_features, ar_labels)
    en_effect_sizes = compute_gender_effect_sizes(en_features, en_labels)
    effect_correlation = np.corrcoef(ar_effect_sizes, en_effect_sizes)[0, 1]

    # Rank correlation of top features
    ar_ranks = np.argsort(np.argsort(-np.abs(ar_effect_sizes)))
    en_ranks = np.argsort(np.argsort(-np.abs(en_effect_sizes)))
    rank_correlation, _ = stats.spearmanr(ar_ranks, en_ranks)

    return {
        'cosine_similarity': float(cosine_sim),
        'effect_size_correlation': float(effect_correlation),
        'rank_correlation': float(rank_correlation),
        'clbas_score': float((cosine_sim + effect_correlation + rank_correlation) / 3)
    }


def analyze_layer(ar_sae_path: str, en_sae_path: str,
                  ar_acts_path: str, en_acts_path: str,
                  layer: int, device: str = 'cpu') -> dict:
    """Run full cross-lingual analysis for a single layer."""
    print(f"\n{'='*60}")
    print(f"  Analyzing Layer {layer}")
    print(f"{'='*60}")

    # Load SAEs
    print("Loading SAEs...")
    ar_sae = load_sae(ar_sae_path, device)
    en_sae = load_sae(en_sae_path, device)

    # Load activations
    print("Loading activations...")
    ar_acts_data = load_activations(ar_acts_path)
    en_acts_data = load_activations(en_acts_path)

    # Get features and labels
    print("Extracting features...")
    ar_features, ar_labels = get_features_and_labels(ar_acts_data, ar_sae, device)
    en_features, en_labels = get_features_and_labels(en_acts_data, en_sae, device)

    print(f"  Arabic: {ar_features.shape[0]} samples, {ar_features.shape[1]} features")
    print(f"  English: {en_features.shape[0]} samples, {en_features.shape[1]} features")

    # Compute effect sizes
    print("Computing effect sizes...")
    ar_effect_sizes = compute_gender_effect_sizes(ar_features, ar_labels)
    en_effect_sizes = compute_gender_effect_sizes(en_features, en_labels)

    # Get top features
    print("Identifying top gender features...")
    ar_top = get_top_gender_features(ar_effect_sizes, k=100)
    en_top = get_top_gender_features(en_effect_sizes, k=100)

    # Compute overlap
    print("Computing feature overlap...")
    overlap_results = compute_feature_overlap(ar_top, en_top)

    # Compute CLBAS
    print("Computing CLBAS...")
    clbas_results = compute_clbas(ar_features, en_features, ar_labels, en_labels)

    return {
        'layer': layer,
        'samples': {
            'arabic': int(ar_features.shape[0]),
            'english': int(en_features.shape[0])
        },
        'overlap': overlap_results,
        'clbas': clbas_results,
        'effect_size_stats': {
            'arabic': {
                'mean_abs': float(np.abs(ar_effect_sizes).mean()),
                'max_abs': float(np.abs(ar_effect_sizes).max()),
                'significant_count': int((np.abs(ar_effect_sizes) > 0.3).sum())
            },
            'english': {
                'mean_abs': float(np.abs(en_effect_sizes).mean()),
                'max_abs': float(np.abs(en_effect_sizes).max()),
                'significant_count': int((np.abs(en_effect_sizes) > 0.3).sum())
            }
        }
    }


def create_summary_visualizations(results: list, output_dir: Path):
    """Create summary visualizations across layers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [r['layer'] for r in results]

    # 1. Feature Overlap by Layer
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, category in enumerate(['male_associated', 'female_associated', 'top_overall']):
        overlaps = [r['overlap'][category]['overlap_pct'] for r in results]
        axes[idx].bar(layers, overlaps, color=['#3498db', '#e74c3c', '#2ecc71'][idx])
        axes[idx].set_xlabel('Layer')
        axes[idx].set_ylabel('Overlap %')
        axes[idx].set_title(f'{category.replace("_", " ").title()} Feature Overlap')
        axes[idx].set_ylim(0, 100)

        # Add value labels
        for i, v in enumerate(overlaps):
            axes[idx].text(layers[i], v + 2, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_overlap_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. CLBAS Components by Layer
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layers))
    width = 0.25

    cosine_sims = [r['clbas']['cosine_similarity'] for r in results]
    effect_corrs = [r['clbas']['effect_size_correlation'] for r in results]
    rank_corrs = [r['clbas']['rank_correlation'] for r in results]

    ax.bar(x - width, cosine_sims, width, label='Cosine Similarity', color='#3498db')
    ax.bar(x, effect_corrs, width, label='Effect Size Correlation', color='#e74c3c')
    ax.bar(x + width, rank_corrs, width, label='Rank Correlation', color='#2ecc71')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Score')
    ax.set_title('CLBAS Components by Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'clbas_components_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Summary Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Male Overlap %', 'Female Overlap %', 'Overall Overlap %',
               'Cosine Sim', 'Effect Corr', 'CLBAS Score']

    data = np.array([
        [r['overlap']['male_associated']['overlap_pct'] for r in results],
        [r['overlap']['female_associated']['overlap_pct'] for r in results],
        [r['overlap']['top_overall']['overlap_pct'] for r in results],
        [r['clbas']['cosine_similarity'] * 100 for r in results],
        [r['clbas']['effect_size_correlation'] * 100 for r in results],
        [r['clbas']['clbas_score'] * 100 for r in results]
    ])

    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                xticklabels=layers, yticklabels=metrics, ax=ax,
                vmin=0, vmax=100)
    ax.set_xlabel('Layer')
    ax.set_title('Cross-Lingual Analysis Summary\n(Lower overlap = More language-specific)')

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_lingual_summary_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cross-lingual feature overlap analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--layers", type=str, default="0,3,6,9,12,15,17",
                        help="Comma-separated list of layers to analyze")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/cross_lingual")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="sae-cross-lingual-bias")
    args = parser.parse_args()
    
    # Initialize W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=f"cross_lingual_overlap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "layers": args.layers,
                "experiment": "cross_lingual_feature_overlap"
            }
        )
        print("W&B logging enabled")

    layers = [int(l) for l in args.layers.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths (adjust these to match your directory structure)
    checkpoints_dir = Path("checkpoints")
    sae_dir = checkpoints_dir / "saes"
    acts_dir = checkpoints_dir / "full_layers_ncc" / "layer_checkpoints"

    all_results = []

    for layer in layers:
        # Construct paths
        ar_sae_path = sae_dir / f"sae_arabic_layer_{layer}.pt"
        en_sae_path = sae_dir / f"sae_english_layer_{layer}.pt"
        ar_acts_path = acts_dir / f"layer_{layer}_arabic.pt"
        en_acts_path = acts_dir / f"layer_{layer}_english.pt"

        # Handle layer 0 special case (shared SAE)
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

        # Run analysis
        try:
            result = analyze_layer(
                str(ar_sae_path), str(en_sae_path),
                str(ar_acts_path), str(en_acts_path),
                layer, args.device
            )
            all_results.append(result)

            # Print key findings
            print(f"\n  KEY FINDINGS for Layer {layer}:")
            print(f"    Feature Overlap: {result['overlap']['top_overall']['overlap_pct']:.1f}%")
            print(f"    CLBAS Score: {result['clbas']['clbas_score']:.3f}")

        except Exception as e:
            print(f"\nError analyzing Layer {layer}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNo layers were successfully analyzed!")
        return

    # Save results
    results_path = output_dir / "cross_lingual_overlap_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Create visualizations
    viz_dir = output_dir / "visualizations"
    create_summary_visualizations(all_results, viz_dir)

    # Print summary table
    print("\n" + "="*80)
    print("  CROSS-LINGUAL FEATURE OVERLAP SUMMARY")
    print("="*80)
    print(f"\n{'Layer':<8}{'Overlap %':<12}{'AR Specific':<14}{'EN Specific':<14}{'CLBAS':<10}")
    print("-"*58)

    for r in all_results:
        overlap = r['overlap']['top_overall']['overlap_pct']
        ar_spec = r['overlap']['top_overall']['ar_specific_count']
        en_spec = r['overlap']['top_overall']['en_specific_count']
        clbas = r['clbas']['clbas_score']
        print(f"{r['layer']:<8}{overlap:<12.1f}{ar_spec:<14}{en_spec:<14}{clbas:<10.3f}")

    # Key insight
    mean_overlap = np.mean([r['overlap']['top_overall']['overlap_pct'] for r in all_results])
    mean_clbas = np.mean([r['clbas']['clbas_score'] for r in all_results])

    print("\n" + "="*80)
    print("  KEY INSIGHTS")
    print("="*80)
    print(f"\n  Mean Feature Overlap: {mean_overlap:.1f}%")
    print(f"  Mean CLBAS Score: {mean_clbas:.3f}")

    if mean_overlap < 20:
        print("\n  LOW OVERLAP: Arabic and English use largely SEPARATE gender features!")
        print("    -> This is a NOVEL FINDING supporting language-specific gender circuits")
    elif mean_overlap < 50:
        print("\n  MODERATE OVERLAP: Some shared features, some language-specific")
    else:
        print("\n  HIGH OVERLAP: Arabic and English share most gender features")

    if mean_clbas < 0.3:
        print(f"\n  LOW CLBAS ({mean_clbas:.2f}): Different bias patterns across languages")
    elif mean_clbas > 0.7:
        print(f"\n  HIGH CLBAS ({mean_clbas:.2f}): Similar bias patterns despite different features")
        print("    -> This suggests 'language-adapted but semantically equivalent bias'")

    # Log to W&B
    if args.wandb and WANDB_AVAILABLE:
        # Log per-layer metrics
        for r in all_results:
            layer = r['layer']
            wandb.log({
                f"layer_{layer}/overlap_pct": r['overlap']['top_overall']['overlap_pct'],
                f"layer_{layer}/clbas_score": r['clbas']['clbas_score'],
                f"layer_{layer}/cosine_similarity": r['clbas']['cosine_similarity'],
                f"layer_{layer}/effect_correlation": r['clbas']['effect_size_correlation'],
            })
        
        # Log summary metrics
        wandb.log({
            "summary/mean_overlap_pct": mean_overlap,
            "summary/mean_clbas": mean_clbas,
        })
        
        # Log visualizations
        for img_path in viz_dir.glob("*.png"):
            wandb.log({img_path.stem: wandb.Image(str(img_path))})
        
        # Log results artifact
        artifact = wandb.Artifact("cross_lingual_overlap_results", type="results")
        artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)
        
        wandb.finish()
        print("\nW&B logging complete.")


if __name__ == "__main__":
    main()
