#!/usr/bin/env python3
"""
Surgical Bias Intervention (SBI) Analysis
==========================================

Tests whether ablating the identified language-specific gender features
actually reduces gender bias in model representations.

Key experiments:
1. Ablate top-k Arabic gender features → measure Arabic probe accuracy drop
2. Ablate top-k English gender features → measure English probe accuracy drop  
3. Cross-lingual test: Ablate Arabic features → test on English (expect no effect)
4. Measure semantic preservation (reconstruction quality)

This validates the causal role of the identified features.

Usage:
    python scripts/26_surgical_bias_intervention.py --config configs/config.yaml
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
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available, logging disabled")

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
    
    return features.cpu().numpy(), binary_labels, acts.cpu()


def get_top_gender_features(features: np.ndarray, labels: np.ndarray, k: int = 100) -> dict:
    """Identify top gender-associated features by effect size."""
    male_mask = labels == 1
    female_mask = labels == 0
    
    effect_sizes = np.zeros(features.shape[1])
    for i in range(features.shape[1]):
        male_vals = features[male_mask, i]
        female_vals = features[female_mask, i]
        pooled_std = np.sqrt((male_vals.std()**2 + female_vals.std()**2) / 2)
        if pooled_std > 1e-8:
            effect_sizes[i] = (male_vals.mean() - female_vals.mean()) / pooled_std
    
    # Top by absolute effect size
    top_idx = np.argsort(np.abs(effect_sizes))[-k:][::-1]
    
    return {
        'top_features': top_idx.tolist(),
        'effect_sizes': effect_sizes[top_idx].tolist(),
        'all_effect_sizes': effect_sizes
    }


def train_gender_probe(features: np.ndarray, labels: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """Train a gender probe and return accuracy."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


def ablate_features(features: np.ndarray, feature_indices: List[int]) -> np.ndarray:
    """Ablate (zero out) specified features."""
    features_ablated = features.copy()
    features_ablated[:, feature_indices] = 0
    return features_ablated


def neutralize_features(features: np.ndarray, labels: np.ndarray, feature_indices: List[int]) -> np.ndarray:
    """Neutralize features by averaging male/female values."""
    features_neutral = features.copy()
    male_mask = labels == 1
    female_mask = labels == 0
    
    for idx in feature_indices:
        avg_val = (features[male_mask, idx].mean() + features[female_mask, idx].mean()) / 2
        features_neutral[:, idx] = avg_val
    
    return features_neutral


def compute_reconstruction_quality(
    original_acts: torch.Tensor, 
    sae: SparseAutoencoder,
    ablated_features: np.ndarray,
    device: str = 'cpu'
) -> float:
    """Measure semantic preservation via reconstruction quality."""
    with torch.no_grad():
        # Original reconstruction
        original_features = sae.encode(original_acts.to(device))
        original_recon = sae.decode(original_features)
        
        # Ablated reconstruction
        ablated_tensor = torch.tensor(ablated_features, dtype=torch.float32, device=device)
        ablated_recon = sae.decode(ablated_tensor)
        
        # Compute similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original_recon.flatten(1), 
            ablated_recon.flatten(1),
            dim=1
        ).mean()
        
    return float(cos_sim)


def run_ablation_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    original_acts: torch.Tensor,
    sae: SparseAutoencoder,
    top_features: List[int],
    k_values: List[int] = [10, 25, 50, 100, 200],
    device: str = 'cpu'
) -> List[dict]:
    """Run ablation experiments with varying number of features."""
    results = []
    
    # Baseline accuracy
    baseline_acc, baseline_std = train_gender_probe(features, labels)
    print(f"  Baseline probe accuracy: {baseline_acc:.3f} ± {baseline_std:.3f}")
    
    for k in k_values:
        if k > len(top_features):
            continue
            
        features_to_ablate = top_features[:k]
        
        # Ablate features
        ablated_features = ablate_features(features, features_to_ablate)
        
        # Measure probe accuracy on ablated features
        ablated_acc, ablated_std = train_gender_probe(ablated_features, labels)
        
        # Measure semantic preservation
        recon_quality = compute_reconstruction_quality(original_acts, sae, ablated_features, device)
        
        accuracy_drop = baseline_acc - ablated_acc
        relative_drop = accuracy_drop / baseline_acc * 100
        
        result = {
            'k': k,
            'baseline_accuracy': float(baseline_acc),
            'ablated_accuracy': float(ablated_acc),
            'accuracy_drop': float(accuracy_drop),
            'relative_drop_pct': float(relative_drop),
            'reconstruction_quality': float(recon_quality),
            'features_ablated': features_to_ablate[:10]  # First 10 for reference
        }
        results.append(result)
        
        print(f"  k={k:3d}: acc {ablated_acc:.3f} (drop: {accuracy_drop:.3f}, {relative_drop:.1f}%), recon: {recon_quality:.3f}")
    
    return results


def run_cross_lingual_ablation(
    ar_features: np.ndarray, ar_labels: np.ndarray,
    en_features: np.ndarray, en_labels: np.ndarray,
    ar_top_features: List[int], en_top_features: List[int],
    k: int = 100
) -> dict:
    """
    Cross-lingual ablation test.
    
    If features are truly language-specific, ablating Arabic features
    should NOT affect English probe accuracy (and vice versa).
    """
    print("\n  Cross-Lingual Ablation Test:")
    
    # Baseline accuracies
    ar_baseline, _ = train_gender_probe(ar_features, ar_labels)
    en_baseline, _ = train_gender_probe(en_features, en_labels)
    
    # Same-language ablation (should reduce accuracy)
    ar_same_ablated = ablate_features(ar_features, ar_top_features[:k])
    en_same_ablated = ablate_features(en_features, en_top_features[:k])
    ar_same_acc, _ = train_gender_probe(ar_same_ablated, ar_labels)
    en_same_acc, _ = train_gender_probe(en_same_ablated, en_labels)
    
    # Cross-language ablation (should NOT reduce accuracy if features are language-specific)
    ar_cross_ablated = ablate_features(ar_features, en_top_features[:k])  # Ablate English features on Arabic
    en_cross_ablated = ablate_features(en_features, ar_top_features[:k])  # Ablate Arabic features on English
    ar_cross_acc, _ = train_gender_probe(ar_cross_ablated, ar_labels)
    en_cross_acc, _ = train_gender_probe(en_cross_ablated, en_labels)
    
    results = {
        'k': k,
        'arabic': {
            'baseline': float(ar_baseline),
            'same_language_ablation': float(ar_same_acc),
            'same_language_drop': float(ar_baseline - ar_same_acc),
            'cross_language_ablation': float(ar_cross_acc),
            'cross_language_drop': float(ar_baseline - ar_cross_acc),
        },
        'english': {
            'baseline': float(en_baseline),
            'same_language_ablation': float(en_same_acc),
            'same_language_drop': float(en_baseline - en_same_acc),
            'cross_language_ablation': float(en_cross_acc),
            'cross_language_drop': float(en_baseline - en_cross_acc),
        }
    }
    
    print(f"    Arabic:  same-lang drop={ar_baseline-ar_same_acc:.3f}, cross-lang drop={ar_baseline-ar_cross_acc:.3f}")
    print(f"    English: same-lang drop={en_baseline-en_same_acc:.3f}, cross-lang drop={en_baseline-en_cross_acc:.3f}")
    
    # Key insight
    ar_specificity = (ar_baseline - ar_same_acc) / max(ar_baseline - ar_cross_acc, 0.001)
    en_specificity = (en_baseline - en_same_acc) / max(en_baseline - en_cross_acc, 0.001)
    
    results['language_specificity'] = {
        'arabic': float(ar_specificity),
        'english': float(en_specificity),
        'interpretation': 'Higher values indicate more language-specific features'
    }
    
    return results


def create_visualizations(all_results: List[dict], output_dir: Path):
    """Create visualizations of SBI results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Accuracy vs K (ablated features) for each layer
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, layer_result in enumerate(all_results):
        if idx >= 7:
            break
        ax = axes[idx]
        layer = layer_result['layer']
        
        # Arabic
        ar_data = layer_result['arabic_ablation']
        k_vals = [r['k'] for r in ar_data]
        ar_accs = [r['ablated_accuracy'] for r in ar_data]
        ar_baseline = ar_data[0]['baseline_accuracy']
        
        # English  
        en_data = layer_result['english_ablation']
        en_accs = [r['ablated_accuracy'] for r in en_data]
        en_baseline = en_data[0]['baseline_accuracy']
        
        ax.axhline(ar_baseline, color='#e74c3c', linestyle='--', alpha=0.5, label='AR baseline')
        ax.axhline(en_baseline, color='#3498db', linestyle='--', alpha=0.5, label='EN baseline')
        ax.plot(k_vals, ar_accs, 'o-', color='#e74c3c', label='Arabic')
        ax.plot(k_vals, en_accs, 's-', color='#3498db', label='English')
        
        ax.set_xlabel('Features Ablated (k)')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title(f'Layer {layer}')
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.0)
    
    # Hide unused subplot
    axes[7].axis('off')
    
    plt.suptitle('SBI: Probe Accuracy After Ablating Top-k Gender Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'sbi_accuracy_vs_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Cross-lingual specificity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = [r['layer'] for r in all_results]
    ar_same = [r['cross_lingual']['arabic']['same_language_drop'] for r in all_results]
    ar_cross = [r['cross_lingual']['arabic']['cross_language_drop'] for r in all_results]
    en_same = [r['cross_lingual']['english']['same_language_drop'] for r in all_results]
    en_cross = [r['cross_lingual']['english']['cross_language_drop'] for r in all_results]
    
    x = np.arange(len(layers))
    width = 0.2
    
    ax.bar(x - 1.5*width, ar_same, width, label='AR same-lang', color='#e74c3c')
    ax.bar(x - 0.5*width, ar_cross, width, label='AR cross-lang', color='#e74c3c', alpha=0.4)
    ax.bar(x + 0.5*width, en_same, width, label='EN same-lang', color='#3498db')
    ax.bar(x + 1.5*width, en_cross, width, label='EN cross-lang', color='#3498db', alpha=0.4)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('Cross-Lingual Ablation: Same-Language vs Cross-Language Effect')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sbi_cross_lingual_specificity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Reconstruction quality vs accuracy drop trade-off
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for layer_result in all_results:
        layer = layer_result['layer']
        ar_data = layer_result['arabic_ablation']
        
        acc_drops = [r['accuracy_drop'] for r in ar_data]
        recon_quals = [r['reconstruction_quality'] for r in ar_data]
        
        ax.plot(acc_drops, recon_quals, 'o-', label=f'Layer {layer}', alpha=0.7)
    
    ax.set_xlabel('Accuracy Drop (Bias Reduction)')
    ax.set_ylabel('Reconstruction Quality (Semantic Preservation)')
    ax.set_title('Trade-off: Bias Reduction vs Semantic Preservation')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sbi_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualizations saved to {output_dir}")


def analyze_layer(
    ar_sae_path: str, en_sae_path: str,
    ar_acts_path: str, en_acts_path: str,
    layer: int, device: str = 'cpu'
) -> dict:
    """Run full SBI analysis for a single layer."""
    print(f"\n{'='*60}")
    print(f"  SBI Analysis - Layer {layer}")
    print(f"{'='*60}")
    
    # Load SAEs
    print("Loading SAEs...")
    ar_sae = load_sae(ar_sae_path, device)
    en_sae = load_sae(en_sae_path, device)
    
    # Load activations and get features
    print("Loading Arabic data...")
    ar_features, ar_labels, ar_acts = get_features_and_labels(load_activations(ar_acts_path), ar_sae, device)
    print(f"  {ar_features.shape[0]} samples")
    
    print("Loading English data...")
    en_features, en_labels, en_acts = get_features_and_labels(load_activations(en_acts_path), en_sae, device)
    print(f"  {en_features.shape[0]} samples")
    
    # Get top gender features
    print("Identifying top gender features...")
    ar_top = get_top_gender_features(ar_features, ar_labels, k=200)
    en_top = get_top_gender_features(en_features, en_labels, k=200)
    
    # Run ablation experiments
    print("\nArabic ablation experiment:")
    ar_ablation_results = run_ablation_experiment(
        ar_features, ar_labels, ar_acts, ar_sae, 
        ar_top['top_features'], device=device
    )
    
    print("\nEnglish ablation experiment:")
    en_ablation_results = run_ablation_experiment(
        en_features, en_labels, en_acts, en_sae,
        en_top['top_features'], device=device
    )
    
    # Cross-lingual ablation test
    cross_lingual_results = run_cross_lingual_ablation(
        ar_features, ar_labels,
        en_features, en_labels,
        ar_top['top_features'], en_top['top_features'],
        k=100
    )
    
    return {
        'layer': layer,
        'arabic_ablation': ar_ablation_results,
        'english_ablation': en_ablation_results,
        'cross_lingual': cross_lingual_results,
        'top_features': {
            'arabic': ar_top['top_features'][:20],
            'english': en_top['top_features'][:20]
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Surgical Bias Intervention analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--layers", type=str, default="0,3,6,9,12,15,17")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/sbi_analysis")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="sae-cross-lingual-bias")
    args = parser.parse_args()
    
    # Initialize W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=f"sbi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "layers": args.layers,
                "device": args.device,
                "experiment": "surgical_bias_intervention"
            }
        )
        print("W&B logging enabled")
    
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
        
        if layer == 0 and not en_sae_path.exists():
            en_sae_path = sae_dir / "sae_layer_0.pt"
        
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
                str(ar_sae_path), str(en_sae_path),
                str(ar_acts_path), str(en_acts_path),
                layer, args.device
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
    results_path = output_dir / "sbi_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Create visualizations
    create_visualizations(all_results, output_dir / "visualizations")
    
    # Log to W&B
    if args.wandb and WANDB_AVAILABLE:
        # Log summary metrics
        for r in all_results:
            layer = r['layer']
            ar_drop = next((x['accuracy_drop'] for x in r['arabic_ablation'] if x['k'] == 100), 0)
            en_drop = next((x['accuracy_drop'] for x in r['english_ablation'] if x['k'] == 100), 0)
            
            wandb.log({
                f"layer_{layer}/arabic_accuracy_drop_k100": ar_drop,
                f"layer_{layer}/english_accuracy_drop_k100": en_drop,
                f"layer_{layer}/arabic_same_lang_drop": r['cross_lingual']['arabic']['same_language_drop'],
                f"layer_{layer}/arabic_cross_lang_drop": r['cross_lingual']['arabic']['cross_language_drop'],
                f"layer_{layer}/english_same_lang_drop": r['cross_lingual']['english']['same_language_drop'],
                f"layer_{layer}/english_cross_lang_drop": r['cross_lingual']['english']['cross_language_drop'],
            })
        
        # Log visualizations
        viz_dir = output_dir / "visualizations"
        for img_path in viz_dir.glob("*.png"):
            wandb.log({img_path.stem: wandb.Image(str(img_path))})
        
        # Log artifact
        artifact = wandb.Artifact("sbi_results", type="results")
        artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)
    
    # Print summary
    print("\n" + "="*80)
    print("  SURGICAL BIAS INTERVENTION SUMMARY")
    print("="*80)
    
    print(f"\n{'Layer':<8}{'AR Drop@100':<14}{'EN Drop@100':<14}{'AR Cross-Drop':<14}{'EN Cross-Drop':<14}")
    print("-"*64)
    
    for r in all_results:
        # Find k=100 results
        ar_drop = next((x['accuracy_drop'] for x in r['arabic_ablation'] if x['k'] == 100), 0)
        en_drop = next((x['accuracy_drop'] for x in r['english_ablation'] if x['k'] == 100), 0)
        ar_cross = r['cross_lingual']['arabic']['cross_language_drop']
        en_cross = r['cross_lingual']['english']['cross_language_drop']
        
        print(f"{r['layer']:<8}{ar_drop:<14.3f}{en_drop:<14.3f}{ar_cross:<14.3f}{en_cross:<14.3f}")
    
    # Key insights
    print("\n" + "="*80)
    print("  KEY INSIGHTS")
    print("="*80)
    
    # Average drops
    avg_ar_same = np.mean([r['cross_lingual']['arabic']['same_language_drop'] for r in all_results])
    avg_ar_cross = np.mean([r['cross_lingual']['arabic']['cross_language_drop'] for r in all_results])
    avg_en_same = np.mean([r['cross_lingual']['english']['same_language_drop'] for r in all_results])
    avg_en_cross = np.mean([r['cross_lingual']['english']['cross_language_drop'] for r in all_results])
    
    print(f"\n  Average Same-Language Ablation Effect:")
    print(f"    Arabic:  {avg_ar_same:.3f} accuracy drop")
    print(f"    English: {avg_en_same:.3f} accuracy drop")
    
    print(f"\n  Average Cross-Language Ablation Effect:")
    print(f"    Arabic features → English: {avg_en_cross:.3f} accuracy drop")
    print(f"    English features → Arabic: {avg_ar_cross:.3f} accuracy drop")
    
    if avg_ar_same > 3 * avg_ar_cross and avg_en_same > 3 * avg_en_cross:
        print("\n  ✓ CONFIRMED: Gender features are CAUSALLY language-specific!")
        print("    Same-language ablation has 3x+ greater effect than cross-language")
        finding = "language_specific_confirmed"
    else:
        print("\n  ~ Mixed results: Some feature overlap in causal effects")
        finding = "mixed_results"
    
    # Log final summary to W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            "summary/avg_arabic_same_lang_drop": avg_ar_same,
            "summary/avg_arabic_cross_lang_drop": avg_ar_cross,
            "summary/avg_english_same_lang_drop": avg_en_same,
            "summary/avg_english_cross_lang_drop": avg_en_cross,
            "summary/arabic_specificity_ratio": avg_ar_same / max(avg_ar_cross, 0.001),
            "summary/english_specificity_ratio": avg_en_same / max(avg_en_cross, 0.001),
            "summary/finding": finding
        })
        wandb.finish()
        print("\nW&B logging complete.")


if __name__ == "__main__":
    main()
