#!/usr/bin/env python3
"""
Llama 3.2 Vision Cross-Lingual Gender Bias Analysis
====================================================

Analyze cross-lingual gender bias in Llama 3.2 Vision using trained SAEs.
Computes CLBAS, cosine similarity, and probe accuracy metrics.

Usage:
    python scripts/40_llama32vision_cross_lingual_analysis.py \
        --layers 0,10,20,30,39 \
        --checkpoints_dir checkpoints/llama32vision \
        --output_dir results/llama32vision_analysis
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import wandb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# Model configuration
MODEL_NAME = "Llama-3.2-Vision-11B"
HIDDEN_SIZE = 4096
NUM_LAYERS = 40


class Llama32VisionSAE(nn.Module):
    """Sparse Autoencoder for Llama 3.2 Vision."""

    def __init__(self, d_model: int = 4096, expansion_factor: int = 8, l1_coefficient: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.l1_coefficient = l1_coefficient
        self.encoder = nn.Linear(d_model, self.d_hidden)
        self.decoder = nn.Linear(self.d_hidden, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))


def load_sae(sae_path: Path, device: str = "cpu") -> Llama32VisionSAE:
    """Load trained SAE."""
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)

    sae = Llama32VisionSAE(
        d_model=checkpoint['d_model'],
        expansion_factor=checkpoint['expansion_factor'],
        l1_coefficient=checkpoint.get('l1_coefficient', 1e-4)
    )
    sae.load_state_dict(checkpoint['state_dict'])
    sae.eval()

    return sae, checkpoint.get('gender_features', {})


def load_activations(checkpoints_dir: Path, language: str, layer: int) -> Tuple[np.ndarray, List[str]]:
    """Load activation files."""
    pattern = f"llama32vision_{language}_layer{layer}_*.npz"
    files = list(checkpoints_dir.glob(pattern))

    if not files:
        return None, None

    all_activations = []
    all_genders = []

    for f in sorted(files):
        data = np.load(f)
        all_activations.append(data['activations'])
        all_genders.extend(data['genders'].tolist())

    return np.concatenate(all_activations, axis=0), all_genders


def compute_cosine_similarity(ar_direction: np.ndarray, en_direction: np.ndarray) -> float:
    """Compute cosine similarity between gender directions."""
    ar_norm = ar_direction / (np.linalg.norm(ar_direction) + 1e-8)
    en_norm = en_direction / (np.linalg.norm(en_direction) + 1e-8)
    return float(np.dot(ar_norm, en_norm))


def compute_feature_overlap(ar_features: List[int], en_features: List[int], top_k: int = 100) -> Dict:
    """Compute overlap between Arabic and English gender features."""
    ar_set = set(ar_features[:top_k])
    en_set = set(en_features[:top_k])

    overlap = ar_set & en_set

    return {
        'overlap_count': len(overlap),
        'overlap_pct': len(overlap) / top_k * 100,
        'overlap_features': list(overlap),
        'jaccard': len(overlap) / len(ar_set | en_set) if ar_set | en_set else 0
    }


def train_gender_probe(activations: np.ndarray, genders: List[str]) -> float:
    """Train logistic regression probe and return accuracy."""
    genders = np.array(genders)
    mask = (genders == "male") | (genders == "female")

    X = activations[mask]
    y = (genders[mask] == "male").astype(int)

    if len(np.unique(y)) < 2 or len(y) < 20:
        return 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5)

    return float(scores.mean())


def analyze_layer(
    layer: int,
    checkpoints_dir: Path,
    saes_dir: Path,
    device: str = "cpu"
) -> Optional[Dict]:
    """Analyze a single layer."""

    # Load activations
    ar_activations, ar_genders = load_activations(checkpoints_dir / "layer_checkpoints", "arabic", layer)
    en_activations, en_genders = load_activations(checkpoints_dir / "layer_checkpoints", "english", layer)

    if ar_activations is None or en_activations is None:
        print(f"  Layer {layer}: Missing activations")
        return None

    # Load SAEs
    ar_sae_path = saes_dir / f"llama32vision_sae_arabic_layer{layer}.pt"
    en_sae_path = saes_dir / f"llama32vision_sae_english_layer{layer}.pt"

    if not ar_sae_path.exists() or not en_sae_path.exists():
        print(f"  Layer {layer}: Missing SAEs")
        return None

    ar_sae, ar_gender_features = load_sae(ar_sae_path, device)
    en_sae, en_gender_features = load_sae(en_sae_path, device)

    # Compute gender probe accuracy
    ar_probe_acc = train_gender_probe(ar_activations, ar_genders)
    en_probe_acc = train_gender_probe(en_activations, en_genders)

    # Compute cosine similarity of gender directions
    ar_direction = np.array(ar_gender_features.get('gender_direction', []))
    en_direction = np.array(en_gender_features.get('gender_direction', []))

    if len(ar_direction) > 0 and len(en_direction) > 0:
        cosine_sim = compute_cosine_similarity(ar_direction, en_direction)
    else:
        cosine_sim = 0.0

    # Compute feature overlap
    ar_male_features = ar_gender_features.get('male_features', [])
    en_male_features = en_gender_features.get('male_features', [])
    overlap = compute_feature_overlap(ar_male_features, en_male_features)

    # Compute CLBAS (Cross-Lingual Bias Alignment Score)
    clbas = abs(cosine_sim) * (1 + overlap['overlap_pct'] / 100) / 2

    return {
        'layer': layer,
        'ar_probe_acc': ar_probe_acc,
        'en_probe_acc': en_probe_acc,
        'cosine_sim': cosine_sim,
        'clbas': clbas,
        'overlap_count': overlap['overlap_count'],
        'overlap_pct': overlap['overlap_pct'],
        'jaccard': overlap['jaccard'],
        'ar_samples': len(ar_genders),
        'en_samples': len(en_genders)
    }


def generate_visualizations(results: List[Dict], output_dir: Path):
    """Generate analysis visualizations."""

    layers = [r['layer'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{MODEL_NAME} Cross-Lingual Gender Bias Analysis', fontsize=14)

    # 1. Probe accuracy comparison
    ax1 = axes[0, 0]
    ax1.plot(layers, [r['ar_probe_acc'] for r in results], 'o-', label='Arabic', color='#e74c3c')
    ax1.plot(layers, [r['en_probe_acc'] for r in results], 's-', label='English', color='#3498db')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Probe Accuracy')
    ax1.set_title('Gender Probe Accuracy by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cosine similarity
    ax2 = axes[0, 1]
    ax2.bar(layers, [r['cosine_sim'] for r in results], color='#9b59b6')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cross-Lingual Gender Direction Similarity')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 3. CLBAS
    ax3 = axes[1, 0]
    ax3.plot(layers, [r['clbas'] for r in results], 'D-', color='#e67e22', markersize=8)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('CLBAS')
    ax3.set_title('Cross-Lingual Bias Alignment Score')
    ax3.grid(True, alpha=0.3)

    # 4. Feature overlap
    ax4 = axes[1, 1]
    ax4.bar(layers, [r['overlap_count'] for r in results], color='#27ae60')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Overlap Count (top-100)')
    ax4.set_title('Shared Gender Features')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'llama32vision_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'llama32vision_analysis.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Llama 3.2 Vision Cross-Lingual Analysis")
    parser.add_argument("--layers", type=str, default="0,5,10,15,20,25,30,35,39",
                        help="Comma-separated list of layers")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/llama32vision")
    parser.add_argument("--output_dir", type=str, default="results/llama32vision_analysis")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb", action="store_true", default=True, help="Log to W&B (enabled by default)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llama32vision-sae-analysis")
    parser.add_argument("--wandb_entity", type=str, default="nourmubarak")
    args = parser.parse_args()

    # Handle wandb flag
    if args.no_wandb:
        args.wandb = False

    # Parse layers
    layers = [int(l.strip()) for l in args.layers.split(",")]

    print("=" * 60)
    print(f"Llama 3.2 Vision Cross-Lingual Analysis")
    print(f"Layers: {layers}")
    print("=" * 60)

    # Initialize W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"llama32vision_cross_lingual_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": MODEL_NAME,
                "d_model": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "layers_analyzed": layers,
                "checkpoints_dir": str(args.checkpoints_dir),
                "output_dir": str(args.output_dir),
                "device": args.device
            },
            tags=["cross-lingual", "analysis", "llama32vision"]
        )
        print(f"W&B initialized: {wandb.run.url}")

    # Setup paths
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saes_dir = checkpoints_dir / "saes"

    # Analyze each layer
    print("\nAnalyzing layers...")
    results = []

    for layer in layers:
        print(f"\nLayer {layer}:")
        result = analyze_layer(layer, checkpoints_dir, saes_dir, args.device)
        if result:
            results.append(result)
            print(f"  AR Probe: {result['ar_probe_acc']:.3f}")
            print(f"  EN Probe: {result['en_probe_acc']:.3f}")
            print(f"  Cosine Sim: {result['cosine_sim']:.4f}")
            print(f"  CLBAS: {result['clbas']:.4f}")
            print(f"  Overlap: {result['overlap_count']}")

            if args.wandb:
                wandb.log({
                    f"layer_{layer}/ar_probe_acc": result['ar_probe_acc'],
                    f"layer_{layer}/en_probe_acc": result['en_probe_acc'],
                    f"layer_{layer}/cosine_sim": result['cosine_sim'],
                    f"layer_{layer}/clbas": result['clbas'],
                    f"layer_{layer}/overlap_pct": result['overlap_pct']
                })

    if not results:
        print("\nNo layers analyzed successfully!")
        return

    # Compute summary statistics
    summary = {
        'model': MODEL_NAME,
        'd_model': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'mean_ar_probe_acc': np.mean([r['ar_probe_acc'] for r in results]),
        'mean_en_probe_acc': np.mean([r['en_probe_acc'] for r in results]),
        'mean_cosine_sim': np.mean([r['cosine_sim'] for r in results]),
        'mean_clbas': np.mean([r['clbas'] for r in results]),
        'mean_overlap': np.mean([r['overlap_count'] for r in results]),
        'probe_gap': np.mean([r['en_probe_acc'] - r['ar_probe_acc'] for r in results])
    }

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(results, output_dir)

    # Save results
    output_data = {
        'model': MODEL_NAME,
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'layer_results': {f"layer_{r['layer']}": r for r in results}
    }

    with open(output_dir / 'cross_lingual_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Analysis Complete!")
    print(f"{'=' * 60}")
    print(f"\nSummary:")
    print(f"  Mean AR Probe Acc: {summary['mean_ar_probe_acc']:.3f}")
    print(f"  Mean EN Probe Acc: {summary['mean_en_probe_acc']:.3f}")
    print(f"  Probe Gap: {summary['probe_gap']:+.3f} (EN - AR)")
    print(f"  Mean Cosine Sim: {summary['mean_cosine_sim']:.4f}")
    print(f"  Mean CLBAS: {summary['mean_clbas']:.4f}")
    print(f"\nResults saved to: {output_dir}")

    if args.wandb:
        wandb.log({
            "summary/mean_ar_probe_acc": summary['mean_ar_probe_acc'],
            "summary/mean_en_probe_acc": summary['mean_en_probe_acc'],
            "summary/mean_cosine": summary['mean_cosine_sim'],
            "summary/mean_clbas": summary['mean_clbas'],
            "summary/mean_overlap": summary['mean_overlap']
        })
        wandb.finish()


if __name__ == "__main__":
    main()
