#!/usr/bin/env python3
"""
Compute SAE Quality Metrics for Llama 3.2 Vision (11B)
======================================================

Computes publication-ready metrics:
1. Explained Variance %
2. Dead Feature Ratio  
3. Mean L0 (active features per sample)
4. Reconstruction Cosine Similarity

Compatible with Llama32VisionSAE checkpoint format (.pt with 'state_dict' key)
and NPZ activation files.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import gc
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")


class Llama32VisionSAE(nn.Module):
    """Sparse Autoencoder matching Llama 3.2 Vision checkpoint format."""

    def __init__(self, d_model: int = 4096, expansion_factor: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.encoder = nn.Linear(d_model, self.d_hidden)
        self.decoder = nn.Linear(self.d_hidden, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features


def load_sae(path: str) -> Llama32VisionSAE:
    """Load SAE from Llama 3.2 Vision checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    d_model = ckpt['d_model']
    expansion_factor = ckpt.get('expansion_factor', 8)

    sae = Llama32VisionSAE(d_model, expansion_factor)
    sae.load_state_dict(ckpt['state_dict'])
    sae.eval()
    return sae


def load_activations_npz(path: str, max_samples: int = 500) -> torch.Tensor:
    """Load activations from NPZ file."""
    data = np.load(path)
    acts = data['activations']

    # Random sample
    n = len(acts)
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        acts = acts[indices]

    return torch.from_numpy(acts).float()


def compute_metrics(sae: Llama32VisionSAE, activations: torch.Tensor, batch_size: int = 256) -> Dict:
    """Compute quality metrics for SAE."""
    sae.eval()
    n_samples = len(activations)
    n_features = sae.d_hidden

    feature_active_count = torch.zeros(n_features)
    total_l0 = 0.0
    total_l0_sq = 0.0
    total_cos_sim = 0.0

    sum_input = torch.zeros(sae.d_model)
    sum_sq_input = torch.zeros(sae.d_model)
    sum_residual = torch.zeros(sae.d_model)
    sum_sq_residual = torch.zeros(sae.d_model)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = activations[i:i+batch_size]

            features = sae.encode(batch)
            reconstruction = sae.decode(features)
            residual = batch - reconstruction

            active = (features > 1e-6).float()
            feature_active_count += active.sum(dim=0)
            l0_batch = active.sum(dim=1)
            total_l0 += l0_batch.sum().item()
            total_l0_sq += (l0_batch ** 2).sum().item()

            cos_batch = nn.functional.cosine_similarity(batch, reconstruction, dim=1)
            total_cos_sim += cos_batch.sum().item()

            sum_input += batch.sum(dim=0)
            sum_sq_input += (batch ** 2).sum(dim=0)
            sum_residual += residual.sum(dim=0)
            sum_sq_residual += (residual ** 2).sum(dim=0)

    # 1. Dead Feature Ratio
    n_alive = (feature_active_count > 0).sum().item()
    dead_ratio = 1 - (n_alive / n_features)

    # 2. Mean L0
    mean_l0 = total_l0 / n_samples
    std_l0 = np.sqrt(max(0, total_l0_sq / n_samples - mean_l0 ** 2))

    # 3. Cosine Similarity
    mean_cos_sim = total_cos_sim / n_samples

    # 4. Explained Variance
    input_var = (sum_sq_input / n_samples) - (sum_input / n_samples) ** 2
    residual_var = (sum_sq_residual / n_samples) - (sum_residual / n_samples) ** 2
    input_var = torch.clamp(input_var, min=1e-10)
    explained_var_per_dim = 1 - (residual_var / input_var)
    explained_var_per_dim = torch.clamp(explained_var_per_dim, 0, 1)
    mean_explained_var = explained_var_per_dim.mean().item()

    # 5. MSE
    mse = (sum_sq_residual.sum() / (n_samples * sae.d_model)).item()

    return {
        'explained_variance_pct': mean_explained_var * 100,
        'dead_feature_pct': dead_ratio * 100,
        'alive_features': n_alive,
        'total_features': n_features,
        'mean_l0': mean_l0,
        'std_l0': std_l0,
        'l0_sparsity_pct': (mean_l0 / n_features) * 100,
        'reconstruction_cosine': mean_cos_sim,
        'reconstruction_mse': mse,
        'n_samples': n_samples
    }


def main():
    print("=" * 70)
    print("SAE Quality Metrics - Llama 3.2 Vision (11B)")
    print("=" * 70)

    sae_dir = BASE_PATH / 'checkpoints/llama32vision/saes'
    act_dir = BASE_PATH / 'checkpoints/llama32vision/layer_checkpoints'
    output_dir = BASE_PATH / 'results/sae_quality_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [0, 5, 10, 15, 20, 25, 30, 35, 39]

    results = {
        'model': 'Llama-3.2-Vision-11B',
        'd_model': 4096,
        'd_hidden': 32768,
        'layers': {}
    }

    for layer in layers:
        print(f"\n  Layer {layer}...")
        layer_results = {}

        for lang in ['arabic', 'english']:
            sae_path = sae_dir / f'llama32vision_sae_{lang}_layer{layer}.pt'
            act_path = act_dir / f'llama32vision_{lang}_layer{layer}_checkpoint0.npz'

            if not sae_path.exists():
                print(f"    {lang}: SAE not found at {sae_path}")
                continue
            if not act_path.exists():
                print(f"    {lang}: Activations not found at {act_path}")
                continue

            try:
                sae = load_sae(str(sae_path))
                acts = load_activations_npz(str(act_path))
                metrics = compute_metrics(sae, acts)
                layer_results[lang] = metrics

                print(f"    {lang}: EV={metrics['explained_variance_pct']:.1f}%, "
                      f"Dead={metrics['dead_feature_pct']:.1f}%, "
                      f"L0={metrics['mean_l0']:.0f}, "
                      f"Cos={metrics['reconstruction_cosine']:.4f}")

                del sae, acts
                gc.collect()

            except Exception as e:
                print(f"    {lang}: Error - {e}")
                import traceback
                traceback.print_exc()

        if layer_results:
            results['layers'][layer] = layer_results

    # Compute summary statistics
    ev_vals, dead_vals, l0_vals, cos_vals = [], [], [], []

    for layer_data in results['layers'].values():
        for metrics in layer_data.values():
            ev_vals.append(metrics['explained_variance_pct'])
            dead_vals.append(metrics['dead_feature_pct'])
            l0_vals.append(metrics['mean_l0'])
            cos_vals.append(metrics['reconstruction_cosine'])

    summary = {
        'd_model': 4096,
        'd_hidden': 32768,
        'explained_variance_pct': {'mean': np.mean(ev_vals), 'std': np.std(ev_vals)} if ev_vals else None,
        'dead_feature_pct': {'mean': np.mean(dead_vals), 'std': np.std(dead_vals)} if dead_vals else None,
        'mean_l0': {'mean': np.mean(l0_vals), 'std': np.std(l0_vals)} if l0_vals else None,
        'reconstruction_cosine': {'mean': np.mean(cos_vals), 'std': np.std(cos_vals)} if cos_vals else None,
    }

    # Save results
    output_file = output_dir / 'llama32vision_metrics.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model_results': results,
            'summary': summary
        }, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 90)
    print("LLAMA 3.2 VISION SAE QUALITY METRICS")
    print("=" * 90)

    if summary['explained_variance_pct']:
        ev = summary['explained_variance_pct']
        dead = summary['dead_feature_pct']
        l0 = summary['mean_l0']
        cos = summary['reconstruction_cosine']

        print(f"\n| Metric | Value |")
        print(f"|--------|-------|")
        print(f"| Explained Variance % | {ev['mean']:.1f} ± {ev['std']:.1f} |")
        print(f"| Dead Feature % | {dead['mean']:.1f} ± {dead['std']:.1f} |")
        print(f"| Mean L0 | {l0['mean']:.0f} ± {l0['std']:.0f} |")
        print(f"| Reconstruction Cosine | {cos['mean']:.4f} ± {cos['std']:.4f} |")

    print(f"\n✓ Results saved to: {output_file}")

    # Print per-layer details
    print(f"\n{'='*90}")
    print("DETAILED PER-LAYER METRICS")
    print(f"{'='*90}")
    print(f"\n| Layer | Language | EV% | Dead% | L0 | Cosine |")
    print(f"|-------|----------|-----|-------|-----|--------|")

    for layer in sorted(results['layers'].keys()):
        for lang, m in results['layers'][layer].items():
            print(f"| {layer} | {lang} | {m['explained_variance_pct']:.1f} | "
                  f"{m['dead_feature_pct']:.1f} | {m['mean_l0']:.0f} | {m['reconstruction_cosine']:.4f} |")

    # Standards comparison
    print(f"\n{'='*90}")
    print("COMPARISON WITH STANDARDS (Anthropic 2024)")
    print(f"{'='*90}")

    if summary['explained_variance_pct']:
        ev_val = summary['explained_variance_pct']['mean']
        dead_val = summary['dead_feature_pct']['mean']
        l0_val = summary['mean_l0']['mean']
        cos_val = summary['reconstruction_cosine']['mean']

        print(f"  Explained Variance: {ev_val:.1f}% (target: >65%) {'✓' if ev_val >= 65 else '⚠️'}")
        print(f"  Dead Features: {dead_val:.1f}% (target: <35%) {'✓' if dead_val < 35 else '⚠️'}")
        print(f"  Mean L0: {l0_val:.0f} (target: 50-300) {'✓' if 50 <= l0_val <= 300 else '⚠️'}")
        print(f"  Recon Cosine: {cos_val:.4f} (target: >0.9) {'✓' if cos_val >= 0.9 else '⚠️'}")


if __name__ == "__main__":
    main()
