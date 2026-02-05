#!/usr/bin/env python3
"""
Compute SAE Quality Metrics from Small Samples
===============================================

Memory-efficient computation of:
1. Explained Variance % 
2. Dead Feature Ratio
3. Mean L0 (active features per sample)
4. Reconstruction Cosine Similarity

Uses small activation samples to fit in memory.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import gc
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")


@dataclass
class SAEConfig:
    d_model: int
    d_hidden: int


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder matching checkpoint format."""
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.decoder.bias))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features


def load_sae(path: str) -> SparseAutoencoder:
    """Load SAE from checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    
    d_model = ckpt['d_model']
    d_hidden = ckpt['d_hidden']
    
    sae = SparseAutoencoder(d_model, d_hidden)
    sae.load_state_dict(ckpt['model_state_dict'])
    sae.eval()
    return sae


def sample_activations(path: str, max_samples: int = 500) -> torch.Tensor:
    """Load a small sample of activations."""
    # Use memory mapping to avoid loading full file
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    if isinstance(data, dict):
        acts = data.get('activations', data.get('features'))
    else:
        acts = data
    
    # Free the original data dict
    if isinstance(data, dict):
        del data
        gc.collect()
    
    # Flatten if needed
    if acts.ndim == 3:
        acts = acts.view(-1, acts.shape[-1])
    
    # Random sample - much smaller to fit in memory
    n = len(acts)
    if n > max_samples:
        indices = torch.randperm(n)[:max_samples]
        acts_sample = acts[indices].clone()
        del acts
        gc.collect()
        return acts_sample.float()
    
    return acts.float()


def compute_metrics(sae: SparseAutoencoder, activations: torch.Tensor, batch_size: int = 256) -> Dict:
    """Compute quality metrics for SAE."""
    sae.eval()
    n_samples = len(activations)
    n_features = sae.d_hidden
    
    # Initialize counters
    feature_active_count = torch.zeros(n_features)
    total_l0 = 0.0
    total_l0_sq = 0.0
    total_cos_sim = 0.0
    
    # Welford's algorithm for online variance calculation
    sum_input = torch.zeros(sae.d_model)
    sum_sq_input = torch.zeros(sae.d_model)
    sum_residual = torch.zeros(sae.d_model)
    sum_sq_residual = torch.zeros(sae.d_model)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = activations[i:i+batch_size]
            
            # Forward pass
            features = sae.encode(batch)
            reconstruction = sae.decode(features)
            residual = batch - reconstruction
            
            # L0 computation
            active = (features > 1e-6).float()
            feature_active_count += active.sum(dim=0)
            l0_batch = active.sum(dim=1)
            total_l0 += l0_batch.sum().item()
            total_l0_sq += (l0_batch ** 2).sum().item()
            
            # Cosine similarity
            cos_batch = nn.functional.cosine_similarity(batch, reconstruction, dim=1)
            total_cos_sim += cos_batch.sum().item()
            
            # Running sums for variance
            sum_input += batch.sum(dim=0)
            sum_sq_input += (batch ** 2).sum(dim=0)
            sum_residual += residual.sum(dim=0)
            sum_sq_residual += (residual ** 2).sum(dim=0)
    
    # Compute final metrics
    
    # 1. Dead Feature Ratio
    n_alive = (feature_active_count > 0).sum().item()
    dead_ratio = 1 - (n_alive / n_features)
    
    # 2. Mean L0
    mean_l0 = total_l0 / n_samples
    std_l0 = np.sqrt(max(0, total_l0_sq / n_samples - mean_l0 ** 2))
    
    # 3. Cosine Similarity
    mean_cos_sim = total_cos_sim / n_samples
    
    # 4. Explained Variance (per-dimension)
    input_var = (sum_sq_input / n_samples) - (sum_input / n_samples) ** 2
    residual_var = (sum_sq_residual / n_samples) - (sum_residual / n_samples) ** 2
    
    # Avoid division by zero
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


def process_model(model_name: str, sae_dir: Path, act_dir: Path, 
                  layers: List[int], sae_pattern: str, act_pattern: str,
                  d_model: int, d_hidden: int) -> Dict:
    """Process all layers for a model."""
    print(f"\n{'='*60}")
    print(f"Computing metrics for {model_name}")
    print(f"{'='*60}")
    
    results = {
        'model': model_name,
        'd_model': d_model,
        'd_hidden': d_hidden,
        'layers': {}
    }
    
    for layer in layers:
        print(f"\n  Layer {layer}...")
        layer_results = {}
        
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / sae_pattern.format(lang=lang, layer=layer)
            act_path = act_dir / act_pattern.format(lang=lang, layer=layer)
            
            if not sae_path.exists():
                print(f"    {lang}: SAE not found")
                continue
            if not act_path.exists():
                print(f"    {lang}: Activations not found")
                continue
            
            try:
                sae = load_sae(str(sae_path))
                acts = sample_activations(str(act_path))
                metrics = compute_metrics(sae, acts)
                layer_results[lang] = metrics
                
                print(f"    {lang}: EV={metrics['explained_variance_pct']:.1f}%, "
                      f"Dead={metrics['dead_feature_pct']:.1f}%, "
                      f"L0={metrics['mean_l0']:.0f}, "
                      f"Cos={metrics['reconstruction_cosine']:.4f}")
                
                # Free memory
                del sae, acts
                gc.collect()
                
            except Exception as e:
                print(f"    {lang}: Error - {e}")
        
        if layer_results:
            results['layers'][layer] = layer_results
    
    return results


def main():
    print("="*70)
    print("SAE Quality Metrics Computation (Publication-Ready)")
    print("="*70)
    
    # PaLiGemma-3B
    paligemma = process_model(
        model_name='PaLiGemma-3B',
        sae_dir=BASE_PATH / 'checkpoints/saes',
        act_dir=BASE_PATH / 'checkpoints/full_layers_ncc/layer_checkpoints',
        layers=[0, 3, 6, 9, 12, 15, 17],
        sae_pattern='sae_{lang}_layer_{layer}.pt',
        act_pattern='layer_{layer}_{lang}.pt',
        d_model=2048,
        d_hidden=16384
    )
    
    # Qwen2-VL-7B
    qwen2vl = process_model(
        model_name='Qwen2-VL-7B',
        sae_dir=BASE_PATH / 'checkpoints/qwen2vl/saes',
        act_dir=BASE_PATH / 'checkpoints/qwen2vl/layer_checkpoints',
        layers=[0, 4, 8, 12, 16, 20, 24, 27],
        sae_pattern='qwen2vl_sae_{lang}_layer_{layer}.pt',
        act_pattern='qwen2vl_layer_{layer}_{lang}.pt',
        d_model=3584,
        d_hidden=28672
    )
    
    # LLaVA-1.5-7B
    llava = process_model(
        model_name='LLaVA-1.5-7B',
        sae_dir=BASE_PATH / 'checkpoints/llava/saes',
        act_dir=BASE_PATH / 'checkpoints/llava/layer_checkpoints',
        layers=[0, 4, 8, 12, 16, 20, 24, 28, 31],
        sae_pattern='llava_sae_{lang}_layer_{layer}.pt',
        act_pattern='llava_layer_{layer}_{lang}.pt',
        d_model=4096,
        d_hidden=32768
    )
    
    all_results = [paligemma, qwen2vl, llava]
    
    # Compute summary statistics
    summary = {}
    for model_data in all_results:
        model = model_data['model']
        ev_vals, dead_vals, l0_vals, cos_vals = [], [], [], []
        
        for layer_data in model_data['layers'].values():
            for metrics in layer_data.values():
                ev_vals.append(metrics['explained_variance_pct'])
                dead_vals.append(metrics['dead_feature_pct'])
                l0_vals.append(metrics['mean_l0'])
                cos_vals.append(metrics['reconstruction_cosine'])
        
        summary[model] = {
            'd_model': model_data['d_model'],
            'd_hidden': model_data['d_hidden'],
            'explained_variance_pct': {'mean': np.mean(ev_vals), 'std': np.std(ev_vals)} if ev_vals else None,
            'dead_feature_pct': {'mean': np.mean(dead_vals), 'std': np.std(dead_vals)} if dead_vals else None,
            'mean_l0': {'mean': np.mean(l0_vals), 'std': np.std(l0_vals)} if l0_vals else None,
            'reconstruction_cosine': {'mean': np.mean(cos_vals), 'std': np.std(cos_vals)} if cos_vals else None,
        }
    
    # Save results
    output_dir = BASE_PATH / 'results/sae_quality_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'computed_metrics.json', 'w') as f:
        json.dump({
            'models': {r['model']: r for r in all_results},
            'summary': summary
        }, f, indent=2, default=str)
    
    # Print summary table
    print("\n" + "="*90)
    print("PUBLICATION-READY SAE QUALITY METRICS")
    print("="*90)
    
    print("\n| Model | d_model | Features | Explained Var% | Dead Features% | Mean L0 | Recon Cosine |")
    print("|-------|---------|----------|----------------|----------------|---------|--------------|")
    
    for model, stats in summary.items():
        ev = f"{stats['explained_variance_pct']['mean']:.1f}±{stats['explained_variance_pct']['std']:.1f}" if stats['explained_variance_pct'] else "N/A"
        dead = f"{stats['dead_feature_pct']['mean']:.1f}±{stats['dead_feature_pct']['std']:.1f}" if stats['dead_feature_pct'] else "N/A"
        l0 = f"{stats['mean_l0']['mean']:.0f}±{stats['mean_l0']['std']:.0f}" if stats['mean_l0'] else "N/A"
        cos = f"{stats['reconstruction_cosine']['mean']:.4f}" if stats['reconstruction_cosine'] else "N/A"
        
        print(f"| {model} | {stats['d_model']:,} | {stats['d_hidden']:,} | {ev} | {dead} | {l0} | {cos} |")
    
    print("\n" + "="*90)
    print("COMPARISON WITH PUBLICATION STANDARDS (Anthropic 2024)")
    print("="*90)
    
    standards = {
        'Explained Variance %': ('>65%', lambda s: s['explained_variance_pct']['mean'] if s['explained_variance_pct'] else 0),
        'Dead Features %': ('<35%', lambda s: s['dead_feature_pct']['mean'] if s['dead_feature_pct'] else 100),
        'Mean L0': ('50-300', lambda s: s['mean_l0']['mean'] if s['mean_l0'] else 0),
        'Reconstruction Cosine': ('>0.9', lambda s: s['reconstruction_cosine']['mean'] if s['reconstruction_cosine'] else 0),
    }
    
    for model, stats in summary.items():
        print(f"\n{model}:")
        for metric, (target, getter) in standards.items():
            val = getter(stats)
            if 'Explained Variance' in metric:
                status = '✓' if val >= 65 else '⚠️'
                print(f"  {metric}: {val:.1f}% (target: {target}) {status}")
            elif 'Dead Features' in metric:
                status = '✓' if val < 35 else '⚠️'
                print(f"  {metric}: {val:.1f}% (target: {target}) {status}")
            elif 'Mean L0' in metric:
                status = '✓' if 50 <= val <= 300 else '⚠️'
                print(f"  {metric}: {val:.0f} (target: {target}) {status}")
            elif 'Cosine' in metric:
                status = '✓' if val >= 0.9 else '⚠️'
                print(f"  {metric}: {val:.4f} (target: {target}) {status}")
    
    # Generate Markdown report
    md_lines = [
        "# SAE Quality Metrics - Publication Report",
        "",
        "## Summary Statistics",
        "",
        "| Model | d_model | Features | Explained Var% | Dead Features% | Mean L0 | Recon Cosine |",
        "|-------|---------|----------|----------------|----------------|---------|--------------|",
    ]
    
    for model, stats in summary.items():
        ev = f"{stats['explained_variance_pct']['mean']:.1f}±{stats['explained_variance_pct']['std']:.1f}" if stats['explained_variance_pct'] else "N/A"
        dead = f"{stats['dead_feature_pct']['mean']:.1f}±{stats['dead_feature_pct']['std']:.1f}" if stats['dead_feature_pct'] else "N/A"
        l0 = f"{stats['mean_l0']['mean']:.0f}±{stats['mean_l0']['std']:.0f}" if stats['mean_l0'] else "N/A"
        cos = f"{stats['reconstruction_cosine']['mean']:.4f}" if stats['reconstruction_cosine'] else "N/A"
        md_lines.append(f"| {model} | {stats['d_model']:,} | {stats['d_hidden']:,} | {ev} | {dead} | {l0} | {cos} |")
    
    md_lines.extend([
        "",
        "## Detailed Per-Layer Metrics",
        ""
    ])
    
    for model_data in all_results:
        md_lines.extend([
            f"### {model_data['model']}",
            "",
            "| Layer | Language | Explained Var% | Dead Features% | Mean L0 | Recon Cosine |",
            "|-------|----------|----------------|----------------|---------|--------------|",
        ])
        
        for layer in sorted(model_data['layers'].keys()):
            for lang, m in model_data['layers'][layer].items():
                md_lines.append(
                    f"| {layer} | {lang} | {m['explained_variance_pct']:.1f} | "
                    f"{m['dead_feature_pct']:.1f} | {m['mean_l0']:.0f} | {m['reconstruction_cosine']:.4f} |"
                )
        md_lines.append("")
    
    with open(output_dir / 'computed_metrics.md', 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("✓ Files: computed_metrics.json, computed_metrics.md")


if __name__ == "__main__":
    main()
