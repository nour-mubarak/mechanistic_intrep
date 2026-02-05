#!/usr/bin/env python3
"""
Compute SAE Quality Metrics for Models with Small Activation Files
===================================================================

Memory-efficient computation for Qwen2-VL and LLaVA.
PaLiGemma requires GPU due to large activation files (21GB each).
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")


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


def load_sae(path: str) -> SparseAutoencoder:
    """Load SAE from checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sae = SparseAutoencoder(ckpt['d_model'], ckpt['d_hidden'])
    sae.load_state_dict(ckpt['model_state_dict'])
    sae.eval()
    return sae


def load_activations(path: str) -> torch.Tensor:
    """Load activations."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        acts = data.get('activations', data.get('features'))
        del data
    else:
        acts = data
    
    if acts.ndim == 3:
        acts = acts.view(-1, acts.shape[-1])
    
    return acts.float()


def compute_metrics(sae: SparseAutoencoder, activations: torch.Tensor, batch_size: int = 128) -> dict:
    """Compute quality metrics."""
    sae.eval()
    n_samples = len(activations)
    n_features = sae.d_hidden
    
    feature_active_count = torch.zeros(n_features)
    total_l0, total_l0_sq = 0.0, 0.0
    total_cos_sim = 0.0
    
    sum_input = torch.zeros(sae.d_model)
    sum_sq_input = torch.zeros(sae.d_model)
    sum_sq_residual = torch.zeros(sae.d_model)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = activations[i:i+batch_size]
            
            features = sae.encode(batch)
            reconstruction = sae.decode(features)
            residual = batch - reconstruction
            
            # L0
            active = (features > 1e-6).float()
            feature_active_count += active.sum(dim=0)
            l0_batch = active.sum(dim=1)
            total_l0 += l0_batch.sum().item()
            total_l0_sq += (l0_batch ** 2).sum().item()
            
            # Cosine sim
            total_cos_sim += nn.functional.cosine_similarity(batch, reconstruction, dim=1).sum().item()
            
            # Variance components
            sum_input += batch.sum(dim=0)
            sum_sq_input += (batch ** 2).sum(dim=0)
            sum_sq_residual += (residual ** 2).sum(dim=0)
    
    # Metrics
    n_alive = (feature_active_count > 0).sum().item()
    dead_ratio = 1 - (n_alive / n_features)
    mean_l0 = total_l0 / n_samples
    std_l0 = np.sqrt(max(0, total_l0_sq / n_samples - mean_l0 ** 2))
    mean_cos_sim = total_cos_sim / n_samples
    
    # Explained variance
    input_var = (sum_sq_input / n_samples) - (sum_input / n_samples) ** 2
    # MSE per dimension
    mse_per_dim = sum_sq_residual / n_samples
    explained_var = 1 - (mse_per_dim / torch.clamp(input_var, min=1e-10))
    explained_var = torch.clamp(explained_var, 0, 1)
    
    return {
        'explained_variance_pct': explained_var.mean().item() * 100,
        'dead_feature_pct': dead_ratio * 100,
        'alive_features': n_alive,
        'total_features': n_features,
        'mean_l0': mean_l0,
        'std_l0': std_l0,
        'l0_sparsity_pct': (mean_l0 / n_features) * 100,
        'reconstruction_cosine': mean_cos_sim,
        'n_samples': n_samples
    }


def process_model(name, sae_dir, act_dir, layers, sae_pat, act_pat, d_model, d_hidden):
    """Process a single model."""
    print(f"\n{'='*60}")
    print(f"Computing metrics for {name}")
    print(f"{'='*60}")
    
    results = {'model': name, 'd_model': d_model, 'd_hidden': d_hidden, 'layers': {}}
    
    for layer in layers:
        print(f"\n  Layer {layer}...")
        layer_data = {}
        
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / sae_pat.format(lang=lang, layer=layer)
            act_path = act_dir / act_pat.format(lang=lang, layer=layer)
            
            if not sae_path.exists() or not act_path.exists():
                print(f"    {lang}: Files not found")
                continue
            
            try:
                sae = load_sae(str(sae_path))
                acts = load_activations(str(act_path))
                m = compute_metrics(sae, acts)
                layer_data[lang] = m
                
                print(f"    {lang}: EV={m['explained_variance_pct']:.1f}%, "
                      f"Dead={m['dead_feature_pct']:.1f}%, "
                      f"L0={m['mean_l0']:.0f}, "
                      f"Cos={m['reconstruction_cosine']:.4f}")
                
                del sae, acts
                gc.collect()
            except Exception as e:
                print(f"    {lang}: Error - {e}")
        
        if layer_data:
            results['layers'][layer] = layer_data
    
    return results


def main():
    print("="*70)
    print("SAE Quality Metrics - Qwen2-VL and LLaVA")
    print("="*70)
    
    # Qwen2-VL-7B
    qwen2vl = process_model(
        name='Qwen2-VL-7B',
        sae_dir=BASE_PATH / 'checkpoints/qwen2vl/saes',
        act_dir=BASE_PATH / 'checkpoints/qwen2vl/layer_checkpoints',
        layers=[0, 4, 8, 12, 16, 20, 24, 27],
        sae_pat='qwen2vl_sae_{lang}_layer_{layer}.pt',
        act_pat='qwen2vl_layer_{layer}_{lang}.pt',
        d_model=3584,
        d_hidden=28672
    )
    
    # LLaVA-1.5-7B
    llava = process_model(
        name='LLaVA-1.5-7B',
        sae_dir=BASE_PATH / 'checkpoints/llava/saes',
        act_dir=BASE_PATH / 'checkpoints/llava/layer_checkpoints',
        layers=[0, 4, 8, 12, 16, 20, 24, 28, 31],
        sae_pat='llava_sae_{lang}_layer_{layer}.pt',
        act_pat='llava_layer_{layer}_{lang}.pt',
        d_model=4096,
        d_hidden=32768
    )
    
    all_results = [qwen2vl, llava]
    
    # Summary
    summary = {}
    for model_data in all_results:
        model = model_data['model']
        vals = {'ev': [], 'dead': [], 'l0': [], 'cos': []}
        
        for layer_data in model_data['layers'].values():
            for m in layer_data.values():
                vals['ev'].append(m['explained_variance_pct'])
                vals['dead'].append(m['dead_feature_pct'])
                vals['l0'].append(m['mean_l0'])
                vals['cos'].append(m['reconstruction_cosine'])
        
        summary[model] = {
            'd_model': model_data['d_model'],
            'd_hidden': model_data['d_hidden'],
            'explained_variance_pct': {'mean': np.mean(vals['ev']), 'std': np.std(vals['ev'])} if vals['ev'] else None,
            'dead_feature_pct': {'mean': np.mean(vals['dead']), 'std': np.std(vals['dead'])} if vals['dead'] else None,
            'mean_l0': {'mean': np.mean(vals['l0']), 'std': np.std(vals['l0'])} if vals['l0'] else None,
            'reconstruction_cosine': {'mean': np.mean(vals['cos']), 'std': np.std(vals['cos'])} if vals['cos'] else None,
        }
    
    # Add PaLiGemma placeholder with training history data
    summary['PaLiGemma-3B'] = {
        'd_model': 2048,
        'd_hidden': 16384,
        'note': 'Requires GPU - activation files are 21GB each',
        'from_training_history': {
            'mean_l0': {'mean': 7440, 'note': 'High L0 indicates L1 regularization may need tuning'},
            'reconstruction_loss': {'mean': 0.016, 'std': 0.03}
        }
    }
    
    # Save
    output_dir = BASE_PATH / 'results/sae_quality_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'computed_metrics_qwen_llava.json', 'w') as f:
        json.dump({
            'models': {r['model']: r for r in all_results},
            'summary': summary
        }, f, indent=2, default=str)
    
    # Print table
    print("\n" + "="*90)
    print("PUBLICATION-READY SAE QUALITY METRICS")
    print("="*90)
    
    print("\n| Model | Features | Explained Var% | Dead Features% | Mean L0 | Recon Cosine |")
    print("|-------|----------|----------------|----------------|---------|--------------|")
    
    for model, s in summary.items():
        if 'explained_variance_pct' not in s or s.get('note'):
            print(f"| {model} | {s['d_hidden']:,} | (GPU required) | (GPU required) | (GPU required) | (GPU required) |")
            continue
        
        ev = f"{s['explained_variance_pct']['mean']:.1f}±{s['explained_variance_pct']['std']:.1f}"
        dead = f"{s['dead_feature_pct']['mean']:.1f}±{s['dead_feature_pct']['std']:.1f}"
        l0 = f"{s['mean_l0']['mean']:.0f}±{s['mean_l0']['std']:.0f}"
        cos = f"{s['reconstruction_cosine']['mean']:.4f}"
        print(f"| {model} | {s['d_hidden']:,} | {ev} | {dead} | {l0} | {cos} |")
    
    print("\n" + "="*90)
    print("COMPARISON WITH ANTHROPIC (2024) STANDARDS")
    print("="*90)
    
    for model, s in summary.items():
        print(f"\n{model}:")
        if 'explained_variance_pct' not in s or s.get('note'):
            print("  (Requires GPU computation - activation files are 21GB)")
            continue
        
        ev = s['explained_variance_pct']['mean']
        dead = s['dead_feature_pct']['mean']
        l0 = s['mean_l0']['mean']
        cos = s['reconstruction_cosine']['mean']
        
        print(f"  Explained Variance: {ev:.1f}% (target: >65%) {'✓' if ev >= 65 else '⚠️'}")
        print(f"  Dead Features: {dead:.1f}% (target: <35%) {'✓' if dead < 35 else '⚠️'}")
        print(f"  Mean L0: {l0:.0f} (target: 50-300) {'✓' if 50 <= l0 <= 300 else '⚠️'}")
        print(f"  Recon Cosine: {cos:.4f} (target: >0.9) {'✓' if cos >= 0.9 else '⚠️'}")
    
    # Generate markdown
    md = [
        "# SAE Quality Metrics - Publication Report",
        "",
        "## Summary (Qwen2-VL-7B and LLaVA-1.5-7B)",
        "",
        "| Model | Features | Explained Var% | Dead Features% | Mean L0 | Recon Cosine |",
        "|-------|----------|----------------|----------------|---------|--------------|"
    ]
    
    for model, s in summary.items():
        if 'explained_variance_pct' not in s or s.get('note'):
            continue
        ev = f"{s['explained_variance_pct']['mean']:.1f}±{s['explained_variance_pct']['std']:.1f}"
        dead = f"{s['dead_feature_pct']['mean']:.1f}±{s['dead_feature_pct']['std']:.1f}"
        l0 = f"{s['mean_l0']['mean']:.0f}±{s['mean_l0']['std']:.0f}"
        cos = f"{s['reconstruction_cosine']['mean']:.4f}"
        md.append(f"| {model} | {s['d_hidden']:,} | {ev} | {dead} | {l0} | {cos} |")
    
    md.extend(["", "## Detailed Per-Layer Metrics", ""])
    
    for r in all_results:
        md.extend([
            f"### {r['model']}",
            "",
            "| Layer | Language | Explained Var% | Dead Features% | Mean L0 | Recon Cosine |",
            "|-------|----------|----------------|----------------|---------|--------------|"
        ])
        for layer in sorted(r['layers'].keys()):
            for lang, m in r['layers'][layer].items():
                md.append(
                    f"| {layer} | {lang} | {m['explained_variance_pct']:.1f} | "
                    f"{m['dead_feature_pct']:.1f} | {m['mean_l0']:.0f} | {m['reconstruction_cosine']:.4f} |"
                )
        md.append("")
    
    md.extend([
        "## Note on PaLiGemma-3B",
        "",
        "PaLiGemma-3B activation files are 21GB each, requiring GPU computation.",
        "From training history:",
        "- Mean L0: ~7,440 (high - suggests L1 regularization may need adjustment)",
        "- Reconstruction Loss: ~0.016"
    ])
    
    with open(output_dir / 'computed_metrics_qwen_llava.md', 'w') as f:
        f.write('\n'.join(md))
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
