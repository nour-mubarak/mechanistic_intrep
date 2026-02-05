#!/usr/bin/env python3
"""
Compute Publication-Ready SAE Metrics for All Models
=====================================================

Computes the recommended metrics for mechanistic interpretability papers:
1. Explained Variance % - How much variance the SAE reconstruction explains
2. Dead Feature Ratio - Percentage of features that never activate
3. Mean L0 - Average number of active features per sample

Models: PaLiGemma-3B, Qwen2-VL-7B, LLaVA-1.5-7B
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Base paths
BASE_PATH = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")
RESULTS_PATH = BASE_PATH / "results"


@dataclass
class SAEConfig:
    """SAE Configuration."""
    d_model: int
    expansion_factor: int = 8
    l1_coefficient: float = 5e-4
    
    @property
    def d_hidden(self) -> int:
        return self.d_model * self.expansion_factor


class SparseAutoencoder(nn.Module):
    """Standard Sparse Autoencoder (matching trained checkpoints)."""
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.d_model, config.d_hidden)
        self.decoder = nn.Linear(config.d_hidden, config.d_model)
        # Note: decoder.bias serves as the decoder_bias (b_dec)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x_centered = x - b_dec, then apply encoder + ReLU
        return torch.relu(self.encoder(x - self.decoder.bias))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        # Linear already adds bias, so reconstruction = W_dec @ features + b_dec
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features


def load_sae(path: str, d_model: int, device: str = 'cpu') -> SparseAutoencoder:
    """Load trained SAE model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get dimensions from checkpoint or state dict
    if 'd_model' in checkpoint and 'd_hidden' in checkpoint:
        cp_d_model = checkpoint['d_model']
        cp_d_hidden = checkpoint['d_hidden']
    else:
        # Infer from encoder weights
        for key in state_dict:
            if 'encoder' in key and 'weight' in key:
                cp_d_hidden, cp_d_model = state_dict[key].shape
                break
    
    # Create config
    config = SAEConfig(d_model=cp_d_model, expansion_factor=cp_d_hidden // cp_d_model)
    
    # Create model and load weights
    sae = SparseAutoencoder(config)
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae.to(device)


def load_activations(path: str) -> Tuple[torch.Tensor, List]:
    """Load activation file and return activations and genders."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    if isinstance(data, dict):
        activations = data.get('activations', data.get('features'))
        genders = data.get('genders', data.get('labels', []))
    else:
        activations = data
        genders = []
    
    return activations, genders


def compute_metrics(sae: SparseAutoencoder, activations: torch.Tensor, device: str = 'cpu', batch_size: int = 256, max_samples: int = 10000) -> Dict:
    """
    Compute all publication metrics for an SAE (memory efficient version).
    
    Returns:
        Dict with explained_variance, dead_feature_ratio, mean_l0, reconstruction_mse
    """
    sae.eval()
    sae.to(device)
    
    # Limit samples for memory efficiency
    n_total_samples = len(activations)
    if n_total_samples > max_samples:
        indices = torch.randperm(n_total_samples)[:max_samples]
        activations = activations[indices]
    
    n_samples = len(activations)
    n_features = sae.config.d_hidden
    
    # Initialize accumulators for streaming computation
    feature_ever_active = torch.zeros(n_features, dtype=torch.bool)
    total_l0 = 0.0
    total_l0_sq = 0.0
    total_mse = 0.0
    total_cos_sim = 0.0
    
    # For explained variance
    sum_input_var = 0.0
    sum_residual_var = 0.0
    
    activation_threshold = 1e-6
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = activations[i:i+batch_size].float().to(device)
            
            # Forward pass
            features = sae.encode(batch)
            reconstruction = sae.decode(features)
            
            # L0 computation
            active_mask = features > activation_threshold
            l0_batch = active_mask.float().sum(dim=1)
            total_l0 += l0_batch.sum().item()
            total_l0_sq += (l0_batch ** 2).sum().item()
            
            # Update feature activity tracker (streaming)
            feature_ever_active |= active_mask.any(dim=0).cpu()
            
            # MSE
            mse_batch = ((batch - reconstruction) ** 2).mean(dim=1)
            total_mse += mse_batch.sum().item()
            
            # Cosine similarity per sample
            cos_batch = nn.functional.cosine_similarity(batch, reconstruction, dim=1)
            total_cos_sim += cos_batch.sum().item()
            
            # Explained variance components (per dimension)
            sum_input_var += batch.var(dim=0).sum().item()
            sum_residual_var += (batch - reconstruction).var(dim=0).sum().item()
            
            # Clear GPU memory
            del batch, features, reconstruction
            if device != 'cpu':
                torch.cuda.empty_cache()
    
    # Compute final metrics
    n_alive = feature_ever_active.sum().item()
    dead_feature_ratio = 1 - (n_alive / n_features)
    
    mean_l0 = total_l0 / n_samples
    std_l0 = np.sqrt(max(0, total_l0_sq / n_samples - mean_l0 ** 2))
    
    mean_mse = total_mse / n_samples
    mean_cos_sim = total_cos_sim / n_samples
    
    # Explained variance = 1 - (residual_var / input_var)
    explained_variance = 1 - (sum_residual_var / (sum_input_var + 1e-10))
    explained_variance = max(0, min(1, explained_variance))  # Clamp to [0, 1]
    
    return {
        'explained_variance': explained_variance * 100,  # As percentage
        'dead_feature_ratio': dead_feature_ratio * 100,  # As percentage
        'alive_features': n_alive,
        'total_features': n_features,
        'mean_l0': mean_l0,
        'std_l0': std_l0,
        'reconstruction_mse': mean_mse,
        'reconstruction_cosine': mean_cos_sim,
        'n_samples': n_samples
    }


def compute_paligemma_metrics(device: str = 'cpu') -> Dict:
    """Compute metrics for PaLiGemma-3B."""
    print("\n" + "="*60)
    print("Computing metrics for PaLiGemma-3B")
    print("="*60)
    
    sae_dir = BASE_PATH / "checkpoints/saes"
    act_dir = BASE_PATH / "checkpoints/full_layers_ncc/layer_checkpoints"
    
    layers = [0, 3, 6, 9, 12, 15, 17]
    d_model = 2048
    
    results = {'model': 'PaLiGemma-3B', 'd_model': d_model, 'layers': {}}
    
    for layer in layers:
        print(f"\n  Layer {layer}...")
        layer_results = {}
        
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / f"sae_{lang}_layer_{layer}.pt"
            act_path = act_dir / f"layer_{layer}_{lang}.pt"
            
            if not sae_path.exists():
                print(f"    Skipping {lang} - SAE not found")
                continue
            if not act_path.exists():
                print(f"    Skipping {lang} - Activations not found")
                continue
            
            try:
                sae = load_sae(str(sae_path), d_model, device)
                activations, _ = load_activations(str(act_path))
                
                if activations.ndim == 3:
                    activations = activations.view(-1, activations.shape[-1])
                
                metrics = compute_metrics(sae, activations, device)
                layer_results[lang] = metrics
                
                print(f"    {lang}: EV={metrics['explained_variance']:.1f}%, "
                      f"Dead={metrics['dead_feature_ratio']:.1f}%, "
                      f"L0={metrics['mean_l0']:.1f}")
            except Exception as e:
                print(f"    Error with {lang}: {e}")
        
        if layer_results:
            results['layers'][layer] = layer_results
    
    return results


def compute_qwen2vl_metrics(device: str = 'cpu') -> Dict:
    """Compute metrics for Qwen2-VL-7B."""
    print("\n" + "="*60)
    print("Computing metrics for Qwen2-VL-7B")
    print("="*60)
    
    sae_dir = BASE_PATH / "checkpoints/qwen2vl/saes"
    act_dir = BASE_PATH / "checkpoints/qwen2vl/layer_checkpoints"
    
    layers = [0, 4, 8, 12, 16, 20, 24, 27]
    d_model = 3584
    
    results = {'model': 'Qwen2-VL-7B', 'd_model': d_model, 'layers': {}}
    
    for layer in layers:
        print(f"\n  Layer {layer}...")
        layer_results = {}
        
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / f"qwen2vl_sae_{lang}_layer_{layer}.pt"
            act_path = act_dir / f"qwen2vl_layer_{layer}_{lang}.pt"
            
            if not sae_path.exists():
                print(f"    Skipping {lang} - SAE not found")
                continue
            if not act_path.exists():
                print(f"    Skipping {lang} - Activations not found")
                continue
            
            try:
                sae = load_sae(str(sae_path), d_model, device)
                activations, _ = load_activations(str(act_path))
                
                if activations.ndim == 3:
                    activations = activations.view(-1, activations.shape[-1])
                
                metrics = compute_metrics(sae, activations, device)
                layer_results[lang] = metrics
                
                print(f"    {lang}: EV={metrics['explained_variance']:.1f}%, "
                      f"Dead={metrics['dead_feature_ratio']:.1f}%, "
                      f"L0={metrics['mean_l0']:.1f}")
            except Exception as e:
                print(f"    Error with {lang}: {e}")
        
        if layer_results:
            results['layers'][layer] = layer_results
    
    return results


def compute_llava_metrics(device: str = 'cpu') -> Dict:
    """Compute metrics for LLaVA-1.5-7B."""
    print("\n" + "="*60)
    print("Computing metrics for LLaVA-1.5-7B")
    print("="*60)
    
    sae_dir = BASE_PATH / "checkpoints/llava/saes"
    act_dir = BASE_PATH / "checkpoints/llava/layer_checkpoints"
    
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    d_model = 4096
    
    results = {'model': 'LLaVA-1.5-7B', 'd_model': d_model, 'layers': {}}
    
    for layer in layers:
        print(f"\n  Layer {layer}...")
        layer_results = {}
        
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / f"llava_sae_{lang}_layer_{layer}.pt"
            act_path = act_dir / f"llava_layer_{layer}_{lang}.pt"
            
            if not sae_path.exists():
                print(f"    Skipping {lang} - SAE not found")
                continue
            if not act_path.exists():
                print(f"    Skipping {lang} - Activations not found")
                continue
            
            try:
                sae = load_sae(str(sae_path), d_model, device)
                activations, _ = load_activations(str(act_path))
                
                if activations.ndim == 3:
                    activations = activations.view(-1, activations.shape[-1])
                
                metrics = compute_metrics(sae, activations, device)
                layer_results[lang] = metrics
                
                print(f"    {lang}: EV={metrics['explained_variance']:.1f}%, "
                      f"Dead={metrics['dead_feature_ratio']:.1f}%, "
                      f"L0={metrics['mean_l0']:.1f}")
            except Exception as e:
                print(f"    Error with {lang}: {e}")
        
        if layer_results:
            results['layers'][layer] = layer_results
    
    return results


def compute_model_averages(model_results: Dict) -> Dict:
    """Compute average metrics across layers for a model."""
    all_ev = []
    all_dead = []
    all_l0 = []
    all_mse = []
    all_cos = []
    
    for layer, langs in model_results['layers'].items():
        for lang, metrics in langs.items():
            all_ev.append(metrics['explained_variance'])
            all_dead.append(metrics['dead_feature_ratio'])
            all_l0.append(metrics['mean_l0'])
            all_mse.append(metrics['reconstruction_mse'])
            all_cos.append(metrics['reconstruction_cosine_sim'])
    
    return {
        'mean_explained_variance': np.mean(all_ev),
        'std_explained_variance': np.std(all_ev),
        'mean_dead_feature_ratio': np.mean(all_dead),
        'std_dead_feature_ratio': np.std(all_dead),
        'mean_l0': np.mean(all_l0),
        'std_l0': np.std(all_l0),
        'mean_reconstruction_mse': np.mean(all_mse),
        'mean_reconstruction_cosine_sim': np.mean(all_cos)
    }


def generate_markdown_table(all_results: Dict) -> str:
    """Generate publication-ready markdown table."""
    
    # Model summary table
    md = "# SAE Quality Metrics - Publication Summary\n\n"
    md += "## Model-Level Summary\n\n"
    md += "| Model | d_model | Features | Mean EV% | Mean Dead% | Mean L0 | Recon MSE | Recon Cos |\n"
    md += "|-------|---------|----------|----------|------------|---------|-----------|----------|\n"
    
    for model_name, results in all_results.items():
        avgs = results['averages']
        d_model = results['d_model']
        n_features = d_model * 8
        
        md += f"| {model_name} | {d_model} | {n_features:,} | "
        md += f"{avgs['mean_explained_variance']:.1f}±{avgs['std_explained_variance']:.1f} | "
        md += f"{avgs['mean_dead_feature_ratio']:.1f}±{avgs['std_dead_feature_ratio']:.1f} | "
        md += f"{avgs['mean_l0']:.1f}±{avgs['std_l0']:.1f} | "
        md += f"{avgs['mean_reconstruction_mse']:.4f} | "
        md += f"{avgs['mean_reconstruction_cosine_sim']:.3f} |\n"
    
    # Per-layer tables for each model
    for model_name, results in all_results.items():
        md += f"\n## {model_name} - Layer-by-Layer Metrics\n\n"
        md += "| Layer | Language | Explained Var% | Dead Feature% | Mean L0 | Recon MSE |\n"
        md += "|-------|----------|----------------|---------------|---------|----------|\n"
        
        for layer in sorted(results['layers'].keys()):
            langs = results['layers'][layer]
            for lang, metrics in langs.items():
                md += f"| {layer} | {lang.capitalize()} | "
                md += f"{metrics['explained_variance']:.1f}% | "
                md += f"{metrics['dead_feature_ratio']:.1f}% | "
                md += f"{metrics['mean_l0']:.1f} | "
                md += f"{metrics['reconstruction_mse']:.4f} |\n"
    
    return md


def generate_latex_table(all_results: Dict) -> str:
    """Generate LaTeX table for publication."""
    latex = "% SAE Quality Metrics Table\n"
    latex += "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{SAE Quality Metrics Across Models}\n"
    latex += "\\label{tab:sae_metrics}\n"
    latex += "\\begin{tabular}{lcccccc}\n"
    latex += "\\toprule\n"
    latex += "Model & $d_{model}$ & Features & Explained Var (\\%) & Dead (\\%) & Mean $L_0$ & Recon Cos \\\\\n"
    latex += "\\midrule\n"
    
    for model_name, results in all_results.items():
        avgs = results['averages']
        d_model = results['d_model']
        n_features = d_model * 8
        
        latex += f"{model_name} & {d_model} & {n_features:,} & "
        latex += f"${avgs['mean_explained_variance']:.1f} \\pm {avgs['std_explained_variance']:.1f}$ & "
        latex += f"${avgs['mean_dead_feature_ratio']:.1f} \\pm {avgs['std_dead_feature_ratio']:.1f}$ & "
        latex += f"${avgs['mean_l0']:.0f} \\pm {avgs['std_l0']:.0f}$ & "
        latex += f"{avgs['mean_reconstruction_cosine_sim']:.3f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute SAE publication metrics')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Compute metrics for all models
    all_results = {}
    
    # PaLiGemma
    try:
        paligemma_results = compute_paligemma_metrics(args.device)
        paligemma_results['averages'] = compute_model_averages(paligemma_results)
        all_results['PaLiGemma-3B'] = paligemma_results
    except Exception as e:
        print(f"Error with PaLiGemma: {e}")
    
    # Qwen2-VL
    try:
        qwen_results = compute_qwen2vl_metrics(args.device)
        qwen_results['averages'] = compute_model_averages(qwen_results)
        all_results['Qwen2-VL-7B'] = qwen_results
    except Exception as e:
        print(f"Error with Qwen2-VL: {e}")
    
    # LLaVA
    try:
        llava_results = compute_llava_metrics(args.device)
        llava_results['averages'] = compute_model_averages(llava_results)
        all_results['LLaVA-1.5-7B'] = llava_results
    except Exception as e:
        print(f"Error with LLaVA: {e}")
    
    # Save results
    output_dir = RESULTS_PATH / "sae_quality_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "publication_metrics.json"
    with open(json_path, 'w') as f:
        # Convert to serializable format
        serializable = {}
        for model, results in all_results.items():
            serializable[model] = {
                'model': results['model'],
                'd_model': results['d_model'],
                'averages': results['averages'],
                'layers': {str(k): v for k, v in results['layers'].items()}
            }
        json.dump(serializable, f, indent=2)
    print(f"\n✓ Saved JSON to: {json_path}")
    
    # Save Markdown
    md_table = generate_markdown_table(all_results)
    md_path = output_dir / "publication_metrics.md"
    with open(md_path, 'w') as f:
        f.write(md_table)
    print(f"✓ Saved Markdown to: {md_path}")
    
    # Save LaTeX
    latex_table = generate_latex_table(all_results)
    latex_path = output_dir / "publication_metrics.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved LaTeX to: {latex_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("PUBLICATION-READY SAE QUALITY METRICS SUMMARY")
    print("="*70)
    print(md_table)
    
    # Print standard targets comparison
    print("\n" + "="*70)
    print("COMPARISON WITH STANDARD TARGETS (Anthropic, 2024)")
    print("="*70)
    print(f"{'Metric':<25} {'Target':<15} {'Your Results':<30}")
    print("-"*70)
    
    for model_name, results in all_results.items():
        avgs = results['averages']
        print(f"\n{model_name}:")
        
        ev = avgs['mean_explained_variance']
        ev_status = "✓" if ev > 65 else "⚠️"
        print(f"  Explained Variance %:    >65%            {ev:.1f}% {ev_status}")
        
        dead = avgs['mean_dead_feature_ratio']
        dead_status = "✓" if dead < 35 else "⚠️"
        print(f"  Dead Feature %:          <35%            {dead:.1f}% {dead_status}")
        
        l0 = avgs['mean_l0']
        l0_status = "✓" if 50 <= l0 <= 300 else "⚠️"
        print(f"  Mean L0:                 50-300          {l0:.1f} {l0_status}")
        
        cos = avgs['mean_reconstruction_cosine_sim']
        cos_status = "✓" if cos > 0.9 else "⚠️"
        print(f"  Reconstruction Cosine:   >0.9            {cos:.3f} {cos_status}")


if __name__ == "__main__":
    main()
