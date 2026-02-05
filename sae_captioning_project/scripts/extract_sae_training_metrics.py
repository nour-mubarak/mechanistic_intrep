#!/usr/bin/env python3
"""
Extract Publication Metrics from SAE Training History
======================================================

Extracts the recommended metrics for publication:
1. Final Training/Validation Loss (proxy for reconstruction quality)
2. L0 Sparsity / Mean Active Features per sample
3. Dead Feature Ratio (where available)

This script extracts metrics from checkpoint history without needing
to load and process large activation files.
"""

import torch
import json
from pathlib import Path
import numpy as np

BASE_PATH = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")
RESULTS_PATH = BASE_PATH / "results/sae_quality_metrics"


def extract_paligemma_metrics():
    """Extract metrics from PaLiGemma SAE checkpoints."""
    sae_dir = BASE_PATH / "checkpoints/saes"
    layers = [0, 3, 6, 9, 12, 15, 17]
    
    results = {
        'model': 'PaLiGemma-3B',
        'd_model': 2048,
        'd_hidden': 16384,
        'expansion_factor': 8,
        'layers': {}
    }
    
    for layer in layers:
        layer_data = {}
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / f"sae_{lang}_layer_{layer}.pt"
            if sae_path.exists():
                ckpt = torch.load(sae_path, map_location='cpu', weights_only=False)
                
                metrics = {
                    'd_model': ckpt.get('d_model', 2048),
                    'd_hidden': ckpt.get('d_hidden', 16384),
                }
                
                # Extract from history
                if 'history' in ckpt and isinstance(ckpt['history'], dict):
                    h = ckpt['history']
                    if 'train_loss' in h:
                        metrics['final_train_loss'] = h['train_loss'][-1] if h['train_loss'] else None
                    if 'val_loss' in h:
                        metrics['final_val_loss'] = h['val_loss'][-1] if h['val_loss'] else None
                    if 'l0_sparsity' in h:
                        # l0_sparsity is typically a ratio - convert to mean active features
                        l0 = h['l0_sparsity'][-1] if h['l0_sparsity'] else None
                        if l0 is not None:
                            # If < 1, it's a ratio; if > 1, it's actual count
                            if l0 < 1:
                                metrics['l0_sparsity_ratio'] = l0
                                metrics['mean_l0'] = l0 * ckpt.get('d_hidden', 16384)
                            else:
                                metrics['mean_l0'] = l0
                                metrics['l0_sparsity_ratio'] = l0 / ckpt.get('d_hidden', 16384)
                    if 'sparsity' in h:
                        metrics['sparsity_ratio'] = h['sparsity'][-1] if h['sparsity'] else None
                    if 'active_features' in h:
                        metrics['mean_l0'] = h['active_features'][-1] if h['active_features'] else None
                
                layer_data[lang] = metrics
        
        if layer_data:
            results['layers'][layer] = layer_data
    
    return results


def extract_qwen2vl_metrics():
    """Extract metrics from Qwen2-VL SAE checkpoints."""
    sae_dir = BASE_PATH / "checkpoints/qwen2vl/saes"
    layers = [0, 4, 8, 12, 16, 20, 24, 27]
    
    results = {
        'model': 'Qwen2-VL-7B',
        'd_model': 3584,
        'd_hidden': 28672,
        'expansion_factor': 8,
        'layers': {}
    }
    
    for layer in layers:
        layer_data = {}
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / f"qwen2vl_sae_{lang}_layer_{layer}.pt"
            if sae_path.exists():
                ckpt = torch.load(sae_path, map_location='cpu', weights_only=False)
                
                metrics = {
                    'd_model': ckpt.get('d_model', 3584),
                    'd_hidden': ckpt.get('d_hidden', 28672),
                }
                
                # Config might have info
                if 'config' in ckpt and isinstance(ckpt['config'], dict):
                    cfg = ckpt['config']
                    metrics['l1_coefficient'] = cfg.get('l1_coefficient', 5e-4)
                
                # No history in Qwen2VL checkpoints, but may have config
                if 'l1_coefficient' in ckpt:
                    metrics['l1_coefficient'] = ckpt['l1_coefficient']
                    
                layer_data[lang] = metrics
        
        if layer_data:
            results['layers'][layer] = layer_data
    
    return results


def extract_llava_metrics():
    """Extract metrics from LLaVA SAE checkpoints."""
    sae_dir = BASE_PATH / "checkpoints/llava/saes"
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    
    results = {
        'model': 'LLaVA-1.5-7B',
        'd_model': 4096,
        'd_hidden': 32768,
        'expansion_factor': 8,
        'layers': {}
    }
    
    for layer in layers:
        layer_data = {}
        for lang in ['arabic', 'english']:
            sae_path = sae_dir / f"llava_sae_{lang}_layer_{layer}.pt"
            if sae_path.exists():
                ckpt = torch.load(sae_path, map_location='cpu', weights_only=False)
                
                metrics = {
                    'd_model': ckpt.get('d_model', 4096),
                    'd_hidden': ckpt.get('d_hidden', 32768),
                }
                
                # Extract final metrics
                if 'final_loss' in ckpt:
                    metrics['final_loss'] = ckpt['final_loss']
                if 'final_sparsity' in ckpt:
                    metrics['sparsity_ratio'] = ckpt['final_sparsity']
                
                # Extract from history
                if 'history' in ckpt and isinstance(ckpt['history'], dict):
                    h = ckpt['history']
                    if 'total_loss' in h:
                        metrics['final_total_loss'] = h['total_loss'][-1] if h['total_loss'] else None
                    if 'recon_loss' in h:
                        metrics['final_recon_loss'] = h['recon_loss'][-1] if h['recon_loss'] else None
                    if 'l1_loss' in h:
                        metrics['final_l1_loss'] = h['l1_loss'][-1] if h['l1_loss'] else None
                    if 'sparsity' in h:
                        metrics['sparsity_ratio'] = h['sparsity'][-1] if h['sparsity'] else None
                    if 'active_features' in h:
                        metrics['mean_l0'] = h['active_features'][-1] if h['active_features'] else None
                        metrics['l0_sparsity_ratio'] = metrics['mean_l0'] / ckpt.get('d_hidden', 32768)
                
                layer_data[lang] = metrics
        
        if layer_data:
            results['layers'][layer] = layer_data
    
    return results


def compute_summary_stats(all_results):
    """Compute summary statistics across all models."""
    summary = {}
    
    for model_data in all_results:
        model_name = model_data['model']
        layers = model_data['layers']
        
        # Collect metrics across layers
        mean_l0_values = []
        loss_values = []
        sparsity_values = []
        
        for layer, lang_data in layers.items():
            for lang, metrics in lang_data.items():
                if 'mean_l0' in metrics and metrics['mean_l0'] is not None:
                    mean_l0_values.append(metrics['mean_l0'])
                if 'final_val_loss' in metrics and metrics['final_val_loss'] is not None:
                    loss_values.append(metrics['final_val_loss'])
                elif 'final_recon_loss' in metrics and metrics['final_recon_loss'] is not None:
                    loss_values.append(metrics['final_recon_loss'])
                if 'sparsity_ratio' in metrics and metrics['sparsity_ratio'] is not None:
                    sparsity_values.append(metrics['sparsity_ratio'])
        
        summary[model_name] = {
            'd_model': model_data['d_model'],
            'd_hidden': model_data['d_hidden'],
            'expansion_factor': model_data['expansion_factor'],
            'n_layers': len(layers),
            'mean_l0': {
                'mean': np.mean(mean_l0_values) if mean_l0_values else None,
                'std': np.std(mean_l0_values) if mean_l0_values else None,
                'values': mean_l0_values
            } if mean_l0_values else None,
            'reconstruction_loss': {
                'mean': np.mean(loss_values) if loss_values else None,
                'std': np.std(loss_values) if loss_values else None,
                'values': loss_values
            } if loss_values else None,
            'sparsity_ratio': {
                'mean': np.mean(sparsity_values) if sparsity_values else None,
                'std': np.std(sparsity_values) if sparsity_values else None,
            } if sparsity_values else None
        }
    
    return summary


def generate_publication_table(all_results, summary):
    """Generate publication-ready table in Markdown and LaTeX."""
    
    md_lines = [
        "# SAE Quality Metrics Summary",
        "",
        "## Model Configuration",
        "",
        "| Model | d_model | Features (8×) | L1 Coefficient |",
        "|-------|---------|---------------|----------------|",
        "| PaLiGemma-3B | 2,048 | 16,384 | 5×10⁻⁴ |",
        "| Qwen2-VL-7B | 3,584 | 28,672 | 5×10⁻⁴ |",
        "| LLaVA-1.5-7B | 4,096 | 32,768 | 5×10⁻⁴ |",
        "",
        "## Training Metrics Summary",
        "",
        "| Model | Mean L0 | Sparsity % | Recon Loss |",
        "|-------|---------|------------|------------|",
    ]
    
    for model_name, stats in summary.items():
        l0 = f"{stats['mean_l0']['mean']:.1f}" if stats.get('mean_l0') and stats['mean_l0']['mean'] else "N/A"
        sparsity = f"{stats['sparsity_ratio']['mean']*100:.2f}%" if stats.get('sparsity_ratio') and stats['sparsity_ratio']['mean'] else "N/A"
        loss = f"{stats['reconstruction_loss']['mean']:.6f}" if stats.get('reconstruction_loss') and stats['reconstruction_loss']['mean'] else "N/A"
        md_lines.append(f"| {model_name} | {l0} | {sparsity} | {loss} |")
    
    md_lines.extend([
        "",
        "## Detailed Per-Layer Metrics",
        ""
    ])
    
    for model_data in all_results:
        model_name = model_data['model']
        md_lines.extend([
            f"### {model_name}",
            "",
            "| Layer | Language | Mean L0 | Sparsity | Loss |",
            "|-------|----------|---------|----------|------|",
        ])
        
        for layer in sorted(model_data['layers'].keys()):
            lang_data = model_data['layers'][layer]
            for lang, metrics in lang_data.items():
                l0 = f"{metrics['mean_l0']:.1f}" if metrics.get('mean_l0') else "N/A"
                sparsity = f"{metrics['sparsity_ratio']*100:.2f}%" if metrics.get('sparsity_ratio') else "N/A"
                loss = metrics.get('final_val_loss') or metrics.get('final_recon_loss') or metrics.get('final_total_loss')
                loss_str = f"{loss:.6f}" if loss else "N/A"
                md_lines.append(f"| {layer} | {lang} | {l0} | {sparsity} | {loss_str} |")
        
        md_lines.append("")
    
    return "\n".join(md_lines)


def main():
    print("Extracting SAE Publication Metrics from Training History")
    print("=" * 60)
    
    # Create output directory
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for each model
    print("\n1. Extracting PaLiGemma-3B metrics...")
    paligemma = extract_paligemma_metrics()
    print(f"   Found {len(paligemma['layers'])} layers")
    
    print("\n2. Extracting Qwen2-VL-7B metrics...")
    qwen2vl = extract_qwen2vl_metrics()
    print(f"   Found {len(qwen2vl['layers'])} layers")
    
    print("\n3. Extracting LLaVA-1.5-7B metrics...")
    llava = extract_llava_metrics()
    print(f"   Found {len(llava['layers'])} layers")
    
    all_results = [paligemma, qwen2vl, llava]
    
    # Compute summary
    print("\n4. Computing summary statistics...")
    summary = compute_summary_stats(all_results)
    
    # Save JSON results
    json_path = RESULTS_PATH / "sae_training_metrics.json"
    with open(json_path, 'w') as f:
        json.dump({
            'models': {r['model']: r for r in all_results},
            'summary': summary
        }, f, indent=2, default=str)
    print(f"\n✓ Saved JSON to: {json_path}")
    
    # Generate and save markdown
    md_content = generate_publication_table(all_results, summary)
    md_path = RESULTS_PATH / "sae_training_metrics.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"✓ Saved Markdown to: {md_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PUBLICATION-READY SAE METRICS SUMMARY")
    print("=" * 60)
    
    print("\n| Model | d_model | Features | Mean L0 | Sparsity |")
    print("|-------|---------|----------|---------|----------|")
    for model_name, stats in summary.items():
        d_model = stats['d_model']
        features = stats['d_hidden']
        l0 = f"{stats['mean_l0']['mean']:.1f}" if stats.get('mean_l0') and stats['mean_l0']['mean'] else "N/A"
        sparsity = f"{stats['sparsity_ratio']['mean']*100:.2f}%" if stats.get('sparsity_ratio') and stats['sparsity_ratio']['mean'] else "N/A"
        print(f"| {model_name} | {d_model:,} | {features:,} | {l0} | {sparsity} |")
    
    print("\n✅ Metrics extraction complete!")


if __name__ == "__main__":
    main()
