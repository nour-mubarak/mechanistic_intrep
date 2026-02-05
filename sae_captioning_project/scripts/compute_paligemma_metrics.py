#!/usr/bin/env python3
"""
Compute SAE Quality Metrics for PaLiGemma-3B
=============================================

This script computes publication-ready metrics using GPU with memory-efficient streaming.
Designed to handle 21GB activation files.

Metrics computed:
1. Explained Variance % - How much variance the SAE reconstruction explains
2. Dead Feature Ratio - Percentage of features that never activate
3. Mean L0 - Average number of active features per sample
4. Reconstruction Cosine Similarity
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import gc
import warnings
warnings.filterwarnings('ignore')

# Base paths
BASE_PATH = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")
RESULTS_PATH = BASE_PATH / "results" / "sae_quality_metrics"
CHECKPOINT_PATH = BASE_PATH / "checkpoints"


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
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.decoder.bias))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features


def load_sae(path: str, device: str = 'cuda') -> SparseAutoencoder:
    """Load trained SAE model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get dimensions from checkpoint
    if 'd_model' in checkpoint and 'd_hidden' in checkpoint:
        d_model = checkpoint['d_model']
        d_hidden = checkpoint['d_hidden']
    else:
        for key in state_dict:
            if 'encoder' in key and 'weight' in key:
                d_hidden, d_model = state_dict[key].shape
                break
    
    config = SAEConfig(d_model=d_model, expansion_factor=d_hidden // d_model)
    sae = SparseAutoencoder(config)
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae.to(device)


def compute_metrics_streaming(sae: SparseAutoencoder, activation_path: str, 
                               device: str = 'cuda', batch_size: int = 1024,
                               max_samples: int = 50000) -> Dict:
    """
    Compute metrics with memory-efficient streaming.
    
    Uses streaming approach to handle large activation files:
    1. Load activations in chunks
    2. Compute running statistics
    3. Track feature activation for dead feature analysis
    """
    print(f"    Loading activations from {activation_path}...")
    
    # Load activation file (this is the memory-intensive part)
    data = torch.load(activation_path, map_location='cpu', weights_only=False)
    
    if isinstance(data, dict):
        activations = data.get('activations', data.get('features'))
    else:
        activations = data
    
    # Subsample if too large
    n_total = len(activations)
    if n_total > max_samples:
        print(f"    Subsampling {max_samples} from {n_total} samples...")
        indices = torch.randperm(n_total)[:max_samples]
        activations = activations[indices]
    
    n_samples = len(activations)
    d_hidden = sae.config.d_hidden
    
    print(f"    Computing metrics on {n_samples} samples...")
    
    # Initialize accumulators
    feature_activation_count = torch.zeros(d_hidden, device='cpu')
    total_l0 = 0
    total_l0_sq = 0
    total_recon_error = 0
    total_variance = 0
    total_cos_sim = 0
    n_processed = 0
    
    sae.eval()
    activation_threshold = 1e-6
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = activations[i:i+batch_size].to(device)
            
            # Forward pass
            features = sae.encode(batch)
            reconstruction = sae.decode(features)
            
            # L0 (number of active features per sample)
            active_mask = (features > activation_threshold).float()
            batch_l0 = active_mask.sum(dim=1)
            total_l0 += batch_l0.sum().item()
            total_l0_sq += (batch_l0 ** 2).sum().item()
            
            # Track which features were active
            feature_activation_count += active_mask.any(dim=0).cpu().float()
            
            # Reconstruction error (MSE)
            recon_error = (batch - reconstruction).pow(2).sum(dim=-1)
            total_recon_error += recon_error.sum().item()
            
            # Input variance
            total_variance += batch.var(dim=-1).sum().item()
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(batch, reconstruction, dim=-1)
            total_cos_sim += cos_sim.sum().item()
            
            n_processed += len(batch)
            
            # Clear GPU memory
            del batch, features, reconstruction, active_mask
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            if (i // batch_size) % 10 == 0:
                print(f"      Processed {n_processed}/{n_samples} samples...")
    
    # Compute final metrics
    mean_l0 = total_l0 / n_processed
    std_l0 = np.sqrt(total_l0_sq / n_processed - mean_l0**2)
    
    # Dead feature ratio
    n_alive = (feature_activation_count > 0).sum().item()
    dead_feature_ratio = 1 - (n_alive / d_hidden)
    
    # Explained variance
    mean_recon_error = total_recon_error / n_processed
    mean_variance = total_variance / n_processed
    explained_variance = max(0, 1 - (mean_recon_error / (mean_variance * sae.config.d_model + 1e-10)))
    
    # Mean cosine similarity
    mean_cos_sim = total_cos_sim / n_processed
    
    # Clear memory
    del activations, data
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'explained_variance_pct': explained_variance * 100,
        'dead_feature_ratio_pct': dead_feature_ratio * 100,
        'mean_l0': mean_l0,
        'std_l0': std_l0,
        'reconstruction_cosine': mean_cos_sim,
        'n_samples': n_processed,
        'n_features': d_hidden,
        'n_alive_features': n_alive,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=50000, help='Max samples per file')
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # PaLiGemma configuration
    PALIGEMMA_CONFIG = {
        'name': 'PaLiGemma-3B',
        'd_model': 2048,
        'd_hidden': 16384,
        'layers': [0, 3, 6, 9, 12, 15, 17],
        'sae_dir': CHECKPOINT_PATH / 'saes',
        'activation_dir': CHECKPOINT_PATH / 'full_layers_ncc' / 'layer_checkpoints',
    }
    
    results = {
        'model': PALIGEMMA_CONFIG['name'],
        'd_model': PALIGEMMA_CONFIG['d_model'],
        'd_hidden': PALIGEMMA_CONFIG['d_hidden'],
        'layers': {},
    }
    
    print("=" * 60)
    print(f"Computing metrics for {PALIGEMMA_CONFIG['name']}")
    print("=" * 60)
    
    for layer in PALIGEMMA_CONFIG['layers']:
        print(f"\n  Layer {layer}...")
        results['layers'][layer] = {}
        
        for lang in ['arabic', 'english']:
            # SAE paths
            sae_path = PALIGEMMA_CONFIG['sae_dir'] / f'sae_{lang}_layer_{layer}.pt'
            
            # Activation paths (try multiple patterns)
            activation_patterns = [
                PALIGEMMA_CONFIG['activation_dir'] / f'layer_{layer}_{lang}.pt',
                PALIGEMMA_CONFIG['activation_dir'] / f'{lang}_layer_{layer}_activations.pt',
                PALIGEMMA_CONFIG['activation_dir'] / f'layer_{layer}_{lang}_activations.pt',
                PALIGEMMA_CONFIG['activation_dir'] / f'{lang}_activations_layer_{layer}.pt',
            ]
            
            activation_path = None
            for pattern in activation_patterns:
                if pattern.exists():
                    activation_path = pattern
                    break
            
            if not sae_path.exists():
                print(f"    Skipping {lang} - SAE not found at {sae_path}")
                continue
            
            if activation_path is None:
                print(f"    Skipping {lang} - Activations not found (tried {len(activation_patterns)} patterns)")
                continue
            
            try:
                print(f"    Processing {lang}...")
                sae = load_sae(str(sae_path), device)
                metrics = compute_metrics_streaming(
                    sae, str(activation_path), device, 
                    batch_size=args.batch_size, max_samples=args.max_samples
                )
                results['layers'][layer][lang] = metrics
                
                print(f"      ✓ EV: {metrics['explained_variance_pct']:.1f}%, "
                      f"Dead: {metrics['dead_feature_ratio_pct']:.1f}%, "
                      f"L0: {metrics['mean_l0']:.0f}, "
                      f"Cos: {metrics['reconstruction_cosine']:.4f}")
                
                # Clear memory
                del sae
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    Error with {lang}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    output_path = RESULTS_PATH / 'paligemma_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PALIGEMMA-3B RESULTS SUMMARY")
    print("=" * 60)
    
    all_ev = []
    all_dead = []
    all_l0 = []
    all_cos = []
    
    print(f"\n{'Layer':<6} {'Lang':<8} {'EV%':<10} {'Dead%':<10} {'L0':<10} {'Cos':<10}")
    print("-" * 60)
    
    for layer in sorted(results['layers'].keys()):
        for lang in ['arabic', 'english']:
            if lang in results['layers'][layer]:
                m = results['layers'][layer][lang]
                print(f"{layer:<6} {lang:<8} {m['explained_variance_pct']:<10.1f} "
                      f"{m['dead_feature_ratio_pct']:<10.1f} {m['mean_l0']:<10.0f} "
                      f"{m['reconstruction_cosine']:<10.4f}")
                all_ev.append(m['explained_variance_pct'])
                all_dead.append(m['dead_feature_ratio_pct'])
                all_l0.append(m['mean_l0'])
                all_cos.append(m['reconstruction_cosine'])
    
    if all_ev:
        print("-" * 60)
        print(f"{'Mean':<6} {'':<8} {np.mean(all_ev):<10.1f} {np.mean(all_dead):<10.1f} "
              f"{np.mean(all_l0):<10.0f} {np.mean(all_cos):<10.4f}")
        print(f"{'Std':<6} {'':<8} {np.std(all_ev):<10.1f} {np.std(all_dead):<10.1f} "
              f"{np.std(all_l0):<10.0f} {np.std(all_cos):<10.4f}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
