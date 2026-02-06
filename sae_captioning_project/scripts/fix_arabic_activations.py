#!/usr/bin/env python3
"""
Fix PaLiGemma Arabic Activation Files

ROOT CAUSE:
    The Arabic activation files were saved with the FULL sequence dimension
    (shape [N, 268, 2048] = 3D), while English files were mean-pooled across
    the sequence dimension (shape [N, 2048] = 2D).
    
    The difference occurred because:
    - English: script 22_extract_english_activations.py applies mean(dim=1) 
      during extraction (line 183)
    - Arabic: script 18_extract_full_activations_ncc.py saves raw hook output
      WITHOUT mean pooling
    
    The SAE expects 2D input [N, d_model] but Arabic provides 3D [N, 268, d_model],
    causing: "output with shape [16384] doesn't match broadcast shape [268, 16384]"

FIX:
    Mean-pool the Arabic chunk files across dim=1 (sequence dimension) to produce
    the same [N, 2048] format as English, then compute SAE quality metrics.
    
    This uses the backup_layer6 chunk files (100 chunks × 100 samples × 268 × 2048)
    and the main Arabic files for other layers (loaded with mmap/streaming).

APPROACH:
    For each layer, process Arabic chunks with mean pooling, 
    then run SAE metrics on the pooled activations.
"""

import torch
import torch.nn as nn
import os
import sys
import json
import gc
import glob
from pathlib import Path

BASE = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")
ACT_DIR = BASE / "checkpoints" / "full_layers_ncc" / "layer_checkpoints"
SAE_DIR = BASE / "checkpoints" / "saes"
RESULTS_DIR = BASE / "results" / "sae_quality_metrics"

# SAE Architecture (matching the checkpoint format)
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)
    
    def encode(self, x):
        return torch.relu(self.encoder(x))
    
    def decode(self, f):
        return self.decoder(f)
    
    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


def load_sae(sae_path, d_model=2048, d_hidden=16384):
    """Load SAE checkpoint."""
    checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
    sae = SparseAutoencoder(d_model, d_hidden)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    return sae


def compute_metrics_batched(sae, activations, batch_size=512):
    """Compute all quality metrics with batched processing."""
    n_samples = activations.shape[0]
    d_hidden = sae.encoder.out_features
    
    all_ev_nums = []
    all_ev_dens = []
    all_cos_sims = []
    max_features = torch.zeros(d_hidden)
    total_l0 = 0.0
    
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = activations[start:end]
            
            x_hat, features = sae(batch)
            
            # Explained variance components
            residual = batch - x_hat
            all_ev_nums.append(residual.var(dim=0).sum().item())
            all_ev_dens.append(batch.var(dim=0).sum().item())
            
            # Cosine similarity
            cos = torch.nn.functional.cosine_similarity(batch, x_hat, dim=-1)
            all_cos_sims.extend(cos.tolist())
            
            # L0 sparsity
            l0 = (features > 0).float().sum(dim=-1)
            total_l0 += l0.sum().item()
            
            # Track max feature activations for dead features
            batch_max = features.max(dim=0).values
            max_features = torch.maximum(max_features, batch_max)
            
            del batch, x_hat, features, residual, cos, l0, batch_max
    
    # Compute final metrics
    ev_num = sum(all_ev_nums) / len(all_ev_nums)
    ev_den = sum(all_ev_dens) / len(all_ev_dens)
    explained_variance = (1.0 - ev_num / ev_den) * 100 if ev_den > 0 else 0.0
    
    dead_ratio = (max_features < 1e-6).float().mean().item() * 100
    mean_l0 = total_l0 / n_samples
    mean_cosine = sum(all_cos_sims) / len(all_cos_sims)
    
    return {
        'explained_variance_pct': round(explained_variance, 2),
        'dead_feature_ratio_pct': round(dead_ratio, 2),
        'mean_l0': round(mean_l0, 1),
        'reconstruction_cosine': round(mean_cosine, 6),
        'n_samples': n_samples,
    }


def process_arabic_from_chunks(layer):
    """
    Process Arabic activations from chunk files with mean pooling.
    Uses backup_layer6 for layer 6, or loads from main file with streaming for other layers.
    """
    print(f"\n  Loading Arabic activations for layer {layer}...")
    
    # Strategy 1: Check for chunk files (backup_layer6 has them for layer 6)
    chunk_dir = ACT_DIR / "backup_layer6"
    chunk_pattern = f"layer_{layer}_arabic_chunk_*.pt"
    chunk_files = sorted(glob.glob(str(chunk_dir / chunk_pattern)))
    
    if chunk_files:
        print(f"  Found {len(chunk_files)} chunk files in backup_layer6")
        all_pooled = []
        for i, cf in enumerate(chunk_files):
            chunk = torch.load(cf, map_location='cpu', weights_only=False)
            act = chunk['activations']
            if act.dim() == 3:
                # Mean pool across sequence dimension
                pooled = act.mean(dim=1)  # [N, seq_len, d_model] -> [N, d_model]
            else:
                pooled = act
            all_pooled.append(pooled)
            if (i + 1) % 20 == 0:
                print(f"    Processed {i+1}/{len(chunk_files)} chunks")
            del chunk, act
        
        activations = torch.cat(all_pooled, dim=0)
        del all_pooled
        gc.collect()
        print(f"  Loaded and pooled: {activations.shape}")
        return activations
    
    # Strategy 2: Load main file with streaming (for 21GB files)
    # We need to load the full file, which requires enough memory
    main_file = ACT_DIR / f"layer_{layer}_arabic.pt"
    if main_file.exists():
        size_gb = os.path.getsize(str(main_file)) / (1024**3)
        print(f"  Loading main file ({size_gb:.1f} GB) - requires sufficient RAM...")
        
        try:
            data = torch.load(str(main_file), map_location='cpu', weights_only=False)
            
            if isinstance(data, dict):
                act = data.get('activations', data.get('layer_activations'))
                if act is None:
                    # Maybe the whole thing is the tensor
                    for k, v in data.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 2:
                            act = v
                            break
            elif isinstance(data, torch.Tensor):
                act = data
            else:
                print(f"  Unexpected data type: {type(data)}")
                return None
            
            if act is not None:
                print(f"  Raw shape: {act.shape}")
                if act.dim() == 3:
                    # Mean pool across sequence dimension
                    # Process in chunks to save memory
                    chunk_size = 500
                    pooled_chunks = []
                    for start in range(0, act.shape[0], chunk_size):
                        end = min(start + chunk_size, act.shape[0])
                        pooled_chunks.append(act[start:end].mean(dim=1))
                    activations = torch.cat(pooled_chunks, dim=0)
                    del pooled_chunks, act, data
                    gc.collect()
                    print(f"  After mean pooling: {activations.shape}")
                    return activations
                elif act.dim() == 2:
                    print(f"  Already 2D: {act.shape}")
                    del data
                    return act
            
            del data
            gc.collect()
            
        except Exception as e:
            print(f"  Error loading main file: {type(e).__name__}: {e}")
            return None
    
    print(f"  No Arabic activation files found for layer {layer}")
    return None


def main():
    print("=" * 70)
    print("FIX: PaLiGemma Arabic Activation File Processing")
    print("=" * 70)
    
    print("\nROOT CAUSE IDENTIFIED:")
    print("  Arabic files: shape [N, 268, 2048] (3D - full sequence, NOT pooled)")
    print("  English files: shape [N, 2048] (2D - mean-pooled across sequence)")
    print("  SAE expects: [N, 2048] (2D)")
    print("  Fix: Apply mean pooling across dim=1 (sequence dimension)")
    
    layers = [3, 6, 9, 12, 15, 17]
    d_model = 2048
    d_hidden = 16384
    
    results = {
        'model': 'PaLiGemma-3B',
        'fix_applied': 'mean_pool_dim1',
        'root_cause': 'Arabic activations saved as [N, 268, 2048] without mean pooling; English saved as [N, 2048] with mean pooling',
        'layers': {},
    }
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")
        
        results['layers'][layer] = {}
        
        # Load SAE
        sae_path = SAE_DIR / f"sae_arabic_layer_{layer}.pt"
        if not sae_path.exists():
            print(f"  SAE not found: {sae_path}")
            continue
        
        sae = load_sae(str(sae_path), d_model, d_hidden)
        print(f"  SAE loaded: {sae_path.name}")
        
        # Process Arabic with mean pooling
        arabic_acts = process_arabic_from_chunks(layer)
        
        if arabic_acts is not None:
            print(f"  Computing Arabic metrics...")
            metrics = compute_metrics_batched(sae, arabic_acts)
            results['layers'][layer]['arabic'] = metrics
            print(f"    EV: {metrics['explained_variance_pct']:.1f}%")
            print(f"    Dead: {metrics['dead_feature_ratio_pct']:.1f}%")
            print(f"    L0: {metrics['mean_l0']:.0f}")
            print(f"    Cos: {metrics['reconstruction_cosine']:.4f}")
            del arabic_acts
        else:
            print(f"  Skipping Arabic - could not load activations")
        
        # Also compute English for comparison
        eng_path = ACT_DIR / f"layer_{layer}_english.pt"
        if eng_path.exists():
            data = torch.load(str(eng_path), map_location='cpu', weights_only=False)
            eng_acts = data['activations']
            print(f"\n  Computing English metrics (shape: {eng_acts.shape})...")
            
            # Load English SAE
            eng_sae_path = SAE_DIR / f"sae_english_layer_{layer}.pt"
            if eng_sae_path.exists():
                eng_sae = load_sae(str(eng_sae_path), d_model, d_hidden)
                metrics = compute_metrics_batched(eng_sae, eng_acts)
                results['layers'][layer]['english'] = metrics
                print(f"    EV: {metrics['explained_variance_pct']:.1f}%")
                print(f"    Dead: {metrics['dead_feature_ratio_pct']:.1f}%")
                print(f"    L0: {metrics['mean_l0']:.0f}")
                print(f"    Cos: {metrics['reconstruction_cosine']:.4f}")
                del eng_sae
            del data, eng_acts
        
        del sae
        gc.collect()
    
    # Save results
    output_path = RESULTS_DIR / "paligemma_arabic_metrics_fixed.json"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: PaLiGemma Arabic vs English Comparison")
    print(f"{'='*70}")
    print(f"{'Layer':>6} | {'Arabic EV%':>10} | {'English EV%':>11} | {'Ar Dead%':>9} | {'En Dead%':>9} | {'Ar L0':>7} | {'En L0':>7} | {'Ar Cos':>8} | {'En Cos':>8}")
    print("-" * 95)
    for layer in layers:
        if layer in results['layers']:
            ar = results['layers'][layer].get('arabic', {})
            en = results['layers'][layer].get('english', {})
            print(f"{layer:>6} | "
                  f"{ar.get('explained_variance_pct', 'N/A'):>10} | "
                  f"{en.get('explained_variance_pct', 'N/A'):>11} | "
                  f"{ar.get('dead_feature_ratio_pct', 'N/A'):>9} | "
                  f"{en.get('dead_feature_ratio_pct', 'N/A'):>9} | "
                  f"{ar.get('mean_l0', 'N/A'):>7} | "
                  f"{en.get('mean_l0', 'N/A'):>7} | "
                  f"{ar.get('reconstruction_cosine', 'N/A'):>8} | "
                  f"{en.get('reconstruction_cosine', 'N/A'):>8}")


if __name__ == '__main__':
    main()
