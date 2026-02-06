#!/bin/bash
#SBATCH --job-name=pali_arabic_fix
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fix_arabic_%j.out
#SBATCH --error=logs/fix_arabic_%j.err

echo "============================================================"
echo "FIX: PaLiGemma Arabic Metrics with Mean Pooling"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project

python3 << 'PYTHON_SCRIPT'
import torch
import torch.nn as nn
import os
import json
import gc
import glob
from pathlib import Path

BASE = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project")
ACT_DIR = BASE / "checkpoints" / "full_layers_ncc" / "layer_checkpoints"
SAE_DIR = BASE / "checkpoints" / "saes"
RESULTS_DIR = BASE / "results" / "sae_quality_metrics"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")

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


def compute_metrics_gpu(sae, activations, batch_size=1024):
    """Compute metrics with GPU acceleration."""
    sae = sae.to(device)
    n_samples = activations.shape[0]
    d_hidden = sae.encoder.out_features
    
    all_ev_nums = []
    all_ev_dens = []
    all_cos_sims = []
    max_features = torch.zeros(d_hidden, device=device)
    total_l0 = 0.0
    
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = activations[start:end].to(device)
            
            x_hat, features = sae(batch)
            
            residual = batch - x_hat
            all_ev_nums.append(residual.var(dim=0).sum().item())
            all_ev_dens.append(batch.var(dim=0).sum().item())
            
            cos = torch.nn.functional.cosine_similarity(batch, x_hat, dim=-1)
            all_cos_sims.extend(cos.cpu().tolist())
            
            l0 = (features > 0).float().sum(dim=-1)
            total_l0 += l0.sum().item()
            
            batch_max = features.max(dim=0).values
            max_features = torch.maximum(max_features, batch_max)
            
            del batch, x_hat, features, residual, cos, l0, batch_max
            torch.cuda.empty_cache()
    
    ev_num = sum(all_ev_nums) / len(all_ev_nums)
    ev_den = sum(all_ev_dens) / len(all_ev_dens)
    ev = (1.0 - ev_num / ev_den) * 100 if ev_den > 0 else 0.0
    
    sae = sae.cpu()
    torch.cuda.empty_cache()
    
    return {
        'explained_variance_pct': round(ev, 2),
        'dead_feature_ratio_pct': round((max_features.cpu() < 1e-6).float().mean().item() * 100, 2),
        'mean_l0': round(total_l0 / n_samples, 1),
        'reconstruction_cosine': round(sum(all_cos_sims) / len(all_cos_sims), 6),
        'n_samples': n_samples,
    }


def load_and_pool_arabic(layer):
    """Load Arabic activation file and apply mean pooling across sequence dim."""
    
    # Strategy 1: Check for chunk files (layer 6 has these)
    chunk_dir = ACT_DIR / "backup_layer6"
    chunk_files = sorted(glob.glob(str(chunk_dir / f"layer_{layer}_arabic_chunk_*.pt")))
    
    if chunk_files:
        print(f"  Using {len(chunk_files)} chunk files from backup_layer6")
        all_pooled = []
        for i, cf in enumerate(chunk_files):
            c = torch.load(cf, map_location='cpu', weights_only=False)
            act = c['activations']
            if act.dim() == 3:
                pooled = act.mean(dim=1)
            else:
                pooled = act
            all_pooled.append(pooled)
            del c, act
            if (i+1) % 25 == 0:
                gc.collect()
        activations = torch.cat(all_pooled, dim=0)
        del all_pooled
        gc.collect()
        return activations
    
    # Strategy 2: Load main 21GB file with streaming mean pooling
    main_file = ACT_DIR / f"layer_{layer}_arabic.pt"
    if main_file.exists():
        size_gb = os.path.getsize(str(main_file)) / (1024**3)
        print(f"  Loading main file ({size_gb:.1f} GB)...")
        
        data = torch.load(str(main_file), map_location='cpu', weights_only=False)
        
        if isinstance(data, dict):
            act = data.get('activations')
            if act is None:
                for k, v in data.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        act = v
                        break
        elif isinstance(data, torch.Tensor):
            act = data
        else:
            print(f"  Unexpected type: {type(data)}")
            return None
        
        if act is not None:
            print(f"  Raw shape: {act.shape}")
            if act.dim() == 3:
                # Streaming mean pool to avoid peak memory
                chunk_size = 200
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
                del data
                return act
        
        del data
        gc.collect()
    
    return None


def main():
    print("=" * 70)
    print("PaLiGemma Arabic Metrics - GPU Fix with Mean Pooling")
    print("=" * 70)
    
    layers = [3, 6, 9, 12, 15, 17]
    results = {'model': 'PaLiGemma-3B', 'layers': {}}
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")
        
        results['layers'][str(layer)] = {}
        
        # Arabic metrics
        sae_path = SAE_DIR / f"sae_arabic_layer_{layer}.pt"
        if not sae_path.exists():
            print(f"  Arabic SAE not found: {sae_path}")
        else:
            print(f"  Loading Arabic SAE...")
            ckpt = torch.load(str(sae_path), map_location='cpu', weights_only=False)
            sae = SparseAutoencoder(2048, 16384)
            sae.load_state_dict(ckpt['model_state_dict'])
            sae.eval()
            del ckpt; gc.collect()
            
            arabic_acts = load_and_pool_arabic(layer)
            if arabic_acts is not None:
                print(f"  Computing Arabic metrics...")
                metrics = compute_metrics_gpu(sae, arabic_acts)
                results['layers'][str(layer)]['arabic'] = metrics
                print(f"    EV:   {metrics['explained_variance_pct']:.1f}%")
                print(f"    Dead: {metrics['dead_feature_ratio_pct']:.1f}%")
                print(f"    L0:   {metrics['mean_l0']:.0f}")
                print(f"    Cos:  {metrics['reconstruction_cosine']:.4f}")
                del arabic_acts
            
            del sae; gc.collect(); torch.cuda.empty_cache()
        
        # English metrics (for comparison)
        eng_sae_path = SAE_DIR / f"sae_english_layer_{layer}.pt"
        eng_act_path = ACT_DIR / f"layer_{layer}_english.pt"
        
        if eng_sae_path.exists() and eng_act_path.exists():
            print(f"  Loading English SAE and activations...")
            ckpt = torch.load(str(eng_sae_path), map_location='cpu', weights_only=False)
            sae = SparseAutoencoder(2048, 16384)
            sae.load_state_dict(ckpt['model_state_dict'])
            sae.eval()
            del ckpt
            
            data = torch.load(str(eng_act_path), map_location='cpu', weights_only=False)
            eng_acts = data['activations']
            
            print(f"  Computing English metrics (shape: {eng_acts.shape})...")
            metrics = compute_metrics_gpu(sae, eng_acts)
            results['layers'][str(layer)]['english'] = metrics
            print(f"    EV:   {metrics['explained_variance_pct']:.1f}%")
            print(f"    Dead: {metrics['dead_feature_ratio_pct']:.1f}%")
            print(f"    L0:   {metrics['mean_l0']:.0f}")
            print(f"    Cos:  {metrics['reconstruction_cosine']:.4f}")
            
            del sae, data, eng_acts; gc.collect(); torch.cuda.empty_cache()
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "paligemma_arabic_fixed_metrics.json"
    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Layer':>6} | {'Ar EV%':>8} | {'En EV%':>8} | {'Ar Dead%':>9} | {'En Dead%':>9} | {'Ar L0':>7} | {'En L0':>7} | {'Ar Cos':>8} | {'En Cos':>8}")
    print("-" * 95)
    for layer in layers:
        lk = str(layer)
        if lk in results['layers']:
            ar = results['layers'][lk].get('arabic', {})
            en = results['layers'][lk].get('english', {})
            print(f"{layer:>6} | "
                  f"{ar.get('explained_variance_pct', '-'):>8} | "
                  f"{en.get('explained_variance_pct', '-'):>8} | "
                  f"{ar.get('dead_feature_ratio_pct', '-'):>9} | "
                  f"{en.get('dead_feature_ratio_pct', '-'):>9} | "
                  f"{ar.get('mean_l0', '-'):>7} | "
                  f"{en.get('mean_l0', '-'):>7} | "
                  f"{ar.get('reconstruction_cosine', '-'):>8} | "
                  f"{en.get('reconstruction_cosine', '-'):>8}")


if __name__ == '__main__':
    main()
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
