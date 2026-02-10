#!/usr/bin/env python3
"""
Llama 3.2 Vision SAE Retraining with Tuned Hyperparameters
===========================================================

The original SAEs had:
- Explained Variance: 36.6% (target: >65%)
- Dead Features: 98.6% (target: <35%)

Tuning strategy:
1. Lower L1 coefficient: 1e-4 → 5e-5 (reduce sparsity pressure → more alive features)
2. More epochs: 50 → 100 (better convergence)
3. Warmup for L1: Start with 0, ramp up over first 10 epochs
4. Learning rate schedule: Cosine annealing with warmup

Usage:
    python scripts/41_llama_sae_retrain_tuned.py --layer 20 --language arabic
    python scripts/41_llama_sae_retrain_tuned.py --all  # Retrain all layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import gc

# Configuration
MODEL_NAME = "Llama-3.2-Vision-11B"
HIDDEN_SIZE = 4096
EXPANSION_FACTOR = 8


class ImprovedLlamaSAE(nn.Module):
    """Improved SAE with better initialization and tied decoder weights option."""
    
    def __init__(self, d_model: int = 4096, expansion_factor: int = 8, 
                 l1_coefficient: float = 5e-5, tied_weights: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.l1_coefficient = l1_coefficient
        self.tied_weights = tied_weights
        
        # Encoder with bias for centering
        self.encoder = nn.Linear(d_model, self.d_hidden, bias=True)
        
        # Decoder (optionally tied to encoder)
        if not tied_weights:
            self.decoder = nn.Linear(self.d_hidden, d_model, bias=True)
        
        # Pre-encoder bias (subtract mean activation)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        
        # Initialize with orthogonal for better gradient flow
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        if not tied_weights:
            nn.init.orthogonal_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with pre-bias subtraction."""
        x_centered = x - self.pre_bias
        return torch.relu(self.encoder(x_centered))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode back to input space."""
        if self.tied_weights:
            return torch.mm(features, self.encoder.weight) + self.pre_bias
        else:
            return self.decoder(features) + self.pre_bias
    
    def forward(self, x: torch.Tensor, l1_coeff_override: float = None) -> tuple:
        """Forward pass with optional L1 coefficient override for warmup."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstruction, x)
        
        # Sparsity loss with optional override
        l1_coeff = l1_coeff_override if l1_coeff_override is not None else self.l1_coefficient
        l1_loss = l1_coeff * features.abs().mean()
        
        # Total loss
        total_loss = recon_loss + l1_loss
        
        # Metrics
        with torch.no_grad():
            sparsity = (features > 0).float().mean()
            active_features = (features > 0).any(dim=0).sum()
            
            # Explained variance
            var_orig = x.var()
            var_resid = (x - reconstruction).var()
            explained_var = 1 - (var_resid / (var_orig + 1e-8))
            
            # Mean L0 (features active per sample)
            l0_per_sample = (features > 0).float().sum(dim=1).mean()
        
        return reconstruction, features, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'l1_loss': l1_loss,
            'sparsity': sparsity,
            'active_features': active_features,
            'explained_variance': explained_var,
            'mean_l0': l0_per_sample
        }


def load_activations(checkpoints_dir: Path, language: str, layer: int) -> tuple:
    """Load activation files."""
    pattern = f"llama32vision_{language}_layer{layer}_*.npz"
    files = list(checkpoints_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    
    all_activations = []
    all_genders = []
    
    for f in sorted(files):
        data = np.load(f)
        all_activations.append(data['activations'])
        all_genders.extend(data['genders'].tolist())
    
    activations = np.concatenate(all_activations, axis=0)
    return torch.from_numpy(activations).float(), all_genders


def train_sae_with_warmup(
    sae: ImprovedLlamaSAE,
    activations: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    warmup_epochs: int = 10,
    device: str = "cuda"
) -> dict:
    """Train SAE with L1 warmup and better scheduling."""
    
    sae = sae.to(device)
    activations = activations.to(device)
    
    # Initialize pre-bias with mean of activations
    with torch.no_grad():
        sae.pre_bias.data = activations.mean(dim=0)
    
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    optimizer = optim.AdamW(sae.parameters(), lr=lr, weight_decay=1e-5)
    
    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    history = {
        'total_loss': [], 'recon_loss': [], 'l1_loss': [],
        'explained_variance': [], 'active_features': [], 'mean_l0': []
    }
    
    best_ev = 0
    best_state = None
    
    for epoch in range(epochs):
        epoch_metrics = {k: [] for k in history.keys()}
        
        # L1 warmup: linearly increase from 0 to target over warmup_epochs
        if epoch < warmup_epochs:
            l1_coeff = sae.l1_coefficient * (epoch / warmup_epochs)
        else:
            l1_coeff = sae.l1_coefficient
        
        for batch in dataloader:
            x = batch[0]
            
            optimizer.zero_grad()
            _, features, losses = sae(x, l1_coeff_override=l1_coeff)
            
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            
            for k in epoch_metrics:
                val = losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
                epoch_metrics[k].append(val)
        
        scheduler.step()
        
        # Record epoch averages
        for k in history:
            avg = np.mean(epoch_metrics[k])
            history[k].append(avg)
        
        # Track best model by explained variance
        current_ev = history['explained_variance'][-1]
        if current_ev > best_ev:
            best_ev = current_ev
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"EV={history['explained_variance'][-1]*100:.1f}%, "
                  f"Active={history['active_features'][-1]:.0f}, "
                  f"L0={history['mean_l0'][-1]:.0f}, "
                  f"Loss={history['total_loss'][-1]:.6f}")
    
    # Restore best state
    if best_state:
        sae.load_state_dict(best_state)
        print(f"  Restored best model with EV={best_ev*100:.1f}%")
    
    return history


def extract_gender_features(sae, activations, genders, device="cuda", top_k=100):
    """Extract gender-associated features."""
    sae = sae.to(device).eval()
    activations = activations.to(device)
    
    with torch.no_grad():
        features = sae.encode(activations).cpu().numpy()
    
    genders = np.array(genders)
    male_mask = genders == "male"
    female_mask = genders == "female"
    
    male_mean = features[male_mask].mean(axis=0)
    female_mean = features[female_mask].mean(axis=0)
    
    gender_direction = male_mean - female_mean
    
    male_top_idx = np.argsort(gender_direction)[-top_k:][::-1]
    female_top_idx = np.argsort(gender_direction)[:top_k]
    
    return {
        'male_features': male_top_idx.tolist(),
        'female_features': female_top_idx.tolist(),
        'gender_direction': gender_direction.tolist(),
        'n_male': int(male_mask.sum()),
        'n_female': int(female_mask.sum())
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain Llama SAEs with tuned hyperparameters")
    parser.add_argument("--layer", type=int, help="Specific layer to train")
    parser.add_argument("--language", type=str, choices=["arabic", "english"])
    parser.add_argument("--all", action="store_true", help="Train all layers and languages")
    parser.add_argument("--input_dir", type=str, default="checkpoints/llama32vision/layer_checkpoints")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llama32vision/saes_tuned")
    parser.add_argument("--l1", type=float, default=5e-5, help="L1 coefficient (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Llama 3.2 Vision SAE Retraining with Tuned Hyperparameters")
    print("=" * 70)
    print(f"L1 coefficient: {args.l1}")
    print(f"Epochs: {args.epochs}")
    print()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine what to train
    if args.all:
        layers = [0, 5, 10, 15, 20, 25, 30, 35, 39]
        languages = ["arabic", "english"]
    else:
        layers = [args.layer] if args.layer is not None else [20]
        languages = [args.language] if args.language else ["arabic", "english"]
    
    results_summary = []
    
    for layer in layers:
        for language in languages:
            print(f"\n{'='*70}")
            print(f"Training: Layer {layer}, {language.upper()}")
            print(f"{'='*70}")
            
            try:
                # Load activations
                activations, genders = load_activations(input_dir, language, layer)
                print(f"Loaded {len(activations)} samples, shape: {activations.shape}")
                
                d_model = activations.shape[1]
                
                # Create improved SAE
                sae = ImprovedLlamaSAE(
                    d_model=d_model,
                    expansion_factor=EXPANSION_FACTOR,
                    l1_coefficient=args.l1,
                    tied_weights=False
                )
                print(f"SAE: {d_model} → {sae.d_hidden} features")
                
                # Train with warmup
                history = train_sae_with_warmup(
                    sae, activations,
                    epochs=args.epochs,
                    batch_size=256,
                    lr=1e-3,
                    warmup_epochs=10,
                    device=args.device
                )
                
                # Extract gender features
                gender_features = extract_gender_features(sae, activations, genders, args.device)
                
                # Final metrics
                final_ev = history['explained_variance'][-1] * 100
                final_active = history['active_features'][-1]
                final_l0 = history['mean_l0'][-1]
                dead_pct = (1 - final_active / sae.d_hidden) * 100
                
                results_summary.append({
                    'layer': layer,
                    'language': language,
                    'explained_variance': final_ev,
                    'dead_features_pct': dead_pct,
                    'mean_l0': final_l0,
                    'active_features': int(final_active)
                })
                
                print(f"\n  Final Results:")
                print(f"    Explained Variance: {final_ev:.1f}%")
                print(f"    Dead Features: {dead_pct:.1f}%")
                print(f"    Mean L0: {final_l0:.0f}")
                print(f"    Active Features: {int(final_active)}/{sae.d_hidden}")
                
                # Save SAE
                sae_path = output_dir / f"llama32vision_sae_{language}_layer{layer}_tuned.pt"
                torch.save({
                    'state_dict': sae.state_dict(),
                    'd_model': d_model,
                    'expansion_factor': EXPANSION_FACTOR,
                    'l1_coefficient': args.l1,
                    'language': language,
                    'layer': layer,
                    'training_history': history,
                    'gender_features': gender_features,
                    'final_metrics': {
                        'explained_variance_pct': final_ev,
                        'dead_feature_pct': dead_pct,
                        'mean_l0': final_l0,
                        'active_features': int(final_active)
                    }
                }, sae_path)
                print(f"  Saved: {sae_path.name}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Layer':<8} {'Language':<10} {'EV%':>8} {'Dead%':>8} {'L0':>8}")
    print("-" * 70)
    for r in results_summary:
        print(f"{r['layer']:<8} {r['language']:<10} {r['explained_variance']:>7.1f}% {r['dead_features_pct']:>7.1f}% {r['mean_l0']:>8.0f}")
    
    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'l1_coefficient': args.l1,
                'epochs': args.epochs,
                'expansion_factor': EXPANSION_FACTOR
            },
            'results': results_summary
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
