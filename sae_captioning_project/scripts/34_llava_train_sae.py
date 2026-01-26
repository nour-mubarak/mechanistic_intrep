#!/usr/bin/env python3
"""
SAE Training for LLaVA-1.5-7B
==============================

Train Sparse Autoencoders on LLaVA activations.
- Hidden size: 4096
- Expansion factor: 8 → 32768 SAE features

Usage:
    python scripts/34_llava_train_sae.py --language arabic --layer 16
    python scripts/34_llava_train_sae.py --language english --layer 16
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
import wandb

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))


class LLaVASAE(nn.Module):
    """Sparse Autoencoder for LLaVA (d_model=4096)."""
    
    def __init__(self, d_model: int = 4096, expansion_factor: int = 8, l1_coefficient: float = 1e-4):
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.l1_coefficient = l1_coefficient
        
        # Encoder: d_model -> d_hidden
        self.encoder = nn.Linear(d_model, self.d_hidden)
        
        # Decoder: d_hidden -> d_model  
        self.decoder = nn.Linear(self.d_hidden, d_model)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.kaiming_normal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features."""
        return torch.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space."""
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass returning reconstruction, features, and losses."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstruction, x)
        
        # Sparsity loss (L1 on features)
        l1_loss = self.l1_coefficient * features.abs().mean()
        
        # Total loss
        total_loss = recon_loss + l1_loss
        
        # Sparsity metrics
        sparsity = (features > 0).float().mean()
        active_features = (features > 0).any(dim=0).sum()
        
        return reconstruction, features, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'l1_loss': l1_loss,
            'sparsity': sparsity,
            'active_features': active_features
        }


def load_llava_activations(path: str) -> tuple:
    """Load LLaVA activation file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    activations = data["activations"]
    # Convert to float32 if needed
    if activations.dtype != torch.float32:
        activations = activations.float()
    
    genders = data.get('genders', [])
    
    print(f"  Loaded: {activations.shape}")
    print(f"  Dtype: {activations.dtype}")
    print(f"  Genders: {sum(1 for g in genders if g == 'male')} male, "
          f"{sum(1 for g in genders if g == 'female')} female")
    
    return activations, genders


def train_sae(
    activations: torch.Tensor,
    d_model: int = 4096,
    expansion_factor: int = 8,
    l1_coefficient: float = 1e-4,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    epochs: int = 50,
    device: str = "cuda",
    use_wandb: bool = False
) -> tuple:
    """Train SAE on activations."""
    
    print(f"\n{'='*60}")
    print("Training LLaVA SAE")
    print(f"{'='*60}")
    print(f"  d_model: {d_model}")
    print(f"  d_hidden: {d_model * expansion_factor}")
    print(f"  L1 coefficient: {l1_coefficient}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    # Create model
    sae = LLaVASAE(d_model, expansion_factor, l1_coefficient).to(device)
    
    # Move activations to device
    activations = activations.to(device)
    
    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(sae.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'l1_loss': [],
        'sparsity': [],
        'active_features': []
    }
    
    # Best model tracking
    best_loss = float('inf')
    best_state = None
    
    # Training loop
    for epoch in range(epochs):
        sae.train()
        epoch_losses = {k: 0.0 for k in history.keys()}
        n_batches = 0
        
        for (batch,) in dataloader:
            optimizer.zero_grad()
            _, _, losses = sae(batch)
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            for k, v in losses.items():
                epoch_losses[k] += v.item() if torch.is_tensor(v) else v
            n_batches += 1
        
        scheduler.step()
        
        # Record epoch averages
        for k in epoch_losses:
            avg = epoch_losses[k] / n_batches
            history[k].append(avg)
        
        # Track best model
        if history['total_loss'][-1] < best_loss:
            best_loss = history['total_loss'][-1]
            best_state = sae.state_dict().copy()
        
        # Log to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": history['total_loss'][-1],
                "train/recon_loss": history['recon_loss'][-1],
                "train/l1_loss": history['l1_loss'][-1],
                "train/sparsity": history['sparsity'][-1],
                "train/active_features": history['active_features'][-1],
                "train/learning_rate": scheduler.get_last_lr()[0]
            })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={history['total_loss'][-1]:.4f}, "
                  f"recon={history['recon_loss'][-1]:.4f}, "
                  f"sparsity={history['sparsity'][-1]:.3f}, "
                  f"active={int(history['active_features'][-1])}")
    
    # Load best model
    if best_state is not None:
        sae.load_state_dict(best_state)
    
    return sae, history


def main():
    parser = argparse.ArgumentParser(description="Train SAE for LLaVA")
    parser.add_argument("--language", type=str, required=True, choices=["arabic", "english"])
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--input_dir", type=str, default="checkpoints/llava/layer_checkpoints")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llava/saes")
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--l1_coefficient", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llava-sae-analysis")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("LLaVA-1.5-7B SAE Training")
    print(f"{'='*60}")
    print(f"Language: {args.language}")
    print(f"Layer: {args.layer}")
    
    # Initialize W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"llava_sae_L{args.layer}_{args.language}",
            config={
                "model": "llava-hf/llava-1.5-7b-hf",
                "language": args.language,
                "layer": args.layer,
                "expansion_factor": args.expansion_factor,
                "l1_coefficient": args.l1_coefficient,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "stage": "sae_training"
            }
        )
    
    # Paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load activations
    input_path = input_dir / f"llava_layer_{args.layer}_{args.language}.pt"
    if not input_path.exists():
        print(f"ERROR: Activation file not found: {input_path}")
        return
    
    print(f"\nLoading activations from: {input_path}")
    activations, genders = load_llava_activations(str(input_path))
    
    # Infer d_model from activations
    d_model = activations.shape[1]
    print(f"Inferred d_model: {d_model}")
    
    # Train SAE
    sae, history = train_sae(
        activations=activations,
        d_model=d_model,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coefficient,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        use_wandb=args.wandb
    )
    
    # Save model
    output_path = output_dir / f"llava_sae_{args.language}_layer_{args.layer}.pt"
    torch.save({
        'model_state_dict': sae.state_dict(),
        'd_model': d_model,
        'd_hidden': d_model * args.expansion_factor,
        'expansion_factor': args.expansion_factor,
        'l1_coefficient': args.l1_coefficient,
        'language': args.language,
        'layer': args.layer,
        'history': history,
        'final_loss': history['total_loss'][-1],
        'final_sparsity': history['sparsity'][-1],
        'model': 'llava-hf/llava-1.5-7b-hf',
        'timestamp': datetime.now().isoformat()
    }, output_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final loss: {history['total_loss'][-1]:.4f}")
    print(f"  Final sparsity: {history['sparsity'][-1]:.3f}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}")
    
    if args.wandb:
        # Log final metrics
        wandb.log({
            "final/total_loss": history['total_loss'][-1],
            "final/recon_loss": history['recon_loss'][-1],
            "final/sparsity": history['sparsity'][-1]
        })
        wandb.finish()


if __name__ == "__main__":
    main()
