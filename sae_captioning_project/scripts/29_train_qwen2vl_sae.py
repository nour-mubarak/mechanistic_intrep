#!/usr/bin/env python3
"""
SAE Training for Qwen2-VL-7B-Instruct
=====================================

Train Sparse Autoencoders on Qwen2-VL activations.
- Hidden size: 3584
- Expansion factor: 8 → 28672 SAE features

Usage:
    python scripts/29_train_qwen2vl_sae.py --language arabic --layer 0
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

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))


class Qwen2VLSAE(nn.Module):
    """Sparse Autoencoder for Qwen2-VL (d_model=3584)."""
    
    def __init__(self, d_model: int = 3584, expansion_factor: int = 8, l1_coefficient: float = 5e-4):
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
        
        return reconstruction, features, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'l1_loss': l1_loss,
            'sparsity': (features > 0).float().mean()
        }


def load_qwen2vl_activations(path: str) -> tuple:
    """Load Qwen2-VL activation file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    activations = data['activations']
    genders = data.get('genders', [])
    
    print(f"  Loaded: {activations.shape}")
    return activations, genders


def train_sae(
    activations: torch.Tensor,
    d_model: int = 3584,
    expansion_factor: int = 8,
    l1_coefficient: float = 5e-4,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    epochs: int = 50,
    device: str = "cuda"
) -> tuple:
    """Train SAE on activations."""
    
    # Create model
    sae = Qwen2VLSAE(d_model, expansion_factor, l1_coefficient).to(device)
    
    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'l1_loss': [],
        'sparsity': []
    }
    
    # Training loop
    for epoch in range(epochs):
        epoch_losses = {'total_loss': 0, 'recon_loss': 0, 'l1_loss': 0, 'sparsity': 0}
        n_batches = 0
        
        for (batch,) in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            _, _, losses = sae(batch)
            losses['total_loss'].backward()
            optimizer.step()
            
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            n_batches += 1
        
        scheduler.step()
        
        # Record epoch averages
        for k in epoch_losses:
            avg = epoch_losses[k] / n_batches
            history[k].append(avg)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={history['total_loss'][-1]:.4f}, "
                  f"recon={history['recon_loss'][-1]:.4f}, "
                  f"sparsity={history['sparsity'][-1]:.3f}")
    
    return sae, history


def main():
    parser = argparse.ArgumentParser(description="Train SAE for Qwen2-VL")
    parser.add_argument("--language", type=str, required=True, choices=["arabic", "english"])
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--input_dir", type=str, default="checkpoints/qwen2vl/layer_checkpoints")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen2vl/saes")
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--l1_coefficient", type=float, default=5e-4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print("="*60)
    print(f"Qwen2-VL SAE Training - Layer {args.layer} ({args.language})")
    print("="*60)
    
    # Load activations
    input_path = Path(args.input_dir) / f"qwen2vl_layer_{args.layer}_{args.language}.pt"
    print(f"Loading: {input_path}")
    activations, genders = load_qwen2vl_activations(str(input_path))
    
    # Get d_model from data
    d_model = activations.shape[-1]
    print(f"d_model: {d_model}")
    print(f"SAE hidden: {d_model * args.expansion_factor}")
    
    # Train SAE
    print("\nTraining SAE...")
    sae, history = train_sae(
        activations=activations,
        d_model=d_model,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coefficient,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device
    )
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"qwen2vl_sae_{args.language}_layer_{args.layer}.pt"
    history_path = output_dir / f"qwen2vl_sae_{args.language}_layer_{args.layer}_history.json"
    
    # Save model state
    torch.save({
        'model_state_dict': sae.state_dict(),
        'd_model': d_model,
        'd_hidden': d_model * args.expansion_factor,
        'expansion_factor': args.expansion_factor,
        'l1_coefficient': args.l1_coefficient,
        'config': {
            'd_model': d_model,
            'expansion_factor': args.expansion_factor,
            'l1_coefficient': args.l1_coefficient
        }
    }, model_path)
    print(f"\nSaved model: {model_path}")
    
    # Save history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved history: {history_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final loss: {history['total_loss'][-1]:.4f}")
    print(f"Final sparsity: {history['sparsity'][-1]:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
