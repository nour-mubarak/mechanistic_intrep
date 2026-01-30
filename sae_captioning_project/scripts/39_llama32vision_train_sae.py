#!/usr/bin/env python3
"""
SAE Training for Llama 3.2 Vision (11B)
========================================

Train Sparse Autoencoders on Llama 3.2 Vision activations.
- Hidden size: 4096
- Expansion factor: 8 → 32768 SAE features

Usage:
    python scripts/39_llama32vision_train_sae.py --language arabic --layer 20
    python scripts/39_llama32vision_train_sae.py --language english --layer 20
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


# Model configuration
MODEL_NAME = "Llama-3.2-Vision-11B"
HIDDEN_SIZE = 4096
DEFAULT_EXPANSION = 8


class Llama32VisionSAE(nn.Module):
    """Sparse Autoencoder for Llama 3.2 Vision (d_model=4096)."""

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


def load_activations(checkpoints_dir: Path, language: str, layer: int) -> tuple:
    """Load activation files for a specific layer."""
    pattern = f"llama32vision_{language}_layer{layer}_*.npz"
    files = list(checkpoints_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No activation files found matching {pattern}")

    print(f"Found {len(files)} checkpoint file(s)")

    all_activations = []
    all_genders = []

    for f in sorted(files):
        data = np.load(f)
        all_activations.append(data['activations'])
        all_genders.extend(data['genders'].tolist())
        print(f"  Loaded {f.name}: {data['activations'].shape}")

    activations = np.concatenate(all_activations, axis=0)
    activations = torch.from_numpy(activations).float()

    return activations, all_genders


def train_sae(
    sae: Llama32VisionSAE,
    activations: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    use_wandb: bool = False
) -> dict:
    """Train SAE on activations."""

    sae = sae.to(device)
    activations = activations.to(device)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(sae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'total_loss': [],
        'recon_loss': [],
        'l1_loss': [],
        'sparsity': [],
        'active_features': []
    }

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        epoch_metrics = {k: [] for k in history.keys()}

        for batch in dataloader:
            x = batch[0]

            optimizer.zero_grad()
            _, features, losses = sae(x)

            losses['total_loss'].backward()
            optimizer.step()

            for k in epoch_metrics:
                val = losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
                epoch_metrics[k].append(val)

        scheduler.step()

        # Average epoch metrics
        for k in history:
            avg = np.mean(epoch_metrics[k])
            history[k].append(avg)

        avg_loss = history['total_loss'][-1]
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}

        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={history['total_loss'][-1]:.6f}, "
                  f"Recon={history['recon_loss'][-1]:.6f}, "
                  f"Sparsity={history['sparsity'][-1]:.4f}, "
                  f"Active={history['active_features'][-1]:.0f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'total_loss': history['total_loss'][-1],
                'recon_loss': history['recon_loss'][-1],
                'l1_loss': history['l1_loss'][-1],
                'sparsity': history['sparsity'][-1],
                'active_features': history['active_features'][-1]
            })

    # Restore best state
    if best_state:
        sae.load_state_dict(best_state)

    return history


def extract_gender_features(
    sae: Llama32VisionSAE,
    activations: torch.Tensor,
    genders: list,
    device: str = "cuda",
    top_k: int = 100
) -> dict:
    """Identify gender-associated SAE features."""

    sae = sae.to(device).eval()
    activations = activations.to(device)

    # Encode all activations
    with torch.no_grad():
        features = sae.encode(activations).cpu().numpy()

    genders = np.array(genders)
    male_mask = genders == "male"
    female_mask = genders == "female"

    # Mean activation per gender
    male_mean = features[male_mask].mean(axis=0)
    female_mean = features[female_mask].mean(axis=0)

    # Gender direction
    gender_direction = male_mean - female_mean

    # Top features
    male_top_idx = np.argsort(gender_direction)[-top_k:][::-1]
    female_top_idx = np.argsort(gender_direction)[:top_k]

    return {
        'male_features': male_top_idx.tolist(),
        'female_features': female_top_idx.tolist(),
        'gender_direction': gender_direction.tolist(),
        'male_mean_activation': male_mean.tolist(),
        'female_mean_activation': female_mean.tolist(),
        'n_male': int(male_mask.sum()),
        'n_female': int(female_mask.sum())
    }


def main():
    parser = argparse.ArgumentParser(description="Train SAE for Llama 3.2 Vision")
    parser.add_argument("--language", type=str, required=True, choices=["arabic", "english"])
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--input_dir", type=str, default="checkpoints/llama32vision/layer_checkpoints")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llama32vision/saes")
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--l1_coefficient", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true", default=True, help="Log to W&B (enabled by default)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llama32vision-sae-analysis")
    parser.add_argument("--wandb_entity", type=str, default="nourmubarak")
    args = parser.parse_args()

    # Handle wandb flag
    if args.no_wandb:
        args.wandb = False

    print("=" * 60)
    print(f"SAE Training: Llama 3.2 Vision")
    print(f"Language: {args.language}, Layer: {args.layer}")
    print("=" * 60)

    # Initialize W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"llama32vision_sae_L{args.layer}_{args.language}",
            config={
                "model": MODEL_NAME,
                "language": args.language,
                "layer": args.layer,
                "d_model": HIDDEN_SIZE,
                "expansion_factor": args.expansion_factor,
                "d_hidden": HIDDEN_SIZE * args.expansion_factor,
                "l1_coefficient": args.l1_coefficient,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "device": args.device
            },
            tags=["sae-training", args.language, f"layer-{args.layer}", "llama32vision"]
        )
        print(f"W&B initialized: {wandb.run.url}")

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load activations
    print(f"\nLoading activations for layer {args.layer}...")
    activations, genders = load_activations(input_dir, args.language, args.layer)
    print(f"Loaded {len(activations)} samples, shape: {activations.shape}")
    print(f"Genders: {genders.count('male')} male, {genders.count('female')} female")

    # Create SAE
    d_model = activations.shape[1]
    print(f"\nCreating SAE: d_model={d_model}, expansion={args.expansion_factor}")
    sae = Llama32VisionSAE(
        d_model=d_model,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coefficient
    )
    print(f"SAE features: {sae.d_hidden}")

    # Train
    print(f"\nTraining SAE for {args.epochs} epochs...")
    history = train_sae(
        sae, activations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_wandb=args.wandb
    )

    # Extract gender features
    print("\nExtracting gender-associated features...")
    gender_features = extract_gender_features(sae, activations, genders, args.device)

    # Save SAE
    sae_path = output_dir / f"llama32vision_sae_{args.language}_layer{args.layer}.pt"
    torch.save({
        'state_dict': sae.state_dict(),
        'd_model': d_model,
        'expansion_factor': args.expansion_factor,
        'l1_coefficient': args.l1_coefficient,
        'language': args.language,
        'layer': args.layer,
        'training_history': history,
        'gender_features': gender_features
    }, sae_path)
    print(f"Saved SAE to {sae_path}")

    # Save gender features separately
    features_path = output_dir / f"llama32vision_gender_features_{args.language}_layer{args.layer}.json"
    with open(features_path, 'w') as f:
        json.dump(gender_features, f, indent=2)
    print(f"Saved gender features to {features_path}")

    if args.wandb:
        wandb.log({
            "final_loss": history['total_loss'][-1],
            "final_sparsity": history['sparsity'][-1],
            "male_samples": gender_features['n_male'],
            "female_samples": gender_features['n_female']
        })
        wandb.finish()

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"Final loss: {history['total_loss'][-1]:.6f}")
    print(f"Final sparsity: {history['sparsity'][-1]:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
