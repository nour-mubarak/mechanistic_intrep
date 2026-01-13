#!/usr/bin/env python3
"""
Train SAE on NCC (Neural Corpus Compilation) format activations.

This script is designed to work with the layer-specific checkpoint format:
  layer_X_english_chunk_YYYY.pt
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
from datetime import datetime
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sae import SAEConfig, SparseAutoencoder, create_sae

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ncc_activations(checkpoint_dir: Path, layer: int, language: str = 'english', max_chunks: int = None, sample_ratio: float = 0.1) -> dict:
    """Load NCC format activations for a specific layer with subsampling to save memory.
    
    Args:
        checkpoint_dir: Directory with NCC checkpoints
        layer: Layer number to load
        language: Language to load ('english' or 'arabic')
        max_chunks: Maximum number of chunks to load
        sample_ratio: Fraction of tokens to sample from each chunk (0.1 = 10%)
    """
    
    # Try to find chunk files first
    chunk_pattern = f'layer_{layer}_{language}_chunk_*.pt'
    chunk_files = sorted(checkpoint_dir.glob(chunk_pattern))
    
    # If no chunk files, check for merged single file
    merged_file = checkpoint_dir / f'layer_{layer}_{language}.pt'
    
    if chunk_files:
        # Load from chunk files
        if max_chunks:
            chunk_files = chunk_files[:max_chunks]
        
        logger.info(f"Loading {len(chunk_files)} chunks for layer {layer} {language} (sampling {sample_ratio*100:.0f}% of tokens)")
        
        all_activations = []
        all_genders = []
        
        for chunk_file in tqdm(chunk_files, desc=f"Loading layer {layer}"):
            data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            
            # Shape: [batch, seq_len, hidden_dim]
            acts = data['activations']
            
            # Flatten to [batch * seq_len, hidden_dim] for SAE training
            batch_size, seq_len, hidden_dim = acts.shape
            acts_flat = acts.reshape(-1, hidden_dim)
            
            # Subsample to reduce memory
            n_samples = acts_flat.shape[0]
            n_keep = max(1, int(n_samples * sample_ratio))
            indices = torch.randperm(n_samples)[:n_keep]
            acts_sampled = acts_flat[indices]
            
            all_activations.append(acts_sampled)
            all_genders.extend(data.get('genders', []))
            
            # Free memory
            del data, acts, acts_flat
        
        # Concatenate all chunks
        activations = torch.cat(all_activations, dim=0)
        del all_activations
        all_genders_final = all_genders
        
    elif merged_file.exists():
        # Load from merged single file
        logger.info(f"Loading merged file for layer {layer} {language} (sampling {sample_ratio*100:.0f}% of tokens)")
        
        data = torch.load(merged_file, map_location='cpu', weights_only=False)
        acts = data['activations']
        
        # Handle both 2D [num_samples, hidden_dim] and 3D [num_samples, seq_len, hidden_dim] formats
        if len(acts.shape) == 2:
            # Already flattened: [num_samples, hidden_dim]
            acts_flat = acts
            hidden_dim = acts.shape[-1]
            logger.info(f"Activations already flattened: {acts.shape}")
        else:
            # 3D format: [num_samples, seq_len, hidden_dim]
            num_samples, seq_len, hidden_dim = acts.shape
            acts_flat = acts.reshape(-1, hidden_dim)
            logger.info(f"Flattened activations from {acts.shape} to {acts_flat.shape}")
        
        # Subsample to reduce memory
        n_samples = acts_flat.shape[0]
        n_keep = max(1, int(n_samples * sample_ratio))
        indices = torch.randperm(n_samples)[:n_keep]
        activations = acts_flat[indices]
        
        all_genders_final = data.get('genders', [])
        
        del data, acts, acts_flat
        
    else:
        raise FileNotFoundError(f"No chunks or merged file found for layer {layer} {language} in {checkpoint_dir}")
    import gc
    gc.collect()
    
    logger.info(f"Loaded activations shape: {activations.shape}")
    logger.info(f"Memory usage: {activations.numel() * 4 / 1e9:.2f} GB")
    
    return {
        'activations': activations,
        'genders': all_genders_final,
        'd_model': activations.shape[1]
    }


def train_sae(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    d_model: int,
    config: dict,
    device: str = 'cuda',
    use_wandb: bool = False,
    layer: int = 0
) -> tuple:
    """Train SAE on activation data."""
    
    sae_config_dict = config.get('sae', {})
    
    # Create SAE config - cast all values to correct types for YAML string parsing
    expansion_factor = int(sae_config_dict.get('expansion_factor', 8))
    
    sae_config = SAEConfig(
        d_model=d_model,
        expansion_factor=expansion_factor,
        l1_coefficient=float(sae_config_dict.get('l1_coefficient', 5e-4)),
        normalize_decoder=bool(sae_config_dict.get('normalize_decoder', True)),
        tied_weights=bool(sae_config_dict.get('tied_weights', False)),
    )
    
    # Create SAE
    sae = SparseAutoencoder(sae_config).to(device)
    d_hidden = sae.d_hidden
    
    logger.info(f"Created SAE: d_model={d_model}, d_hidden={d_hidden}")
    logger.info(f"Total parameters: {sum(p.numel() for p in sae.parameters()):,}")
    
    # Training setup - cast to correct types in case YAML parsed as strings
    batch_size = int(sae_config_dict.get('batch_size', 256))
    epochs = int(sae_config_dict.get('epochs', 50))
    lr = float(sae_config_dict.get('learning_rate', 1e-4))
    
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'l0_sparsity': []}
    best_val_loss = float('inf')
    patience = int(sae_config_dict.get('patience', 10))
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        sae.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, features, aux_info = sae(x)
            # Compute loss using SAE's compute_loss method
            loss = sae.compute_loss(x, recon, features)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        sae.eval()
        val_losses = []
        l0_sparsities = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon, features, aux_info = sae(x)
                val_loss = sae.compute_loss(x, recon, features)
                val_losses.append(val_loss.item())
                
                # L0 sparsity (fraction of non-zero features)
                l0 = (features.abs() > 1e-6).float().mean().item()
                l0_sparsities.append(l0)
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_l0 = sum(l0_sparsities) / len(l0_sparsities)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['l0_sparsity'].append(avg_l0)
        
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, L0={avg_l0:.4f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                f'layer_{layer}/train_loss': avg_train_loss,
                f'layer_{layer}/val_loss': avg_val_loss,
                f'layer_{layer}/l0_sparsity': avg_l0,
                f'layer_{layer}/learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch + 1,
            })
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_state = sae.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # Load best model
    sae.load_state_dict(best_state)
    
    return sae, history


def main():
    parser = argparse.ArgumentParser(description="Train SAE on NCC activations")
    parser.add_argument('--layer', type=int, required=True, help='Layer to train SAE for')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'arabic'],
                        help='Language to train on (english or arabic)')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/full_layers_ncc/layer_checkpoints',
                        help='Directory with NCC checkpoints')
    parser.add_argument('--output-dir', type=str, default='checkpoints/saes', help='Output directory for SAEs')
    parser.add_argument('--max-chunks', type=int, default=None, help='Max chunks to load (for debugging)')
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='Fraction of tokens to sample (0.1 = 10%%, saves memory)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb (always disabled in this script)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training SAE for layer {args.layer}, language: {args.language}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    # Load activations (with subsampling to save memory)
    data = load_ncc_activations(checkpoint_dir, args.layer, args.language, args.max_chunks, args.sample_ratio)
    activations = data['activations']
    d_model = data['d_model']
    
    # Split into train/val
    val_split = config.get('data', {}).get('val_split', 0.1)
    n_samples = len(activations)
    n_val = int(n_samples * val_split)
    
    # Shuffle indices
    indices = torch.randperm(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_data = activations[train_indices]
    val_data = activations[val_indices]
    
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Free memory
    del activations, data
    import gc
    gc.collect()
    
    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config.get('logging', {}).get('wandb_project', 'sae-captioning-bias'),
            entity=config.get('logging', {}).get('wandb_entity', None),
            name=f"sae-{args.language}-layer-{args.layer}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                'layer': args.layer,
                'language': args.language,
                'd_model': d_model,
                'sae_config': config.get('sae', {}),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
            },
            tags=['sae-training', f'layer-{args.layer}', args.language, 'ncc']
        )
        logger.info(f"Wandb initialized: {wandb.run.url}")
    
    # Train SAE
    sae, history = train_sae(train_data, val_data, d_model, config, args.device, use_wandb, args.layer)
    
    # Save SAE
    sae_path = output_dir / f'sae_{args.language}_layer_{args.layer}.pt'
    torch.save({
        'model_state_dict': sae.state_dict(),
        'd_model': d_model,
        'd_hidden': sae.d_hidden,
        'layer': args.layer,
        'language': args.language,
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }, sae_path)
    
    logger.info(f"Saved SAE to {sae_path}")
    
    # Save training history
    history_path = output_dir / f'sae_{args.language}_layer_{args.layer}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Log final metrics to wandb
    if use_wandb:
        wandb.log({
            f'{args.language}/layer_{args.layer}/final_train_loss': history['train_loss'][-1],
            f'{args.language}/layer_{args.layer}/final_val_loss': history['val_loss'][-1],
            f'{args.language}/layer_{args.layer}/final_l0_sparsity': history['l0_sparsity'][-1],
            f'{args.language}/layer_{args.layer}/best_val_loss': min(history['val_loss']),
        })
        wandb.finish()
    
    logger.info("Training complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
