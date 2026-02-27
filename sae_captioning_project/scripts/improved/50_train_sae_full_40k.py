#!/usr/bin/env python3
"""
50_train_sae_full_40k.py — Train SAEs on full 40K dataset
=========================================================

Previous training: 10,000 samples → 71.7-99.9% EV
This script: ~40,000 samples → expected improved EV and feature quality

Trains SAEs per-language per-layer from the full extracted activations.

Usage:
  python scripts/improved/50_train_sae_full_40k.py \
      --layers 9 17 --language english --epochs 50 --batch_size 256
"""

import argparse
import logging
import sys
import gc
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.sae import SAEConfig, SparseAutoencoder, SAETrainer, create_sae
from src.analysis.metrics import compute_reconstruction_metrics, compute_sparsity_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')


def load_full_activations(activation_dir, language, layer):
    """Load all activation chunks for a language/layer and concatenate."""
    activation_dir = Path(activation_dir)
    
    # Find all chunks
    chunk_files = sorted(activation_dir.glob(f'activations_{language}_chunk_*.pt'))
    
    if not chunk_files:
        raise FileNotFoundError(f"No activation chunks found in {activation_dir} for {language}")
    
    logger.info(f"Found {len(chunk_files)} chunks for {language}")
    
    all_acts = []
    for chunk_path in tqdm(chunk_files, desc=f"Loading {language} L{layer}"):
        chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
        if layer in chunk['activations']:
            all_acts.append(chunk['activations'][layer])
        elif str(layer) in chunk['activations']:
            all_acts.append(chunk['activations'][str(layer)])
    
    if not all_acts:
        raise ValueError(f"No activations found for layer {layer}")
    
    combined = torch.cat(all_acts, dim=0)
    logger.info(f"Combined {language} L{layer}: {combined.shape}")
    return combined


def train_sae_full(train_data, val_data, config, layer, device='cuda'):
    """Train SAE with full dataset."""
    d_model = train_data.shape[-1]
    
    # Create SAE
    sae = create_sae(
        d_model=d_model,
        expansion_factor=config['expansion_factor'],
        l1_coefficient=config['l1_coefficient'],
        sae_type="standard",
        dtype=torch.float32
    )
    
    logger.info(f"SAE: d_model={d_model}, d_hidden={sae.d_hidden}, "
                f"L1={config['l1_coefficient']}, expansion={config['expansion_factor']}")
    
    # Trainer
    trainer = SAETrainer(
        sae=sae,
        learning_rate=config['learning_rate'],
        warmup_steps=config.get('warmup_steps', 1000),
        device=device
    )
    
    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(train_data.float()),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(val_data.float()),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Training loop
    history = {'loss': [], 'recon_loss': [], 'l1_loss': [], 'l0_sparsity': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(config['epochs']):
        epoch_metrics = []
        for batch, in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False):
            metrics = trainer.train_step(batch)
            
            if torch.isnan(torch.tensor(metrics['loss'])):
                logger.error(f"NaN at epoch {epoch+1}")
                break
            epoch_metrics.append(metrics)
        
        if not epoch_metrics:
            break
        
        avg_loss = np.mean([m['loss'] for m in epoch_metrics])
        avg_recon = np.mean([m['recon_loss'] for m in epoch_metrics])
        avg_l1 = np.mean([m['l1_loss'] for m in epoch_metrics])
        avg_l0 = np.mean([m['l0_sparsity'] for m in epoch_metrics])
        
        history['loss'].append(float(avg_loss))
        history['recon_loss'].append(float(avg_recon))
        history['l1_loss'].append(float(avg_l1))
        history['l0_sparsity'].append(float(avg_l0))
        
        val_metrics = trainer.evaluate(val_loader)
        history['val_loss'].append(float(val_metrics['val_loss']))
        
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - "
                     f"Loss: {avg_loss:.6f} | Recon: {avg_recon:.6f} | "
                     f"L1: {avg_l1:.6f} | L0: {avg_l0:.1f} | "
                     f"Val: {val_metrics['val_loss']:.6f}")
        
        # Early stopping
        if val_metrics['val_loss'] < best_val_loss - config.get('min_delta', 1e-5):
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        sae.load_state_dict(best_state)
    
    return sae, history


def evaluate_sae_full(sae, val_data, device='cuda'):
    """Evaluate trained SAE on validation data."""
    sae = sae.to(device)
    sae.eval()
    
    val_tensor = val_data.float().to(device)
    
    # Process in chunks to avoid OOM
    chunk_size = 10000
    all_recon = []
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(val_tensor), chunk_size):
            chunk = val_tensor[i:i+chunk_size]
            recon, features, _ = sae(chunk)
            all_recon.append(recon.cpu())
            all_features.append(features.cpu())
    
    recon_combined = torch.cat(all_recon, dim=0)
    features_combined = torch.cat(all_features, dim=0)
    
    recon_metrics = compute_reconstruction_metrics(val_data, recon_combined)
    sparsity_metrics = compute_sparsity_metrics(features_combined)
    
    return {**recon_metrics, **sparsity_metrics}


def main():
    parser = argparse.ArgumentParser(description='Train SAEs on full 40K dataset')
    parser.add_argument('--layers', type=int, nargs='+', default=[9, 17])
    parser.add_argument('--language', type=str, default='english', choices=['english', 'arabic'])
    parser.add_argument('--activation_dir', type=str, 
                       default=str(BASE_DIR / 'checkpoints/full_40k_activations'))
    parser.add_argument('--output_dir', type=str,
                       default=str(BASE_DIR / 'checkpoints/saes_full_40k'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--expansion_factor', type=int, default=8)
    parser.add_argument('--l1_coefficient', type=float, default=1e-4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'expansion_factor': args.expansion_factor,
        'l1_coefficient': args.l1_coefficient,
        'learning_rate': args.learning_rate,
        'warmup_steps': 1000,
        'patience': 10,
        'min_delta': 1e-5,
    }
    
    logger.info("=" * 60)
    logger.info("SAE Training on Full 40K Dataset")
    logger.info(f"Language: {args.language}, Layers: {args.layers}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    logger.info("=" * 60)
    
    all_results = {}
    
    for layer in args.layers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training SAE for Layer {layer} ({args.language})")
        logger.info(f"{'='*50}")
        
        # Load activations
        activations = load_full_activations(args.activation_dir, args.language, layer)
        
        # Split train/val
        n_val = int(len(activations) * args.val_split)
        perm = torch.randperm(len(activations), generator=torch.Generator().manual_seed(42))
        train_indices = perm[n_val:]
        val_indices = perm[:n_val]
        
        train_data = activations[train_indices]
        val_data = activations[val_indices]
        
        logger.info(f"Train: {train_data.shape}, Val: {val_data.shape}")
        
        del activations
        gc.collect()
        
        # Train
        sae, history = train_sae_full(train_data, val_data, config, layer, args.device)
        
        # Evaluate
        eval_metrics = evaluate_sae_full(sae, val_data, args.device)
        logger.info("Final evaluation:")
        for k, v in eval_metrics.items():
            logger.info(f"  {k}: {v:.6f}")
        
        # Save SAE
        sae_path = output_dir / f'sae_{args.language}_layer_{layer}.pt'
        torch.save({
            'model_state_dict': sae.state_dict(),
            'config': sae.config,
            'd_model': sae.d_model,
            'd_hidden': sae.d_hidden,
            'layer': layer,
            'language': args.language,
            'n_training_samples': len(train_data),
            'eval_metrics': eval_metrics,
            'timestamp': datetime.now().isoformat(),
        }, sae_path)
        logger.info(f"Saved SAE to {sae_path}")
        
        # Save history
        history_path = output_dir / f'sae_{args.language}_layer_{layer}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        all_results[f'layer_{layer}'] = {
            'eval_metrics': {k: float(v) for k, v in eval_metrics.items()},
            'n_train': len(train_data),
            'n_val': len(val_data),
            'best_epoch': len(history['loss']),
            'sae_path': str(sae_path),
        }
        
        del train_data, val_data, sae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save summary
    summary_path = output_dir / f'training_summary_{args.language}.json'
    all_results['config'] = config
    all_results['timestamp'] = datetime.now().isoformat()
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nTraining complete! Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
