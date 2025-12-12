#!/usr/bin/env python3
"""
Script 03: SAE Training
=======================

Trains Sparse Autoencoders on extracted activations.

Usage:
    python scripts/03_train_sae.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
from datetime import datetime
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sae import SAEConfig, SparseAutoencoder, SAETrainer, create_sae
from src.analysis.metrics import compute_reconstruction_metrics, compute_sparsity_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_activations(checkpoint_dir: Path, language: str, layer: int = None) -> dict:
    """Load extracted activations from checkpoints or merged file."""
    # First try to load merged file
    merged_path = checkpoint_dir / f'activations_{language}.pt'
    if merged_path.exists():
        data = torch.load(merged_path, weights_only=False)
        logger.info(f"Loaded {language} activations from merged file")
        
        # If layer specified, keep only that layer's activations to save memory
        if layer is not None and 'activations' in data:
            layer_act = data['activations'].get(layer)
            data['activations'] = {layer: layer_act} if layer_act is not None else {}
        
        return data
    
    # If merged file doesn't exist, load from checkpoint chunks
    logger.info(f"Merged file not found, loading {language} checkpoints...")
    chunk_files = sorted(checkpoint_dir.glob(f'activations_{language}_chunk_*.pt'))
    
    if not chunk_files:
        raise FileNotFoundError(f"No activations found for {language} in {checkpoint_dir}")
    
    logger.info(f"Found {len(chunk_files)} checkpoint files for {language}")
    
    # Load and merge all chunks
    all_activations = {}
    for chunk_file in tqdm(chunk_files, desc=f"Loading {language} chunks"):
        chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
        
        if 'activations' in chunk_data:
            for layer_idx, acts in chunk_data['activations'].items():
                # Only load the specified layer if provided
                if layer is not None and layer_idx != layer:
                    continue
                    
                if layer_idx not in all_activations:
                    all_activations[layer_idx] = []
                all_activations[layer_idx].append(acts)
    
    # Concatenate all chunks for each layer
    merged_activations = {}
    for layer_idx, acts_list in all_activations.items():
        merged_activations[layer_idx] = torch.cat(acts_list, dim=0)
        logger.info(f"Layer {layer_idx}: {merged_activations[layer_idx].shape}")
    
    return {'activations': merged_activations}


def prepare_training_data(
    english_data: dict,
    arabic_data: dict,
    layer: int,
    val_split: float = 0.1,
    max_tokens_per_sample: int = 50
) -> tuple:
    """Prepare training and validation data for SAE with token sampling."""

    # Get activations for the layer (keep in float16)
    en_acts = english_data['activations'][layer]
    ar_acts = arabic_data['activations'][layer]

    logger.info(f"English activations shape: {en_acts.shape}, dtype: {en_acts.dtype}")
    logger.info(f"Arabic activations shape: {ar_acts.shape}, dtype: {ar_acts.dtype}")

    # Sample tokens from sequence to reduce memory usage
    # Instead of using all 278 tokens, sample max_tokens_per_sample tokens per image
    if len(en_acts.shape) == 3:
        batch_size, seq_len, hidden = en_acts.shape
        # Randomly sample token positions (same for all samples)
        if seq_len > max_tokens_per_sample:
            token_indices = torch.randperm(seq_len)[:max_tokens_per_sample]
            en_acts = en_acts[:, token_indices, :]
            ar_acts = ar_acts[:, token_indices, :]
            logger.info(f"Sampled {max_tokens_per_sample} tokens from {seq_len} to reduce memory")

        # Flatten: (batch, seq, hidden) -> (batch*seq, hidden)
        en_flat = en_acts.reshape(-1, en_acts.shape[-1])
        ar_flat = ar_acts.reshape(-1, ar_acts.shape[-1])
    else:
        en_flat = en_acts
        ar_flat = ar_acts

    logger.info(f"After sampling - English: {en_flat.shape}, Arabic: {ar_flat.shape}")

    # Calculate split sizes
    en_samples = en_flat.shape[0]
    ar_samples = ar_flat.shape[0]

    # Take validation from end of each dataset
    en_val_size = int(en_samples * val_split)
    ar_val_size = int(ar_samples * val_split)

    # Split without creating full combined tensor
    en_train = en_flat[:-en_val_size]
    en_val = en_flat[-en_val_size:]
    ar_train = ar_flat[:-ar_val_size]
    ar_val = ar_flat[-ar_val_size:]

    logger.info(f"Training samples: {en_train.shape[0] + ar_train.shape[0]}")
    logger.info(f"Validation samples: {en_val.shape[0] + ar_val.shape[0]}")

    # Return as separate tensors - will be combined in dataloader
    return (en_train, ar_train), (en_val, ar_val)


def train_sae_for_layer(
    train_data: tuple,
    val_data: tuple,
    config: dict,
    layer: int,
    device: str = "cuda",
    use_wandb: bool = True
) -> tuple:
    """Train SAE for a specific layer."""

    sae_config = config['sae']

    # Handle tuple input (en_data, ar_data)
    if isinstance(train_data, tuple):
        d_model = train_data[0].shape[-1]
        # Combine for dataloaders
        train_combined = torch.cat([train_data[0], train_data[1]], dim=0)
        val_combined = torch.cat([val_data[0], val_data[1]], dim=0)
    else:
        d_model = train_data.shape[-1]
        train_combined = train_data
        val_combined = val_data

    # Log input statistics
    logger.info(f"Training data stats - mean: {train_combined.float().mean():.6f}, std: {train_combined.float().std():.6f}")
    logger.info(f"Training data range - min: {train_combined.float().min():.6f}, max: {train_combined.float().max():.6f}")

    # Create SAE (keep float32 for training stability)
    sae = create_sae(
        d_model=d_model,
        expansion_factor=int(sae_config.get('expansion_factor', 8)),
        l1_coefficient=float(sae_config.get('l1_coefficient', 5e-4)),
        sae_type="standard",
        dtype=torch.float32
    )

    logger.info(f"Created SAE: d_model={d_model}, d_hidden={sae.d_hidden}")

    # Log SAE config to wandb
    if use_wandb:
        wandb.config.update({
            f'layer_{layer}_d_model': d_model,
            f'layer_{layer}_d_hidden': sae.d_hidden,
            f'layer_{layer}_expansion_factor': sae_config.get('expansion_factor', 8),
            f'layer_{layer}_l1_coefficient': sae_config.get('l1_coefficient', 5e-4),
        })

    # Create trainer
    trainer = SAETrainer(
        sae=sae,
        learning_rate=float(sae_config.get('learning_rate', 1e-4)),
        warmup_steps=int(sae_config.get('warmup_steps', 1000)),
        device=device
    )

    # Create dataloaders - convert to float32 here
    batch_size = sae_config.get('batch_size', 2048)
    train_loader = DataLoader(
        TensorDataset(train_combined.float()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(val_combined.float()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Training loop
    epochs = sae_config.get('epochs', 50)
    patience = sae_config.get('patience', 10)
    min_delta = float(sae_config.get('min_delta', 1e-5))
    
    history = {
        'loss': [],
        'recon_loss': [],
        'l1_loss': [],
        'l0_sparsity': [],
        'val_loss': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        epoch_metrics = []
        for batch, in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            metrics = trainer.train_step(batch)

            # Check for NaN
            if torch.isnan(torch.tensor(metrics['loss'])):
                logger.error(f"NaN detected in loss at epoch {epoch+1}")
                logger.error(f"Metrics: {metrics}")
                break

            epoch_metrics.append(metrics)
        
        # Average training metrics
        avg_loss = sum(m['loss'] for m in epoch_metrics) / len(epoch_metrics)
        avg_recon = sum(m['recon_loss'] for m in epoch_metrics) / len(epoch_metrics)
        avg_l1 = sum(m['l1_loss'] for m in epoch_metrics) / len(epoch_metrics)
        avg_l0 = sum(m['l0_sparsity'] for m in epoch_metrics) / len(epoch_metrics)
        
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['l1_loss'].append(avg_l1)
        history['l0_sparsity'].append(avg_l0)
        
        # Validation
        val_metrics = trainer.evaluate(val_loader)
        history['val_loss'].append(val_metrics['val_loss'])
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {avg_loss:.6f} | "
            f"Recon: {avg_recon:.6f} | "
            f"L1: {avg_l1:.6f} | "
            f"L0: {avg_l0:.1f} | "
            f"Val: {val_metrics['val_loss']:.6f}"
        )

        # Log to wandb
        if use_wandb:
            wandb.log({
                f'layer_{layer}/train_loss': avg_loss,
                f'layer_{layer}/recon_loss': avg_recon,
                f'layer_{layer}/l1_loss': avg_l1,
                f'layer_{layer}/l0_sparsity': avg_l0,
                f'layer_{layer}/val_loss': val_metrics['val_loss'],
                f'layer_{layer}/epoch': epoch + 1,
            })

        # Early stopping
        if val_metrics['val_loss'] < best_val_loss - min_delta:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            best_state = sae.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best state
    if best_state is not None:
        sae.load_state_dict(best_state)
    
    return sae, history


def evaluate_sae(sae, val_data, device: str = "cuda") -> dict:
    """Evaluate trained SAE."""
    sae.eval()
    
    # Handle tuple input (en_data, ar_data)
    if isinstance(val_data, tuple):
        en_val, ar_val = val_data
        en_val = en_val.to(device)
        ar_val = ar_val.to(device)
        val_tensors = [en_val, ar_val]
    else:
        val_tensors = [val_data.to(device)]
    
    all_reconstruction = []
    all_features = []
    all_input = []
    
    with torch.no_grad():
        for val_tensor in val_tensors:
            reconstruction, features, _ = sae(val_tensor)
            all_reconstruction.append(reconstruction.cpu())
            all_features.append(features.cpu())
            all_input.append(val_tensor.cpu())
    
    val_data_combined = torch.cat(all_input, dim=0)
    reconstruction_combined = torch.cat(all_reconstruction, dim=0)
    features_combined = torch.cat(all_features, dim=0)
    
    # Reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(val_data_combined, reconstruction_combined)
    
    # Sparsity metrics
    sparsity_metrics = compute_sparsity_metrics(features_combined)
    
    return {**recon_metrics, **sparsity_metrics}


def main():
    parser = argparse.ArgumentParser(description="Train SAEs on activations")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Override layers to train')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config.get('wandb', {}).get('project', 'sae-captioning-bias'),
            name=f"sae-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                'model': config['model']['name'],
                'sae_config': config['sae'],
                'data_config': config['data'],
            },
            tags=['sae-training', 'cross-lingual']
        )
        logger.info(f"Wandb initialized: {wandb.run.url}")

    # Setup paths
    checkpoint_dir = Path(config['paths']['checkpoints'])
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine layers first (need to peek at activations file)
    try:
        # Try merged file first
        merged_path = checkpoint_dir / 'activations_english.pt'
        if merged_path.exists():
            temp_data = torch.load(merged_path, weights_only=False)
            available_layers = list(temp_data['activations'].keys())
            del temp_data
            import gc
            gc.collect()
        else:
            # Load from first checkpoint to get available layers
            chunk_files = sorted(checkpoint_dir.glob('activations_english_chunk_*.pt'))
            if not chunk_files:
                raise FileNotFoundError("No English activation files found")
            
            temp_data = torch.load(chunk_files[0], map_location='cpu', weights_only=False)
            available_layers = list(temp_data['activations'].keys())
            del temp_data
            import gc
            gc.collect()
    except (FileNotFoundError, IndexError) as e:
        logger.error(f"Activations not found: {e}. Run 02_extract_activations.py first.")
        return 1

    layers = args.layers or config['layers'].get('primary_analysis', available_layers)
    logger.info(f"Training SAEs for layers: {layers}")

    # Train SAE for each layer
    all_results = {}

    for layer in layers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training SAE for Layer {layer}")
        logger.info(f"{'='*50}")

        # Load activations for this layer only to save memory
        try:
            english_data = load_activations(checkpoint_dir, 'english', layer=layer)
            arabic_data = load_activations(checkpoint_dir, 'arabic', layer=layer)
        except FileNotFoundError as e:
            logger.error(f"{e}")
            logger.error("Run 02_extract_activations.py first.")
            return 1

        # Prepare data
        train_data, val_data = prepare_training_data(
            english_data, arabic_data, layer,
            val_split=config['data'].get('val_split', 0.1)
        )

        # Free memory
        del english_data, arabic_data
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train
        sae, history = train_sae_for_layer(
            train_data, val_data, config, layer, args.device, use_wandb
        )

        # Evaluate
        eval_metrics = evaluate_sae(sae, val_data, args.device)
        logger.info(f"Final evaluation metrics:")
        for k, v in eval_metrics.items():
            logger.info(f"  {k}: {v:.6f}")

        # Log final metrics to wandb
        if use_wandb:
            wandb.log({
                f'layer_{layer}/final_{k}': v
                for k, v in eval_metrics.items()
            })
        
        # Save SAE
        sae_path = checkpoint_dir / f'sae_layer_{layer}.pt'
        torch.save({
            'model_state_dict': sae.state_dict(),
            'config': sae.config,
            'd_model': sae.d_model,
            'd_hidden': sae.d_hidden,
            'layer': layer,
            'timestamp': datetime.now().isoformat(),
        }, sae_path)
        logger.info(f"Saved SAE to {sae_path}")
        
        # Store results
        all_results[layer] = {
            'history': history,
            'eval_metrics': eval_metrics,
            'sae_path': str(sae_path),
        }
    
    # Save training summary
    summary_path = results_dir / 'sae_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved training summary to {summary_path}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    logger.info("\nSAE training complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
