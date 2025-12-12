#!/usr/bin/env python3
"""
Script 08: SAE Feature Visualization and Analysis
==================================================

Analyzes and visualizes learned SAE features:
- Feature activation patterns
- Top activated features per language
- Feature importance and selectivity
- Cross-lingual feature comparison
- t-SNE/UMAP embeddings of features

Usage:
    python scripts/08_visualize_sae_features.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint_batch(checkpoint_dir: Path, language: str, layer: int,
                          max_samples: int = 500) -> dict:
    """Load a subset of activation checkpoints for analysis."""
    chunk_files = sorted(checkpoint_dir.glob(f'activations_{language}_chunk_*.pt'))
    
    if not chunk_files:
        raise FileNotFoundError(f"No checkpoints found for {language}")
    
    # Load first few chunks for analysis
    activations = []
    captions = []
    samples_loaded = 0
    
    logger.info(f"Loading {language} activations (max {max_samples} samples)...")
    
    for chunk_file in chunk_files[:10]:  # Load first 10 chunks
        if samples_loaded >= max_samples:
            break
            
        chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
        
        if layer in chunk_data['activations']:
            acts = chunk_data['activations'][layer]  # [batch, seq, hidden]
            batch_size = acts.shape[0]
            samples_to_take = min(batch_size, max_samples - samples_loaded)
            
            activations.append(acts[:samples_to_take])
            
            if 'captions' in chunk_data:
                captions.extend(chunk_data['captions'][:samples_to_take])
            
            samples_loaded += samples_to_take
    
    if not activations:
        raise ValueError(f"No activations found for layer {layer}")
    
    activations = torch.cat(activations, dim=0)
    logger.info(f"Loaded {language}: {activations.shape}")
    
    return {
        'activations': activations,
        'captions': captions if captions else None
    }


def analyze_feature_activation_patterns(sae, activations, language, layer):
    """Analyze which features activate and their patterns."""
    logger.info(f"Analyzing feature activation patterns for {language} layer {layer}...")
    
    # Flatten activations: [batch, seq, hidden] -> [batch*seq, hidden]
    if len(activations.shape) == 3:
        batch, seq, hidden = activations.shape
        activations_flat = activations.reshape(-1, hidden)
    else:
        activations_flat = activations
    
    # Get SAE features
    device = next(sae.parameters()).device
    sae.eval()
    
    all_features = []
    batch_size = 1000
    
    with torch.no_grad():
        for i in tqdm(range(0, len(activations_flat), batch_size), desc="Extracting features"):
            batch = activations_flat[i:i+batch_size].to(device)
            _, features, _ = sae(batch)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)  # [N, d_hidden]
    
    # Compute statistics
    feature_stats = {
        'mean_activation': all_features.mean(dim=0).numpy(),
        'max_activation': all_features.max(dim=0)[0].numpy(),
        'std_activation': all_features.std(dim=0).numpy(),
        'activation_frequency': (all_features > 0).float().mean(dim=0).numpy(),
        'l0_per_sample': (all_features > 0).sum(dim=1).float().mean().item(),
    }
    
    # Find dead features
    dead_features = (feature_stats['activation_frequency'] == 0).sum()
    logger.info(f"Dead features: {dead_features} / {len(feature_stats['activation_frequency'])}")
    
    return feature_stats, all_features


def find_top_features(feature_stats, n_top=50):
    """Find most important features by various metrics."""
    metrics = {}
    
    # Most frequently activated
    freq_idx = np.argsort(feature_stats['activation_frequency'])[-n_top:][::-1]
    metrics['most_frequent'] = freq_idx.tolist()
    
    # Highest mean activation
    mean_idx = np.argsort(feature_stats['mean_activation'])[-n_top:][::-1]
    metrics['highest_mean'] = mean_idx.tolist()
    
    # Highest max activation
    max_idx = np.argsort(feature_stats['max_activation'])[-n_top:][::-1]
    metrics['highest_max'] = max_idx.tolist()
    
    # Most selective (high max, low frequency)
    selectivity = feature_stats['max_activation'] / (feature_stats['activation_frequency'] + 1e-8)
    selective_idx = np.argsort(selectivity)[-n_top:][::-1]
    metrics['most_selective'] = selective_idx.tolist()
    
    return metrics


def visualize_feature_statistics(en_stats, ar_stats, layer, output_dir):
    """Create visualizations of feature statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'SAE Feature Statistics - Layer {layer}', fontsize=16)
    
    # 1. Activation frequency distribution
    axes[0, 0].hist(en_stats['activation_frequency'], bins=50, alpha=0.6, label='English', color='blue')
    axes[0, 0].hist(ar_stats['activation_frequency'], bins=50, alpha=0.6, label='Arabic', color='red')
    axes[0, 0].set_xlabel('Activation Frequency')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Feature Activation Frequency')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. Mean activation distribution
    axes[0, 1].hist(en_stats['mean_activation'], bins=50, alpha=0.6, label='English', color='blue')
    axes[0, 1].hist(ar_stats['mean_activation'], bins=50, alpha=0.6, label='Arabic', color='red')
    axes[0, 1].set_xlabel('Mean Activation')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Mean Feature Activation')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # 3. Max activation distribution
    axes[0, 2].hist(en_stats['max_activation'], bins=50, alpha=0.6, label='English', color='blue')
    axes[0, 2].hist(ar_stats['max_activation'], bins=50, alpha=0.6, label='Arabic', color='red')
    axes[0, 2].set_xlabel('Max Activation')
    axes[0, 2].set_ylabel('Number of Features')
    axes[0, 2].set_title('Maximum Feature Activation')
    axes[0, 2].legend()
    axes[0, 2].set_yscale('log')
    
    # 4. Frequency vs Mean scatter
    axes[1, 0].scatter(en_stats['activation_frequency'], en_stats['mean_activation'], 
                      alpha=0.3, s=1, label='English', color='blue')
    axes[1, 0].scatter(ar_stats['activation_frequency'], ar_stats['mean_activation'], 
                      alpha=0.3, s=1, label='Arabic', color='red')
    axes[1, 0].set_xlabel('Activation Frequency')
    axes[1, 0].set_ylabel('Mean Activation')
    axes[1, 0].set_title('Frequency vs Mean Activation')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # 5. Dead features
    en_dead = (en_stats['activation_frequency'] == 0).sum()
    ar_dead = (ar_stats['activation_frequency'] == 0).sum()
    axes[1, 1].bar(['English', 'Arabic'], [en_dead, ar_dead], color=['blue', 'red'], alpha=0.6)
    axes[1, 1].set_ylabel('Number of Dead Features')
    axes[1, 1].set_title('Dead Features by Language')
    
    # 6. Selectivity comparison
    en_selectivity = en_stats['max_activation'] / (en_stats['activation_frequency'] + 1e-8)
    ar_selectivity = ar_stats['max_activation'] / (ar_stats['activation_frequency'] + 1e-8)
    
    axes[1, 2].hist(np.log10(en_selectivity + 1), bins=50, alpha=0.6, label='English', color='blue')
    axes[1, 2].hist(np.log10(ar_selectivity + 1), bins=50, alpha=0.6, label='Arabic', color='red')
    axes[1, 2].set_xlabel('Log10(Selectivity + 1)')
    axes[1, 2].set_ylabel('Number of Features')
    axes[1, 2].set_title('Feature Selectivity')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    save_path = output_dir / f'layer_{layer}_feature_statistics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved feature statistics plot: {save_path}")
    plt.close()
    
    return save_path


def visualize_top_features(en_top, ar_top, en_stats, ar_stats, layer, output_dir):
    """Visualize top features for each language."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top 50 Features - Layer {layer}', fontsize=16)
    
    # English most frequent
    axes[0, 0].barh(range(50), en_stats['activation_frequency'][en_top['most_frequent'][:50]], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Activation Frequency')
    axes[0, 0].set_ylabel('Feature Index (sorted)')
    axes[0, 0].set_title('English: Most Frequently Activated Features')
    axes[0, 0].invert_yaxis()
    
    # Arabic most frequent
    axes[0, 1].barh(range(50), ar_stats['activation_frequency'][ar_top['most_frequent'][:50]], color='red', alpha=0.6)
    axes[0, 1].set_xlabel('Activation Frequency')
    axes[0, 1].set_ylabel('Feature Index (sorted)')
    axes[0, 1].set_title('Arabic: Most Frequently Activated Features')
    axes[0, 1].invert_yaxis()
    
    # English most selective
    en_selectivity = en_stats['max_activation'] / (en_stats['activation_frequency'] + 1e-8)
    axes[1, 0].barh(range(50), en_selectivity[en_top['most_selective'][:50]], color='blue', alpha=0.6)
    axes[1, 0].set_xlabel('Selectivity (Max / Frequency)')
    axes[1, 0].set_ylabel('Feature Index (sorted)')
    axes[1, 0].set_title('English: Most Selective Features')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xscale('log')
    
    # Arabic most selective
    ar_selectivity = ar_stats['max_activation'] / (ar_stats['activation_frequency'] + 1e-8)
    axes[1, 1].barh(range(50), ar_selectivity[ar_top['most_selective'][:50]], color='red', alpha=0.6)
    axes[1, 1].set_xlabel('Selectivity (Max / Frequency)')
    axes[1, 1].set_ylabel('Feature Index (sorted)')
    axes[1, 1].set_title('Arabic: Most Selective Features')
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    
    save_path = output_dir / f'layer_{layer}_top_features.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved top features plot: {save_path}")
    plt.close()
    
    return save_path


def compute_feature_embeddings(en_features, ar_features, method='tsne', n_components=2):
    """Compute low-dimensional embeddings of features."""
    logger.info(f"Computing {method.upper()} embeddings...")
    
    # Sample features for visualization (too many for t-SNE)
    n_samples = min(5000, len(en_features), len(ar_features))
    
    en_sample_idx = np.random.choice(len(en_features), n_samples, replace=False)
    ar_sample_idx = np.random.choice(len(ar_features), n_samples, replace=False)
    
    en_sample = en_features[en_sample_idx].numpy()
    ar_sample = ar_features[ar_sample_idx].numpy()
    
    # Combine for joint embedding
    combined = np.vstack([en_sample, ar_sample])
    labels = np.array(['English'] * len(en_sample) + ['Arabic'] * len(ar_sample))
    
    if method == 'tsne':
        embedder = TSNE(n_components=n_components, random_state=42, perplexity=30)
    elif method == 'pca':
        embedder = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings = embedder.fit_transform(combined)
    
    return embeddings, labels


def visualize_feature_embeddings(embeddings, labels, layer, method, output_dir):
    """Visualize feature embeddings."""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    en_mask = labels == 'English'
    ar_mask = labels == 'Arabic'
    
    ax.scatter(embeddings[en_mask, 0], embeddings[en_mask, 1], 
              c='blue', alpha=0.3, s=10, label='English')
    ax.scatter(embeddings[ar_mask, 0], embeddings[ar_mask, 1], 
              c='red', alpha=0.3, s=10, label='Arabic')
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Feature Space {method.upper()} - Layer {layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / f'layer_{layer}_features_{method}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved {method} plot: {save_path}")
    plt.close()
    
    return save_path


def analyze_cross_lingual_features(en_stats, ar_stats, en_top, ar_top):
    """Analyze overlap and differences between languages."""
    logger.info("Analyzing cross-lingual feature patterns...")
    
    # Find shared top features
    en_freq_set = set(en_top['most_frequent'][:100])
    ar_freq_set = set(ar_top['most_frequent'][:100])
    shared_frequent = en_freq_set & ar_freq_set
    
    en_sel_set = set(en_top['most_selective'][:100])
    ar_sel_set = set(ar_top['most_selective'][:100])
    shared_selective = en_sel_set & ar_sel_set
    
    # Compute correlation between activation patterns
    freq_corr = np.corrcoef(en_stats['activation_frequency'], 
                            ar_stats['activation_frequency'])[0, 1]
    mean_corr = np.corrcoef(en_stats['mean_activation'], 
                            ar_stats['mean_activation'])[0, 1]
    
    analysis = {
        'shared_frequent_features': len(shared_frequent),
        'shared_selective_features': len(shared_selective),
        'frequency_correlation': float(freq_corr),
        'mean_activation_correlation': float(mean_corr),
        'english_unique_frequent': len(en_freq_set - ar_freq_set),
        'arabic_unique_frequent': len(ar_freq_set - en_freq_set),
        'english_unique_selective': len(en_sel_set - ar_sel_set),
        'arabic_unique_selective': len(ar_sel_set - en_sel_set),
    }
    
    logger.info(f"Shared frequent features: {analysis['shared_frequent_features']}/100")
    logger.info(f"Frequency correlation: {analysis['frequency_correlation']:.4f}")
    logger.info(f"Mean activation correlation: {analysis['mean_activation_correlation']:.4f}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Visualize SAE features')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--layers', type=int, nargs='+', help='Specific layers to analyze')
    parser.add_argument('--max-samples', type=int, default=500, help='Max samples per language')
    parser.add_argument('--use-wandb', action='store_true', help='Log to W&B')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    checkpoint_dir = Path(config['paths']['checkpoints'])
    results_dir = Path(config['paths']['results'])
    vis_dir = Path(config['paths']['visualizations'])
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=config['wandb']['project'],
            name=f"feature-visualization-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config
        )
    
    # Determine layers
    layers = args.layers or config['layers'].get('primary_analysis', [10, 14, 18, 22])
    logger.info(f"Analyzing layers: {layers}")
    
    all_results = {}
    
    for layer in layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Layer {layer}")
        logger.info(f"{'='*60}")
        
        try:
            # Load SAE
            sae_path = checkpoint_dir / f'sae_layer_{layer}.pt'
            if not sae_path.exists():
                logger.warning(f"SAE not found: {sae_path}")
                continue
            
            checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
            sae = checkpoint['model']
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sae = sae.to(device)
            logger.info(f"Loaded SAE from {sae_path}")
            
            # Load activations
            en_data = load_checkpoint_batch(checkpoint_dir, 'english', layer, args.max_samples)
            ar_data = load_checkpoint_batch(checkpoint_dir, 'arabic', layer, args.max_samples)
            
            # Analyze features
            en_stats, en_features = analyze_feature_activation_patterns(
                sae, en_data['activations'], 'English', layer
            )
            ar_stats, ar_features = analyze_feature_activation_patterns(
                sae, ar_data['activations'], 'Arabic', layer
            )
            
            # Find top features
            en_top = find_top_features(en_stats, n_top=100)
            ar_top = find_top_features(ar_stats, n_top=100)
            
            # Cross-lingual analysis
            cross_lingual = analyze_cross_lingual_features(en_stats, ar_stats, en_top, ar_top)
            
            # Visualizations
            stat_plot = visualize_feature_statistics(en_stats, ar_stats, layer, vis_dir)
            top_plot = visualize_top_features(en_top, ar_top, en_stats, ar_stats, layer, vis_dir)
            
            # Embeddings
            tsne_emb, labels = compute_feature_embeddings(en_features, ar_features, method='tsne')
            tsne_plot = visualize_feature_embeddings(tsne_emb, labels, layer, 'tsne', vis_dir)
            
            pca_emb, _ = compute_feature_embeddings(en_features, ar_features, method='pca')
            pca_plot = visualize_feature_embeddings(pca_emb, labels, layer, 'pca', vis_dir)
            
            # Save results
            layer_results = {
                'layer': layer,
                'english_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in en_stats.items()},
                'arabic_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in ar_stats.items()},
                'english_top_features': en_top,
                'arabic_top_features': ar_top,
                'cross_lingual_analysis': cross_lingual,
                'visualizations': {
                    'statistics': str(stat_plot),
                    'top_features': str(top_plot),
                    'tsne': str(tsne_plot),
                    'pca': str(pca_plot)
                }
            }
            
            all_results[layer] = layer_results
            
            # Log to W&B
            if args.use_wandb:
                wandb.log({
                    f'layer_{layer}/en_dead_features': (en_stats['activation_frequency'] == 0).sum(),
                    f'layer_{layer}/ar_dead_features': (ar_stats['activation_frequency'] == 0).sum(),
                    f'layer_{layer}/en_l0_per_sample': en_stats['l0_per_sample'],
                    f'layer_{layer}/ar_l0_per_sample': ar_stats['l0_per_sample'],
                    f'layer_{layer}/frequency_correlation': cross_lingual['frequency_correlation'],
                    f'layer_{layer}/shared_frequent': cross_lingual['shared_frequent_features'],
                })
                
                wandb.log({
                    f'layer_{layer}/statistics': wandb.Image(str(stat_plot)),
                    f'layer_{layer}/top_features': wandb.Image(str(top_plot)),
                    f'layer_{layer}/tsne': wandb.Image(str(tsne_plot)),
                    f'layer_{layer}/pca': wandb.Image(str(pca_plot)),
                })
            
            logger.info(f"Completed analysis for layer {layer}")
            
        except Exception as e:
            logger.error(f"Error processing layer {layer}: {e}", exc_info=True)
            continue
    
    # Save summary
    summary_path = results_dir / 'feature_analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved analysis summary to {summary_path}")
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info("\nFeature analysis and visualization complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
