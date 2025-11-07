#!/usr/bin/env python3
"""
Script 04: Feature Analysis
===========================

Analyzes SAE features for cross-lingual gender patterns.

Usage:
    python scripts/04_analyze_features.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder, SAEConfig
from src.analysis.features import (
    GenderFeatureAnalyzer,
    FeatureClustering,
    compute_embedding_space_analysis,
    serialize_analysis_results
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sae(checkpoint_path: Path, device: str = "cuda") -> SparseAutoencoder:
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    sae = SparseAutoencoder(config)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    return sae


def load_activations(checkpoint_dir: Path, language: str) -> dict:
    """Load extracted activations."""
    path = checkpoint_dir / f'activations_{language}.pt'
    if not path.exists():
        raise FileNotFoundError(f"Activations not found: {path}")
    return torch.load(path)


def analyze_layer(
    sae: SparseAutoencoder,
    english_activations: torch.Tensor,
    arabic_activations: torch.Tensor,
    english_genders: list,
    arabic_genders: list,
    config: dict,
    device: str = "cuda"
) -> dict:
    """Perform complete analysis for a layer."""
    
    analysis_config = config.get('analysis', {})
    
    # Initialize analyzer
    analyzer = GenderFeatureAnalyzer(
        sae=sae,
        device=device,
        significance_level=analysis_config.get('significance_level', 0.05),
        effect_size_threshold=analysis_config.get('effect_size_threshold', 0.3)
    )
    
    # Compute feature activations
    logger.info("Computing feature activations...")
    en_features = analyzer.compute_feature_activations(english_activations)
    ar_features = analyzer.compute_feature_activations(arabic_activations)
    
    logger.info(f"English features shape: {en_features.shape}")
    logger.info(f"Arabic features shape: {ar_features.shape}")
    
    # Compute statistics for each language
    logger.info("Computing English feature statistics...")
    en_stats = analyzer.compute_feature_stats(en_features, english_genders)
    
    logger.info("Computing Arabic feature statistics...")
    ar_stats = analyzer.compute_feature_stats(ar_features, arabic_genders)
    
    # Identify gender-associated features
    logger.info("Identifying gender-associated features...")
    en_gender_features = analyzer.identify_gender_features(
        en_stats, top_k=analysis_config.get('top_k_features', 100)
    )
    ar_gender_features = analyzer.identify_gender_features(
        ar_stats, top_k=analysis_config.get('top_k_features', 100)
    )
    
    # Cross-lingual comparison
    logger.info("Computing cross-lingual comparison...")
    comparison = analyzer.compare_languages(
        en_features, ar_features,
        english_genders, arabic_genders
    )
    
    # Feature clustering
    logger.info("Clustering features...")
    clustering = FeatureClustering(
        n_clusters=analysis_config.get('n_clusters', 10)
    )
    feature_directions = sae.get_feature_directions()
    cluster_labels = clustering.fit(feature_directions)
    cluster_summary = clustering.get_cluster_summary(cluster_labels, en_stats)
    
    # 2D embedding for visualization
    logger.info("Computing embedding space...")
    embeddings, lang_labels = compute_embedding_space_analysis(
        en_features, ar_features, english_genders
    )
    
    return {
        'english_stats': [vars(s) for s in en_stats],
        'arabic_stats': [vars(s) for s in ar_stats],
        'english_gender_features': {
            'male_associated': en_gender_features['male_associated'],
            'female_associated': en_gender_features['female_associated'],
        },
        'arabic_gender_features': {
            'male_associated': ar_gender_features['male_associated'],
            'female_associated': ar_gender_features['female_associated'],
        },
        'cross_lingual_comparison': {
            'shared_gender_features': comparison.shared_gender_features,
            'english_specific_features': comparison.english_specific_features,
            'arabic_specific_features': comparison.arabic_specific_features,
            'overlap_ratio': comparison.overlap_ratio,
            'correlation': comparison.correlation,
            'feature_correlations': comparison.feature_correlations,
        },
        'cluster_summary': cluster_summary,
        'embeddings': embeddings.tolist(),
        'embedding_languages': lang_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE features")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Override layers to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup paths
    checkpoint_dir = Path(config['paths']['checkpoints'])
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load activations
    logger.info("Loading activations...")
    english_data = load_activations(checkpoint_dir, 'english')
    arabic_data = load_activations(checkpoint_dir, 'arabic')
    
    # Determine layers
    if args.layers:
        layers = args.layers
    else:
        # Find available SAE checkpoints
        sae_files = list(checkpoint_dir.glob('sae_layer_*.pt'))
        layers = sorted([int(f.stem.split('_')[-1]) for f in sae_files])
    
    if not layers:
        logger.error("No SAE checkpoints found. Run 03_train_sae.py first.")
        return 1
    
    logger.info(f"Analyzing layers: {layers}")
    
    # Analyze each layer
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'layers': {},
        'layer_divergence': {},
    }
    
    for layer in layers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing Layer {layer}")
        logger.info(f"{'='*50}")
        
        # Load SAE
        sae_path = checkpoint_dir / f'sae_layer_{layer}.pt'
        if not sae_path.exists():
            logger.warning(f"SAE not found for layer {layer}, skipping")
            continue
        
        sae = load_sae(sae_path, args.device)
        
        # Get activations for this layer
        en_acts = english_data['activations'][layer]
        ar_acts = arabic_data['activations'][layer]
        
        # Analyze
        layer_results = analyze_layer(
            sae=sae,
            english_activations=en_acts,
            arabic_activations=ar_acts,
            english_genders=english_data['genders'],
            arabic_genders=arabic_data['genders'],
            config=config,
            device=args.device
        )
        
        # Store results
        all_results['layers'][layer] = layer_results
        all_results['layer_divergence'][layer] = layer_results['cross_lingual_comparison']['overlap_ratio']
        
        # Log summary
        comparison = layer_results['cross_lingual_comparison']
        logger.info(f"\nLayer {layer} Summary:")
        logger.info(f"  Shared gender features: {len(comparison['shared_gender_features'])}")
        logger.info(f"  English-specific features: {len(comparison['english_specific_features'])}")
        logger.info(f"  Arabic-specific features: {len(comparison['arabic_specific_features'])}")
        logger.info(f"  Overlap ratio: {comparison['overlap_ratio']:.2%}")
        logger.info(f"  Cross-lingual correlation: {comparison['correlation']:.3f}")
    
    # Save results
    output_path = results_dir / 'feature_analysis.json'
    
    # Convert to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"\nSaved analysis results to {output_path}")
    
    # Print overall summary
    logger.info("\n" + "="*50)
    logger.info("OVERALL SUMMARY")
    logger.info("="*50)
    
    logger.info("\nLayer Divergence (Cross-lingual Gender Feature Overlap):")
    for layer, overlap in sorted(all_results['layer_divergence'].items()):
        bar = "█" * int(overlap * 20)
        logger.info(f"  Layer {layer:2d}: {overlap:5.1%} {bar}")
    
    # Find layer with minimum overlap (most divergent)
    if all_results['layer_divergence']:
        min_layer = min(all_results['layer_divergence'], key=all_results['layer_divergence'].get)
        max_layer = max(all_results['layer_divergence'], key=all_results['layer_divergence'].get)
        
        logger.info(f"\nMost divergent layer: {min_layer} ({all_results['layer_divergence'][min_layer]:.1%} overlap)")
        logger.info(f"Most convergent layer: {max_layer} ({all_results['layer_divergence'][max_layer]:.1%} overlap)")
    
    logger.info("\nFeature analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
