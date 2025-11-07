#!/usr/bin/env python3
"""
Script 06: Generate Visualizations
==================================

Creates all visualizations from analysis results.

Usage:
    python scripts/06_generate_visualizations.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    VisualizationConfig,
    SAEVisualization,
    CrossLingualVisualization,
    SteeringVisualization,
    InteractiveVisualization,
    create_all_visualizations
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


def load_results(results_dir: Path) -> dict:
    """Load all analysis results."""
    results = {}
    
    # Feature analysis
    feature_path = results_dir / 'feature_analysis.json'
    if feature_path.exists():
        with open(feature_path, 'r') as f:
            results['feature_analysis'] = json.load(f)
        logger.info("Loaded feature analysis results")
    
    # SAE training summary
    training_path = results_dir / 'sae_training_summary.json'
    if training_path.exists():
        with open(training_path, 'r') as f:
            results['training_summary'] = json.load(f)
        logger.info("Loaded SAE training summary")
    
    # Steering experiments
    steering_path = results_dir / 'steering_experiments.json'
    if steering_path.exists():
        with open(steering_path, 'r') as f:
            results['steering_experiments'] = json.load(f)
        logger.info("Loaded steering experiments")
    
    return results


def create_sae_visualizations(
    training_summary: dict,
    viz: SAEVisualization
) -> None:
    """Create SAE training visualizations."""
    
    for layer_str, layer_data in training_summary.items():
        try:
            layer = int(layer_str)
        except ValueError:
            continue
        
        if 'history' in layer_data:
            viz.plot_training_curves(
                layer_data['history'],
                save_name=f"training_curves_layer_{layer}"
            )
            logger.info(f"Created training curves for layer {layer}")


def create_feature_visualizations(
    feature_analysis: dict,
    viz: CrossLingualVisualization
) -> None:
    """Create feature analysis visualizations."""
    
    # Layer divergence plot
    if 'layer_divergence' in feature_analysis:
        layer_divergence = {
            int(k): v for k, v in feature_analysis['layer_divergence'].items()
        }
        viz.plot_layer_divergence(layer_divergence)
        logger.info("Created layer divergence plot")
    
    # Gender feature heatmaps for each layer
    for layer_str, layer_data in feature_analysis.get('layers', {}).items():
        try:
            layer = int(layer_str)
        except ValueError:
            continue
        
        en_stats = layer_data.get('english_stats', [])
        ar_stats = layer_data.get('arabic_stats', [])
        
        if en_stats and ar_stats:
            viz.plot_gender_feature_heatmap(
                en_stats, ar_stats,
                top_k=50,
                save_name=f"gender_heatmap_layer_{layer}"
            )
            logger.info(f"Created gender heatmap for layer {layer}")
        
        # Embedding visualization
        if 'embeddings' in layer_data and 'embedding_languages' in layer_data:
            import numpy as np
            embeddings = np.array(layer_data['embeddings'])
            lang_labels = layer_data['embedding_languages']
            
            # Need gender labels - use first half (English genders)
            n_per_lang = len(embeddings) // 2
            genders = ['unknown'] * n_per_lang  # Placeholder if not available
            
            viz.plot_embedding_space(
                embeddings, lang_labels, genders,
                save_name=f"embedding_space_layer_{layer}"
            )
            logger.info(f"Created embedding visualization for layer {layer}")


def create_steering_visualizations(
    steering_results: dict,
    viz: SteeringVisualization
) -> None:
    """Create steering experiment visualizations."""
    
    for language, lang_results in steering_results.get('languages', {}).items():
        viz.plot_steering_effect(
            lang_results,
            save_name=f"steering_effect_{language}"
        )
        logger.info(f"Created steering plot for {language}")


def create_interactive_dashboard(
    all_results: dict,
    viz: InteractiveVisualization
) -> None:
    """Create interactive dashboard."""
    
    # Prepare combined results for dashboard
    dashboard_data = {}
    
    if 'feature_analysis' in all_results:
        fa = all_results['feature_analysis']
        dashboard_data['layer_divergence'] = {
            int(k): v for k, v in fa.get('layer_divergence', {}).items()
        }
        
        # Get feature stats from first available layer
        for layer_str, layer_data in fa.get('layers', {}).items():
            if 'english_stats' in layer_data:
                dashboard_data['feature_stats'] = layer_data['english_stats']
                dashboard_data['cross_lingual_comparison'] = layer_data.get('cross_lingual_comparison', {})
                break
    
    if 'steering_experiments' in all_results:
        dashboard_data['steering_results'] = all_results['steering_experiments']
    
    if dashboard_data:
        viz.create_interactive_dashboard(dashboard_data)
        logger.info("Created interactive dashboard")
        
        if 'feature_stats' in dashboard_data:
            viz.create_feature_explorer(dashboard_data['feature_stats'])
            logger.info("Created feature explorer")


def create_summary_report(
    all_results: dict,
    output_dir: Path
) -> None:
    """Create a text summary report."""
    
    report_lines = [
        "=" * 60,
        "SAE CROSS-LINGUAL ANALYSIS SUMMARY REPORT",
        "=" * 60,
        "",
    ]
    
    # Feature analysis summary
    if 'feature_analysis' in all_results:
        fa = all_results['feature_analysis']
        
        report_lines.extend([
            "LAYER DIVERGENCE ANALYSIS",
            "-" * 40,
        ])
        
        layer_div = fa.get('layer_divergence', {})
        for layer, overlap in sorted(layer_div.items(), key=lambda x: int(x[0])):
            bar = "█" * int(float(overlap) * 20)
            report_lines.append(f"  Layer {layer:>2}: {float(overlap)*100:5.1f}% {bar}")
        
        if layer_div:
            min_layer = min(layer_div.keys(), key=lambda k: layer_div[k])
            max_layer = max(layer_div.keys(), key=lambda k: layer_div[k])
            report_lines.extend([
                "",
                f"  Most divergent layer: {min_layer}",
                f"  Most convergent layer: {max_layer}",
                "",
            ])
    
    # Steering summary
    if 'steering_experiments' in all_results:
        se = all_results['steering_experiments']
        
        report_lines.extend([
            "STEERING EXPERIMENTS",
            "-" * 40,
        ])
        
        for lang, results in se.get('languages', {}).items():
            best = results.get('best_steering', {})
            baseline = results.get('baseline', {})
            
            report_lines.extend([
                f"  {lang.upper()}:",
                f"    Baseline accuracy: {baseline.get('accuracy', 0)*100:.1f}%",
                f"    Best steering: strength={best.get('strength', 'N/A')}",
                f"    Best accuracy: {best.get('steered_accuracy', 0)*100:.1f}%",
                f"    Improvement: {best.get('improvement', 0)*100:+.1f}%",
                "",
            ])
    
    report_lines.extend([
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    # Save report
    report_path = output_dir / 'analysis_summary.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved summary report to {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--interactive-only', action='store_true',
                       help='Only create interactive plots')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup paths
    results_dir = Path(config['paths']['results'])
    vis_dir = Path(config['paths']['visualizations'])
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info("Loading analysis results...")
    all_results = load_results(results_dir)
    
    if not all_results:
        logger.error("No analysis results found. Run analysis scripts first.")
        return 1
    
    # Create visualization config
    vis_config = config.get('visualization', {})
    viz_config = VisualizationConfig(
        dpi=vis_config.get('dpi', 150),
        fig_width=vis_config.get('figure_width', 12),
        fig_height=vis_config.get('figure_height', 8),
        english_color=vis_config.get('english_color', '#2ecc71'),
        arabic_color=vis_config.get('arabic_color', '#e74c3c'),
        male_color=vis_config.get('male_color', '#3498db'),
        female_color=vis_config.get('female_color', '#e91e63'),
        save_formats=vis_config.get('save_formats', ['png', 'pdf']),
        output_dir=vis_dir
    )
    
    logger.info(f"Saving visualizations to {vis_dir}")
    
    if not args.interactive_only:
        # Create SAE visualizations
        if 'training_summary' in all_results:
            logger.info("\nCreating SAE training visualizations...")
            sae_viz = SAEVisualization(viz_config)
            create_sae_visualizations(all_results['training_summary'], sae_viz)
        
        # Create feature visualizations
        if 'feature_analysis' in all_results:
            logger.info("\nCreating feature analysis visualizations...")
            cross_viz = CrossLingualVisualization(viz_config)
            create_feature_visualizations(all_results['feature_analysis'], cross_viz)
        
        # Create steering visualizations
        if 'steering_experiments' in all_results:
            logger.info("\nCreating steering experiment visualizations...")
            steer_viz = SteeringVisualization(viz_config)
            create_steering_visualizations(all_results['steering_experiments'], steer_viz)
    
    # Create interactive dashboard
    logger.info("\nCreating interactive visualizations...")
    interactive_viz = InteractiveVisualization(viz_config)
    create_interactive_dashboard(all_results, interactive_viz)
    
    # Create summary report
    logger.info("\nCreating summary report...")
    create_summary_report(all_results, vis_dir)
    
    # List created files
    logger.info("\nCreated files:")
    for f in sorted(vis_dir.glob('*')):
        logger.info(f"  {f.name}")
    
    logger.info("\nVisualization generation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
