#!/usr/bin/env python3
"""
Cross-Layer Feature Analysis
==============================

Compare how gender-biased features behave across different network layers.
Tests if features become more/less gender-specific in deeper layers.

Key Questions:
1. Do the same features appear across multiple layers?
2. Do ablation effects strengthen or weaken in deeper layers?
3. Which layers have the strongest gender-specific features?

Usage:
    python scripts/16_cross_layer_analysis.py --config configs/config.yaml
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import argparse
from tqdm import tqdm
import pandas as pd
from typing import Dict, List
import yaml
import wandb
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossLayerAnalyzer:
    """Analyze gender-biased features across multiple network layers."""

    def __init__(self, config: Dict, layers: List[int], device: str = 'cuda'):
        self.config = config
        self.layers = layers
        self.device = device

        # Load paths
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.output_dir = Path(config['paths']['visualizations']) / 'cross_layer_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        results_file = Path(config['paths']['results']) / 'comprehensive_analysis_results.json'
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        # Load SAEs for all layers
        self.saes = {}
        for layer in layers:
            self.saes[layer] = self._load_sae(layer)

    def _load_sae(self, layer: int) -> SparseAutoencoder:
        """Load trained SAE for a specific layer."""
        sae_path = self.checkpoint_dir / f'sae_layer_{layer}.pt'
        checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
        sae_config = checkpoint['config']

        sae = SparseAutoencoder(sae_config)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(self.device)
        sae.eval()

        logger.info(f"Loaded SAE for layer {layer}")
        return sae

    def analyze_layer_ablation(
        self,
        layer: int,
        activations: torch.Tensor,
        features_to_ablate: List[int],
        genders: List[str]
    ) -> Dict:
        """
        Analyze ablation effects for a specific layer.

        Args:
            layer: Layer index
            activations: Activations for this layer
            features_to_ablate: Features to ablate
            genders: Gender labels

        Returns:
            Dictionary with ablation results
        """
        logger.info(f"  Analyzing layer {layer}...")

        sae = self.saes[layer]

        # Flatten activations
        activations_flat = activations.reshape(-1, activations.shape[-1])
        batch_size = 512
        all_diff_norms = []

        with torch.no_grad():
            for i in range(0, len(activations_flat), batch_size):
                batch = activations_flat[i:i+batch_size].to(self.device)

                # Original reconstruction
                recon_orig, features_orig, _ = sae(batch)

                # Ablated reconstruction
                features_ablated = features_orig.clone()
                features_ablated[:, features_to_ablate] = 0
                recon_ablated = sae.decode(features_ablated)

                # Compute differences
                recon_diff = (recon_orig - recon_ablated).cpu()
                diff_norm = torch.norm(recon_diff, dim=1)
                all_diff_norms.append(diff_norm)

                # Clean up
                del batch, recon_orig, features_orig, features_ablated, recon_ablated, recon_diff
                torch.cuda.empty_cache()

        recon_diff_norm = torch.cat(all_diff_norms)

        # Aggregate by sample
        num_samples = len(genders)
        seq_length = len(recon_diff_norm) // num_samples

        sample_diff_norms = []
        for i in range(num_samples):
            sample_slice = slice(i * seq_length, (i + 1) * seq_length)
            sample_diff_norms.append(recon_diff_norm[sample_slice].mean().item())

        # Separate by gender
        male_changes = [sample_diff_norms[i] for i in range(num_samples) if genders[i] == 'male']
        female_changes = [sample_diff_norms[i] for i in range(num_samples) if genders[i] == 'female']

        return {
            'layer': layer,
            'male_mean': np.mean(male_changes),
            'male_std': np.std(male_changes),
            'female_mean': np.mean(female_changes),
            'female_std': np.std(female_changes),
            'differential': np.mean(male_changes) - np.mean(female_changes),
            'all_changes': sample_diff_norms,
            'male_changes': male_changes,
            'female_changes': female_changes
        }

    def run_cross_layer_analysis(self, num_features: int = 3):
        """Run cross-layer ablation analysis."""

        logger.info(f"\n{'='*60}")
        logger.info("Cross-Layer Feature Analysis")
        logger.info(f"{'='*60}\n")

        # Load activations checkpoint
        logger.info("Loading English activations...")
        checkpoint = torch.load(
            self.checkpoint_dir / 'activations_english_sample_small.pt',
            map_location='cpu', weights_only=False
        )

        genders = checkpoint['genders']
        image_ids = checkpoint['image_ids']

        results = {
            'layers': self.layers,
            'male_biased_ablation': {},
            'female_biased_ablation': {}
        }

        # Analyze male-biased features across layers
        logger.info("\n" + "="*60)
        logger.info("Ablating MALE-BIASED features across layers")
        logger.info("="*60)

        for layer in self.layers:
            if str(layer) not in self.results['layers']:
                logger.warning(f"Layer {layer} not found in results, skipping")
                continue

            layer_results = self.results['layers'][str(layer)]
            male_biased = layer_results['gender_bias']['english']['top_male_biased'][:num_features]

            logger.info(f"\nLayer {layer} - Male-biased features: {male_biased}")

            activations = checkpoint['activations'][layer]

            ablation_results = self.analyze_layer_ablation(
                layer, activations, male_biased, genders
            )

            results['male_biased_ablation'][layer] = ablation_results

        # Analyze female-biased features across layers
        logger.info("\n" + "="*60)
        logger.info("Ablating FEMALE-BIASED features across layers")
        logger.info("="*60)

        for layer in self.layers:
            if str(layer) not in self.results['layers']:
                continue

            layer_results = self.results['layers'][str(layer)]
            female_biased = layer_results['gender_bias']['english']['top_female_biased'][:num_features]

            logger.info(f"\nLayer {layer} - Female-biased features: {female_biased}")

            activations = checkpoint['activations'][layer]

            ablation_results = self.analyze_layer_ablation(
                layer, activations, female_biased, genders
            )

            results['female_biased_ablation'][layer] = ablation_results

        # Visualize
        self.visualize_cross_layer_effects(results)

        # Save results
        output_file = self.output_dir / 'cross_layer_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

        return results

    def visualize_cross_layer_effects(self, results: Dict):
        """Create comprehensive cross-layer visualization."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Cross-Layer Gender Feature Analysis',
                     fontsize=18, fontweight='bold')

        layers = results['layers']
        male_results = results['male_biased_ablation']
        female_results = results['female_biased_ablation']

        # 1. Male feature effects by layer
        ax = axes[0, 0]
        male_means = [male_results[l]['male_mean'] for l in layers]
        female_means_for_male = [male_results[l]['female_mean'] for l in layers]

        x = np.arange(len(layers))
        width = 0.35

        ax.bar(x - width/2, male_means, width, label='Male Images', color='steelblue')
        ax.bar(x + width/2, female_means_for_male, width, label='Female Images', color='lightpink')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Reconstruction Change')
        ax.set_title('Ablating Male-Biased Features\nEffect on Different Genders')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Female feature effects by layer
        ax = axes[0, 1]
        male_means_for_female = [female_results[l]['male_mean'] for l in layers]
        female_means = [female_results[l]['female_mean'] for l in layers]

        ax.bar(x - width/2, male_means_for_female, width, label='Male Images', color='lightblue')
        ax.bar(x + width/2, female_means, width, label='Female Images', color='orchid')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Reconstruction Change')
        ax.set_title('Ablating Female-Biased Features\nEffect on Different Genders')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Differential effects across layers
        ax = axes[0, 2]
        male_diffs = [male_results[l]['differential'] for l in layers]
        female_diffs = [female_results[l]['differential'] for l in layers]

        ax.plot(layers, male_diffs, 'o-', label='Male Features\n(Male - Female)',
                color='steelblue', linewidth=2, markersize=8)
        ax.plot(layers, [-d for d in female_diffs], 's-', label='Female Features\n(Female - Male)',
                color='orchid', linewidth=2, markersize=8)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Differential Effect')
        ax.set_title('Gender-Specific Differential Effects\nAcross Layers')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        # 4. Distribution boxplots - Male features
        ax = axes[1, 0]
        male_data = []
        positions = []
        for i, layer in enumerate(layers):
            male_data.append(male_results[layer]['male_changes'])
            male_data.append(male_results[layer]['female_changes'])
            positions.extend([i*2, i*2+0.8])

        bp = ax.boxplot(male_data, positions=positions, widths=0.6, patch_artist=True)

        # Color boxes
        for i, box in enumerate(bp['boxes']):
            if i % 2 == 0:
                box.set_facecolor('steelblue')
            else:
                box.set_facecolor('lightpink')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Reconstruction Change')
        ax.set_title('Distribution: Ablating Male Features')
        ax.set_xticks([i*2 + 0.4 for i in range(len(layers))])
        ax.set_xticklabels(layers)
        ax.grid(True, alpha=0.3, axis='y')

        # 5. Distribution boxplots - Female features
        ax = axes[1, 1]
        female_data = []
        positions = []
        for i, layer in enumerate(layers):
            female_data.append(female_results[layer]['male_changes'])
            female_data.append(female_results[layer]['female_changes'])
            positions.extend([i*2, i*2+0.8])

        bp = ax.boxplot(female_data, positions=positions, widths=0.6, patch_artist=True)

        for i, box in enumerate(bp['boxes']):
            if i % 2 == 0:
                box.set_facecolor('lightblue')
            else:
                box.set_facecolor('orchid')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Reconstruction Change')
        ax.set_title('Distribution: Ablating Female Features')
        ax.set_xticks([i*2 + 0.4 for i in range(len(layers))])
        ax.set_xticklabels(layers)
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Summary statistics table
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = "Key Findings:\n\n"

        # Find layer with strongest male differential
        strongest_male_layer = max(layers, key=lambda l: abs(male_results[l]['differential']))
        strongest_male_diff = male_results[strongest_male_layer]['differential']

        # Find layer with strongest female differential
        strongest_female_layer = max(layers, key=lambda l: abs(female_results[l]['differential']))
        strongest_female_diff = female_results[strongest_female_layer]['differential']

        summary_text += f"Male Features:\n"
        summary_text += f"  Strongest at layer {strongest_male_layer}\n"
        summary_text += f"  Differential: {strongest_male_diff:.3f}\n\n"

        summary_text += f"Female Features:\n"
        summary_text += f"  Strongest at layer {strongest_female_layer}\n"
        summary_text += f"  Differential: {abs(strongest_female_diff):.3f}\n\n"

        # Check if effects increase with depth
        male_trend = male_diffs[-1] > male_diffs[0]
        female_trend = abs(female_diffs[-1]) > abs(female_diffs[0])

        if male_trend:
            summary_text += "✓ Male features strengthen in deeper layers\n"
        else:
            summary_text += "✗ Male features weaken in deeper layers\n"

        if female_trend:
            summary_text += "✓ Female features strengthen in deeper layers\n"
        else:
            summary_text += "✗ Female features weaken in deeper layers\n"

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / 'cross_layer_effects.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization: {output_path}")

        # Create detailed comparison table
        self._create_comparison_table(results)

    def _create_comparison_table(self, results: Dict):
        """Create detailed comparison table across layers."""

        layers = results['layers']

        # Prepare data
        table_data = []
        for layer in layers:
            male_res = results['male_biased_ablation'][layer]
            female_res = results['female_biased_ablation'][layer]

            table_data.append({
                'Layer': layer,
                'Male Features - Male Images': f"{male_res['male_mean']:.3f} ± {male_res['male_std']:.3f}",
                'Male Features - Female Images': f"{male_res['female_mean']:.3f} ± {male_res['female_std']:.3f}",
                'Male Differential': f"{male_res['differential']:.3f}",
                'Female Features - Male Images': f"{female_res['male_mean']:.3f} ± {female_res['male_std']:.3f}",
                'Female Features - Female Images': f"{female_res['female_mean']:.3f} ± {female_res['female_std']:.3f}",
                'Female Differential': f"{abs(female_res['differential']):.3f}"
            })

        df = pd.DataFrame(table_data)

        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Cross-Layer Ablation Analysis - Detailed Statistics',
                 fontsize=14, fontweight='bold', pad=20)

        output_path = self.output_dir / 'cross_layer_table.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved comparison table: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Layer Feature Analysis')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[10, 14, 18, 22],
                       help='Layers to analyze (default: 10 14 18 22)')
    parser.add_argument('--num-features', type=int, default=3)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    analyzer = CrossLayerAnalyzer(config, args.layers)
    analyzer.run_cross_layer_analysis(num_features=args.num_features)

    logger.info("\n" + "="*60)
    logger.info("Cross-layer analysis complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
