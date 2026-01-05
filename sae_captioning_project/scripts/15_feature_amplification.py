#!/usr/bin/env python3
"""
Feature Amplification Experiments
==================================

Test if amplifying gender-biased features increases gender signals in reconstructions.
Complements ablation analysis by testing the opposite direction.

Usage:
    python scripts/15_feature_amplification.py --config configs/config.yaml --layer 10
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
from PIL import Image
from typing import Dict, List
import yaml

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureAmplificationAnalyzer:
    """Analyze effects of amplifying gender-biased features."""

    def __init__(self, config: Dict, layer: int, device: str = 'cuda'):
        self.config = config
        self.layer = layer
        self.device = device

        # Load paths
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.output_dir = Path(config['paths']['visualizations']) / f'feature_amplification_layer_{layer}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        results_file = Path(config['paths']['results']) / 'comprehensive_analysis_results.json'
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        # Load SAE
        self.sae = self._load_sae()

    def _load_sae(self) -> SparseAutoencoder:
        """Load trained SAE."""
        sae_path = self.checkpoint_dir / f'sae_layer_{self.layer}.pt'
        checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
        sae_config = checkpoint['config']

        sae = SparseAutoencoder(sae_config)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(self.device)
        sae.eval()

        return sae

    def analyze_amplification_effects(
        self,
        activations: torch.Tensor,
        features_to_amplify: List[int],
        genders: List[str],
        image_ids: List[str],
        amplification_factors: List[float] = [1.5, 2.0, 3.0]
    ) -> Dict:
        """
        Analyze effects of amplifying features by different factors.

        Args:
            activations: Original activations
            features_to_amplify: Features to amplify
            genders: Gender labels
            image_ids: Image IDs
            amplification_factors: Multiplication factors to test

        Returns:
            Dictionary with amplification results
        """
        logger.info(f"Testing amplification factors: {amplification_factors}")

        # Flatten activations
        activations_flat = activations.reshape(-1, activations.shape[-1])
        batch_size = 512

        results = {
            'amplification_factors': amplification_factors,
            'per_factor_results': {}
        }

        for factor in amplification_factors:
            logger.info(f"  Testing {factor}x amplification...")

            all_diff_norms = []

            with torch.no_grad():
                for i in range(0, len(activations_flat), batch_size):
                    batch = activations_flat[i:i+batch_size].to(self.device)

                    # Original reconstruction
                    recon_orig, features_orig, _ = self.sae(batch)

                    # Amplified reconstruction
                    features_amplified = features_orig.clone()
                    features_amplified[:, features_to_amplify] *= factor
                    recon_amplified = self.sae.decode(features_amplified)

                    # Compute differences
                    recon_diff = (recon_amplified - recon_orig).cpu()
                    diff_norm = torch.norm(recon_diff, dim=1)
                    all_diff_norms.append(diff_norm)

                    # Clean up
                    del batch, recon_orig, features_orig, features_amplified, recon_amplified, recon_diff
                    torch.cuda.empty_cache()

            recon_diff_norm = torch.cat(all_diff_norms)

            # Aggregate by sample
            num_samples = len(genders)
            seq_length = len(recon_diff_norm) // num_samples

            sample_diff_norms = []
            for i in range(num_samples):
                sample_slice = slice(i * seq_length, (i + 1) * seq_length)
                sample_diff_norms.append(recon_diff_norm[sample_slice].mean().item())

            # Store per-sample results
            per_sample = []
            for i in range(num_samples):
                per_sample.append({
                    'image_id': image_ids[i],
                    'gender': genders[i],
                    'reconstruction_change': sample_diff_norms[i]
                })

            results['per_factor_results'][str(factor)] = {
                'per_sample': per_sample,
                'mean_change': np.mean(sample_diff_norms),
                'std_change': np.std(sample_diff_norms)
            }

        return results

    def visualize_amplification_effects(
        self,
        male_results: Dict,
        female_results: Dict
    ):
        """Visualize amplification experiment results."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Layer {self.layer} - Feature Amplification Effects',
                     fontsize=18, fontweight='bold')

        factors = male_results['amplification_factors']

        # Extract data for each factor
        male_data_by_factor = {}
        female_data_by_factor = {}

        for factor in factors:
            male_df = pd.DataFrame(male_results['per_factor_results'][str(factor)]['per_sample'])
            female_df = pd.DataFrame(female_results['per_factor_results'][str(factor)]['per_sample'])

            male_data_by_factor[factor] = male_df
            female_data_by_factor[factor] = female_df

        # 1. Male feature amplification - effect by gender
        ax = axes[0, 0]
        for factor in factors:
            df = male_data_by_factor[factor]
            male_changes = df[df['gender'] == 'male']['reconstruction_change']
            female_changes = df[df['gender'] == 'female']['reconstruction_change']

            positions = [factor - 0.1, factor + 0.1]
            ax.boxplot([male_changes, female_changes], positions=positions,
                      widths=0.15, patch_artist=True,
                      boxprops=dict(facecolor='lightblue' if factor == factors[0] else 'lightpink'))

        ax.set_xlabel('Amplification Factor')
        ax.set_ylabel('Reconstruction Change')
        ax.set_title('Amplifying Male-Biased Features')
        ax.set_xticks(factors)
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Female feature amplification - effect by gender
        ax = axes[0, 1]
        for factor in factors:
            df = female_data_by_factor[factor]
            male_changes = df[df['gender'] == 'male']['reconstruction_change']
            female_changes = df[df['gender'] == 'female']['reconstruction_change']

            positions = [factor - 0.1, factor + 0.1]
            ax.boxplot([male_changes, female_changes], positions=positions,
                      widths=0.15, patch_artist=True,
                      boxprops=dict(facecolor='lightblue' if factor == factors[0] else 'lightpink'))

        ax.set_xlabel('Amplification Factor')
        ax.set_ylabel('Reconstruction Change')
        ax.set_title('Amplifying Female-Biased Features')
        ax.set_xticks(factors)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Dose-response curve - Male features
        ax = axes[0, 2]
        for gender in ['male', 'female']:
            means = []
            stds = []
            for factor in factors:
                df = male_data_by_factor[factor]
                changes = df[df['gender'] == gender]['reconstruction_change']
                means.append(changes.mean())
                stds.append(changes.std())

            ax.plot(factors, means, 'o-', label=f'{gender.capitalize()} images',
                   linewidth=2, markersize=8)
            ax.fill_between(factors,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2)

        ax.set_xlabel('Amplification Factor')
        ax.set_ylabel('Mean Reconstruction Change')
        ax.set_title('Dose-Response: Male Features')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Dose-response curve - Female features
        ax = axes[1, 0]
        for gender in ['male', 'female']:
            means = []
            stds = []
            for factor in factors:
                df = female_data_by_factor[factor]
                changes = df[df['gender'] == gender]['reconstruction_change']
                means.append(changes.mean())
                stds.append(changes.std())

            ax.plot(factors, means, 'o-', label=f'{gender.capitalize()} images',
                   linewidth=2, markersize=8)
            ax.fill_between(factors,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2)

        ax.set_xlabel('Amplification Factor')
        ax.set_ylabel('Mean Reconstruction Change')
        ax.set_title('Dose-Response: Female Features')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Differential effects
        ax = axes[1, 1]
        male_diffs = []
        female_diffs = []

        for factor in factors:
            # Male features
            male_df = male_data_by_factor[factor]
            male_on_male = male_df[male_df['gender'] == 'male']['reconstruction_change'].mean()
            male_on_female = male_df[male_df['gender'] == 'female']['reconstruction_change'].mean()
            male_diffs.append(male_on_male - male_on_female)

            # Female features
            female_df = female_data_by_factor[factor]
            female_on_male = female_df[female_df['gender'] == 'male']['reconstruction_change'].mean()
            female_on_female = female_df[female_df['gender'] == 'female']['reconstruction_change'].mean()
            female_diffs.append(female_on_male - female_on_female)

        x = np.arange(len(factors))
        width = 0.35

        ax.bar(x - width/2, male_diffs, width, label='Male Features\n(Male - Female)',
              color='steelblue')
        ax.bar(x + width/2, [-d for d in female_diffs], width,
              label='Female Features\n(Female - Male)', color='orchid')

        ax.set_xlabel('Amplification Factor')
        ax.set_ylabel('Differential Effect')
        ax.set_title('Gender-Specific Differential Effects')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f}x' for f in factors])
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Summary statistics table
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = "Key Findings:\n\n"
        summary_text += f"Amplification Factors: {factors}\n\n"

        # Check if effects scale with amplification
        male_scaling = male_diffs[-1] / male_diffs[0] if male_diffs[0] != 0 else 0
        female_scaling = abs(female_diffs[-1] / female_diffs[0]) if female_diffs[0] != 0 else 0

        summary_text += f"Male Features:\n"
        summary_text += f"  Effect scaling: {male_scaling:.2f}x\n"
        summary_text += f"  Max differential: {max(male_diffs):.3f}\n\n"

        summary_text += f"Female Features:\n"
        summary_text += f"  Effect scaling: {female_scaling:.2f}x\n"
        summary_text += f"  Max differential: {max([abs(d) for d in female_diffs]):.3f}\n\n"

        if male_scaling > 1.5:
            summary_text += "✓ Male features show dose-dependent effects\n"
        if female_scaling > 1.5:
            summary_text += "✓ Female features show dose-dependent effects\n"

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / 'amplification_effects.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization: {output_path}")

    def run_analysis(self, num_features: int = 3):
        """Run amplification analysis."""

        logger.info(f"\n{'='*60}")
        logger.info(f"Feature Amplification Study - Layer {self.layer}")
        logger.info(f"{'='*60}\n")

        # Get top features
        layer_results = self.results['layers'][str(self.layer)]
        male_biased = layer_results['gender_bias']['english']['top_male_biased'][:num_features]
        female_biased = layer_results['gender_bias']['english']['top_female_biased'][:num_features]

        logger.info(f"Male-biased features: {male_biased}")
        logger.info(f"Female-biased features: {female_biased}")

        # Load activations
        logger.info("\nLoading English activations...")
        checkpoint = torch.load(
            self.checkpoint_dir / 'activations_english_sample_small.pt',
            map_location='cpu', weights_only=False
        )

        activations = checkpoint['activations'][self.layer]
        genders = checkpoint['genders']
        image_ids = checkpoint['image_ids']

        # Test amplification
        logger.info("\nTesting MALE-BIASED feature amplification...")
        male_results = self.analyze_amplification_effects(
            activations, male_biased, genders, image_ids
        )

        logger.info("\nTesting FEMALE-BIASED feature amplification...")
        female_results = self.analyze_amplification_effects(
            activations, female_biased, genders, image_ids
        )

        # Visualize
        self.visualize_amplification_effects(male_results, female_results)

        # Save results
        output_file = self.output_dir / 'amplification_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'male_biased': male_results,
                'female_biased': female_results
            }, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

        return male_results, female_results


def main():
    parser = argparse.ArgumentParser(description='Feature Amplification Analysis')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--num-features', type=int, default=3)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    analyzer = FeatureAmplificationAnalyzer(config, args.layer)
    analyzer.run_analysis(num_features=args.num_features)

    logger.info("\n" + "="*60)
    logger.info("Feature amplification analysis complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
