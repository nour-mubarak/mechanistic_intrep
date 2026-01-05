#!/usr/bin/env python3
"""
Lightweight Feature Ablation Analysis
======================================

Analyze the effect of ablating gender-biased features on SAE reconstructions
without requiring full VLM model inference. This is much faster and gives
insights into feature importance.

Usage:
    python scripts/13_feature_ablation_analysis.py --config configs/config.yaml --layer 10
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
from typing import Dict, List, Tuple
import yaml

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureAblationAnalyzer:
    """Analyze feature importance through ablation without full model inference."""

    def __init__(self, config: Dict, layer: int, device: str = 'cuda'):
        self.config = config
        self.layer = layer
        self.device = device

        # Load paths
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.data_file = Path(config['paths']['processed_data']) / 'samples.csv'
        self.image_dir = Path(config['paths']['processed_data']) / 'images'
        self.output_dir = Path(config['paths']['visualizations']) / f'feature_ablation_layer_{layer}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        results_file = Path(config['paths']['results']) / 'comprehensive_analysis_results.json'
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        # Load SAE
        self.sae = self._load_sae()

        # Load data
        self.data_df = pd.read_csv(self.data_file)

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

    def analyze_reconstruction_changes(
        self,
        activations: torch.Tensor,
        features_to_ablate: List[int],
        genders: List[str],
        image_ids: List[str]
    ) -> Dict:
        """
        Analyze how ablating features changes reconstructions.

        Args:
            activations: Original activations [batch, seq, hidden]
            features_to_ablate: Feature indices to ablate
            genders: Gender labels
            image_ids: Image IDs

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing reconstruction changes when ablating {len(features_to_ablate)} features...")

        # Flatten activations
        orig_shape = activations.shape
        activations_flat = activations.reshape(-1, activations.shape[-1])

        # Process in batches to avoid OOM
        batch_size = 512
        all_diff_norms = []

        with torch.no_grad():
            for i in range(0, len(activations_flat), batch_size):
                batch = activations_flat[i:i+batch_size].to(self.device)

                # Original reconstruction
                recon_orig, features_orig, _ = self.sae(batch)

                # Ablated reconstruction
                features_ablated = features_orig.clone()
                features_ablated[:, features_to_ablate] = 0
                recon_ablated = self.sae.decode(features_ablated)

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
            # Use mean norm across sequence
            sample_diff_norms.append(recon_diff_norm[sample_slice].mean().item())

        results = {
            'feature_indices': features_to_ablate,
            'num_samples': num_samples,
            'per_sample_results': []
        }

        for i in range(num_samples):
            results['per_sample_results'].append({
                'image_id': image_ids[i],
                'gender': genders[i],
                'reconstruction_change': sample_diff_norms[i]
            })

        return results

    def run_ablation_study(self, num_features: int = 3):
        """Run comprehensive ablation study."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Feature Ablation Study - Layer {self.layer}")
        logger.info(f"{'='*60}\n")

        # Get top biased features
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

        logger.info(f"Loaded {len(genders)} samples")

        # Analyze male-biased feature ablation
        logger.info("\n" + "="*60)
        logger.info("Ablating MALE-BIASED features")
        logger.info("="*60)

        male_ablation = self.analyze_reconstruction_changes(
            activations, male_biased, genders, image_ids
        )

        # Analyze female-biased feature ablation
        logger.info("\n" + "="*60)
        logger.info("Ablating FEMALE-BIASED features")
        logger.info("="*60)

        female_ablation = self.analyze_reconstruction_changes(
            activations, female_biased, genders, image_ids
        )

        # Visualize results
        self.visualize_ablation_effects(male_ablation, female_ablation)

        # Save results
        output_file = self.output_dir / 'ablation_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'male_biased_ablation': male_ablation,
                'female_biased_ablation': female_ablation
            }, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

        return male_ablation, female_ablation

    def visualize_ablation_effects(
        self,
        male_ablation: Dict,
        female_ablation: Dict
    ):
        """Visualize the effects of feature ablation."""

        # Extract data
        male_results = pd.DataFrame(male_ablation['per_sample_results'])
        female_results = pd.DataFrame(female_ablation['per_sample_results'])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Layer {self.layer} - Feature Ablation Effects',
                     fontsize=18, fontweight='bold')

        # 1. Ablate male features - effect by gender
        ax = axes[0, 0]
        male_male = male_results[male_results['gender'] == 'male']['reconstruction_change']
        male_female = male_results[male_results['gender'] == 'female']['reconstruction_change']

        ax.boxplot([male_male, male_female], labels=['Male Images', 'Female Images'])
        ax.set_ylabel('Reconstruction Change (L2 norm)')
        ax.set_title('Ablating Male-Biased Features\nEffect on Different Genders')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics
        male_mean_diff = male_male.mean() - male_female.mean()
        ax.text(0.5, 0.95, f'Mean difference: {male_mean_diff:.4f}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Ablate female features - effect by gender
        ax = axes[0, 1]
        female_male = female_results[female_results['gender'] == 'male']['reconstruction_change']
        female_female = female_results[female_results['gender'] == 'female']['reconstruction_change']

        ax.boxplot([female_male, female_female], labels=['Male Images', 'Female Images'])
        ax.set_ylabel('Reconstruction Change (L2 norm)')
        ax.set_title('Ablating Female-Biased Features\nEffect on Different Genders')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics
        female_mean_diff = female_male.mean() - female_female.mean()
        ax.text(0.5, 0.95, f'Mean difference: {female_mean_diff:.4f}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3. Distribution comparison
        ax = axes[1, 0]
        ax.hist(male_male, bins=20, alpha=0.5, label='Male imgs (ablate male feats)', color='blue')
        ax.hist(male_female, bins=20, alpha=0.5, label='Female imgs (ablate male feats)', color='pink')
        ax.hist(female_male, bins=20, alpha=0.5, label='Male imgs (ablate female feats)', color='lightblue')
        ax.hist(female_female, bins=20, alpha=0.5, label='Female imgs (ablate female feats)', color='hotpink')
        ax.set_xlabel('Reconstruction Change')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Reconstruction Changes')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('off')

        stats_data = [
            ['Ablation', 'Gender', 'Mean Change', 'Std', 'Max'],
            ['Male Features', 'Male', f'{male_male.mean():.4f}', f'{male_male.std():.4f}', f'{male_male.max():.4f}'],
            ['Male Features', 'Female', f'{male_female.mean():.4f}', f'{male_female.std():.4f}', f'{male_female.max():.4f}'],
            ['Female Features', 'Male', f'{female_male.mean():.4f}', f'{female_male.std():.4f}', f'{female_male.max():.4f}'],
            ['Female Features', 'Female', f'{female_female.mean():.4f}', f'{female_female.std():.4f}', f'{female_female.max():.4f}'],
        ]

        table = ax.table(cellText=stats_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Reconstruction Change Statistics', fontweight='bold', pad=20)

        # Key finding text
        key_finding = ""
        if abs(male_mean_diff) > abs(female_mean_diff):
            key_finding = "Male-biased features have STRONGER gender-specific effects"
        else:
            key_finding = "Female-biased features have STRONGER gender-specific effects"

        fig.text(0.5, 0.02, f"Key Finding: {key_finding}",
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / 'ablation_effects.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ablation visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Feature Ablation Analysis')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--num-features', type=int, default=3)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    analyzer = FeatureAblationAnalyzer(config, args.layer)
    analyzer.run_ablation_study(num_features=args.num_features)

    logger.info("\n" + "="*60)
    logger.info("Feature ablation analysis complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
