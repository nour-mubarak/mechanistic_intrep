#!/usr/bin/env python3
"""
Visual Pattern Analysis for Gender-Biased Features
===================================================

Deep dive into what specific visual patterns trigger gender-biased features.
Creates detailed visualizations showing:
- Top activating image regions
- Common visual attributes
- Token-level activation heatmaps

Usage:
    python scripts/14_visual_pattern_analysis.py --config configs/config.yaml --layer 10
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
import wandb
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualPatternAnalyzer:
    """Analyze visual patterns that activate specific SAE features."""

    def __init__(self, config: Dict, layer: int, device: str = 'cuda'):
        self.config = config
        self.layer = layer
        self.device = device

        # Load paths
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.data_file = Path(config['paths']['processed_data']) / 'samples.csv'
        self.image_dir = Path(config['paths']['processed_data']) / 'images'
        self.output_dir = Path(config['paths']['visualizations']) / f'visual_patterns_layer_{layer}'
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

    def analyze_feature_patterns(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        genders: List[str],
        image_ids: List[str],
        top_k: int = 20
    ) -> Dict:
        """
        Analyze visual patterns for a specific feature.

        Args:
            feature_idx: Feature to analyze
            activations: Activations [batch, seq, hidden]
            genders: Gender labels
            image_ids: Image IDs
            top_k: Number of top examples to analyze

        Returns:
            Dictionary with pattern analysis
        """
        logger.info(f"Analyzing feature {feature_idx}...")

        # Extract SAE features
        activations_flat = activations.reshape(-1, activations.shape[-1])
        batch_size = 512
        all_features = []

        with torch.no_grad():
            for i in range(0, len(activations_flat), batch_size):
                batch = activations_flat[i:i+batch_size].to(self.device)
                _, features, _ = self.sae(batch)
                all_features.append(features.cpu())
                del batch, features
                torch.cuda.empty_cache()

        features = torch.cat(all_features, dim=0)  # [num_tokens, num_features]

        # Get this feature's activations
        feature_acts = features[:, feature_idx].numpy()

        # Reshape to [num_samples, seq_length]
        num_samples = len(genders)
        seq_length = len(feature_acts) // num_samples
        feature_acts_per_sample = feature_acts.reshape(num_samples, seq_length)

        # Find top activating samples
        max_acts_per_sample = feature_acts_per_sample.max(axis=1)
        top_indices = np.argsort(max_acts_per_sample)[-top_k:][::-1]

        results = {
            'feature_idx': feature_idx,
            'top_examples': [],
            'gender_distribution': {},
            'activation_stats': {}
        }

        # Analyze top examples
        for rank, idx in enumerate(top_indices):
            # Find which token had max activation
            token_idx = feature_acts_per_sample[idx].argmax()

            results['top_examples'].append({
                'rank': rank + 1,
                'sample_idx': int(idx),
                'image_id': image_ids[idx],
                'gender': genders[idx],
                'max_activation': float(max_acts_per_sample[idx]),
                'max_token_position': int(token_idx),
                'token_activations': feature_acts_per_sample[idx].tolist()
            })

        # Gender distribution in top examples
        top_genders = [genders[idx] for idx in top_indices]
        for gender in ['male', 'female', 'unknown']:
            results['gender_distribution'][gender] = top_genders.count(gender)

        # Activation statistics
        results['activation_stats'] = {
            'mean': float(feature_acts.mean()),
            'std': float(feature_acts.std()),
            'max': float(feature_acts.max()),
            'pct_active': float((feature_acts > 0).mean() * 100),
            'mean_when_active': float(feature_acts[feature_acts > 0].mean()) if (feature_acts > 0).any() else 0
        }

        return results

    def visualize_feature_patterns(
        self,
        feature_idx: int,
        analysis: Dict,
        feature_type: str
    ):
        """Create comprehensive visual pattern visualization."""

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

        fig.suptitle(f'Feature {feature_idx} - Visual Pattern Analysis ({feature_type})',
                     fontsize=18, fontweight='bold')

        # Top 15 examples (3 rows x 5 cols)
        top_examples = analysis['top_examples'][:15]

        for idx, example in enumerate(top_examples):
            row = idx // 5
            col = idx % 5
            ax = fig.add_subplot(gs[row, col])

            # Load and display image
            image_path = self.image_dir / example['image_id']
            if image_path.exists():
                img = Image.open(image_path)
                ax.imshow(img)

            # Title with activation info
            title = f"#{example['rank']} | {example['gender']}\n"
            title += f"Act: {example['max_activation']:.1f}"
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.axis('off')

        # Bottom row: Statistics and patterns
        # 1. Gender distribution
        ax = fig.add_subplot(gs[3, 0])
        gender_dist = analysis['gender_distribution']
        colors = ['lightblue', 'lightpink', 'lightgray']
        ax.bar(gender_dist.keys(), gender_dist.values(), color=colors)
        ax.set_title('Gender Distribution\nin Top 20', fontweight='bold')
        ax.set_ylabel('Count')

        # 2. Token position heatmap
        ax = fig.add_subplot(gs[3, 1:3])
        token_acts = np.array([ex['token_activations'] for ex in top_examples])
        im = ax.imshow(token_acts, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_title('Token-level Activations\n(Top 15 Examples)', fontweight='bold')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Example Rank')
        plt.colorbar(im, ax=ax, label='Activation')

        # 3. Activation statistics
        ax = fig.add_subplot(gs[3, 3:5])
        ax.axis('off')

        stats = analysis['activation_stats']
        stats_text = "Activation Statistics:\n\n"
        stats_text += f"Mean: {stats['mean']:.3f}\n"
        stats_text += f"Std: {stats['std']:.3f}\n"
        stats_text += f"Max: {stats['max']:.3f}\n"
        stats_text += f"% Active: {stats['pct_active']:.1f}%\n"
        stats_text += f"Mean (when active): {stats['mean_when_active']:.3f}\n\n"

        stats_text += f"Gender in Top 20:\n"
        stats_text += f"  Male: {gender_dist.get('male', 0)}\n"
        stats_text += f"  Female: {gender_dist.get('female', 0)}\n"
        stats_text += f"  Unknown: {gender_dist.get('unknown', 0)}"

        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = self.output_dir / f'feature_{feature_idx}_patterns.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization: {output_path}")

    def run_analysis(self, num_features: int = 3):
        """Run visual pattern analysis for top features."""

        logger.info(f"\n{'='*60}")
        logger.info(f"Visual Pattern Analysis - Layer {self.layer}")
        logger.info(f"{'='*60}\n")

        # Get top features
        layer_results = self.results['layers'][str(self.layer)]
        male_biased = layer_results['gender_bias']['english']['top_male_biased'][:num_features]
        female_biased = layer_results['gender_bias']['english']['top_female_biased'][:num_features]

        # Load activations
        logger.info("Loading English activations...")
        checkpoint = torch.load(
            self.checkpoint_dir / 'activations_english_sample_small.pt',
            map_location='cpu', weights_only=False
        )

        activations = checkpoint['activations'][self.layer]
        genders = checkpoint['genders']
        image_ids = checkpoint['image_ids']

        all_results = {
            'layer': self.layer,
            'male_biased': {},
            'female_biased': {}
        }

        # Analyze male-biased features
        logger.info("\nAnalyzing MALE-BIASED features...")
        for feature_idx in tqdm(male_biased, desc="Male features"):
            analysis = self.analyze_feature_patterns(
                feature_idx, activations, genders, image_ids, top_k=20
            )
            all_results['male_biased'][feature_idx] = analysis
            self.visualize_feature_patterns(feature_idx, analysis, 'Male-Biased')

        # Analyze female-biased features
        logger.info("\nAnalyzing FEMALE-BIASED features...")
        for feature_idx in tqdm(female_biased, desc="Female features"):
            analysis = self.analyze_feature_patterns(
                feature_idx, activations, genders, image_ids, top_k=20
            )
            all_results['female_biased'][feature_idx] = analysis
            self.visualize_feature_patterns(feature_idx, analysis, 'Female-Biased')

        # Save results
        output_file = self.output_dir / 'visual_pattern_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Visual Pattern Analysis')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--num-features', type=int, default=3)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    analyzer = VisualPatternAnalyzer(config, args.layer)
    analyzer.run_analysis(num_features=args.num_features)

    logger.info("\n" + "="*60)
    logger.info("Visual pattern analysis complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
