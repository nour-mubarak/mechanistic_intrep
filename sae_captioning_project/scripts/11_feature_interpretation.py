#!/usr/bin/env python3
"""
Feature Interpretation & Visualization
=======================================

Understand what top gender-biased features actually represent:
1. Extract images/captions that maximally activate specific features
2. Create feature dashboards showing top activating examples
3. Visualize feature activation patterns across different genders
4. Perform activation patching to verify causal importance

Usage:
    python scripts/11_feature_interpretation.py --config configs/config.yaml --layer 10
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
from typing import Dict, List, Tuple, Optional
import yaml
import wandb
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureInterpreter:
    """Interpret SAE features by examining their top activating examples."""

    def __init__(self, config: Dict, layer: int, device: str = 'cuda'):
        """
        Initialize feature interpreter.

        Args:
            config: Configuration dictionary
            layer: Layer to analyze
            device: Device to use
        """
        self.config = config
        self.layer = layer
        self.device = device

        # Load paths
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.data_file = Path(config['paths']['processed_data']) / 'samples.csv'
        self.image_dir = Path(config['paths']['processed_data']) / 'images'
        self.output_dir = Path(config['paths']['visualizations']) / f'feature_interpretation_layer_{layer}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results to get top biased features
        results_file = Path(config['paths']['results']) / 'comprehensive_analysis_results.json'
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        # Load SAE
        self.sae = self._load_sae()

        # Load data
        self.data_df = pd.read_csv(self.data_file)
        logger.info(f"Loaded {len(self.data_df)} samples from {self.data_file}")

    def _load_sae(self) -> SparseAutoencoder:
        """Load trained SAE for the layer."""
        sae_path = self.checkpoint_dir / f'sae_layer_{self.layer}.pt'
        logger.info(f"Loading SAE from {sae_path}")

        checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
        sae_config = checkpoint['config']

        sae = SparseAutoencoder(sae_config)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(self.device)
        sae.eval()

        return sae

    def load_activations_and_features(self, language: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        """
        Load activations and compute SAE features for a language.

        Args:
            language: 'english' or 'arabic'

        Returns:
            Tuple of (activations, features, genders, image_ids)
        """
        logger.info(f"Loading {language} activations...")

        # Load activation checkpoint
        checkpoint_path = self.checkpoint_dir / f'activations_{language}_sample_small.pt'
        if not checkpoint_path.exists():
            checkpoint_path = self.checkpoint_dir / f'activations_{language}_sample.pt'

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        activations = checkpoint['activations'][self.layer]  # [batch, seq, hidden]
        genders = checkpoint['genders']
        image_ids = checkpoint['image_ids']

        logger.info(f"Loaded activations shape: {activations.shape}")

        # Compute SAE features
        logger.info("Computing SAE features...")
        batch_size = 256
        activations_flat = activations.reshape(-1, activations.shape[-1])

        all_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(activations_flat), batch_size), desc="Computing features"):
                batch = activations_flat[i:i+batch_size].to(self.device)
                _, features, _ = self.sae(batch)
                all_features.append(features.cpu())
                del batch, features
                torch.cuda.empty_cache()

        features = torch.cat(all_features, dim=0)  # [batch*seq, num_features]
        logger.info(f"Features shape: {features.shape}")

        return activations, features, genders, image_ids

    def find_top_activating_examples(
        self,
        features: torch.Tensor,
        feature_idx: int,
        genders: List[str],
        image_ids: List[str],
        top_k: int = 10
    ) -> Dict[str, List]:
        """
        Find top-k examples that maximally activate a feature.

        Args:
            features: Feature activations [num_tokens, num_features]
            feature_idx: Index of feature to analyze
            genders: List of gender labels (one per sample)
            image_ids: List of image IDs (one per sample)
            top_k: Number of top examples to return

        Returns:
            Dictionary with top activating examples
        """
        # Get feature activations
        feature_acts = features[:, feature_idx].numpy()

        # Calculate sequence length
        num_samples = len(genders)
        seq_length = len(feature_acts) // num_samples

        # Aggregate activations per sample (using max pooling)
        sample_acts = feature_acts.reshape(num_samples, seq_length).max(axis=1)

        # Get top-k indices
        top_indices = np.argsort(sample_acts)[-top_k:][::-1]

        results = {
            'feature_idx': feature_idx,
            'top_examples': []
        }

        for rank, idx in enumerate(top_indices):
            results['top_examples'].append({
                'rank': rank + 1,
                'sample_idx': int(idx),
                'image_id': image_ids[idx],
                'gender': genders[idx],
                'activation': float(sample_acts[idx]),
                'token_activations': feature_acts[idx*seq_length:(idx+1)*seq_length].tolist()
            })

        return results

    def visualize_feature_dashboard(
        self,
        feature_idx: int,
        en_examples: Dict,
        ar_examples: Dict,
        language: str = 'english'
    ):
        """
        Create a dashboard showing top activating images for a feature.

        Args:
            feature_idx: Feature index
            en_examples: English top examples
            ar_examples: Arabic top examples
            language: Which language examples to visualize
        """
        examples = en_examples if language == 'english' else ar_examples
        top_examples = examples['top_examples'][:6]  # Show top 6

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Feature {feature_idx} - Top Activating {language.capitalize()} Examples',
                     fontsize=18, fontweight='bold')

        for idx, (ax, example) in enumerate(zip(axes.flat, top_examples)):
            # Load image
            image_id = example['image_id']
            # image_id already includes .jpg extension
            image_path = self.image_dir / image_id

            if image_path.exists():
                img = Image.open(image_path)
                ax.imshow(img)

                # Add info
                title = f"Rank {example['rank']} | Gender: {example['gender']}\n"
                title += f"Max Activation: {example['activation']:.2f}"
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')

                # Get caption if available
                # CSV uses 'image' column, not 'image_id'
                sample_data = self.data_df[self.data_df['image'] == image_id]
                if not sample_data.empty:
                    caption_col = 'en_caption' if language == 'english' else 'ar_caption'
                    caption = sample_data[caption_col].iloc[0] if caption_col in sample_data.columns else "N/A"
                    # Truncate caption
                    if len(caption) > 100:
                        caption = caption[:100] + "..."
                    ax.text(0.5, -0.15, caption, transform=ax.transAxes,
                           ha='center', va='top', fontsize=9, wrap=True)
            else:
                ax.text(0.5, 0.5, f"Image not found:\n{image_id}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        plt.tight_layout()
        output_path = self.output_dir / f'feature_{feature_idx}_dashboard_{language}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved dashboard: {output_path}")

    def visualize_activation_distribution(
        self,
        feature_idx: int,
        en_features: torch.Tensor,
        ar_features: torch.Tensor,
        en_genders: List[str],
        ar_genders: List[str]
    ):
        """
        Visualize activation distribution across genders and languages.

        Args:
            feature_idx: Feature to analyze
            en_features: English features
            ar_features: Arabic features
            en_genders: English gender labels
            ar_genders: Arabic gender labels
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Feature {feature_idx} - Activation Distribution Analysis',
                     fontsize=18, fontweight='bold')

        # Helper to get sample-level activations
        def get_sample_acts(features, genders):
            feature_acts = features[:, feature_idx].numpy()
            num_samples = len(genders)
            seq_length = len(feature_acts) // num_samples
            return feature_acts.reshape(num_samples, seq_length).max(axis=1), genders

        en_acts, _ = get_sample_acts(en_features, en_genders)
        ar_acts, _ = get_sample_acts(ar_features, ar_genders)

        # 1. Histogram by gender - English
        ax = axes[0, 0]
        male_acts_en = en_acts[[g == 'male' for g in en_genders]]
        female_acts_en = en_acts[[g == 'female' for g in en_genders]]

        ax.hist(male_acts_en, bins=30, alpha=0.6, label='Male', color='blue')
        ax.hist(female_acts_en, bins=30, alpha=0.6, label='Female', color='pink')
        ax.set_xlabel('Activation')
        ax.set_ylabel('Frequency')
        ax.set_title('English - Activation Distribution by Gender')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Histogram by gender - Arabic
        ax = axes[0, 1]
        male_acts_ar = ar_acts[[g == 'male' for g in ar_genders]]
        female_acts_ar = ar_acts[[g == 'female' for g in ar_genders]]

        ax.hist(male_acts_ar, bins=30, alpha=0.6, label='Male', color='blue')
        ax.hist(female_acts_ar, bins=30, alpha=0.6, label='Female', color='pink')
        ax.set_xlabel('Activation')
        ax.set_ylabel('Frequency')
        ax.set_title('Arabic - Activation Distribution by Gender')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Boxplot comparison
        ax = axes[1, 0]
        data_to_plot = [male_acts_en, female_acts_en, male_acts_ar, female_acts_ar]
        labels = ['EN Male', 'EN Female', 'AR Male', 'AR Female']
        colors = ['lightblue', 'lightpink', 'cornflowerblue', 'hotpink']

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Activation')
        ax.set_title('Activation Comparison Across Gender & Language')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('off')

        stats_data = [
            ['Metric', 'EN Male', 'EN Female', 'AR Male', 'AR Female'],
            ['Mean', f'{male_acts_en.mean():.3f}', f'{female_acts_en.mean():.3f}',
             f'{male_acts_ar.mean():.3f}', f'{female_acts_ar.mean():.3f}'],
            ['Std', f'{male_acts_en.std():.3f}', f'{female_acts_en.std():.3f}',
             f'{male_acts_ar.std():.3f}', f'{female_acts_ar.std():.3f}'],
            ['Max', f'{male_acts_en.max():.3f}', f'{female_acts_en.max():.3f}',
             f'{male_acts_ar.max():.3f}', f'{female_acts_ar.max():.3f}'],
            ['% Active', f'{(male_acts_en > 0).mean()*100:.1f}%',
             f'{(female_acts_en > 0).mean()*100:.1f}%',
             f'{(male_acts_ar > 0).mean()*100:.1f}%',
             f'{(female_acts_ar > 0).mean()*100:.1f}%']
        ]

        table = ax.table(cellText=stats_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Activation Statistics', fontweight='bold', pad=20)

        plt.tight_layout()
        output_path = self.output_dir / f'feature_{feature_idx}_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved distribution plot: {output_path}")

    def analyze_top_biased_features(self, num_features: int = 10):
        """
        Analyze top gender-biased features for the layer.

        Args:
            num_features: Number of top features to analyze
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing Top {num_features} Biased Features for Layer {self.layer}")
        logger.info(f"{'='*60}\n")

        # Load activations and features for both languages
        logger.info("Loading English data...")
        _, en_features, en_genders, en_image_ids = self.load_activations_and_features('english')

        logger.info("Loading Arabic data...")
        _, ar_features, ar_genders, ar_image_ids = self.load_activations_and_features('arabic')

        # Get top biased features from results
        layer_results = self.results['layers'][str(self.layer)]

        male_biased = layer_results['gender_bias']['english']['top_male_biased'][:num_features]
        female_biased = layer_results['gender_bias']['english']['top_female_biased'][:num_features]

        logger.info(f"\nTop {num_features} Male-Biased Features: {male_biased}")
        logger.info(f"Top {num_features} Female-Biased Features: {female_biased}")

        # Analyze each feature
        all_results = {
            'layer': self.layer,
            'male_biased_features': {},
            'female_biased_features': {}
        }

        logger.info("\n" + "="*60)
        logger.info("MALE-BIASED FEATURES")
        logger.info("="*60)

        for feature_idx in tqdm(male_biased, desc="Analyzing male-biased features"):
            # Find top activating examples
            en_examples = self.find_top_activating_examples(
                en_features, feature_idx, en_genders, en_image_ids, top_k=10
            )
            ar_examples = self.find_top_activating_examples(
                ar_features, feature_idx, ar_genders, ar_image_ids, top_k=10
            )

            # Visualize
            self.visualize_feature_dashboard(feature_idx, en_examples, ar_examples, 'english')
            self.visualize_feature_dashboard(feature_idx, en_examples, ar_examples, 'arabic')
            self.visualize_activation_distribution(
                feature_idx, en_features, ar_features, en_genders, ar_genders
            )

            all_results['male_biased_features'][feature_idx] = {
                'english': en_examples,
                'arabic': ar_examples
            }

        logger.info("\n" + "="*60)
        logger.info("FEMALE-BIASED FEATURES")
        logger.info("="*60)

        for feature_idx in tqdm(female_biased, desc="Analyzing female-biased features"):
            # Find top activating examples
            en_examples = self.find_top_activating_examples(
                en_features, feature_idx, en_genders, en_image_ids, top_k=10
            )
            ar_examples = self.find_top_activating_examples(
                ar_features, feature_idx, ar_genders, ar_image_ids, top_k=10
            )

            # Visualize
            self.visualize_feature_dashboard(feature_idx, en_examples, ar_examples, 'english')
            self.visualize_feature_dashboard(feature_idx, en_examples, ar_examples, 'arabic')
            self.visualize_activation_distribution(
                feature_idx, en_features, ar_features, en_genders, ar_genders
            )

            all_results['female_biased_features'][feature_idx] = {
                'english': en_examples,
                'arabic': ar_examples
            }

        # Save results
        output_file = self.output_dir / 'feature_interpretation_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nSaved interpretation results to: {output_file}")
        logger.info(f"Visualizations saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Feature Interpretation & Visualization')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--layer', type=int, required=True, help='Layer to analyze')
    parser.add_argument('--num-features', type=int, default=5,
                       help='Number of top features to analyze (default: 5)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize WandB
    wandb.init(
        project=config.get('wandb', {}).get('project', 'sae-captioning-bias'),
        name=f"feature-interpretation-layer{args.layer}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            'stage': 'feature_interpretation',
            'layer': args.layer,
            'num_features': args.num_features,
        },
        tags=['feature-interpretation', f'layer-{args.layer}']
    )
    logger.info(f"WandB initialized: {wandb.run.url}")

    # Create interpreter
    interpreter = FeatureInterpreter(config, args.layer)

    # Run analysis
    interpreter.analyze_top_biased_features(num_features=args.num_features)

    # Log completion to WandB
    wandb.log({'status': 'complete', 'layer': args.layer})
    wandb.finish()

    logger.info("\n" + "="*60)
    logger.info("Feature interpretation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
