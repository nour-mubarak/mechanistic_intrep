#!/usr/bin/env python3
"""
Qualitative Visual Analysis of Top Activating Images
======================================================

Manually inspect and categorize the visual patterns in top-activating images
for gender-biased features. This script generates detailed reports showing:
- Image grids organized by feature
- Common visual attributes and themes
- Detailed captions and metadata
- Comparative analysis across genders

Usage:
    python scripts/17_qualitative_visual_analysis.py --config configs/config.yaml --layer 10
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
from collections import Counter

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualitativeVisualAnalyzer:
    """Generate detailed qualitative reports for gender-biased features."""

    def __init__(self, config: Dict, layer: int, device: str = 'cuda'):
        self.config = config
        self.layer = layer
        self.device = device

        # Load paths
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.data_file = Path(config['paths']['processed_data']) / 'samples.csv'
        self.image_dir = Path(config['paths']['processed_data']) / 'images'
        self.output_dir = Path(config['paths']['visualizations']) / f'qualitative_analysis_layer_{layer}'
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

    def create_detailed_feature_report(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        genders: List[str],
        image_ids: List[str],
        feature_type: str,
        top_k: int = 30
    ) -> Dict:
        """
        Create detailed qualitative report for a feature.

        Args:
            feature_idx: Feature to analyze
            activations: Original activations
            genders: Gender labels
            image_ids: Image IDs
            feature_type: 'male-biased' or 'female-biased'
            top_k: Number of top examples to analyze

        Returns:
            Dictionary with detailed analysis
        """
        logger.info(f"Creating detailed report for feature {feature_idx} ({feature_type})...")

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

        features = torch.cat(all_features, dim=0)

        # Get this feature's activations
        feature_acts = features[:, feature_idx].numpy()

        # Reshape to [num_samples, seq_length]
        num_samples = len(genders)
        seq_length = len(feature_acts) // num_samples
        feature_acts_per_sample = feature_acts.reshape(num_samples, seq_length)

        # Find top activating samples
        max_acts_per_sample = feature_acts_per_sample.max(axis=1)
        top_indices = np.argsort(max_acts_per_sample)[-top_k:][::-1]

        # Collect detailed information
        top_examples = []
        for rank, idx in enumerate(top_indices):
            image_id = image_ids[idx]
            gender = genders[idx]
            max_act = max_acts_per_sample[idx]
            token_idx = feature_acts_per_sample[idx].argmax()

            # Get caption
            sample_data = self.data_df[self.data_df['image'] == image_id]
            caption = "N/A"
            if not sample_data.empty:
                caption = sample_data['en_caption'].iloc[0] if 'en_caption' in sample_data.columns else "N/A"

            top_examples.append({
                'rank': rank + 1,
                'image_id': image_id,
                'gender': gender,
                'max_activation': float(max_act),
                'token_position': int(token_idx),
                'caption': caption
            })

        # Gender distribution
        gender_dist = Counter([genders[idx] for idx in top_indices])

        # Analyze captions for themes (simple keyword extraction)
        captions = [ex['caption'] for ex in top_examples if ex['caption'] != "N/A"]
        caption_themes = self._extract_caption_themes(captions)

        report = {
            'feature_idx': feature_idx,
            'feature_type': feature_type,
            'top_examples': top_examples,
            'gender_distribution': dict(gender_dist),
            'caption_themes': caption_themes,
            'statistics': {
                'mean_activation': float(feature_acts.mean()),
                'max_activation': float(feature_acts.max()),
                'activation_sparsity': float((feature_acts > 0).mean()),
            }
        }

        return report

    def _extract_caption_themes(self, captions: List[str]) -> Dict:
        """Extract common themes from captions."""
        # Common keywords to look for
        appearance_words = ['wearing', 'shirt', 'dress', 'hair', 'beard', 'glasses', 'hat', 'tie', 'suit']
        activity_words = ['standing', 'sitting', 'walking', 'playing', 'holding', 'smiling', 'looking']
        setting_words = ['indoor', 'outdoor', 'street', 'room', 'field', 'beach', 'building']
        people_words = ['man', 'woman', 'person', 'people', 'group', 'young', 'old']

        themes = {
            'appearance': [],
            'activity': [],
            'setting': [],
            'people_descriptors': []
        }

        for caption in captions:
            caption_lower = caption.lower()

            for word in appearance_words:
                if word in caption_lower:
                    themes['appearance'].append(word)

            for word in activity_words:
                if word in caption_lower:
                    themes['activity'].append(word)

            for word in setting_words:
                if word in caption_lower:
                    themes['setting'].append(word)

            for word in people_words:
                if word in caption_lower:
                    themes['people_descriptors'].append(word)

        # Count frequencies
        return {
            'appearance': dict(Counter(themes['appearance']).most_common(5)),
            'activity': dict(Counter(themes['activity']).most_common(5)),
            'setting': dict(Counter(themes['setting']).most_common(5)),
            'people_descriptors': dict(Counter(themes['people_descriptors']).most_common(5))
        }

    def visualize_detailed_report(self, report: Dict):
        """Create comprehensive visual report for a feature."""

        feature_idx = report['feature_idx']
        feature_type = report['feature_type']
        top_examples = report['top_examples'][:20]  # Show top 20

        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.3)

        fig.suptitle(f'Feature {feature_idx} - Detailed Qualitative Analysis ({feature_type})',
                     fontsize=20, fontweight='bold', y=0.995)

        # Top 20 images (5 rows × 4 cols)
        for idx, example in enumerate(top_examples):
            row = idx // 4
            col = idx % 4
            ax = fig.add_subplot(gs[row, col])

            # Load image
            image_path = self.image_dir / example['image_id']
            if image_path.exists():
                img = Image.open(image_path)
                ax.imshow(img)

            # Title with metadata
            title = f"#{example['rank']} | {example['gender']}\n"
            title += f"Act: {example['max_activation']:.1f}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')

            # Add caption below image
            caption = example['caption']
            if len(caption) > 80:
                caption = caption[:80] + "..."
            ax.text(0.5, -0.05, caption, transform=ax.transAxes,
                   ha='center', va='top', fontsize=7, wrap=True)

        # Bottom row: Statistics and themes
        # Gender distribution
        ax = fig.add_subplot(gs[5, 0])
        gender_dist = report['gender_distribution']
        colors = {'male': 'lightblue', 'female': 'lightpink', 'unknown': 'lightgray'}
        bar_colors = [colors.get(g, 'gray') for g in gender_dist.keys()]
        ax.bar(gender_dist.keys(), gender_dist.values(), color=bar_colors)
        ax.set_title('Gender Distribution\n(Top 30)', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

        # Caption themes - Appearance
        ax = fig.add_subplot(gs[5, 1])
        ax.axis('off')
        themes_text = "Caption Themes:\n\n"
        themes_text += "Appearance:\n"
        for word, count in list(report['caption_themes']['appearance'].items())[:5]:
            themes_text += f"  {word}: {count}\n"
        themes_text += "\nActivity:\n"
        for word, count in list(report['caption_themes']['activity'].items())[:5]:
            themes_text += f"  {word}: {count}\n"
        ax.text(0.1, 0.5, themes_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Caption themes - Setting & People
        ax = fig.add_subplot(gs[5, 2])
        ax.axis('off')
        themes_text2 = "Caption Themes:\n\n"
        themes_text2 += "Setting:\n"
        for word, count in list(report['caption_themes']['setting'].items())[:5]:
            themes_text2 += f"  {word}: {count}\n"
        themes_text2 += "\nPeople:\n"
        for word, count in list(report['caption_themes']['people_descriptors'].items())[:5]:
            themes_text2 += f"  {word}: {count}\n"
        ax.text(0.1, 0.5, themes_text2, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Statistics
        ax = fig.add_subplot(gs[5, 3])
        ax.axis('off')
        stats = report['statistics']
        stats_text = "Activation Statistics:\n\n"
        stats_text += f"Mean: {stats['mean_activation']:.3f}\n"
        stats_text += f"Max: {stats['max_activation']:.3f}\n"
        stats_text += f"Sparsity: {stats['activation_sparsity']*100:.1f}%\n\n"
        stats_text += "Gender in Top 30:\n"
        for gender, count in gender_dist.items():
            stats_text += f"  {gender.capitalize()}: {count}\n"
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()
        output_path = self.output_dir / f'feature_{feature_idx}_detailed_report.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved detailed report: {output_path}")

    def run_analysis(self, num_features: int = 3):
        """Run qualitative visual analysis."""

        logger.info(f"\n{'='*60}")
        logger.info(f"Qualitative Visual Analysis - Layer {self.layer}")
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

        all_reports = {
            'layer': self.layer,
            'male_biased': {},
            'female_biased': {}
        }

        # Analyze male-biased features
        logger.info("\nAnalyzing MALE-BIASED features...")
        for feature_idx in tqdm(male_biased, desc="Male features"):
            report = self.create_detailed_feature_report(
                feature_idx, activations, genders, image_ids, 'male-biased', top_k=30
            )
            all_reports['male_biased'][feature_idx] = report
            self.visualize_detailed_report(report)

        # Analyze female-biased features
        logger.info("\nAnalyzing FEMALE-BIASED features...")
        for feature_idx in tqdm(female_biased, desc="Female features"):
            report = self.create_detailed_feature_report(
                feature_idx, activations, genders, image_ids, 'female-biased', top_k=30
            )
            all_reports['female_biased'][feature_idx] = report
            self.visualize_detailed_report(report)

        # Save detailed reports
        output_file = self.output_dir / 'qualitative_reports.json'
        with open(output_file, 'w') as f:
            json.dump(all_reports, f, indent=2)

        logger.info(f"\nDetailed reports saved to: {output_file}")

        # Create summary comparison
        self._create_summary_comparison(all_reports)

        return all_reports

    def _create_summary_comparison(self, all_reports: Dict):
        """Create summary comparison across all analyzed features."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Layer {self.layer} - Qualitative Analysis Summary',
                     fontsize=18, fontweight='bold')

        # Extract data
        male_features = list(all_reports['male_biased'].keys())
        female_features = list(all_reports['female_biased'].keys())

        # 1. Gender distributions - Male features
        ax = axes[0, 0]
        male_gender_data = []
        labels = []
        for feat_idx in male_features:
            report = all_reports['male_biased'][feat_idx]
            dist = report['gender_distribution']
            male_gender_data.append([
                dist.get('male', 0),
                dist.get('female', 0),
                dist.get('unknown', 0)
            ])
            labels.append(f"F{feat_idx}")

        male_gender_data = np.array(male_gender_data)
        x = np.arange(len(male_features))
        width = 0.25

        ax.bar(x - width, male_gender_data[:, 0], width, label='Male', color='steelblue')
        ax.bar(x, male_gender_data[:, 1], width, label='Female', color='salmon')
        ax.bar(x + width, male_gender_data[:, 2], width, label='Unknown', color='gray')

        ax.set_ylabel('Count (Top 30)')
        ax.set_title('Male-Biased Features\nGender Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Gender distributions - Female features
        ax = axes[0, 1]
        female_gender_data = []
        labels = []
        for feat_idx in female_features:
            report = all_reports['female_biased'][feat_idx]
            dist = report['gender_distribution']
            female_gender_data.append([
                dist.get('male', 0),
                dist.get('female', 0),
                dist.get('unknown', 0)
            ])
            labels.append(f"F{feat_idx}")

        female_gender_data = np.array(female_gender_data)
        x = np.arange(len(female_features))

        ax.bar(x - width, female_gender_data[:, 0], width, label='Male', color='steelblue')
        ax.bar(x, female_gender_data[:, 1], width, label='Female', color='salmon')
        ax.bar(x + width, female_gender_data[:, 2], width, label='Unknown', color='gray')

        ax.set_ylabel('Count (Top 30)')
        ax.set_title('Female-Biased Features\nGender Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Activation statistics
        ax = axes[0, 2]
        male_max_acts = [all_reports['male_biased'][f]['statistics']['max_activation'] for f in male_features]
        female_max_acts = [all_reports['female_biased'][f]['statistics']['max_activation'] for f in female_features]

        x_male = np.arange(len(male_features))
        x_female = np.arange(len(female_features)) + len(male_features) + 0.5

        ax.bar(x_male, male_max_acts, color='steelblue', alpha=0.7, label='Male-biased')
        ax.bar(x_female, female_max_acts, color='salmon', alpha=0.7, label='Female-biased')

        ax.set_ylabel('Max Activation')
        ax.set_title('Maximum Activation Strength')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4-6. Theme analysis
        theme_categories = ['appearance', 'activity', 'setting']
        for col_idx, theme_cat in enumerate(theme_categories):
            ax = axes[1, col_idx]
            ax.axis('off')

            # Aggregate themes across features
            male_themes = Counter()
            female_themes = Counter()

            for feat_idx in male_features:
                themes = all_reports['male_biased'][feat_idx]['caption_themes'][theme_cat]
                male_themes.update(themes)

            for feat_idx in female_features:
                themes = all_reports['female_biased'][feat_idx]['caption_themes'][theme_cat]
                female_themes.update(themes)

            # Display
            text = f"Common {theme_cat.capitalize()} Themes:\n\n"
            text += "Male-Biased Features:\n"
            for word, count in male_themes.most_common(5):
                text += f"  {word}: {count}\n"
            text += "\nFemale-Biased Features:\n"
            for word, count in female_themes.most_common(5):
                text += f"  {word}: {count}\n"

            ax.text(0.1, 0.5, text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

            ax.set_title(f'{theme_cat.capitalize()} Themes', fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'qualitative_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved summary comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Qualitative Visual Analysis')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--num-features', type=int, default=3)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    analyzer = QualitativeVisualAnalyzer(config, args.layer)
    analyzer.run_analysis(num_features=args.num_features)

    logger.info("\n" + "="*60)
    logger.info("Qualitative visual analysis complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
