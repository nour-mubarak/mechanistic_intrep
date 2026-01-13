#!/usr/bin/env python3
"""
Full Analysis Pipeline with W&B Tracking and Visualizations
============================================================

This script performs comprehensive analysis of trained SAE models:
1. Feature extraction and statistics
2. Gender bias analysis per layer
3. Cross-lingual comparison (using same-image paired data)
4. Visualization generation
5. W&B experiment tracking

Usage:
    python scripts/20_full_analysis_pipeline.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from scipy import stats
import gc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sae import SparseAutoencoder

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'analysis_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


class AnalysisPipeline:
    """Comprehensive SAE feature analysis pipeline."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.results = {}
        self.figures = {}
        
        # Paths
        self.checkpoint_dir = PROJECT_ROOT / config['paths']['checkpoints']
        self.results_dir = PROJECT_ROOT / config['paths']['results']
        self.viz_dir = PROJECT_ROOT / config['paths']['visualizations']
        self.data_dir = PROJECT_ROOT / config['paths']['data_dir']
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
        
        # Colors from config
        self.colors = {
            'english': config['visualization'].get('english_color', '#2ecc71'),
            'arabic': config['visualization'].get('arabic_color', '#e74c3c'),
            'male': config['visualization'].get('male_color', '#3498db'),
            'female': config['visualization'].get('female_color', '#e91e63'),
        }
        
        # Layers to analyze
        self.layers = [0, 3, 6, 9, 12, 15, 17]
        
        # W&B setup
        self.use_wandb = config['logging'].get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb.init(
            project=self.config['logging'].get('wandb_project', 'sae-captioning-bias'),
            entity=self.config['logging'].get('wandb_entity', None),
            name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config,
            tags=['analysis', 'cross-lingual', 'gender-bias']
        )
        logger.info("W&B initialized successfully")
    
    def load_sae(self, layer: int, language: str = 'arabic') -> SparseAutoencoder:
        """Load a trained SAE model."""
        if language == 'english' and layer == 0:
            path = self.checkpoint_dir / 'saes' / f'sae_layer_0.pt'
        else:
            path = self.checkpoint_dir / 'saes' / f'sae_{language}_layer_{layer}.pt'
        
        if not path.exists():
            raise FileNotFoundError(f"SAE not found: {path}")
        
        logger.info(f"Loading SAE from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        config = checkpoint['config']
        sae = SparseAutoencoder(config)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(self.device)
        sae.eval()
        
        return sae, checkpoint.get('history', {})
    
    def load_activations(self, layer: int, language: str = 'arabic', 
                         max_samples: int = 2000) -> dict:
        """Load activations for a specific layer with memory management."""
        layer_dir = self.checkpoint_dir / 'full_layers_ncc' / 'layer_checkpoints'
        path = layer_dir / f'layer_{layer}_{language}.pt'
        
        if not path.exists():
            raise FileNotFoundError(f"Activations not found: {path}")
        
        logger.info(f"Loading activations from {path}")
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        # Get activations
        activations = data.get('activations', data.get('layer_activations'))
        if isinstance(activations, dict) and layer in activations:
            activations = activations[layer]
        
        # Sample if too large
        if activations.shape[0] > max_samples:
            indices = torch.randperm(activations.shape[0])[:max_samples]
            activations = activations[indices]
            genders = [data['genders'][i] for i in indices.tolist()] if 'genders' in data else None
        else:
            genders = data.get('genders', None)
        
        return {
            'activations': activations,
            'genders': genders,
            'image_ids': data.get('image_ids', None),
        }
    
    def extract_sae_features(self, sae: SparseAutoencoder, 
                             activations: torch.Tensor,
                             batch_size: int = 256) -> torch.Tensor:
        """Extract SAE hidden features from activations."""
        logger.info(f"Extracting SAE features from {activations.shape[0]} samples...")
        
        # Handle different activation shapes
        if len(activations.shape) == 3:
            # [batch, seq, hidden] -> take mean over sequence
            activations = activations.mean(dim=1)
        
        sae.eval()
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(activations), batch_size), desc="Extracting features"):
                batch = activations[i:i+batch_size].to(self.device)
                features = sae.encode(batch)
                all_features.append(features.cpu())
                del batch, features
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return torch.cat(all_features, dim=0)
    
    def compute_feature_statistics(self, features: torch.Tensor, 
                                   genders: list) -> pd.DataFrame:
        """Compute per-feature statistics by gender."""
        features_np = features.numpy()
        
        # Create gender mask
        gender_array = np.array(genders)
        male_mask = gender_array == 'male'
        female_mask = gender_array == 'female'
        
        stats_list = []
        n_features = features_np.shape[1]
        
        for i in tqdm(range(n_features), desc="Computing feature stats"):
            feat = features_np[:, i]
            
            # Basic stats
            mean_all = feat.mean()
            std_all = feat.std()
            sparsity = (feat < 0.01).mean()
            
            # Gender-specific stats
            if male_mask.sum() > 0 and female_mask.sum() > 0:
                mean_male = feat[male_mask].mean()
                mean_female = feat[female_mask].mean()
                
                # T-test for significance
                t_stat, p_value = stats.ttest_ind(
                    feat[male_mask], 
                    feat[female_mask],
                    equal_var=False
                )
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (feat[male_mask].std()**2 + feat[female_mask].std()**2) / 2
                )
                effect_size = (mean_male - mean_female) / (pooled_std + 1e-8)
            else:
                mean_male = mean_female = np.nan
                t_stat = p_value = effect_size = np.nan
            
            stats_list.append({
                'feature_idx': i,
                'mean': mean_all,
                'std': std_all,
                'sparsity': sparsity,
                'mean_male': mean_male,
                'mean_female': mean_female,
                'gender_diff': mean_male - mean_female if not np.isnan(mean_male) else np.nan,
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
            })
        
        return pd.DataFrame(stats_list)
    
    def identify_gender_features(self, stats_df: pd.DataFrame, 
                                 top_k: int = 50,
                                 significance: float = 0.05) -> dict:
        """Identify features significantly associated with gender."""
        # Filter by significance
        significant = stats_df[stats_df['p_value'] < significance].copy()
        
        # Sort by effect size
        significant_sorted = significant.sort_values('effect_size', ascending=False)
        
        # Male-associated (positive effect size)
        male_features = significant_sorted[
            significant_sorted['effect_size'] > 0
        ].head(top_k)['feature_idx'].tolist()
        
        # Female-associated (negative effect size)
        female_features = significant_sorted[
            significant_sorted['effect_size'] < 0
        ].tail(top_k)['feature_idx'].tolist()
        
        return {
            'male_associated': male_features,
            'female_associated': female_features,
            'total_significant': len(significant),
            'percent_significant': len(significant) / len(stats_df) * 100,
        }
    
    def train_gender_probe(self, features: torch.Tensor, 
                           genders: list) -> dict:
        """Train linear probe to predict gender from features."""
        # Filter unknown genders
        valid_mask = np.array([g in ['male', 'female'] for g in genders])
        
        X = features.numpy()[valid_mask]
        y = np.array([1 if g == 'male' else 0 for g in genders])[valid_mask]
        
        if len(np.unique(y)) < 2:
            return {'accuracy': np.nan, 'cv_scores': []}
        
        # Train logistic regression with cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        
        # Train final model
        clf.fit(X, y)
        
        # Get feature importances
        importances = np.abs(clf.coef_[0])
        top_features = np.argsort(importances)[::-1][:20]
        
        return {
            'accuracy': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'top_predictive_features': top_features.tolist(),
            'feature_importances': importances,
        }
    
    def analyze_layer(self, layer: int) -> dict:
        """Perform complete analysis for a single layer."""
        logger.info(f"\n{'='*60}\nAnalyzing Layer {layer}\n{'='*60}")
        
        results = {'layer': layer}
        
        try:
            # Load SAE
            sae, history = self.load_sae(layer, 'arabic')
            results['sae_history'] = history
            
            # Load activations
            act_data = self.load_activations(layer, 'arabic', max_samples=2000)
            activations = act_data['activations']
            genders = act_data['genders']
            
            # Extract features
            features = self.extract_sae_features(sae, activations)
            results['feature_shape'] = features.shape
            
            # Compute statistics
            logger.info("Computing feature statistics...")
            stats_df = self.compute_feature_statistics(features, genders)
            results['feature_stats'] = stats_df.describe().to_dict()
            
            # Identify gender features
            gender_features = self.identify_gender_features(stats_df)
            results['gender_features'] = gender_features
            
            # Train gender probe
            logger.info("Training gender probe...")
            probe_results = self.train_gender_probe(features, genders)
            results['probe_results'] = {
                'accuracy': probe_results['accuracy'],
                'accuracy_std': probe_results['accuracy_std'],
            }
            
            # Generate visualizations for this layer
            self._create_layer_visualizations(layer, features, genders, stats_df)
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    f'layer_{layer}/probe_accuracy': probe_results['accuracy'],
                    f'layer_{layer}/significant_features': gender_features['total_significant'],
                    f'layer_{layer}/percent_significant': gender_features['percent_significant'],
                    f'layer_{layer}/sparsity_mean': stats_df['sparsity'].mean(),
                })
            
            logger.info(f"Layer {layer} analysis complete:")
            logger.info(f"  - Probe accuracy: {probe_results['accuracy']:.3f}")
            logger.info(f"  - Significant features: {gender_features['total_significant']}")
            logger.info(f"  - Mean sparsity: {stats_df['sparsity'].mean():.3f}")
            
            # Cleanup
            del sae, activations, features
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error analyzing layer {layer}: {e}")
            results['error'] = str(e)
        
        return results
    
    def _create_layer_visualizations(self, layer: int, features: torch.Tensor,
                                     genders: list, stats_df: pd.DataFrame):
        """Create visualizations for a specific layer."""
        viz_dir = self.viz_dir / f'layer_{layer}'
        viz_dir.mkdir(exist_ok=True)
        
        features_np = features.numpy()
        gender_array = np.array(genders)
        
        # 1. Feature activation distribution by gender
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Mean activation by gender
        male_mask = gender_array == 'male'
        female_mask = gender_array == 'female'
        
        if male_mask.sum() > 0 and female_mask.sum() > 0:
            mean_male = features_np[male_mask].mean(axis=0)
            mean_female = features_np[female_mask].mean(axis=0)
            
            axes[0].scatter(mean_male, mean_female, alpha=0.3, s=5)
            axes[0].plot([0, mean_male.max()], [0, mean_male.max()], 'r--', label='y=x')
            axes[0].set_xlabel('Male Mean Activation')
            axes[0].set_ylabel('Female Mean Activation')
            axes[0].set_title(f'Layer {layer}: Feature Activations by Gender')
            axes[0].legend()
        
        # Effect size distribution
        effect_sizes = stats_df['effect_size'].dropna()
        axes[1].hist(effect_sizes, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--')
        axes[1].set_xlabel("Effect Size (Cohen's d)")
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Layer {layer}: Gender Effect Size Distribution')
        
        # Sparsity distribution
        axes[2].hist(stats_df['sparsity'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[2].set_xlabel('Sparsity (fraction near zero)')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'Layer {layer}: Feature Sparsity Distribution')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
        if self.use_wandb:
            wandb.log({f'layer_{layer}/feature_distributions': wandb.Image(fig)})
        plt.close()
        
        # 2. t-SNE visualization
        logger.info(f"Creating t-SNE visualization for layer {layer}...")
        valid_mask = np.isin(gender_array, ['male', 'female'])
        if valid_mask.sum() > 100:
            # Sample for speed
            sample_size = min(1000, valid_mask.sum())
            valid_indices = np.where(valid_mask)[0]
            sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
            
            features_sample = features_np[sample_indices]
            genders_sample = gender_array[sample_indices]
            
            # Apply PCA first for speed
            pca = PCA(n_components=50)
            features_pca = pca.fit_transform(features_sample)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings = tsne.fit_transform(features_pca)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            for gender, color in [('male', self.colors['male']), 
                                  ('female', self.colors['female'])]:
                mask = genders_sample == gender
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                          c=color, label=gender, alpha=0.6, s=20)
            
            ax.set_title(f'Layer {layer}: t-SNE of SAE Features by Gender')
            ax.legend()
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            
            plt.savefig(viz_dir / 'tsne_gender.png', dpi=150, bbox_inches='tight')
            if self.use_wandb:
                wandb.log({f'layer_{layer}/tsne_gender': wandb.Image(fig)})
            plt.close()
        
        # 3. Top gender-associated features
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top male features
        top_male = stats_df.nlargest(20, 'effect_size')
        axes[0].barh(range(len(top_male)), top_male['effect_size'], color=self.colors['male'])
        axes[0].set_yticks(range(len(top_male)))
        axes[0].set_yticklabels([f"Feature {i}" for i in top_male['feature_idx']])
        axes[0].set_xlabel("Effect Size (Cohen's d)")
        axes[0].set_title(f'Layer {layer}: Top Male-Associated Features')
        
        # Top female features
        top_female = stats_df.nsmallest(20, 'effect_size')
        axes[1].barh(range(len(top_female)), -top_female['effect_size'], color=self.colors['female'])
        axes[1].set_yticks(range(len(top_female)))
        axes[1].set_yticklabels([f"Feature {i}" for i in top_female['feature_idx']])
        axes[1].set_xlabel("Effect Size (|Cohen's d|)")
        axes[1].set_title(f'Layer {layer}: Top Female-Associated Features')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'top_gender_features.png', dpi=150, bbox_inches='tight')
        if self.use_wandb:
            wandb.log({f'layer_{layer}/top_gender_features': wandb.Image(fig)})
        plt.close()
    
    def create_summary_visualizations(self):
        """Create cross-layer summary visualizations."""
        logger.info("Creating summary visualizations...")
        
        # Collect metrics across layers
        layers = []
        probe_accuracies = []
        significant_percents = []
        sparsities = []
        
        for layer, result in self.results.items():
            if 'error' not in result:
                layers.append(layer)
                probe_accuracies.append(result.get('probe_results', {}).get('accuracy', np.nan))
                significant_percents.append(
                    result.get('gender_features', {}).get('percent_significant', np.nan)
                )
                sparsities.append(
                    result.get('feature_stats', {}).get('sparsity', {}).get('mean', np.nan)
                )
        
        if not layers:
            logger.warning("No layers to summarize")
            return
        
        # 1. Layer-wise comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Probe accuracy
        axes[0].bar(layers, probe_accuracies, color='steelblue', edgecolor='black')
        axes[0].axhline(0.5, color='red', linestyle='--', label='Chance')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Gender Probe Accuracy by Layer')
        axes[0].legend()
        axes[0].set_ylim(0.4, 1.0)
        
        # Significant features
        axes[1].bar(layers, significant_percents, color='coral', edgecolor='black')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('% Significant')
        axes[1].set_title('Gender-Significant Features by Layer')
        
        # Sparsity
        axes[2].bar(layers, sparsities, color='green', edgecolor='black')
        axes[2].set_xlabel('Layer')
        axes[2].set_ylabel('Mean Sparsity')
        axes[2].set_title('Feature Sparsity by Layer')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'layer_comparison.png', dpi=150, bbox_inches='tight')
        if self.use_wandb:
            wandb.log({'summary/layer_comparison': wandb.Image(fig)})
        plt.close()
        
        # 2. Heatmap of gender encoding across layers
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data_matrix = np.array([probe_accuracies, significant_percents, sparsities])
        row_labels = ['Probe Accuracy', '% Significant Features', 'Mean Sparsity']
        
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=layers, yticklabels=row_labels, ax=ax)
        ax.set_xlabel('Layer')
        ax.set_title('Gender Encoding Metrics Across Layers')
        
        plt.savefig(self.viz_dir / 'layer_heatmap.png', dpi=150, bbox_inches='tight')
        if self.use_wandb:
            wandb.log({'summary/layer_heatmap': wandb.Image(fig)})
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'layers_analyzed': list(self.results.keys()),
            'results': {},
        }
        
        # Add per-layer results
        for layer, result in self.results.items():
            report['results'][f'layer_{layer}'] = {
                'probe_accuracy': result.get('probe_results', {}).get('accuracy'),
                'gender_features': result.get('gender_features', {}),
                'feature_shape': result.get('feature_shape', []),
            }
        
        # Summary statistics
        accuracies = [r.get('probe_results', {}).get('accuracy', np.nan) 
                     for r in self.results.values()]
        valid_accuracies = [a for a in accuracies if not np.isnan(a)]
        
        report['summary'] = {
            'total_layers': len(self.results),
            'mean_probe_accuracy': np.mean(valid_accuracies) if valid_accuracies else None,
            'max_probe_accuracy': max(valid_accuracies) if valid_accuracies else None,
            'best_layer': list(self.results.keys())[np.argmax(accuracies)] if valid_accuracies else None,
        }
        
        # Save report
        report_path = self.results_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        
        # Create markdown report
        self._create_markdown_report(report)
        
        return report
    
    def _create_markdown_report(self, report: dict):
        """Create human-readable markdown report."""
        md_content = f"""# SAE Feature Analysis Report
**Generated**: {report['timestamp']}

## Summary

| Metric | Value |
|--------|-------|
| Layers Analyzed | {report['summary']['total_layers']} |
| Mean Probe Accuracy | {report['summary']['mean_probe_accuracy']:.3f if report['summary']['mean_probe_accuracy'] else 'N/A'} |
| Best Layer | {report['summary']['best_layer']} |
| Max Accuracy | {report['summary']['max_probe_accuracy']:.3f if report['summary']['max_probe_accuracy'] else 'N/A'} |

## Layer-by-Layer Results

"""
        for layer in sorted(self.results.keys()):
            result = self.results[layer]
            acc = result.get('probe_results', {}).get('accuracy', 'N/A')
            gender_feats = result.get('gender_features', {})
            
            md_content += f"""### Layer {layer}

- **Probe Accuracy**: {acc:.3f if isinstance(acc, float) else acc}
- **Significant Gender Features**: {gender_feats.get('total_significant', 'N/A')} ({gender_feats.get('percent_significant', 'N/A'):.1f}%)
- **Male-Associated Features**: {len(gender_feats.get('male_associated', []))}
- **Female-Associated Features**: {len(gender_feats.get('female_associated', []))}

"""
        
        md_content += """## Visualizations

See the `visualizations/` directory for:
- `layer_comparison.png` - Cross-layer metrics comparison
- `layer_heatmap.png` - Gender encoding heatmap
- `layer_X/` - Per-layer visualizations

## Methodology

1. **Feature Extraction**: SAE hidden activations extracted from model activations
2. **Statistical Analysis**: T-tests comparing male vs female feature activations
3. **Probe Training**: Logistic regression classifier with 5-fold cross-validation
4. **Visualization**: t-SNE embeddings, effect size distributions, feature rankings
"""
        
        report_path = self.results_dir / 'ANALYSIS_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report saved to {report_path}")
    
    def run(self):
        """Run the complete analysis pipeline."""
        logger.info("="*60)
        logger.info("Starting Full Analysis Pipeline")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Analyze each layer
        for layer in self.layers:
            try:
                result = self.analyze_layer(layer)
                self.results[layer] = result
            except FileNotFoundError as e:
                logger.warning(f"Skipping layer {layer}: {e}")
                continue
        
        # Create summary visualizations
        self.create_summary_visualizations()
        
        # Generate report
        report = self.generate_report()
        
        # Final logging
        elapsed = datetime.now() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Analysis Complete!")
        logger.info(f"Elapsed time: {elapsed}")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"Visualizations saved to: {self.viz_dir}")
        logger.info(f"{'='*60}")
        
        if self.use_wandb:
            wandb.log({'total_runtime_seconds': elapsed.total_seconds()})
            wandb.finish()
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Full SAE Analysis Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Specific layers to analyze (default: all)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')
    
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override W&B if requested
    if args.no_wandb:
        config['logging']['use_wandb'] = False
    
    # Initialize and run pipeline
    pipeline = AnalysisPipeline(config, device=args.device)
    
    if args.layers:
        pipeline.layers = args.layers
    
    pipeline.run()


if __name__ == '__main__':
    main()
