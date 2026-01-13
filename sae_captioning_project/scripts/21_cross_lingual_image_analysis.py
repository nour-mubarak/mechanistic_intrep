#!/usr/bin/env python3
"""
Cross-Lingual Analysis with Image Visualization
================================================

This script performs:
1. Cross-lingual comparison (English vs Arabic SAE features)
2. CLBAS (Cross-Lingual Bias Alignment Score) computation
3. Image visualization with predicted vs actual gender
4. Feature importance visualization per sample

Usage:
    python scripts/21_cross_lingual_image_analysis.py --config configs/config.yaml
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
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy import stats
from scipy.optimize import linear_sum_assignment
import gc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sae import SparseAutoencoder, SAEConfig

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class CrossLingualAnalyzer:
    """Cross-lingual SAE feature analysis with image visualization."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.results = {}
        
        # Paths
        self.checkpoint_dir = PROJECT_ROOT / config['paths']['checkpoints']
        self.results_dir = PROJECT_ROOT / config['paths']['results']
        self.viz_dir = PROJECT_ROOT / config['paths']['visualizations']
        self.data_dir = PROJECT_ROOT / config['paths']['data_dir']
        self.image_dir = PROJECT_ROOT / 'data' / 'raw' / 'images'
        
        # Create output directories
        (self.viz_dir / 'cross_lingual').mkdir(parents=True, exist_ok=True)
        (self.viz_dir / 'sample_predictions').mkdir(parents=True, exist_ok=True)
        
        # Colors
        self.colors = {
            'english': '#2ecc71',
            'arabic': '#e74c3c',
            'male': '#3498db',
            'female': '#e91e63',
            'correct': '#27ae60',
            'incorrect': '#c0392b',
        }
        
        # Load captions data
        self.captions_df = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'captions.csv')
        
        # W&B
        self.use_wandb = config['logging'].get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config['logging'].get('wandb_project', 'sae-captioning-bias'),
                name=f"cross_lingual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
                tags=['cross-lingual', 'image-analysis']
            )
    
    def load_sae(self, layer: int, language: str) -> SparseAutoencoder:
        """Load SAE model."""
        if language == 'english' and layer == 0:
            path = self.checkpoint_dir / 'saes' / f'sae_layer_0.pt'
        else:
            path = self.checkpoint_dir / 'saes' / f'sae_{language}_layer_{layer}.pt'
        
        if not path.exists():
            raise FileNotFoundError(f"SAE not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        d_model = checkpoint.get('d_model', 2048)
        d_hidden = checkpoint.get('d_hidden', 16384)
        config = SAEConfig(d_model=d_model, expansion_factor=d_hidden // d_model)
        
        sae = SparseAutoencoder(config)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(self.device)
        sae.eval()
        
        return sae
    
    def load_activations(self, layer: int, language: str, max_samples: int = 500):
        """Load activations with memory management."""
        layer_dir = self.checkpoint_dir / 'full_layers_ncc' / 'layer_checkpoints'
        path = layer_dir / f'layer_{layer}_{language}.pt'
        
        if not path.exists():
            raise FileNotFoundError(f"Activations not found: {path}")
        
        logger.info(f"Loading {language} activations for layer {layer}...")
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        activations = data.get('activations', data.get('layer_activations'))
        genders = data.get('genders', [])
        image_ids = data.get('image_ids', [])
        
        # Sample
        if len(activations) > max_samples:
            indices = torch.randperm(len(activations))[:max_samples]
            activations = activations[indices].clone()
            genders = [genders[i] for i in indices.tolist()]
            image_ids = [image_ids[i] for i in indices.tolist()] if image_ids else []
        
        del data
        gc.collect()
        
        return activations, genders, image_ids
    
    def extract_features(self, sae, activations, batch_size=256):
        """Extract SAE features."""
        if len(activations.shape) == 3:
            activations = activations.mean(dim=1)
        
        all_features = []
        with torch.no_grad():
            for i in range(0, len(activations), batch_size):
                batch = activations[i:i+batch_size].to(self.device)
                features = sae.encode(batch)
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def train_gender_classifier(self, features, genders):
        """Train gender classifier and return model + predictions."""
        valid_mask = np.array([g in ['male', 'female'] for g in genders])
        X = features.numpy()[valid_mask]
        y = np.array([1 if g == 'male' else 0 for g in genders])[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(np.unique(y)) < 2:
            return None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, valid_indices, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        return clf, {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'idx_test': idx_test,
            'accuracy': accuracy,
        }
    
    def compute_clbas(self, en_features, ar_features, en_genders, ar_genders):
        """Compute Cross-Lingual Bias Alignment Score (CLBAS)."""
        logger.info("Computing CLBAS...")
        
        # Get gender-specific stats for both languages
        def compute_gender_bias(features, genders):
            features_np = features.numpy()
            gender_array = np.array(genders)
            male_mask = gender_array == 'male'
            female_mask = gender_array == 'female'
            
            if male_mask.sum() == 0 or female_mask.sum() == 0:
                return np.zeros(features_np.shape[1])
            
            mean_male = features_np[male_mask].mean(axis=0)
            mean_female = features_np[female_mask].mean(axis=0)
            
            # Bias = difference in means (positive = male-associated)
            bias = mean_male - mean_female
            return bias
        
        en_bias = compute_gender_bias(en_features, en_genders)
        ar_bias = compute_gender_bias(ar_features, ar_genders)
        
        # Align features using cosine similarity
        en_norm = en_features.numpy().T  # [features, samples]
        ar_norm = ar_features.numpy().T
        
        # Compute mean feature vectors
        en_mean = en_features.numpy().mean(axis=0)
        ar_mean = ar_features.numpy().mean(axis=0)
        
        # Normalize
        en_mean_norm = en_mean / (np.linalg.norm(en_mean) + 1e-8)
        ar_mean_norm = ar_mean / (np.linalg.norm(ar_mean) + 1e-8)
        
        # Cosine similarity between feature biases
        similarity = np.abs(np.corrcoef(en_bias, ar_bias)[0, 1])
        
        # CLBAS: weighted difference in bias
        # Low = same stereotypes, High = different stereotypes
        bias_diff = np.abs(en_bias - ar_bias)
        clbas = np.mean(bias_diff * similarity)
        
        return {
            'clbas': float(clbas),
            'bias_correlation': float(np.corrcoef(en_bias, ar_bias)[0, 1]),
            'en_bias_magnitude': float(np.abs(en_bias).mean()),
            'ar_bias_magnitude': float(np.abs(ar_bias).mean()),
            'similarity': float(similarity),
        }
    
    def visualize_sample_predictions(self, clf, test_results, genders, image_ids, 
                                     features, layer, language, num_samples=16):
        """Visualize individual samples with predictions."""
        logger.info(f"Creating sample prediction visualizations...")
        
        viz_dir = self.viz_dir / 'sample_predictions' / f'layer_{layer}_{language}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        idx_test = test_results['idx_test']
        y_test = test_results['y_test']
        y_pred = test_results['y_pred']
        y_prob = test_results['y_prob']
        
        # Get samples: 4 correct male, 4 correct female, 4 wrong male, 4 wrong female
        correct_mask = y_test == y_pred
        male_mask = y_test == 1
        female_mask = y_test == 0
        
        sample_indices = []
        categories = [
            ('Correct Male', correct_mask & male_mask),
            ('Correct Female', correct_mask & female_mask),
            ('Incorrect (Actual Male)', ~correct_mask & male_mask),
            ('Incorrect (Actual Female)', ~correct_mask & female_mask),
        ]
        
        for cat_name, mask in categories:
            cat_indices = np.where(mask)[0][:4]
            for i in cat_indices:
                sample_indices.append((i, cat_name))
        
        if not sample_indices:
            logger.warning("No samples to visualize")
            return
        
        # Create grid visualization
        n_samples = min(len(sample_indices), 16)
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(16, 4 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
        
        for idx, (sample_idx, category) in enumerate(sample_indices[:n_samples]):
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
            
            # Get original index and image
            orig_idx = idx_test[sample_idx]
            
            if image_ids and orig_idx < len(image_ids):
                img_name = image_ids[orig_idx]
                img_path = self.image_dir / img_name
                
                if img_path.exists():
                    try:
                        img = mpimg.imread(str(img_path))
                        ax.imshow(img)
                    except:
                        ax.text(0.5, 0.5, 'Image\nLoad Error', ha='center', va='center',
                               transform=ax.transAxes, fontsize=12)
                else:
                    ax.text(0.5, 0.5, f'Image\nNot Found', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, f'Sample {orig_idx}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
            
            ax.axis('off')
            
            # Add prediction info
            actual = 'Male' if y_test[sample_idx] == 1 else 'Female'
            predicted = 'Male' if y_pred[sample_idx] == 1 else 'Female'
            confidence = y_prob[sample_idx].max() * 100
            
            is_correct = y_test[sample_idx] == y_pred[sample_idx]
            color = self.colors['correct'] if is_correct else self.colors['incorrect']
            
            title = f"{category}\nActual: {actual} | Pred: {predicted}\nConf: {confidence:.1f}%"
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        
        plt.suptitle(f'Sample Predictions - Layer {layer} ({language.title()})', 
                    fontsize=14, fontweight='bold')
        
        plt.savefig(viz_dir / 'sample_grid.png', dpi=150, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({f'{language}/sample_predictions_layer_{layer}': wandb.Image(fig)})
        
        plt.close()
        
        # Create detailed view for misclassified samples
        self._create_detailed_misclassifications(
            clf, test_results, genders, image_ids, features, layer, language, viz_dir
        )
    
    def _create_detailed_misclassifications(self, clf, test_results, genders, 
                                            image_ids, features, layer, language, viz_dir):
        """Create detailed visualization of misclassified samples."""
        idx_test = test_results['idx_test']
        y_test = test_results['y_test']
        y_pred = test_results['y_pred']
        y_prob = test_results['y_prob']
        
        # Find misclassified
        wrong_mask = y_test != y_pred
        wrong_indices = np.where(wrong_mask)[0][:8]
        
        if len(wrong_indices) == 0:
            return
        
        # Get feature importances
        importances = np.abs(clf.coef_[0])
        top_features = np.argsort(importances)[::-1][:10]
        
        fig, axes = plt.subplots(len(wrong_indices), 2, figsize=(14, 3*len(wrong_indices)))
        if len(wrong_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_idx in enumerate(wrong_indices):
            orig_idx = idx_test[sample_idx]
            
            # Image
            ax_img = axes[i, 0]
            if image_ids and orig_idx < len(image_ids):
                img_name = image_ids[orig_idx]
                img_path = self.image_dir / img_name
                
                if img_path.exists():
                    try:
                        img = mpimg.imread(str(img_path))
                        ax_img.imshow(img)
                    except:
                        ax_img.text(0.5, 0.5, 'Error', ha='center', va='center')
                else:
                    ax_img.text(0.5, 0.5, 'Not Found', ha='center', va='center')
            
            ax_img.axis('off')
            
            actual = 'Male' if y_test[sample_idx] == 1 else 'Female'
            predicted = 'Male' if y_pred[sample_idx] == 1 else 'Female'
            ax_img.set_title(f'Actual: {actual} | Predicted: {predicted}\n'
                           f'Confidence: {y_prob[sample_idx].max()*100:.1f}%',
                           fontsize=10, color='red')
            
            # Top feature activations
            ax_feat = axes[i, 1]
            sample_features = test_results['X_test'][sample_idx]
            top_vals = sample_features[top_features]
            
            colors = [self.colors['male'] if clf.coef_[0][f] > 0 else self.colors['female'] 
                     for f in top_features]
            
            ax_feat.barh(range(len(top_features)), top_vals, color=colors)
            ax_feat.set_yticks(range(len(top_features)))
            ax_feat.set_yticklabels([f'F{f}' for f in top_features])
            ax_feat.set_xlabel('Activation')
            ax_feat.set_title('Top Predictive Features', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'misclassified_detail.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_cross_lingual_comparison(self, en_results, ar_results, layer):
        """Create cross-lingual comparison visualizations."""
        logger.info("Creating cross-lingual comparison plots...")
        
        viz_dir = self.viz_dir / 'cross_lingual'
        
        # 1. Accuracy comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Accuracy bars
        accuracies = [en_results['accuracy'], ar_results['accuracy']]
        ax = axes[0]
        bars = ax.bar(['English', 'Arabic'], accuracies, 
                     color=[self.colors['english'], self.colors['arabic']])
        ax.axhline(0.5, color='gray', linestyle='--', label='Chance')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Layer {layer}: Gender Probe Accuracy')
        ax.set_ylim(0.4, 1.0)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{acc:.1%}', ha='center', fontsize=11)
        
        # Confusion matrices
        for idx, (name, results, color) in enumerate([
            ('English', en_results, self.colors['english']),
            ('Arabic', ar_results, self.colors['arabic'])
        ]):
            ax = axes[idx + 1]
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{name} Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'layer_{layer}_comparison.png', dpi=150, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({f'cross_lingual/layer_{layer}_comparison': wandb.Image(fig)})
        
        plt.close()
    
    def visualize_clbas_results(self, clbas_results, layer):
        """Visualize CLBAS results."""
        viz_dir = self.viz_dir / 'cross_lingual'
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # CLBAS components
        ax = axes[0]
        metrics = ['CLBAS', 'Bias Corr.', 'EN Bias', 'AR Bias']
        values = [
            clbas_results['clbas'],
            clbas_results['bias_correlation'],
            clbas_results['en_bias_magnitude'],
            clbas_results['ar_bias_magnitude']
        ]
        colors = ['purple', 'orange', self.colors['english'], self.colors['arabic']]
        
        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylabel('Value')
        ax.set_title(f'Layer {layer}: Cross-Lingual Bias Metrics')
        ax.axhline(0, color='black', linewidth=0.5)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', fontsize=10)
        
        # Interpretation
        ax = axes[1]
        ax.axis('off')
        
        clbas = clbas_results['clbas']
        corr = clbas_results['bias_correlation']
        
        if clbas < 0.1:
            interpretation = "LOW CLBAS: Similar gender stereotypes in both languages"
        elif clbas < 0.3:
            interpretation = "MODERATE CLBAS: Some language-specific stereotypes"
        else:
            interpretation = "HIGH CLBAS: Different gender stereotypes across languages"
        
        if corr > 0.7:
            corr_interp = "Strong positive correlation: biases align across languages"
        elif corr > 0.3:
            corr_interp = "Moderate correlation: partial bias alignment"
        elif corr > -0.3:
            corr_interp = "Weak correlation: largely independent biases"
        else:
            corr_interp = "Negative correlation: opposite biases!"
        
        text = f"""
CLBAS Analysis for Layer {layer}
{'='*40}

CLBAS Score: {clbas:.4f}
{interpretation}

Bias Correlation: {corr:.4f}
{corr_interp}

English Bias Magnitude: {clbas_results['en_bias_magnitude']:.4f}
Arabic Bias Magnitude: {clbas_results['ar_bias_magnitude']:.4f}
"""
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'layer_{layer}_clbas.png', dpi=150, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({
                f'clbas/layer_{layer}': clbas,
                f'clbas/layer_{layer}_correlation': corr,
            })
        
        plt.close()
    
    def analyze_layer(self, layer: int):
        """Run full cross-lingual analysis for a layer."""
        logger.info(f"\n{'='*60}\nCross-Lingual Analysis - Layer {layer}\n{'='*60}")
        
        results = {'layer': layer}
        
        try:
            # Load Arabic (we have activations for Arabic)
            ar_sae = self.load_sae(layer, 'arabic')
            ar_acts, ar_genders, ar_image_ids = self.load_activations(layer, 'arabic')
            ar_features = self.extract_features(ar_sae, ar_acts)
            
            # Train Arabic classifier
            ar_clf, ar_test_results = self.train_gender_classifier(ar_features, ar_genders)
            
            if ar_clf is None:
                logger.warning(f"Could not train Arabic classifier for layer {layer}")
                return results
            
            results['arabic'] = {
                'accuracy': ar_test_results['accuracy'],
                'n_samples': len(ar_genders),
            }
            
            logger.info(f"Arabic probe accuracy: {ar_test_results['accuracy']:.3f}")
            
            # Visualize Arabic predictions with images
            self.visualize_sample_predictions(
                ar_clf, ar_test_results, ar_genders, ar_image_ids,
                ar_features, layer, 'arabic'
            )
            
            # Try loading English SAE (we may not have English activations)
            try:
                en_sae = self.load_sae(layer, 'english')
                
                # Use same Arabic activations but with English SAE to compare feature spaces
                # This shows how the English SAE encodes the same samples
                en_features = self.extract_features(en_sae, ar_acts)
                
                # Train English classifier
                en_clf, en_test_results = self.train_gender_classifier(en_features, ar_genders)
                
                if en_clf is not None:
                    results['english'] = {
                        'accuracy': en_test_results['accuracy'],
                        'n_samples': len(ar_genders),
                    }
                    logger.info(f"English SAE (on Arabic data) accuracy: {en_test_results['accuracy']:.3f}")
                    
                    # Visualize English predictions
                    self.visualize_sample_predictions(
                        en_clf, en_test_results, ar_genders, ar_image_ids,
                        en_features, layer, 'english'
                    )
                    
                    # Cross-lingual comparison
                    self.visualize_cross_lingual_comparison(en_test_results, ar_test_results, layer)
                    
                    # Compute CLBAS
                    clbas_results = self.compute_clbas(en_features, ar_features, ar_genders, ar_genders)
                    results['clbas'] = clbas_results
                    self.visualize_clbas_results(clbas_results, layer)
                    
                    logger.info(f"CLBAS: {clbas_results['clbas']:.4f}")
                    logger.info(f"Bias correlation: {clbas_results['bias_correlation']:.4f}")
                    
            except FileNotFoundError:
                logger.info(f"No English SAE for layer {layer}")
            
            # Cleanup
            del ar_sae, ar_acts, ar_features
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error analyzing layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def create_summary(self):
        """Create summary visualizations and report."""
        logger.info("Creating summary...")
        
        viz_dir = self.viz_dir / 'cross_lingual'
        
        # Collect results across layers
        layers = []
        ar_accuracies = []
        en_accuracies = []
        clbas_scores = []
        
        for layer, result in self.results.items():
            if 'error' in result:
                continue
            layers.append(layer)
            ar_accuracies.append(result.get('arabic', {}).get('accuracy', np.nan))
            en_accuracies.append(result.get('english', {}).get('accuracy', np.nan))
            clbas_scores.append(result.get('clbas', {}).get('clbas', np.nan))
        
        if not layers:
            return
        
        # Sort by layer
        sort_idx = np.argsort(layers)
        layers = [layers[i] for i in sort_idx]
        ar_accuracies = [ar_accuracies[i] for i in sort_idx]
        en_accuracies = [en_accuracies[i] for i in sort_idx]
        clbas_scores = [clbas_scores[i] for i in sort_idx]
        
        # Summary plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        ax = axes[0]
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ar_accuracies, width, label='Arabic SAE',
                      color=self.colors['arabic'], alpha=0.8)
        bars2 = ax.bar(x + width/2, en_accuracies, width, label='English SAE',
                      color=self.colors['english'], alpha=0.8)
        
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title('Cross-Lingual Gender Probe Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        
        # CLBAS across layers
        ax = axes[1]
        valid_clbas = [(l, c) for l, c in zip(layers, clbas_scores) if not np.isnan(c)]
        if valid_clbas:
            ls, cs = zip(*valid_clbas)
            ax.plot(ls, cs, 'o-', color='purple', linewidth=2, markersize=10)
            ax.fill_between(ls, 0, cs, alpha=0.3, color='purple')
        ax.set_xlabel('Layer')
        ax.set_ylabel('CLBAS Score')
        ax.set_title('Cross-Lingual Bias Alignment Score by Layer')
        ax.set_ylim(0, max(clbas_scores) * 1.2 if any(not np.isnan(c) for c in clbas_scores) else 1)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'summary.png', dpi=150, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({'cross_lingual/summary': wandb.Image(fig)})
        
        plt.close()
        
        # Save results JSON
        report = {
            'timestamp': datetime.now().isoformat(),
            'layers': layers,
            'results': {str(k): v for k, v in self.results.items()},
            'summary': {
                'mean_arabic_accuracy': np.nanmean(ar_accuracies),
                'mean_english_accuracy': np.nanmean(en_accuracies),
                'mean_clbas': np.nanmean(clbas_scores),
            }
        }
        
        with open(self.results_dir / 'cross_lingual_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Summary saved to {self.results_dir / 'cross_lingual_report.json'}")
    
    def run(self, layers=None):
        """Run cross-lingual analysis."""
        if layers is None:
            layers = [0, 3, 9, 12, 15, 17]  # Skip 6 (corrupted)
        
        logger.info("="*60)
        logger.info("Starting Cross-Lingual Analysis")
        logger.info("="*60)
        
        for layer in layers:
            result = self.analyze_layer(layer)
            self.results[layer] = result
        
        self.create_summary()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info("\n" + "="*60)
        logger.info("Cross-Lingual Analysis Complete!")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()
    
    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)
    
    if args.no_wandb:
        config['logging']['use_wandb'] = False
    
    analyzer = CrossLingualAnalyzer(config, device=args.device)
    analyzer.run(layers=args.layers)


if __name__ == '__main__':
    main()
