#!/usr/bin/env python3
"""
Script 23: Proper Cross-Lingual Analysis
=========================================

Compares English SAE on English data vs Arabic SAE on Arabic data.
This is the correct methodology for cross-lingual bias comparison.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class ProperCrossLingualAnalyzer:
    """Analyze English and Arabic SAEs on their respective data."""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.checkpoints_dir = self.project_dir / 'checkpoints'
        self.results_dir = self.project_dir / 'results' / 'proper_cross_lingual'
        self.viz_dir = self.project_dir / 'visualizations' / 'proper_cross_lingual'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.layers = [0, 3, 9, 12, 15, 17]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.colors = {
            'english': '#2ecc71',
            'arabic': '#e74c3c',
            'male': '#3498db',
            'female': '#e91e63'
        }
    
    def load_sae(self, layer: int, language: str):
        """Load SAE model."""
        if language == 'english' and layer == 0:
            sae_path = self.checkpoints_dir / 'saes' / f'sae_layer_0.pt'
        else:
            sae_path = self.checkpoints_dir / 'saes' / f'sae_{language}_layer_{layer}.pt'
        
        if not sae_path.exists():
            print(f"SAE not found: {sae_path}")
            return None
        
        sae_data = torch.load(sae_path, map_location='cpu', weights_only=False)
        return sae_data
    
    def load_activations(self, layer: int, language: str):
        """Load activations for a layer/language."""
        act_path = self.checkpoints_dir / 'full_layers_ncc' / 'layer_checkpoints' / f'layer_{layer}_{language}.pt'
        
        if not act_path.exists():
            print(f"Activations not found: {act_path}")
            return None, None, None
        
        data = torch.load(act_path, map_location='cpu', weights_only=False)
        activations = data['activations']
        
        # Handle Arabic format (may have sequence dimension)
        if len(activations.shape) == 3:
            activations = activations.mean(dim=1)
        
        genders = data.get('genders', ['unknown'] * len(activations))
        image_ids = data.get('image_ids', [str(i) for i in range(len(activations))])
        
        return activations, genders, image_ids
    
    def encode_with_sae(self, activations: torch.Tensor, sae_data: dict) -> torch.Tensor:
        """Encode activations through SAE."""
        encoder = sae_data['model_state_dict']['encoder.weight']
        encoder_bias = sae_data['model_state_dict']['encoder.bias']
        
        activations = activations.float()
        features = torch.relu(activations @ encoder.T + encoder_bias)
        return features
    
    def train_gender_probe(self, features: np.ndarray, genders: list) -> dict:
        """Train a linear probe to predict gender from SAE features."""
        # Filter to known genders
        mask = np.array([g in ['male', 'female'] for g in genders])
        if mask.sum() < 100:
            return {'accuracy': np.nan, 'n_samples': mask.sum()}
        
        X = features[mask]
        y = np.array([1 if g == 'male' else 0 for g in genders])[mask]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train with cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        
        # Get feature importance
        clf.fit(X_scaled, y)
        importance = np.abs(clf.coef_[0])
        top_features = np.argsort(importance)[-20:][::-1]
        
        return {
            'accuracy': scores.mean(),
            'std': scores.std(),
            'n_samples': mask.sum(),
            'n_male': (y == 1).sum(),
            'n_female': (y == 0).sum(),
            'top_features': top_features.tolist(),
            'top_importance': importance[top_features].tolist()
        }
    
    def compute_feature_statistics(self, features: np.ndarray, genders: list) -> dict:
        """Compute feature-level gender statistics."""
        male_mask = np.array([g == 'male' for g in genders])
        female_mask = np.array([g == 'female' for g in genders])
        
        if male_mask.sum() < 10 or female_mask.sum() < 10:
            return {}
        
        male_mean = features[male_mask].mean(axis=0)
        female_mean = features[female_mask].mean(axis=0)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (features[male_mask].std(axis=0)**2 + features[female_mask].std(axis=0)**2) / 2
        )
        effect_size = (male_mean - female_mean) / (pooled_std + 1e-8)
        
        # Sparsity
        sparsity = (features < 0.01).mean(axis=0)
        
        return {
            'male_mean': male_mean,
            'female_mean': female_mean,
            'effect_size': effect_size,
            'sparsity': sparsity,
            'overall_sparsity': float((features < 0.01).mean()),
            'mean_activation': float(features.mean()),
            'active_features': int((features > 0.01).any(axis=0).sum())
        }
    
    def compute_cross_lingual_alignment(self, en_features: np.ndarray, ar_features: np.ndarray,
                                         en_genders: list, ar_genders: list) -> dict:
        """Compute cross-lingual feature alignment."""
        # Get gender-biased features for each language
        en_stats = self.compute_feature_statistics(en_features, en_genders)
        ar_stats = self.compute_feature_statistics(ar_features, ar_genders)
        
        if not en_stats or not ar_stats:
            return {'clbas': np.nan}
        
        en_effect = en_stats['effect_size']
        ar_effect = ar_stats['effect_size']
        
        # CLBAS: Cross-Lingual Bias Alignment Score
        # Correlation between effect sizes
        correlation = np.corrcoef(en_effect, ar_effect)[0, 1]
        
        # Top biased features overlap
        en_top = set(np.argsort(np.abs(en_effect))[-100:])
        ar_top = set(np.argsort(np.abs(ar_effect))[-100:])
        overlap = len(en_top & ar_top) / 100
        
        # Mean absolute difference in effect sizes
        mean_diff = np.abs(en_effect - ar_effect).mean()
        
        return {
            'clbas': mean_diff,
            'bias_correlation': float(correlation),
            'top_feature_overlap': float(overlap),
            'en_mean_effect': float(np.abs(en_effect).mean()),
            'ar_mean_effect': float(np.abs(ar_effect).mean())
        }
    
    def analyze_layer(self, layer: int) -> dict:
        """Analyze a single layer for both languages."""
        print(f"\n{'='*50}")
        print(f"Analyzing Layer {layer}")
        print('='*50)
        
        results = {'layer': layer}
        
        # Load SAEs
        en_sae = self.load_sae(layer, 'english')
        ar_sae = self.load_sae(layer, 'arabic')
        
        if en_sae is None or ar_sae is None:
            print(f"Missing SAE for layer {layer}")
            return results
        
        # Load activations
        en_acts, en_genders, en_ids = self.load_activations(layer, 'english')
        ar_acts, ar_genders, ar_ids = self.load_activations(layer, 'arabic')
        
        if en_acts is None or ar_acts is None:
            print(f"Missing activations for layer {layer}")
            return results
        
        print(f"English: {en_acts.shape}, {Counter(en_genders)}")
        print(f"Arabic: {ar_acts.shape}, {Counter(ar_genders)}")
        
        # Encode through SAEs
        print("Encoding through SAEs...")
        en_features = self.encode_with_sae(en_acts, en_sae).numpy()
        ar_features = self.encode_with_sae(ar_acts, ar_sae).numpy()
        
        print(f"English features: {en_features.shape}, sparsity: {(en_features < 0.01).mean():.3f}")
        print(f"Arabic features: {ar_features.shape}, sparsity: {(ar_features < 0.01).mean():.3f}")
        
        # Train gender probes
        print("Training gender probes...")
        en_probe = self.train_gender_probe(en_features, en_genders)
        ar_probe = self.train_gender_probe(ar_features, ar_genders)
        
        print(f"English probe accuracy: {en_probe['accuracy']:.3f} ± {en_probe['std']:.3f}")
        print(f"Arabic probe accuracy: {ar_probe['accuracy']:.3f} ± {ar_probe['std']:.3f}")
        
        results['english'] = {
            'probe_accuracy': en_probe['accuracy'],
            'probe_std': en_probe['std'],
            'n_samples': en_probe['n_samples'],
            'sparsity': float((en_features < 0.01).mean()),
            'top_features': en_probe.get('top_features', [])
        }
        
        results['arabic'] = {
            'probe_accuracy': ar_probe['accuracy'],
            'probe_std': ar_probe['std'],
            'n_samples': ar_probe['n_samples'],
            'sparsity': float((ar_features < 0.01).mean()),
            'top_features': ar_probe.get('top_features', [])
        }
        
        # Cross-lingual alignment
        print("Computing cross-lingual alignment...")
        alignment = self.compute_cross_lingual_alignment(
            en_features, ar_features, en_genders, ar_genders
        )
        results['alignment'] = alignment
        print(f"CLBAS: {alignment['clbas']:.4f}, Bias correlation: {alignment['bias_correlation']:.3f}")
        
        # Feature statistics
        en_stats = self.compute_feature_statistics(en_features, en_genders)
        ar_stats = self.compute_feature_statistics(ar_features, ar_genders)
        results['english']['stats'] = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                        for k, v in en_stats.items() if not isinstance(v, np.ndarray)}
        results['arabic']['stats'] = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                       for k, v in ar_stats.items() if not isinstance(v, np.ndarray)}
        
        # Create visualizations
        self.create_layer_visualization(layer, en_features, ar_features, 
                                        en_genders, ar_genders, en_stats, ar_stats, alignment)
        
        return results
    
    def create_layer_visualization(self, layer: int, en_features: np.ndarray, ar_features: np.ndarray,
                                    en_genders: list, ar_genders: list, en_stats: dict, ar_stats: dict,
                                    alignment: dict):
        """Create visualization for a layer."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Layer {layer} - Cross-Lingual Gender Bias Analysis', fontsize=14, fontweight='bold')
        
        # 1. Effect size comparison
        ax = axes[0, 0]
        if 'effect_size' in en_stats and 'effect_size' in ar_stats:
            ax.scatter(en_stats['effect_size'], ar_stats['effect_size'], alpha=0.3, s=5)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Highlight top biased features
            en_top = np.argsort(np.abs(en_stats['effect_size']))[-20:]
            ar_top = np.argsort(np.abs(ar_stats['effect_size']))[-20:]
            ax.scatter(en_stats['effect_size'][en_top], ar_stats['effect_size'][en_top], 
                      c=self.colors['english'], s=30, label='Top English', alpha=0.8)
            ax.scatter(en_stats['effect_size'][ar_top], ar_stats['effect_size'][ar_top], 
                      c=self.colors['arabic'], s=30, label='Top Arabic', alpha=0.8)
            
            ax.set_xlabel('English Effect Size (Cohen\'s d)')
            ax.set_ylabel('Arabic Effect Size (Cohen\'s d)')
            ax.set_title(f'Gender Effect Size Comparison\n(r={alignment["bias_correlation"]:.3f})')
            ax.legend()
        
        # 2. Accuracy comparison bar chart
        ax = axes[0, 1]
        x = ['English\n(on English data)', 'Arabic\n(on Arabic data)']
        accuracies = [en_stats.get('accuracy', 0.5), ar_stats.get('accuracy', 0.5)]
        # Use probe accuracies from the results
        en_mask = np.array([g in ['male', 'female'] for g in en_genders])
        ar_mask = np.array([g in ['male', 'female'] for g in ar_genders])
        
        bars = ax.bar(x, [0.5, 0.5], color=[self.colors['english'], self.colors['arabic']], alpha=0.7)
        ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title('Gender Classification Accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
        
        # 3. Sparsity distribution
        ax = axes[1, 0]
        en_sparsity = (en_features < 0.01).mean(axis=0)
        ar_sparsity = (ar_features < 0.01).mean(axis=0)
        ax.hist(en_sparsity, bins=50, alpha=0.6, color=self.colors['english'], label='English', density=True)
        ax.hist(ar_sparsity, bins=50, alpha=0.6, color=self.colors['arabic'], label='Arabic', density=True)
        ax.set_xlabel('Feature Sparsity')
        ax.set_ylabel('Density')
        ax.set_title('Feature Sparsity Distribution')
        ax.legend()
        
        # 4. Top features heatmap
        ax = axes[1, 1]
        if 'effect_size' in en_stats and 'effect_size' in ar_stats:
            # Get top 20 most gender-biased features for each language
            en_top20 = np.argsort(np.abs(en_stats['effect_size']))[-20:][::-1]
            ar_top20 = np.argsort(np.abs(ar_stats['effect_size']))[-20:][::-1]
            
            combined = list(set(en_top20) | set(ar_top20))[:30]
            
            data = np.array([en_stats['effect_size'][combined], ar_stats['effect_size'][combined]])
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['English', 'Arabic'])
            ax.set_xlabel('Top Gender-Biased Features')
            ax.set_title('Effect Size Heatmap')
            plt.colorbar(im, ax=ax, label="Cohen's d")
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'layer_{layer}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_summary_visualization(self, all_results: dict):
        """Create summary visualization across all layers."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Cross-Lingual Gender Bias Analysis Summary', fontsize=14, fontweight='bold')
        
        layers = []
        en_accs = []
        ar_accs = []
        clbas_scores = []
        correlations = []
        
        for layer, result in all_results.items():
            if isinstance(layer, int) or layer.isdigit():
                layers.append(int(layer))
                en_accs.append(result.get('english', {}).get('probe_accuracy', np.nan))
                ar_accs.append(result.get('arabic', {}).get('probe_accuracy', np.nan))
                clbas_scores.append(result.get('alignment', {}).get('clbas', np.nan))
                correlations.append(result.get('alignment', {}).get('bias_correlation', np.nan))
        
        # Sort by layer
        sort_idx = np.argsort(layers)
        layers = [layers[i] for i in sort_idx]
        en_accs = [en_accs[i] for i in sort_idx]
        ar_accs = [ar_accs[i] for i in sort_idx]
        clbas_scores = [clbas_scores[i] for i in sort_idx]
        correlations = [correlations[i] for i in sort_idx]
        
        # 1. Accuracy by layer
        ax = axes[0, 0]
        x = np.arange(len(layers))
        width = 0.35
        ax.bar(x - width/2, en_accs, width, label='English', color=self.colors['english'], alpha=0.8)
        ax.bar(x + width/2, ar_accs, width, label='Arabic', color=self.colors['arabic'], alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.set_ylabel('Probe Accuracy')
        ax.set_title('Gender Classification Accuracy by Layer')
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        
        # 2. CLBAS by layer
        ax = axes[0, 1]
        ax.plot(layers, clbas_scores, 'o-', color='purple', linewidth=2, markersize=8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('CLBAS')
        ax.set_title('Cross-Lingual Bias Alignment Score by Layer\n(Lower = More Similar Bias)')
        ax.set_xticks(layers)
        
        # 3. Bias correlation by layer
        ax = axes[1, 0]
        ax.plot(layers, correlations, 's-', color='orange', linewidth=2, markersize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Bias Correlation')
        ax.set_title('Cross-Lingual Bias Pattern Correlation\n(Higher = More Similar Patterns)')
        ax.set_xticks(layers)
        ax.set_ylim(-0.5, 1.0)
        
        # 4. Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary text
        mean_en = np.nanmean(en_accs)
        mean_ar = np.nanmean(ar_accs)
        mean_clbas = np.nanmean(clbas_scores)
        mean_corr = np.nanmean(correlations)
        
        summary_text = f"""
SUMMARY STATISTICS
==================

Mean English Accuracy: {mean_en:.3f}
Mean Arabic Accuracy: {mean_ar:.3f}

Mean CLBAS: {mean_clbas:.4f}
Mean Bias Correlation: {mean_corr:.3f}

Best English Layer: L{layers[np.nanargmax(en_accs)]} ({max(en_accs):.3f})
Best Arabic Layer: L{layers[np.nanargmax(ar_accs)]} ({max(ar_accs):.3f})

Interpretation:
- High accuracy = SAE features encode gender well
- Low CLBAS = Similar bias patterns across languages
- High correlation = Aligned gender representations
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run the full cross-lingual analysis."""
        print("="*60)
        print("PROPER CROSS-LINGUAL ANALYSIS")
        print("English SAE on English Data vs Arabic SAE on Arabic Data")
        print("="*60)
        
        # Initialize W&B
        if WANDB_AVAILABLE:
            wandb.init(
                project='sae-captioning-bias',
                name=f'proper-cross-lingual-{datetime.now().strftime("%Y%m%d_%H%M")}',
                config={'layers': self.layers, 'device': self.device}
            )
        
        all_results = {}
        
        for layer in self.layers:
            result = self.analyze_layer(layer)
            all_results[layer] = result
            
            # Log to W&B
            if WANDB_AVAILABLE and result.get('english'):
                wandb.log({
                    f'layer_{layer}/english_accuracy': result['english'].get('probe_accuracy', np.nan),
                    f'layer_{layer}/arabic_accuracy': result['arabic'].get('probe_accuracy', np.nan),
                    f'layer_{layer}/clbas': result.get('alignment', {}).get('clbas', np.nan),
                    f'layer_{layer}/bias_correlation': result.get('alignment', {}).get('bias_correlation', np.nan),
                })
        
        # Create summary visualization
        print("\nCreating summary visualization...")
        self.create_summary_visualization(all_results)
        
        # Save results
        results_file = self.results_dir / 'cross_lingual_results.json'
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        with open(results_file, 'w') as f:
            json.dump(convert_numpy(all_results), f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        print(f"Visualizations saved to {self.viz_dir}")
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        for layer in self.layers:
            r = all_results.get(layer, {})
            en_acc = r.get('english', {}).get('probe_accuracy', np.nan)
            ar_acc = r.get('arabic', {}).get('probe_accuracy', np.nan)
            clbas = r.get('alignment', {}).get('clbas', np.nan)
            corr = r.get('alignment', {}).get('bias_correlation', np.nan)
            print(f"Layer {layer:2d}: EN={en_acc:.3f}, AR={ar_acc:.3f}, CLBAS={clbas:.4f}, Corr={corr:.3f}")
        
        if WANDB_AVAILABLE:
            wandb.finish()
        
        return all_results


if __name__ == '__main__':
    project_dir = '/home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project'
    analyzer = ProperCrossLingualAnalyzer(project_dir)
    analyzer.run_analysis()
