"""
Visualization Module for SAE Analysis
=====================================

Comprehensive visualization tools for:
- Feature analysis plots
- Cross-lingual comparison
- Steering experiments
- Interactive dashboards
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualizationConfig:
    """Configuration for visualizations."""
    
    def __init__(
        self,
        dpi: int = 150,
        fig_width: float = 12,
        fig_height: float = 8,
        style: str = "seaborn-v0_8-whitegrid",
        english_color: str = "#2ecc71",
        arabic_color: str = "#e74c3c",
        male_color: str = "#3498db",
        female_color: str = "#e91e63",
        save_formats: List[str] = None,
        output_dir: Union[str, Path] = "./visualizations"
    ):
        self.dpi = dpi
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.style = style
        self.english_color = english_color
        self.arabic_color = arabic_color
        self.male_color = male_color
        self.female_color = female_color
        self.save_formats = save_formats or ["png"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')


class SAEVisualization:
    """
    Visualizations for SAE training and analysis.
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def plot_training_curves(
        self,
        training_history: Dict[str, List[float]],
        save_name: str = "training_curves"
    ) -> plt.Figure:
        """
        Plot SAE training curves.
        
        Args:
            training_history: Dictionary with keys like 'loss', 'recon_loss', 'l1_loss'
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.config.fig_width, self.config.fig_height))
        
        # Total loss
        if 'loss' in training_history:
            axes[0, 0].plot(training_history['loss'], color='#2c3e50', linewidth=2)
            axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_yscale('log')
        
        # Reconstruction loss
        if 'recon_loss' in training_history:
            axes[0, 1].plot(training_history['recon_loss'], color='#3498db', linewidth=2)
            axes[0, 1].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].set_yscale('log')
        
        # L1 loss
        if 'l1_loss' in training_history:
            axes[1, 0].plot(training_history['l1_loss'], color='#e74c3c', linewidth=2)
            axes[1, 0].set_title('Sparsity Loss (L1)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('L1 Loss')
        
        # L0 sparsity
        if 'l0_sparsity' in training_history:
            axes[1, 1].plot(training_history['l0_sparsity'], color='#27ae60', linewidth=2)
            axes[1, 1].set_title('L0 Sparsity (Active Features)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Avg Active Features')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_feature_activation_distribution(
        self,
        features: torch.Tensor,
        save_name: str = "feature_distribution"
    ) -> plt.Figure:
        """
        Plot distribution of feature activations.
        
        Args:
            features: Feature activations (n_samples, n_features)
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(self.config.fig_width, 4))
        
        # Per-sample active features
        active_per_sample = (features > 0).float().sum(dim=1).numpy()
        axes[0].hist(active_per_sample, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
        axes[0].axvline(np.mean(active_per_sample), color='#e74c3c', linestyle='--', linewidth=2)
        axes[0].set_title('Active Features per Sample', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Number of Active Features')
        axes[0].set_ylabel('Count')
        
        # Per-feature activation frequency
        freq_per_feature = (features > 0).float().mean(dim=0).numpy()
        axes[1].hist(freq_per_feature, bins=50, color='#27ae60', edgecolor='white', alpha=0.8)
        axes[1].set_title('Feature Activation Frequency', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Fraction of Samples')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_xlim(0, 1)
        
        # Activation magnitude distribution (non-zero only)
        nonzero_acts = features[features > 0].numpy()
        axes[2].hist(nonzero_acts, bins=100, color='#9b59b6', edgecolor='white', alpha=0.8)
        axes[2].set_title('Activation Magnitude (Non-zero)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Activation Value')
        axes[2].set_ylabel('Count')
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_dead_feature_analysis(
        self,
        features: torch.Tensor,
        save_name: str = "dead_features"
    ) -> plt.Figure:
        """
        Analyze dead and near-dead features.
        
        Args:
            features: Feature activations
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.config.fig_width, 4))
        
        # Activation frequency per feature
        freq = (features > 0).float().mean(dim=0).numpy()
        
        # Sort by frequency
        sorted_idx = np.argsort(freq)[::-1]
        sorted_freq = freq[sorted_idx]
        
        # Cumulative plot
        axes[0].plot(range(len(freq)), sorted_freq, color='#3498db', linewidth=2)
        axes[0].fill_between(range(len(freq)), sorted_freq, alpha=0.3, color='#3498db')
        axes[0].axhline(0.01, color='#e74c3c', linestyle='--', label='1% threshold')
        axes[0].set_title('Feature Activation Frequency (Sorted)', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Feature Rank')
        axes[0].set_ylabel('Activation Frequency')
        axes[0].legend()
        
        # Dead feature breakdown
        dead = (freq == 0).sum()
        near_dead = ((freq > 0) & (freq < 0.01)).sum()
        active = (freq >= 0.01).sum()
        
        categories = ['Dead\n(0%)', 'Near-dead\n(<1%)', 'Active\n(≥1%)']
        counts = [dead, near_dead, active]
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        bars = axes[1].bar(categories, counts, color=colors, edgecolor='white', linewidth=2)
        axes[1].set_title('Feature Health Distribution', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Features')
        
        # Add percentage labels
        total = len(freq)
        for bar, count in zip(bars, counts):
            axes[1].text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{count}\n({100*count/total:.1f}%)',
                ha='center', va='bottom', fontsize=10
            )
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure in configured formats."""
        for fmt in self.config.save_formats:
            path = self.config.output_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved figure: {path}")


class CrossLingualVisualization:
    """
    Visualizations for cross-lingual comparison.
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def plot_layer_divergence(
        self,
        layer_overlaps: Dict[int, float],
        save_name: str = "layer_divergence"
    ) -> plt.Figure:
        """
        Plot how feature overlap changes across layers.
        
        Args:
            layer_overlaps: Dict mapping layer -> overlap ratio
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = sorted(layer_overlaps.keys())
        overlaps = [layer_overlaps[l] for l in layers]
        
        # Line plot with markers
        ax.plot(layers, overlaps, marker='o', markersize=10, linewidth=3,
                color='#2c3e50', markerfacecolor='#e74c3c', markeredgecolor='white',
                markeredgewidth=2)
        
        # Fill area under curve
        ax.fill_between(layers, overlaps, alpha=0.2, color='#3498db')
        
        # Add annotations for significant changes
        for i in range(1, len(layers)):
            change = overlaps[i] - overlaps[i-1]
            if abs(change) > 0.1:
                mid_x = (layers[i] + layers[i-1]) / 2
                mid_y = (overlaps[i] + overlaps[i-1]) / 2
                ax.annotate(
                    f'{change:+.2f}',
                    xy=(mid_x, mid_y),
                    fontsize=9,
                    color='#e74c3c' if change < 0 else '#27ae60'
                )
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Cross-Lingual Feature Overlap', fontsize=12)
        ax.set_title('Gender Feature Overlap Across Layers\n(English vs Arabic)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add interpretation zone annotations
        ax.axhline(0.5, color='#95a5a6', linestyle=':', alpha=0.5)
        ax.text(layers[-1], 0.52, 'Moderate overlap', fontsize=9, color='#95a5a6', ha='right')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_feature_comparison_scatter(
        self,
        english_activations: np.ndarray,
        arabic_activations: np.ndarray,
        feature_idx: int,
        gender_labels: List[str],
        save_name: str = "feature_scatter"
    ) -> plt.Figure:
        """
        Scatter plot comparing feature activations between languages.
        
        Args:
            english_activations: English feature activations for specific feature
            arabic_activations: Arabic feature activations for specific feature
            feature_idx: Index of the feature being plotted
            gender_labels: Gender labels for each sample
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create color map for genders
        colors = [
            self.config.male_color if g == 'male' else 
            self.config.female_color if g == 'female' else '#95a5a6'
            for g in gender_labels
        ]
        
        # Scatter plot
        ax.scatter(english_activations, arabic_activations, c=colors,
                  alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # Add diagonal line
        max_val = max(english_activations.max(), arabic_activations.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2)
        
        # Compute and display correlation
        corr = np.corrcoef(english_activations, arabic_activations)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontweight='bold')
        
        ax.set_xlabel('English Activation', fontsize=12)
        ax.set_ylabel('Arabic Activation', fontsize=12)
        ax.set_title(f'Feature {feature_idx} Cross-Lingual Comparison',
                    fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.config.male_color, label='Male'),
            mpatches.Patch(color=self.config.female_color, label='Female'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_gender_feature_heatmap(
        self,
        english_stats: List[Dict],
        arabic_stats: List[Dict],
        top_k: int = 50,
        save_name: str = "gender_heatmap"
    ) -> plt.Figure:
        """
        Heatmap of gender effect sizes for top features.
        
        Args:
            english_stats: Feature statistics for English
            arabic_stats: Feature statistics for Arabic
            top_k: Number of top features to show
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        # Get top features by absolute effect size (from English)
        en_effects = [(s['feature_idx'], s.get('gender_effect_size', 0) or 0) 
                      for s in english_stats]
        top_features = sorted(en_effects, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        feature_indices = [f[0] for f in top_features]
        
        # Build heatmap data
        en_dict = {s['feature_idx']: s.get('gender_effect_size', 0) or 0 for s in english_stats}
        ar_dict = {s['feature_idx']: s.get('gender_effect_size', 0) or 0 for s in arabic_stats}
        
        data = np.array([
            [en_dict.get(idx, 0) for idx in feature_indices],
            [ar_dict.get(idx, 0) for idx in feature_indices]
        ])
        
        fig, ax = plt.subplots(figsize=(self.config.fig_width, 4))
        
        # Create heatmap
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Labels
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['English', 'Arabic'])
        ax.set_xlabel('Feature Index (sorted by English effect size)')
        ax.set_title('Gender Effect Size by Language\n(Blue = Female, Red = Male)',
                    fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label("Cohen's d", rotation=270, labelpad=15)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_embedding_space(
        self,
        embeddings: np.ndarray,
        language_labels: List[str],
        gender_labels: List[str],
        save_name: str = "embedding_space"
    ) -> plt.Figure:
        """
        2D embedding visualization with language and gender coloring.
        
        Args:
            embeddings: 2D embeddings (n_samples*2, 2)
            language_labels: Language for each point
            gender_labels: Gender labels (repeated for both languages)
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.config.fig_width, 5))
        
        # Color by language
        colors_lang = [
            self.config.english_color if l == 'english' else self.config.arabic_color
            for l in language_labels
        ]
        axes[0].scatter(embeddings[:, 0], embeddings[:, 1], c=colors_lang,
                       alpha=0.5, s=20)
        axes[0].set_title('Colored by Language', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        
        # Legend for language
        legend_elements = [
            mpatches.Patch(color=self.config.english_color, label='English'),
            mpatches.Patch(color=self.config.arabic_color, label='Arabic'),
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')
        
        # Color by gender (using doubled labels)
        n_per_lang = len(embeddings) // 2
        full_genders = gender_labels + gender_labels  # Duplicate for both languages
        colors_gender = [
            self.config.male_color if g == 'male' else
            self.config.female_color if g == 'female' else '#95a5a6'
            for g in full_genders
        ]
        axes[1].scatter(embeddings[:, 0], embeddings[:, 1], c=colors_gender,
                       alpha=0.5, s=20)
        axes[1].set_title('Colored by Gender', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
        
        # Legend for gender
        legend_elements = [
            mpatches.Patch(color=self.config.male_color, label='Male'),
            mpatches.Patch(color=self.config.female_color, label='Female'),
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure in configured formats."""
        for fmt in self.config.save_formats:
            path = self.config.output_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved figure: {path}")


class SteeringVisualization:
    """
    Visualizations for steering experiments.
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
    
    def plot_steering_effect(
        self,
        steering_results: Dict[str, Any],
        save_name: str = "steering_effect"
    ) -> plt.Figure:
        """
        Plot the effect of steering on gender bias.
        
        Args:
            steering_results: Results from steering experiments
            save_name: Name for saved figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(self.config.fig_width, 4))
        
        strengths = steering_results.get('strengths', [-2, -1, 0, 1, 2])
        
        # Accuracy by steering strength
        if 'accuracy_by_strength' in steering_results:
            axes[0].plot(strengths, steering_results['accuracy_by_strength'],
                        marker='o', linewidth=2, markersize=8, color='#3498db')
            axes[0].axhline(steering_results['accuracy_by_strength'][len(strengths)//2],
                           linestyle='--', color='#95a5a6', alpha=0.5)
            axes[0].set_xlabel('Steering Strength')
            axes[0].set_ylabel('Gender Accuracy')
            axes[0].set_title('Accuracy vs Steering', fontsize=11, fontweight='bold')
        
        # Bias score by steering strength
        if 'bias_by_strength' in steering_results:
            axes[1].plot(strengths, steering_results['bias_by_strength'],
                        marker='s', linewidth=2, markersize=8, color='#e74c3c')
            axes[1].axhline(0, linestyle='--', color='#95a5a6', alpha=0.5)
            axes[1].set_xlabel('Steering Strength')
            axes[1].set_ylabel('Bias Score')
            axes[1].set_title('Bias vs Steering', fontsize=11, fontweight='bold')
            axes[1].set_ylim(-1, 1)
        
        # Before/After comparison for best steering
        if 'best_steering' in steering_results:
            best = steering_results['best_steering']
            categories = ['Original', 'Steered']
            accuracy = [best['original_accuracy'], best['steered_accuracy']]
            
            bars = axes[2].bar(categories, accuracy, color=['#95a5a6', '#27ae60'],
                              edgecolor='white', linewidth=2)
            axes[2].set_ylabel('Accuracy')
            axes[2].set_title(f'Best Steering (strength={best["strength"]})',
                            fontsize=11, fontweight='bold')
            axes[2].set_ylim(0, 1)
            
            # Add improvement annotation
            improvement = best['steered_accuracy'] - best['original_accuracy']
            axes[2].annotate(
                f'+{improvement:.1%}',
                xy=(1, best['steered_accuracy']),
                xytext=(1.2, best['steered_accuracy']),
                fontsize=12, fontweight='bold',
                color='#27ae60' if improvement > 0 else '#e74c3c'
            )
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure in configured formats."""
        for fmt in self.config.save_formats:
            path = self.config.output_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved figure: {path}")


# ============================================================================
# Interactive Visualizations (Plotly)
# ============================================================================

class InteractiveVisualization:
    """
    Interactive visualizations using Plotly.
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive plots disabled.")
    
    def create_interactive_dashboard(
        self,
        analysis_results: Dict[str, Any],
        save_name: str = "dashboard"
    ) -> Optional[go.Figure]:
        """
        Create interactive dashboard for analysis results.
        
        Args:
            analysis_results: Complete analysis results
            save_name: Name for saved HTML file
            
        Returns:
            Plotly figure or None if Plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Layer Divergence',
                'Feature Effect Sizes',
                'Cross-Lingual Correlation',
                'Sparsity Distribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ]
        )
        
        # Layer divergence
        if 'layer_divergence' in analysis_results:
            layers = sorted(analysis_results['layer_divergence'].keys())
            overlaps = [analysis_results['layer_divergence'][l] for l in layers]
            
            fig.add_trace(
                go.Scatter(
                    x=layers, y=overlaps,
                    mode='lines+markers',
                    name='Overlap Ratio',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=10)
                ),
                row=1, col=1
            )
        
        # Feature effect sizes
        if 'feature_stats' in analysis_results:
            effects = [
                s.get('gender_effect_size', 0) or 0
                for s in analysis_results['feature_stats'][:100]
            ]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(effects))),
                    y=effects,
                    name='Effect Size',
                    marker_color=['#e74c3c' if e > 0 else '#3498db' for e in effects]
                ),
                row=1, col=2
            )
        
        # Cross-lingual correlation
        if 'cross_lingual_comparison' in analysis_results:
            corrs = analysis_results['cross_lingual_comparison'].get('feature_correlations', {})
            if corrs:
                features = list(corrs.keys())[:50]
                correlations = [corrs[f] for f in features]
                
                fig.add_trace(
                    go.Scatter(
                        x=features,
                        y=correlations,
                        mode='markers',
                        name='Correlation',
                        marker=dict(
                            size=10,
                            color=correlations,
                            colorscale='RdBu',
                            showscale=True
                        )
                    ),
                    row=2, col=1
                )
        
        # Sparsity distribution
        if 'feature_stats' in analysis_results:
            sparsities = [s.get('sparsity', 0) for s in analysis_results['feature_stats']]
            
            fig.add_trace(
                go.Histogram(
                    x=sparsities,
                    nbinsx=50,
                    name='Sparsity',
                    marker_color='#27ae60'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text='SAE Cross-Lingual Analysis Dashboard',
            showlegend=True
        )
        
        # Save
        path = self.config.output_dir / f"{save_name}.html"
        fig.write_html(str(path))
        logger.info(f"Saved interactive dashboard: {path}")
        
        return fig
    
    def create_feature_explorer(
        self,
        feature_stats: List[Dict],
        save_name: str = "feature_explorer"
    ) -> Optional[go.Figure]:
        """
        Create interactive feature explorer.
        
        Args:
            feature_stats: List of feature statistics
            save_name: Name for saved HTML file
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Prepare data
        df_data = {
            'Feature': [s['feature_idx'] for s in feature_stats],
            'Sparsity': [s.get('sparsity', 0) for s in feature_stats],
            'Effect Size': [s.get('gender_effect_size', 0) or 0 for s in feature_stats],
            'Mean Activation': [s.get('mean_activation', 0) for s in feature_stats],
            'P-value': [s.get('gender_p_value', 1) or 1 for s in feature_stats],
        }
        
        fig = px.scatter(
            df_data,
            x='Sparsity',
            y='Effect Size',
            size='Mean Activation',
            color='P-value',
            hover_data=['Feature'],
            color_continuous_scale='Viridis_r',
            title='Feature Explorer: Sparsity vs Gender Effect'
        )
        
        fig.update_layout(
            xaxis_title='Activation Frequency (Sparsity)',
            yaxis_title="Gender Effect Size (Cohen's d)",
            coloraxis_colorbar_title='P-value'
        )
        
        # Add significance threshold line
        fig.add_hline(y=0.3, line_dash='dash', line_color='gray',
                     annotation_text='Effect size threshold')
        fig.add_hline(y=-0.3, line_dash='dash', line_color='gray')
        
        # Save
        path = self.config.output_dir / f"{save_name}.html"
        fig.write_html(str(path))
        logger.info(f"Saved feature explorer: {path}")
        
        return fig


def create_all_visualizations(
    analysis_results: Dict[str, Any],
    config: VisualizationConfig
) -> Dict[str, Any]:
    """
    Create all visualizations from analysis results.
    
    Args:
        analysis_results: Complete analysis results
        config: Visualization configuration
        
    Returns:
        Dictionary of created figures
    """
    figures = {}
    
    # SAE visualizations
    sae_viz = SAEVisualization(config)
    
    if 'training_history' in analysis_results:
        figures['training'] = sae_viz.plot_training_curves(
            analysis_results['training_history']
        )
    
    # Cross-lingual visualizations
    cross_viz = CrossLingualVisualization(config)
    
    if 'layer_divergence' in analysis_results:
        figures['layer_divergence'] = cross_viz.plot_layer_divergence(
            analysis_results['layer_divergence']
        )
    
    if 'feature_stats' in analysis_results:
        figures['gender_heatmap'] = cross_viz.plot_gender_feature_heatmap(
            analysis_results.get('english_stats', analysis_results['feature_stats']),
            analysis_results.get('arabic_stats', analysis_results['feature_stats'])
        )
    
    # Steering visualizations
    if 'steering_results' in analysis_results:
        steer_viz = SteeringVisualization(config)
        figures['steering'] = steer_viz.plot_steering_effect(
            analysis_results['steering_results']
        )
    
    # Interactive dashboard
    interactive_viz = InteractiveVisualization(config)
    figures['dashboard'] = interactive_viz.create_interactive_dashboard(analysis_results)
    
    if 'feature_stats' in analysis_results:
        figures['feature_explorer'] = interactive_viz.create_feature_explorer(
            analysis_results['feature_stats']
        )
    
    logger.info(f"Created {len(figures)} visualizations")
    return figures
