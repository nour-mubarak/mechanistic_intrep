"""
Enhanced Visualization Module for Mechanistic Interpretability
============================================================

This module provides comprehensive visualization tools for analyzing gender bias
in English-Arabic image captioning models through mechanistic interpretability.
It includes activation visualizations, attention pattern analysis, bias detection
plots, and interactive dashboards.

Key Features:
- Activation heatmaps and distribution plots
- Attention pattern visualizations
- Gender bias analysis charts
- Circuit discovery visualizations
- Interactive dashboards with Plotly
- Cross-lingual comparison tools
- Statistical significance plots

Example usage:
    from visualizations import BiasVisualizer
    
    visualizer = BiasVisualizer()
    visualizer.plot_activation_patterns(activations, labels)
    visualizer.create_bias_dashboard(results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import json

# Set up matplotlib for Arabic text support
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

class BiasVisualizer:
    """Comprehensive visualization toolkit for gender bias analysis."""
    
    def __init__(self, output_dir: str = "visualizations", style: str = "whitegrid"):
        """Initialize the visualizer with output directory and style settings."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting styles
        sns.set_style(style)
        plt.style.use('default')
        
        # Color palettes for different visualizations
        self.gender_colors = {'male': '#1f77b4', 'female': '#ff7f0e', 'neutral': '#2ca02c'}
        self.language_colors = {'english': '#d62728', 'arabic': '#9467bd'}
        self.bias_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
        
    def plot_activation_patterns(self, 
                               activations: Dict[str, np.ndarray], 
                               labels: List[str],
                               sample_ids: List[str] = None,
                               save_path: str = None) -> None:
        """
        Visualize activation patterns across different layers and samples.
        
        Args:
            activations: Dictionary mapping layer names to activation arrays
            labels: Gender labels for each sample
            sample_ids: Optional sample identifiers
            save_path: Path to save the visualization
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(labels))]
            
        # Create subplots for different layers
        n_layers = len(activations)
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_layers > 1 else [axes]
        
        for idx, (layer_name, layer_activations) in enumerate(activations.items()):
            if idx >= len(axes):
                break
                
            # Compute mean activation per sample
            mean_activations = np.mean(layer_activations, axis=1)
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame({
                'activation': mean_activations,
                'gender': labels,
                'sample_id': sample_ids
            })
            
            # Plot activation distribution by gender
            sns.boxplot(data=df, x='gender', y='activation', 
                       palette=self.gender_colors, ax=axes[idx])
            axes[idx].set_title(f'Layer {layer_name} Activations')
            axes[idx].set_ylabel('Mean Activation')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'activation_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_attention_heatmap(self, 
                             attention_weights: np.ndarray,
                             tokens: List[str],
                             layer_idx: int = 0,
                             head_idx: int = 0,
                             save_path: str = None) -> None:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weight matrix [seq_len, seq_len]
            tokens: List of tokens corresponding to sequence positions
            layer_idx: Layer index for labeling
            head_idx: Attention head index for labeling
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(attention_weights, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues', 
                   annot=False,
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f'Attention Patterns - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'attention_heatmap_L{layer_idx}_H{head_idx}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_bias_metrics(self, 
                         metrics_data: Dict[str, Dict[str, float]],
                         save_path: str = None) -> None:
        """
        Visualize bias metrics comparison between baseline and fine-tuned models.
        
        Args:
            metrics_data: Dictionary containing metrics for different models
            save_path: Path to save the visualization
        """
        # Prepare data for plotting
        models = list(metrics_data.keys())
        metrics = ['accuracy', 'male_accuracy', 'female_accuracy', 'bias_gap']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [metrics_data[model].get(metric, 0) for model in models]
            
            bars = axes[idx].bar(models, values, 
                               color=[self.bias_colors[i % len(self.bias_colors)] for i in range(len(models))])
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_ylabel('Score')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'bias_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_bias_dashboard(self, 
                                        results_data: Dict,
                                        save_path: str = None) -> go.Figure:
        """
        Create an interactive dashboard for bias analysis results.
        
        Args:
            results_data: Dictionary containing comprehensive results
            save_path: Path to save the HTML dashboard
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Bias Metrics Over Time', 'Gender Classification Accuracy',
                          'Caption Quality Metrics', 'Activation Distributions',
                          'Cross-lingual Comparison', 'Statistical Significance'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "violin"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Bias metrics over training steps
        if 'training_metrics' in results_data:
            training_data = results_data['training_metrics']
            fig.add_trace(
                go.Scatter(x=training_data['steps'], y=training_data['bias_gap'],
                          name='Bias Gap', line=dict(color='red')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=training_data['steps'], y=training_data['accuracy'],
                          name='Overall Accuracy', line=dict(color='blue')),
                row=1, col=1, secondary_y=True
            )
        
        # 2. Gender classification accuracy
        if 'gender_metrics' in results_data:
            gender_data = results_data['gender_metrics']
            fig.add_trace(
                go.Bar(x=list(gender_data.keys()), y=list(gender_data.values()),
                      name='Gender Accuracy', marker_color='lightblue'),
                row=1, col=2
            )
        
        # 3. Caption quality metrics
        if 'quality_metrics' in results_data:
            quality_data = results_data['quality_metrics']
            metrics = ['BLEU', 'ROUGE-L', 'METEOR']
            baseline_scores = [quality_data.get(f'baseline_{m.lower()}', 0) for m in metrics]
            tuned_scores = [quality_data.get(f'tuned_{m.lower()}', 0) for m in metrics]
            
            fig.add_trace(
                go.Scatter(x=metrics, y=baseline_scores, name='Baseline',
                          mode='markers+lines', marker_size=10),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=metrics, y=tuned_scores, name='Fine-tuned',
                          mode='markers+lines', marker_size=10),
                row=2, col=1
            )
        
        # 4. Activation distributions
        if 'activations' in results_data:
            activation_data = results_data['activations']
            for gender in ['male', 'female', 'neutral']:
                if gender in activation_data:
                    fig.add_trace(
                        go.Violin(y=activation_data[gender], name=f'{gender.title()}',
                                 box_visible=True, meanline_visible=True),
                        row=2, col=2
                    )
        
        # 5. Cross-lingual comparison
        if 'cross_lingual' in results_data:
            cross_data = results_data['cross_lingual']
            languages = ['English', 'Arabic']
            bias_scores = [cross_data.get('en_bias', 0), cross_data.get('ar_bias', 0)]
            
            fig.add_trace(
                go.Bar(x=languages, y=bias_scores, name='Cross-lingual Bias',
                      marker_color=['#d62728', '#9467bd']),
                row=3, col=1
            )
        
        # 6. Statistical significance
        if 'significance_tests' in results_data:
            sig_data = results_data['significance_tests']
            tests = list(sig_data.keys())
            p_values = list(sig_data.values())
            
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            fig.add_trace(
                go.Scatter(x=tests, y=p_values, mode='markers',
                          marker=dict(size=12, color=colors),
                          name='P-values'),
                row=3, col=2
            )
            fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Gender Bias Analysis Dashboard",
            showlegend=True,
            height=1200,
            template="plotly_white"
        )
        
        # Save as HTML if path provided
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.output_dir / 'bias_dashboard.html')
            
        return fig
        
    def plot_circuit_discovery(self, 
                             circuit_data: Dict[str, np.ndarray],
                             feature_names: List[str] = None,
                             save_path: str = None) -> None:
        """
        Visualize discovered circuits and their activations.
        
        Args:
            circuit_data: Dictionary mapping circuit names to activation patterns
            feature_names: Names of features/neurons in the circuit
            save_path: Path to save the visualization
        """
        n_circuits = len(circuit_data)
        fig, axes = plt.subplots(1, n_circuits, figsize=(5*n_circuits, 6))
        
        if n_circuits == 1:
            axes = [axes]
            
        for idx, (circuit_name, activations) in enumerate(circuit_data.items()):
            # Create heatmap of circuit activations
            sns.heatmap(activations.T, 
                       ax=axes[idx],
                       cmap='RdBu_r',
                       center=0,
                       cbar_kws={'label': 'Activation Strength'})
            
            axes[idx].set_title(f'Circuit: {circuit_name}')
            axes[idx].set_xlabel('Samples')
            axes[idx].set_ylabel('Features')
            
            if feature_names:
                axes[idx].set_yticklabels(feature_names[:activations.shape[1]], rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'circuit_discovery.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_intervention_effects(self, 
                                intervention_results: Dict[str, Dict[str, float]],
                                save_path: str = None) -> None:
        """
        Visualize the effects of different interventions on model behavior.
        
        Args:
            intervention_results: Dictionary mapping intervention types to their effects
            save_path: Path to save the visualization
        """
        interventions = list(intervention_results.keys())
        metrics = ['bias_reduction', 'quality_preservation', 'overall_effect']
        
        # Prepare data for grouped bar chart
        data = []
        for intervention in interventions:
            for metric in metrics:
                data.append({
                    'intervention': intervention,
                    'metric': metric,
                    'value': intervention_results[intervention].get(metric, 0)
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='intervention', y='value', hue='metric', 
                   palette='viridis')
        
        plt.title('Intervention Effects on Model Behavior')
        plt.xlabel('Intervention Type')
        plt.ylabel('Effect Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric')
        
        # Add horizontal line at zero for reference
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'intervention_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_cross_lingual_analysis(self, 
                                    english_results: Dict,
                                    arabic_results: Dict,
                                    save_path: str = None) -> go.Figure:
        """
        Create comprehensive cross-lingual bias analysis visualization.
        
        Args:
            english_results: Results for English captions
            arabic_results: Results for Arabic captions
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bias Comparison', 'Quality Metrics',
                          'Gender Distribution', 'Correlation Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "heatmap"}]]
        )
        
        # 1. Bias comparison
        languages = ['English', 'Arabic']
        bias_scores = [english_results.get('bias_score', 0), arabic_results.get('bias_score', 0)]
        
        fig.add_trace(
            go.Bar(x=languages, y=bias_scores, name='Bias Score',
                  marker_color=['#d62728', '#9467bd']),
            row=1, col=1
        )
        
        # 2. Quality metrics scatter
        en_quality = english_results.get('quality_metrics', {})
        ar_quality = arabic_results.get('quality_metrics', {})
        
        metrics = ['bleu', 'rouge', 'meteor']
        en_scores = [en_quality.get(m, 0) for m in metrics]
        ar_scores = [ar_quality.get(m, 0) for m in metrics]
        
        fig.add_trace(
            go.Scatter(x=en_scores, y=ar_scores, mode='markers+text',
                      text=metrics, textposition="top center",
                      marker=dict(size=12, color='blue'),
                      name='Quality Correlation'),
            row=1, col=2
        )
        
        # Add diagonal line for perfect correlation
        max_score = max(max(en_scores), max(ar_scores))
        fig.add_trace(
            go.Scatter(x=[0, max_score], y=[0, max_score],
                      mode='lines', line=dict(dash='dash', color='gray'),
                      name='Perfect Correlation', showlegend=False),
            row=1, col=2
        )
        
        # 3. Gender distribution pie charts
        en_gender_dist = english_results.get('gender_distribution', {})
        ar_gender_dist = arabic_results.get('gender_distribution', {})
        
        # English pie chart
        fig.add_trace(
            go.Pie(labels=list(en_gender_dist.keys()), 
                  values=list(en_gender_dist.values()),
                  name="English", hole=0.3),
            row=2, col=1
        )
        
        # 4. Correlation heatmap (placeholder - would need actual correlation data)
        correlation_matrix = np.random.rand(4, 4)  # Placeholder
        correlation_labels = ['Bias', 'Quality', 'Fluency', 'Accuracy']
        
        fig.add_trace(
            go.Heatmap(z=correlation_matrix, 
                      x=correlation_labels, 
                      y=correlation_labels,
                      colorscale='RdBu', zmid=0),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Cross-lingual Gender Bias Analysis",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.output_dir / 'cross_lingual_analysis.html')
            
        return fig
        
    def save_all_visualizations(self, results_data: Dict) -> None:
        """
        Generate and save all visualization types for a complete analysis.
        
        Args:
            results_data: Comprehensive results dictionary containing all analysis data
        """
        print("Generating comprehensive visualization suite...")
        
        # Create activation patterns if available
        if 'activations' in results_data:
            self.plot_activation_patterns(
                results_data['activations'],
                results_data.get('labels', []),
                save_path=self.output_dir / 'activation_patterns.png'
            )
        
        # Create bias metrics comparison
        if 'bias_metrics' in results_data:
            self.plot_bias_metrics(
                results_data['bias_metrics'],
                save_path=self.output_dir / 'bias_metrics.png'
            )
        
        # Create interactive dashboard
        self.create_interactive_bias_dashboard(
            results_data,
            save_path=self.output_dir / 'interactive_dashboard.html'
        )
        
        # Create circuit discovery visualization
        if 'circuits' in results_data:
            self.plot_circuit_discovery(
                results_data['circuits'],
                save_path=self.output_dir / 'circuits.png'
            )
        
        # Create intervention effects plot
        if 'interventions' in results_data:
            self.plot_intervention_effects(
                results_data['interventions'],
                save_path=self.output_dir / 'interventions.png'
            )
        
        # Create cross-lingual analysis
        if 'english_results' in results_data and 'arabic_results' in results_data:
            self.create_cross_lingual_analysis(
                results_data['english_results'],
                results_data['arabic_results'],
                save_path=self.output_dir / 'cross_lingual.html'
            )
        
        print(f"All visualizations saved to {self.output_dir}")


def create_sample_dashboard():
    """Create a sample dashboard with mock data for demonstration."""
    visualizer = BiasVisualizer()
    
    # Mock results data
    sample_results = {
        'training_metrics': {
            'steps': list(range(0, 1000, 100)),
            'bias_gap': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15],
            'accuracy': [0.6, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
        },
        'gender_metrics': {
            'male_accuracy': 0.85,
            'female_accuracy': 0.72,
            'neutral_accuracy': 0.68,
            'overall_accuracy': 0.75
        },
        'quality_metrics': {
            'baseline_bleu': 0.45,
            'tuned_bleu': 0.43,
            'baseline_rouge-l': 0.52,
            'tuned_rouge-l': 0.51,
            'baseline_meteor': 0.38,
            'tuned_meteor': 0.37
        },
        'activations': {
            'male': np.random.normal(0.5, 0.2, 100),
            'female': np.random.normal(0.3, 0.15, 100),
            'neutral': np.random.normal(0.4, 0.1, 100)
        },
        'cross_lingual': {
            'en_bias': 0.25,
            'ar_bias': 0.18
        },
        'significance_tests': {
            'gender_difference': 0.001,
            'language_difference': 0.023,
            'intervention_effect': 0.008,
            'quality_preservation': 0.156
        }
    }
    
    # Generate dashboard
    fig = visualizer.create_interactive_bias_dashboard(sample_results)
    return fig


if __name__ == "__main__":
    # Create sample visualizations
    create_sample_dashboard()
    print("Sample dashboard created successfully!")

