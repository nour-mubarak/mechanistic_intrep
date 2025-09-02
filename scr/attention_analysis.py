"""
Attention Analysis and Circuit Visualization Module
=================================================

This module provides specialized tools for analyzing attention patterns and
visualizing neural circuits in transformer models, with a focus on gender
bias detection in multilingual image captioning.

Key Features:
- Multi-head attention pattern analysis
- Circuit discovery and visualization
- Causal intervention analysis
- Gender-specific attention probing
- Cross-lingual attention comparison
- Interactive attention flow diagrams

Example usage:
    from attention_analysis import AttentionAnalyzer
    
    analyzer = AttentionAnalyzer()
    analyzer.analyze_gender_attention(attention_weights, tokens, gender_labels)
    analyzer.visualize_circuits(circuit_activations)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import torch
from pathlib import Path
import json

class AttentionAnalyzer:
    """Specialized analyzer for attention patterns and circuit discovery."""
    
    def __init__(self, output_dir: str = "attention_analysis"):
        """Initialize the attention analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Gender-related keywords for analysis
        self.gender_keywords = {
            'male': ['man', 'boy', 'father', 'son', 'brother', 'husband', 'he', 'his', 'him'],
            'female': ['woman', 'girl', 'mother', 'daughter', 'sister', 'wife', 'she', 'her', 'hers'],
            'arabic_male': ['رجل', 'ولد', 'أب', 'ابن', 'أخ', 'زوج', 'هو'],
            'arabic_female': ['امرأة', 'فتاة', 'أم', 'ابنة', 'أخت', 'زوجة', 'هي']
        }
    
    def extract_attention_patterns(self, 
                                 model_outputs: Dict,
                                 layer_indices: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract attention patterns from model outputs.
        
        Args:
            model_outputs: Dictionary containing model outputs with attention weights
            layer_indices: Specific layers to analyze (if None, analyze all)
            
        Returns:
            Dictionary mapping layer names to attention weight arrays
        """
        attention_patterns = {}
        
        if 'attentions' in model_outputs:
            attentions = model_outputs['attentions']
            
            if layer_indices is None:
                layer_indices = list(range(len(attentions)))
            
            for layer_idx in layer_indices:
                if layer_idx < len(attentions):
                    # Average across batch and heads for simplicity
                    attention_weights = attentions[layer_idx].mean(dim=(0, 1)).cpu().numpy()
                    attention_patterns[f'layer_{layer_idx}'] = attention_weights
        
        return attention_patterns
    
    def analyze_gender_attention(self, 
                               attention_weights: np.ndarray,
                               tokens: List[str],
                               gender_label: str,
                               save_path: str = None) -> Dict[str, float]:
        """
        Analyze attention patterns for gender-specific tokens.
        
        Args:
            attention_weights: Attention weight matrix [seq_len, seq_len]
            tokens: List of tokens in the sequence
            gender_label: Gender label for this sample
            save_path: Path to save the analysis
            
        Returns:
            Dictionary containing gender attention statistics
        """
        # Identify gender-related token positions
        male_positions = []
        female_positions = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if any(keyword in token_lower for keyword in self.gender_keywords['male']):
                male_positions.append(i)
            elif any(keyword in token_lower for keyword in self.gender_keywords['female']):
                female_positions.append(i)
        
        # Calculate attention statistics
        stats = {
            'total_tokens': len(tokens),
            'male_token_count': len(male_positions),
            'female_token_count': len(female_positions),
            'gender_label': gender_label
        }
        
        if male_positions:
            male_attention = attention_weights[:, male_positions].mean()
            stats['male_attention_received'] = float(male_attention)
            stats['male_attention_given'] = float(attention_weights[male_positions, :].mean())
        
        if female_positions:
            female_attention = attention_weights[:, female_positions].mean()
            stats['female_attention_received'] = float(female_attention)
            stats['female_attention_given'] = float(attention_weights[female_positions, :].mean())
        
        # Calculate attention bias score
        if male_positions and female_positions:
            male_avg = attention_weights[:, male_positions].mean()
            female_avg = attention_weights[:, female_positions].mean()
            stats['attention_bias_score'] = float(abs(male_avg - female_avg))
        else:
            stats['attention_bias_score'] = 0.0
        
        # Save analysis if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=2)
        
        return stats
    
    def visualize_attention_flow(self, 
                               attention_weights: np.ndarray,
                               tokens: List[str],
                               threshold: float = 0.1,
                               save_path: str = None) -> go.Figure:
        """
        Create an interactive attention flow visualization.
        
        Args:
            attention_weights: Attention weight matrix
            tokens: List of tokens
            threshold: Minimum attention weight to display
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, token in enumerate(tokens):
            G.add_node(i, label=token)
        
        # Add edges based on attention weights
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if attention_weights[i, j] > threshold:
                    G.add_edge(i, j, weight=attention_weights[i, j])
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [tokens[node] for node in G.nodes()]
        
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]]['weight'])
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            name='Attention Flow'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Tokens'
        ))
        
        fig.update_layout(
            title="Attention Flow Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Attention flow visualization - edges represent attention weights above threshold",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.output_dir / 'attention_flow.html')
        
        return fig
    
    def compare_cross_lingual_attention(self, 
                                      english_attention: np.ndarray,
                                      arabic_attention: np.ndarray,
                                      english_tokens: List[str],
                                      arabic_tokens: List[str],
                                      save_path: str = None) -> Dict[str, float]:
        """
        Compare attention patterns between English and Arabic captions.
        
        Args:
            english_attention: Attention weights for English caption
            arabic_attention: Attention weights for Arabic caption
            english_tokens: English tokens
            arabic_tokens: Arabic tokens
            save_path: Path to save the comparison
            
        Returns:
            Dictionary containing comparison statistics
        """
        # Analyze gender attention for both languages
        en_stats = self.analyze_gender_attention(english_attention, english_tokens, 'english')
        ar_stats = self.analyze_gender_attention(arabic_attention, arabic_tokens, 'arabic')
        
        # Calculate cross-lingual differences
        comparison = {
            'english_bias_score': en_stats.get('attention_bias_score', 0),
            'arabic_bias_score': ar_stats.get('attention_bias_score', 0),
            'cross_lingual_bias_difference': abs(
                en_stats.get('attention_bias_score', 0) - 
                ar_stats.get('attention_bias_score', 0)
            ),
            'english_male_attention': en_stats.get('male_attention_received', 0),
            'arabic_male_attention': ar_stats.get('male_attention_received', 0),
            'english_female_attention': en_stats.get('female_attention_received', 0),
            'arabic_female_attention': ar_stats.get('female_attention_received', 0)
        }
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # English attention heatmap
        sns.heatmap(english_attention, ax=ax1, cmap='Blues', cbar_kws={'label': 'Attention Weight'})
        ax1.set_title('English Attention Patterns')
        ax1.set_xlabel('Key Tokens')
        ax1.set_ylabel('Query Tokens')
        
        # Arabic attention heatmap
        sns.heatmap(arabic_attention, ax=ax2, cmap='Reds', cbar_kws={'label': 'Attention Weight'})
        ax2.set_title('Arabic Attention Patterns')
        ax2.set_xlabel('Key Tokens')
        ax2.set_ylabel('Query Tokens')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also save comparison stats
            stats_path = save_path.replace('.png', '_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(comparison, f, indent=2)
        else:
            plt.savefig(self.output_dir / 'cross_lingual_attention.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return comparison
    
    def discover_gender_circuits(self, 
                               activations: Dict[str, np.ndarray],
                               gender_labels: List[str],
                               threshold: float = 0.8) -> Dict[str, Dict]:
        """
        Discover neural circuits associated with gender processing.
        
        Args:
            activations: Dictionary mapping layer names to activation arrays
            gender_labels: Gender labels for each sample
            threshold: Correlation threshold for circuit discovery
            
        Returns:
            Dictionary containing discovered circuits
        """
        circuits = {}
        
        for layer_name, layer_activations in activations.items():
            # Calculate correlations between activations and gender labels
            gender_numeric = [1 if label == 'male' else -1 if label == 'female' else 0 
                            for label in gender_labels]
            
            correlations = []
            for neuron_idx in range(layer_activations.shape[1]):
                neuron_activations = layer_activations[:, neuron_idx]
                correlation = np.corrcoef(neuron_activations, gender_numeric)[0, 1]
                correlations.append(correlation)
            
            correlations = np.array(correlations)
            
            # Find neurons with high correlation to gender
            high_corr_indices = np.where(np.abs(correlations) > threshold)[0]
            
            if len(high_corr_indices) > 0:
                circuits[f'{layer_name}_gender_circuit'] = {
                    'neuron_indices': high_corr_indices.tolist(),
                    'correlations': correlations[high_corr_indices].tolist(),
                    'circuit_size': len(high_corr_indices),
                    'mean_correlation': float(np.mean(np.abs(correlations[high_corr_indices])))
                }
        
        return circuits
    
    def visualize_circuits(self, 
                         circuits: Dict[str, Dict],
                         activations: Dict[str, np.ndarray] = None,
                         save_path: str = None) -> go.Figure:
        """
        Create visualization of discovered neural circuits.
        
        Args:
            circuits: Dictionary containing circuit information
            activations: Optional activation data for detailed visualization
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        # Create subplots for each circuit
        n_circuits = len(circuits)
        fig = make_subplots(
            rows=1, cols=n_circuits,
            subplot_titles=list(circuits.keys()),
            specs=[[{"type": "bar"}] * n_circuits]
        )
        
        for idx, (circuit_name, circuit_data) in enumerate(circuits.items()):
            # Plot correlation strengths
            neuron_indices = circuit_data['neuron_indices']
            correlations = circuit_data['correlations']
            
            colors = ['red' if corr > 0 else 'blue' for corr in correlations]
            
            fig.add_trace(
                go.Bar(
                    x=[f'N{i}' for i in neuron_indices],
                    y=correlations,
                    marker_color=colors,
                    name=f'{circuit_name}',
                    showlegend=False
                ),
                row=1, col=idx+1
            )
            
            fig.update_xaxes(title_text="Neuron Index", row=1, col=idx+1)
            fig.update_yaxes(title_text="Gender Correlation", row=1, col=idx+1)
        
        fig.update_layout(
            title_text="Discovered Gender Processing Circuits",
            height=500,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(self.output_dir / 'gender_circuits.html')
        
        return fig
    
    def intervention_analysis(self, 
                            original_activations: np.ndarray,
                            intervened_activations: np.ndarray,
                            intervention_type: str,
                            save_path: str = None) -> Dict[str, float]:
        """
        Analyze the effects of causal interventions on model activations.
        
        Args:
            original_activations: Original activation patterns
            intervened_activations: Activations after intervention
            intervention_type: Type of intervention performed
            save_path: Path to save the analysis
            
        Returns:
            Dictionary containing intervention analysis results
        """
        # Calculate intervention effects
        activation_diff = intervened_activations - original_activations
        
        analysis = {
            'intervention_type': intervention_type,
            'mean_activation_change': float(np.mean(activation_diff)),
            'max_activation_change': float(np.max(np.abs(activation_diff))),
            'activation_variance_change': float(np.var(intervened_activations) - np.var(original_activations)),
            'affected_neurons_count': int(np.sum(np.abs(activation_diff) > 0.1)),
            'intervention_strength': float(np.linalg.norm(activation_diff))
        }
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original vs intervened activations
        axes[0, 0].hist(original_activations.flatten(), alpha=0.7, label='Original', bins=50)
        axes[0, 0].hist(intervened_activations.flatten(), alpha=0.7, label='Intervened', bins=50)
        axes[0, 0].set_title('Activation Distributions')
        axes[0, 0].legend()
        
        # Activation differences
        axes[0, 1].hist(activation_diff.flatten(), bins=50, color='red', alpha=0.7)
        axes[0, 1].set_title('Activation Changes')
        axes[0, 1].axvline(x=0, color='black', linestyle='--')
        
        # Heatmap of changes
        if activation_diff.ndim == 2:
            sns.heatmap(activation_diff, ax=axes[1, 0], cmap='RdBu_r', center=0)
            axes[1, 0].set_title('Spatial Pattern of Changes')
        
        # Summary statistics
        axes[1, 1].bar(range(len(analysis)), list(analysis.values())[1:])  # Skip intervention_type
        axes[1, 1].set_xticks(range(len(analysis)-1))
        axes[1, 1].set_xticklabels(list(analysis.keys())[1:], rotation=45, ha='right')
        axes[1, 1].set_title('Intervention Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Save analysis data
            stats_path = save_path.replace('.png', '_analysis.json')
            with open(stats_path, 'w') as f:
                json.dump(analysis, f, indent=2)
        else:
            plt.savefig(self.output_dir / f'intervention_{intervention_type}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return analysis


def create_sample_attention_analysis():
    """Create sample attention analysis for demonstration."""
    analyzer = AttentionAnalyzer()
    
    # Mock attention data
    seq_len = 20
    attention_weights = np.random.rand(seq_len, seq_len)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)  # Normalize
    
    tokens = ['USER:', '<image>', 'Describe', 'the', 'person', 'in', 'the', 'image', 
              'ASSISTANT:', 'A', 'woman', 'is', 'standing', 'in', 'a', 'park', 'wearing', 
              'a', 'blue', 'dress']
    
    # Analyze gender attention
    gender_stats = analyzer.analyze_gender_attention(attention_weights, tokens, 'female')
    print("Gender attention analysis:", gender_stats)
    
    # Create attention flow visualization
    flow_fig = analyzer.visualize_attention_flow(attention_weights, tokens)
    
    # Mock circuit discovery
    mock_activations = {
        'layer_0': np.random.randn(100, 512),
        'layer_1': np.random.randn(100, 512)
    }
    mock_labels = ['male'] * 30 + ['female'] * 40 + ['neutral'] * 30
    
    circuits = analyzer.discover_gender_circuits(mock_activations, mock_labels, threshold=0.3)
    print("Discovered circuits:", circuits)
    
    if circuits:
        circuit_fig = analyzer.visualize_circuits(circuits)
    
    return analyzer


if __name__ == "__main__":
    # Create sample analysis
    analyzer = create_sample_attention_analysis()
    print("Sample attention analysis completed!")

