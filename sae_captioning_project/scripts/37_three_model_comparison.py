#!/usr/bin/env python3
"""
Three-Model Cross-Lingual Comparison
=====================================

Comprehensive comparison of cross-lingual gender bias across three VLMs:
1. PaLiGemma-3B - Native multilingual support (trained on Arabic)
2. Qwen2-VL-7B - Native Arabic support with dedicated tokenizer
3. LLaVA-1.5-7B - Byte-fallback Arabic (UTF-8 token representation)

This script analyzes how each model's multilingual design affects:
- Cross-lingual feature alignment (CLBAS scores)
- Gender representation patterns
- Layer-wise bias distribution
- Feature interpretability

Usage:
    python scripts/37_three_model_comparison.py \\
        --paligemma_results results/cross_lingual_overlap/cross_lingual_overlap_results.json \\
        --qwen2vl_results results/qwen2vl_analysis/cross_lingual_results.json \\
        --llava_results results/llava_analysis/cross_lingual_results.json \\
        --output_dir results/three_model_comparison
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Statistical tests
from scipy import stats


# =============================================================================
# Model Metadata
# =============================================================================

MODEL_INFO = {
    'paligemma': {
        'name': 'PaLiGemma-3B',
        'full_name': 'google/paligemma-3b-pt-224',
        'params': '3B',
        'hidden_dim': 2048,
        'num_layers': 26,
        'arabic_support': 'Native multilingual',
        'tokenizer': 'SentencePiece (multilingual)',
        'color': '#2ecc71',  # Green
        'layers_analyzed': [0, 3, 6, 9, 12, 15, 17],
    },
    'qwen2vl': {
        'name': 'Qwen2-VL-7B',
        'full_name': 'Qwen/Qwen2-VL-7B-Instruct',
        'params': '7B',
        'hidden_dim': 3584,
        'num_layers': 28,
        'arabic_support': 'Native Arabic tokens',
        'tokenizer': 'Custom (151k vocab with Arabic)',
        'color': '#3498db',  # Blue
        'layers_analyzed': [0, 4, 8, 12, 16, 20, 24, 27],
    },
    'llava': {
        'name': 'LLaVA-1.5-7B',
        'full_name': 'llava-hf/llava-1.5-7b-hf',
        'params': '7B',
        'hidden_dim': 4096,
        'num_layers': 32,
        'arabic_support': 'Byte-fallback (UTF-8)',
        'tokenizer': 'SentencePiece (32k + byte tokens)',
        'color': '#e74c3c',  # Red
        'layers_analyzed': [0, 4, 8, 12, 16, 20, 24, 28, 31],
    }
}


# =============================================================================
# Data Loading
# =============================================================================

def load_results(results_path: str, model_key: str) -> Optional[Dict]:
    """Load results JSON file for a model."""
    path = Path(results_path)
    if not path.exists():
        print(f"  Warning: {model_key} results not found at {path}")
        return None
    
    with open(path) as f:
        data = json.load(f)
    
    print(f"  ✓ Loaded {model_key}: {len(data.get('layer_results', data.get('layers', {})))} layers")
    return data


def normalize_layer_index(layer_idx: int, num_layers: int) -> float:
    """Normalize layer index to [0, 1] range for cross-model comparison."""
    return layer_idx / (num_layers - 1)


def extract_metrics(results: Dict, model_key: str) -> pd.DataFrame:
    """Extract key metrics into a standardized DataFrame."""
    info = MODEL_INFO[model_key]
    rows = []
    
    # Handle different result formats
    layer_data = results.get('layer_results', results.get('layers', {}))
    
    for layer_key, layer_results in layer_data.items():
        # Parse layer number
        if isinstance(layer_key, str):
            layer_idx = int(layer_key.replace('layer_', ''))
        else:
            layer_idx = int(layer_key)
        
        # Normalize layer position
        norm_pos = normalize_layer_index(layer_idx, info['num_layers'])
        
        row = {
            'model': model_key,
            'model_name': info['name'],
            'layer': layer_idx,
            'layer_normalized': norm_pos,
            'layer_position': 'early' if norm_pos < 0.33 else ('middle' if norm_pos < 0.67 else 'late'),
            # CLBAS / Feature overlap
            'clbas': layer_results.get('clbas', layer_results.get('feature_overlap', {}).get('clbas', np.nan)),
            'jaccard': layer_results.get('jaccard', layer_results.get('feature_overlap', {}).get('jaccard', np.nan)),
            'cosine_sim': layer_results.get('cosine_similarity', layer_results.get('feature_overlap', {}).get('cosine', np.nan)),
            # Probe accuracy
            'probe_arabic': layer_results.get('probe_arabic', layer_results.get('probes', {}).get('arabic', {}).get('accuracy', np.nan)),
            'probe_english': layer_results.get('probe_english', layer_results.get('probes', {}).get('english', {}).get('accuracy', np.nan)),
            # Feature counts
            'features_arabic': layer_results.get('active_features_arabic', layer_results.get('feature_counts', {}).get('arabic', np.nan)),
            'features_english': layer_results.get('active_features_english', layer_results.get('feature_counts', {}).get('english', np.nan)),
            'shared_features': layer_results.get('shared_features', np.nan),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for each model."""
    summary = {}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        summary[model] = {
            'n_layers': len(model_df),
            'clbas': {
                'mean': model_df['clbas'].mean(),
                'std': model_df['clbas'].std(),
                'min': model_df['clbas'].min(),
                'max': model_df['clbas'].max(),
            },
            'probe_accuracy': {
                'arabic_mean': model_df['probe_arabic'].mean(),
                'english_mean': model_df['probe_english'].mean(),
                'gap': (model_df['probe_english'] - model_df['probe_arabic']).mean(),
            },
            'cosine_sim': {
                'mean': model_df['cosine_sim'].mean(),
                'std': model_df['cosine_sim'].std(),
            }
        }
    
    return summary


def perform_statistical_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical tests comparing models."""
    tests = {}
    
    models = df['model'].unique()
    if len(models) < 2:
        return tests
    
    # Kruskal-Wallis test for CLBAS across models
    groups = [df[df['model'] == m]['clbas'].dropna().values for m in models]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        tests['kruskal_clbas'] = {'H': h_stat, 'p': p_val}
    
    # Pairwise Mann-Whitney U tests
    tests['pairwise'] = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            g1 = df[df['model'] == m1]['clbas'].dropna()
            g2 = df[df['model'] == m2]['clbas'].dropna()
            if len(g1) > 0 and len(g2) > 0:
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                tests['pairwise'][f'{m1}_vs_{m2}'] = {'U': u_stat, 'p': p_val}
    
    return tests


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_clbas_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot CLBAS scores across models with normalized layer positions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Line plot with normalized positions
    ax1 = axes[0]
    for model in df['model'].unique():
        model_df = df[df['model'] == model].sort_values('layer_normalized')
        info = MODEL_INFO[model]
        ax1.plot(model_df['layer_normalized'], model_df['clbas'], 
                 'o-', color=info['color'], label=info['name'], 
                 linewidth=2, markersize=8)
    
    ax1.set_xlabel('Normalized Layer Position (0=first, 1=last)', fontsize=12)
    ax1.set_ylabel('CLBAS Score', fontsize=12)
    ax1.set_title('Cross-Lingual Bias Alignment Score by Layer Depth', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    
    # Right: Box plot comparison
    ax2 = axes[1]
    models = df['model'].unique()
    positions = range(len(models))
    colors = [MODEL_INFO[m]['color'] for m in models]
    
    bp = ax2.boxplot([df[df['model'] == m]['clbas'].dropna() for m in models],
                      positions=positions, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels([MODEL_INFO[m]['name'] for m in models], fontsize=11)
    ax2.set_ylabel('CLBAS Score', fontsize=12)
    ax2.set_title('CLBAS Distribution by Model', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clbas_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'clbas_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved CLBAS comparison plots")


def plot_probe_accuracy_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot gender probe accuracy across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Arabic vs English probe accuracy
    ax1 = axes[0]
    width = 0.35
    models = df['model'].unique()
    x = np.arange(len(models))
    
    arabic_means = [df[df['model'] == m]['probe_arabic'].mean() for m in models]
    english_means = [df[df['model'] == m]['probe_english'].mean() for m in models]
    arabic_stds = [df[df['model'] == m]['probe_arabic'].std() for m in models]
    english_stds = [df[df['model'] == m]['probe_english'].std() for m in models]
    
    bars1 = ax1.bar(x - width/2, arabic_means, width, yerr=arabic_stds,
                    label='Arabic', color='#e74c3c', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, english_means, width, yerr=english_stds,
                    label='English', color='#3498db', alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Probe Accuracy', fontsize=12)
    ax1.set_title('Gender Probe Accuracy by Language', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_INFO[m]['name'] for m in models], fontsize=11)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Accuracy gap (English - Arabic)
    ax2 = axes[1]
    gaps = [e - a for e, a in zip(english_means, arabic_means)]
    colors = [MODEL_INFO[m]['color'] for m in models]
    
    bars = ax2.bar(x, gaps, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy Gap (English - Arabic)', fontsize=12)
    ax2.set_title('Cross-Lingual Probe Accuracy Gap', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_INFO[m]['name'] for m in models], fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax2.annotate(f'{gap:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -10),
                     textcoords="offset points",
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'probe_accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved probe accuracy comparison plots")


def plot_layer_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of metrics by model and layer position."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['clbas', 'probe_arabic', 'probe_english']
    titles = ['CLBAS Score', 'Arabic Probe Accuracy', 'English Probe Accuracy']
    
    for ax, metric, title in zip(axes, metrics, titles):
        # Create pivot table
        pivot_data = df.pivot_table(
            values=metric, 
            index='model', 
            columns='layer_position',
            aggfunc='mean'
        )
        
        # Reorder columns
        col_order = ['early', 'middle', 'late']
        pivot_data = pivot_data[[c for c in col_order if c in pivot_data.columns]]
        
        # Reorder rows by model
        row_order = ['paligemma', 'qwen2vl', 'llava']
        pivot_data = pivot_data.reindex([r for r in row_order if r in pivot_data.index])
        pivot_data.index = [MODEL_INFO[m]['name'] for m in pivot_data.index]
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=ax, cbar_kws={'label': metric.upper()})
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Layer Position')
        ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_position_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'layer_position_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved layer position heatmap")


def plot_comprehensive_dashboard(df: pd.DataFrame, summary: Dict, output_dir: Path):
    """Create a comprehensive dashboard with all key visualizations."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    models = list(MODEL_INFO.keys())
    present_models = [m for m in models if m in df['model'].values]
    
    # 1. Model comparison summary (top-left, wide)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Create summary table
    summary_data = []
    for model in present_models:
        info = MODEL_INFO[model]
        model_df = df[df['model'] == model]
        summary_data.append({
            'Model': info['name'],
            'Parameters': info['params'],
            'Arabic Support': info['arabic_support'],
            'Avg CLBAS': f"{model_df['clbas'].mean():.3f}",
            'Probe Gap': f"{(model_df['probe_english'] - model_df['probe_arabic']).mean():.3f}",
        })
    
    ax1.axis('off')
    table = ax1.table(
        cellText=[[d['Model'], d['Parameters'], d['Arabic Support'], d['Avg CLBAS'], d['Probe Gap']] 
                  for d in summary_data],
        colLabels=['Model', 'Parameters', 'Arabic Support', 'Avg CLBAS', 'Probe Gap'],
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 5
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    ax1.set_title('Three-Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Model architecture comparison (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(present_models))
    width = 0.25
    
    hidden_dims = [MODEL_INFO[m]['hidden_dim'] / 1000 for m in present_models]  # in K
    num_layers = [MODEL_INFO[m]['num_layers'] for m in present_models]
    
    ax2.bar(x - width/2, hidden_dims, width, label='Hidden Dim (K)', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, num_layers, width, label='Num Layers', color='#e74c3c', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_INFO[m]['name'] for m in present_models], fontsize=9)
    ax2.legend()
    ax2.set_title('Model Architecture', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. CLBAS by layer depth (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    for model in present_models:
        model_df = df[df['model'] == model].sort_values('layer_normalized')
        info = MODEL_INFO[model]
        ax3.plot(model_df['layer_normalized'], model_df['clbas'], 
                 'o-', color=info['color'], label=info['name'], linewidth=2, markersize=6)
    ax3.set_xlabel('Normalized Layer Depth')
    ax3.set_ylabel('CLBAS')
    ax3.set_title('CLBAS by Layer Depth', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Probe accuracy comparison (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    for model in present_models:
        model_df = df[df['model'] == model].sort_values('layer_normalized')
        info = MODEL_INFO[model]
        ax4.plot(model_df['layer_normalized'], model_df['probe_arabic'], 
                 '--', color=info['color'], alpha=0.7, linewidth=1.5)
        ax4.plot(model_df['layer_normalized'], model_df['probe_english'], 
                 '-', color=info['color'], linewidth=2, label=info['name'])
    ax4.set_xlabel('Normalized Layer Depth')
    ax4.set_ylabel('Probe Accuracy')
    ax4.set_title('Probe Accuracy (solid=EN, dashed=AR)', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # 5. CLBAS distribution (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    data_for_violin = [df[df['model'] == m]['clbas'].dropna().values for m in present_models]
    parts = ax5.violinplot(data_for_violin, positions=range(len(present_models)), showmeans=True)
    
    for i, (pc, model) in enumerate(zip(parts['bodies'], present_models)):
        pc.set_facecolor(MODEL_INFO[model]['color'])
        pc.set_alpha(0.7)
    
    ax5.set_xticks(range(len(present_models)))
    ax5.set_xticklabels([MODEL_INFO[m]['name'] for m in present_models], fontsize=9)
    ax5.set_ylabel('CLBAS')
    ax5.set_title('CLBAS Distribution', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cosine similarity (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    for model in present_models:
        model_df = df[df['model'] == model].sort_values('layer_normalized')
        info = MODEL_INFO[model]
        if 'cosine_sim' in model_df.columns and not model_df['cosine_sim'].isna().all():
            ax6.plot(model_df['layer_normalized'], model_df['cosine_sim'], 
                     'o-', color=info['color'], label=info['name'], linewidth=2, markersize=6)
    ax6.set_xlabel('Normalized Layer Depth')
    ax6.set_ylabel('Cosine Similarity')
    ax6.set_title('Feature Cosine Similarity', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Arabic support impact (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    
    support_order = ['Native multilingual', 'Native Arabic tokens', 'Byte-fallback (UTF-8)']
    support_colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    clbas_by_support = []
    labels = []
    for support, color in zip(support_order, support_colors):
        models_with_support = [m for m in present_models if MODEL_INFO[m]['arabic_support'] == support]
        if models_with_support:
            clbas_vals = df[df['model'].isin(models_with_support)]['clbas'].dropna().values
            if len(clbas_vals) > 0:
                clbas_by_support.append(clbas_vals)
                labels.append(support.replace(' ', '\n'))
    
    if clbas_by_support:
        bp = ax7.boxplot(clbas_by_support, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], support_colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax7.set_ylabel('CLBAS')
    ax7.set_title('CLBAS by Arabic Support Type', fontsize=12)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Key findings text (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Generate key findings
    findings = []
    if present_models:
        # Best CLBAS
        best_clbas_model = max(present_models, key=lambda m: df[df['model'] == m]['clbas'].mean())
        findings.append(f"• Highest CLBAS: {MODEL_INFO[best_clbas_model]['name']}")
        
        # Smallest probe gap
        gaps = {m: abs((df[df['model'] == m]['probe_english'] - df[df['model'] == m]['probe_arabic']).mean()) 
                for m in present_models}
        smallest_gap_model = min(gaps, key=gaps.get)
        findings.append(f"• Smallest probe gap: {MODEL_INFO[smallest_gap_model]['name']}")
        
        # Most consistent CLBAS
        stds = {m: df[df['model'] == m]['clbas'].std() for m in present_models}
        most_consistent = min(stds, key=stds.get)
        findings.append(f"• Most consistent: {MODEL_INFO[most_consistent]['name']}")
    
    findings_text = "Key Findings:\n\n" + "\n\n".join(findings)
    ax8.text(0.1, 0.9, findings_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Cross-Lingual Gender Bias: Three-Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(output_dir / 'comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'comprehensive_dashboard.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved comprehensive dashboard")


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(df: pd.DataFrame, summary: Dict, tests: Dict, output_dir: Path):
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# Three-Model Cross-Lingual Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Model overview
    report.append("## 1. Models Analyzed\n")
    report.append("| Model | Parameters | Hidden Dim | Layers | Arabic Support |")
    report.append("|-------|------------|------------|--------|----------------|")
    
    for model_key in ['paligemma', 'qwen2vl', 'llava']:
        if model_key in df['model'].values:
            info = MODEL_INFO[model_key]
            report.append(f"| {info['name']} | {info['params']} | {info['hidden_dim']} | {info['num_layers']} | {info['arabic_support']} |")
    
    # Summary statistics
    report.append("\n## 2. Summary Statistics\n")
    
    for model_key, stats in summary.items():
        info = MODEL_INFO[model_key]
        report.append(f"### {info['name']}\n")
        report.append(f"- **CLBAS**: {stats['clbas']['mean']:.4f} ± {stats['clbas']['std']:.4f}")
        report.append(f"  - Range: [{stats['clbas']['min']:.4f}, {stats['clbas']['max']:.4f}]")
        report.append(f"- **Probe Accuracy**:")
        report.append(f"  - Arabic: {stats['probe_accuracy']['arabic_mean']:.4f}")
        report.append(f"  - English: {stats['probe_accuracy']['english_mean']:.4f}")
        report.append(f"  - Gap: {stats['probe_accuracy']['gap']:.4f}")
        report.append(f"- **Cosine Similarity**: {stats['cosine_sim']['mean']:.4f} ± {stats['cosine_sim']['std']:.4f}\n")
    
    # Statistical tests
    report.append("## 3. Statistical Analysis\n")
    
    if 'kruskal_clbas' in tests:
        report.append(f"### Kruskal-Wallis Test (CLBAS across models)")
        report.append(f"- H-statistic: {tests['kruskal_clbas']['H']:.4f}")
        report.append(f"- p-value: {tests['kruskal_clbas']['p']:.6f}")
        sig = "**Significant**" if tests['kruskal_clbas']['p'] < 0.05 else "Not significant"
        report.append(f"- Result: {sig} at α=0.05\n")
    
    if 'pairwise' in tests and tests['pairwise']:
        report.append("### Pairwise Comparisons (Mann-Whitney U)\n")
        report.append("| Comparison | U-statistic | p-value | Significant |")
        report.append("|------------|-------------|---------|-------------|")
        for comparison, result in tests['pairwise'].items():
            sig = "Yes" if result['p'] < 0.05 else "No"
            report.append(f"| {comparison.replace('_', ' ')} | {result['U']:.2f} | {result['p']:.6f} | {sig} |")
    
    # Key findings
    report.append("\n## 4. Key Findings\n")
    
    present_models = df['model'].unique().tolist()
    
    if present_models:
        # Best CLBAS
        clbas_means = {m: summary[m]['clbas']['mean'] for m in present_models if m in summary}
        if clbas_means:
            best_model = max(clbas_means, key=clbas_means.get)
            report.append(f"1. **Highest Cross-Lingual Alignment**: {MODEL_INFO[best_model]['name']} (CLBAS = {clbas_means[best_model]:.4f})")
        
        # Probe gap analysis
        gaps = {m: summary[m]['probe_accuracy']['gap'] for m in present_models if m in summary}
        if gaps:
            smallest_gap = min(gaps, key=lambda x: abs(gaps[x]))
            report.append(f"2. **Most Language-Balanced Probes**: {MODEL_INFO[smallest_gap]['name']} (gap = {gaps[smallest_gap]:.4f})")
        
        # Arabic support impact
        native_models = [m for m in present_models if 'Native' in MODEL_INFO[m]['arabic_support']]
        byte_models = [m for m in present_models if 'Byte' in MODEL_INFO[m]['arabic_support']]
        
        if native_models and byte_models:
            native_avg = np.mean([clbas_means[m] for m in native_models if m in clbas_means])
            byte_avg = np.mean([clbas_means[m] for m in byte_models if m in clbas_means])
            report.append(f"3. **Arabic Support Impact**:")
            report.append(f"   - Native support models avg CLBAS: {native_avg:.4f}")
            report.append(f"   - Byte-fallback models avg CLBAS: {byte_avg:.4f}")
    
    # Recommendations
    report.append("\n## 5. Recommendations\n")
    report.append("Based on the analysis:\n")
    report.append("1. For cross-lingual bias studies, prioritize models with higher CLBAS scores")
    report.append("2. Consider Arabic support mechanism when interpreting results")
    report.append("3. Layer selection should account for model-specific bias patterns")
    
    # Write report
    report_path = output_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"  ✓ Saved comparison report to {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Three-Model Cross-Lingual Comparison')
    parser.add_argument('--paligemma_results', type=str, 
                        default='results/cross_lingual_overlap/cross_lingual_overlap_results.json',
                        help='Path to PaLiGemma results JSON')
    parser.add_argument('--qwen2vl_results', type=str,
                        default='results/qwen2vl_analysis/cross_lingual_results.json',
                        help='Path to Qwen2-VL results JSON')
    parser.add_argument('--llava_results', type=str,
                        default='results/llava_analysis/cross_lingual_results.json',
                        help='Path to LLaVA results JSON')
    parser.add_argument('--output_dir', type=str, default='results/three_model_comparison',
                        help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Three-Model Cross-Lingual Comparison")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results = {}
    
    paligemma_data = load_results(args.paligemma_results, 'paligemma')
    if paligemma_data:
        results['paligemma'] = paligemma_data
    
    qwen2vl_data = load_results(args.qwen2vl_results, 'qwen2vl')
    if qwen2vl_data:
        results['qwen2vl'] = qwen2vl_data
    
    llava_data = load_results(args.llava_results, 'llava')
    if llava_data:
        results['llava'] = llava_data
    
    if not results:
        print("\nERROR: No results files found. Please run the individual model analyses first.")
        return
    
    print(f"\nLoaded {len(results)} model(s): {list(results.keys())}")
    
    # Extract metrics into unified DataFrame
    print("\nExtracting metrics...")
    dfs = []
    for model_key, data in results.items():
        df = extract_metrics(data, model_key)
        dfs.append(df)
        print(f"  {model_key}: {len(df)} layers")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined DataFrame: {len(combined_df)} rows")
    
    # Compute statistics
    print("\nComputing statistics...")
    summary = compute_summary_statistics(combined_df)
    tests = perform_statistical_tests(combined_df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_clbas_comparison(combined_df, output_dir)
    plot_probe_accuracy_comparison(combined_df, output_dir)
    plot_layer_heatmap(combined_df, output_dir)
    plot_comprehensive_dashboard(combined_df, summary, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_markdown_report(combined_df, summary, tests, output_dir)
    
    # Save raw data
    combined_df.to_csv(output_dir / 'combined_metrics.csv', index=False)
    
    with open(output_dir / 'summary_statistics.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(summary, f, indent=2, default=convert)
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")
    
    # Print key findings
    print("\n" + "-" * 60)
    print("Key Findings:")
    print("-" * 60)
    
    for model_key in results.keys():
        model_df = combined_df[combined_df['model'] == model_key]
        info = MODEL_INFO[model_key]
        print(f"\n{info['name']}:")
        print(f"  CLBAS: {model_df['clbas'].mean():.4f} ± {model_df['clbas'].std():.4f}")
        print(f"  Probe gap: {(model_df['probe_english'] - model_df['probe_arabic']).mean():.4f}")


if __name__ == '__main__':
    main()
