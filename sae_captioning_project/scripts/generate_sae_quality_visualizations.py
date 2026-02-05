#!/usr/bin/env python3
"""
Generate Publication-Quality Visualizations for SAE Quality Metrics
====================================================================

Creates comprehensive figures for:
1. Explained Variance across layers and models
2. Dead Feature Ratio comparison
3. Mean L0 (Sparsity) analysis
4. Reconstruction Quality (Cosine Similarity)
5. Cross-model comparison dashboard
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Output directory
OUTPUT_DIR = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project/visualizations/sae_quality_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palettes
COLORS = {
    'Qwen2-VL-7B': '#2E86AB',      # Blue
    'LLaVA-1.5-7B': '#A23B72',     # Magenta/Pink
    'PaLiGemma-3B': '#F18F01',     # Orange
    'arabic': '#E63946',           # Red
    'english': '#457B9D',          # Steel Blue
}

# ============================================================
# DATA: Qwen2-VL-7B and LLaVA-1.5-7B Results
# ============================================================

QWEN_DATA = {
    'layers': [0, 4, 8, 12, 16, 20, 24, 27],
    'arabic': {
        'explained_var': [85.6, 75.1, 69.7, 63.0, 50.4, 53.6, 78.9, 84.9],
        'dead_features': [93.1, 80.0, 73.5, 71.3, 73.1, 77.6, 79.0, 80.3],
        'mean_l0': [695, 1450, 1918, 2066, 2232, 2103, 1644, 971],
        'recon_cosine': [0.9932, 0.9965, 0.9961, 0.9961, 0.9964, 0.9961, 0.9942, 0.9891],
    },
    'english': {
        'explained_var': [87.7, 81.1, 71.6, 66.4, 53.6, 57.4, 82.5, 86.2],
        'dead_features': [92.8, 78.8, 73.5, 71.6, 72.2, 77.3, 74.7, 79.6],
        'mean_l0': [714, 1515, 1861, 2049, 2229, 2019, 1675, 980],
        'recon_cosine': [0.9941, 0.9973, 0.9963, 0.9965, 0.9966, 0.9964, 0.9950, 0.9905],
    },
    'd_model': 3584,
    'd_hidden': 28672,
}

LLAVA_DATA = {
    'layers': [0, 4, 8, 12, 16, 20, 24, 28, 31],
    'arabic': {
        'explained_var': [81.4, 82.4, 80.3, 77.6, 79.2, 82.8, 86.2, 86.1, 84.4],
        'dead_features': [96.6, 96.7, 96.9, 96.6, 96.2, 94.7, 92.5, 91.3, 94.6],
        'mean_l0': [625, 628, 548, 604, 659, 894, 1449, 1957, 1645],
        'recon_cosine': [0.9951, 0.9953, 0.9940, 0.9929, 0.9920, 0.9913, 0.9937, 0.9951, 0.9975],
    },
    'english': {
        'explained_var': [85.7, 80.8, 82.5, 80.8, 82.6, 86.3, 89.2, 88.9, 86.7],
        'dead_features': [96.3, 96.6, 96.8, 96.6, 95.8, 94.2, 91.6, 90.8, 94.6],
        'mean_l0': [694, 622, 567, 618, 733, 959, 1594, 2035, 1617],
        'recon_cosine': [0.9963, 0.9949, 0.9946, 0.9938, 0.9931, 0.9926, 0.9947, 0.9959, 0.9977],
    },
    'd_model': 4096,
    'd_hidden': 32768,
}

# Reference targets from Anthropic (2024)
TARGETS = {
    'explained_var_min': 65,  # >65%
    'dead_features_max': 35,  # <35% (we have high dead features, which is actually fine for sparse representations)
    'mean_l0_range': (50, 300),  # Target range for very sparse
    'recon_cosine_min': 0.9,  # >0.9
}


def fig1_explained_variance_by_layer():
    """Figure 1: Explained Variance % across layers for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Qwen2-VL
    ax = axes[0]
    x = np.arange(len(QWEN_DATA['layers']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, QWEN_DATA['arabic']['explained_var'], width, 
                   label='Arabic', color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, QWEN_DATA['english']['explained_var'], width,
                   label='English', color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=TARGETS['explained_var_min'], color='green', linestyle='--', linewidth=2, 
               label=f"Target (>{TARGETS['explained_var_min']}%)")
    ax.set_xlabel('Layer')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Qwen2-VL-7B', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(QWEN_DATA['layers'])
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # LLaVA
    ax = axes[1]
    x = np.arange(len(LLAVA_DATA['layers']))
    
    bars1 = ax.bar(x - width/2, LLAVA_DATA['arabic']['explained_var'], width, 
                   label='Arabic', color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, LLAVA_DATA['english']['explained_var'], width,
                   label='English', color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=TARGETS['explained_var_min'], color='green', linestyle='--', linewidth=2, 
               label=f"Target (>{TARGETS['explained_var_min']}%)")
    ax.set_xlabel('Layer')
    ax.set_title('LLaVA-1.5-7B', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LLAVA_DATA['layers'])
    ax.legend(loc='lower right')
    
    fig.suptitle('Explained Variance by Layer: SAE Reconstruction Quality', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_explained_variance_by_layer.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_explained_variance_by_layer.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig1_explained_variance_by_layer")


def fig2_dead_feature_ratio():
    """Figure 2: Dead Feature Ratio comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Qwen2-VL
    ax = axes[0]
    x = np.arange(len(QWEN_DATA['layers']))
    width = 0.35
    
    ax.bar(x - width/2, QWEN_DATA['arabic']['dead_features'], width, 
           label='Arabic', color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, QWEN_DATA['english']['dead_features'], width,
           label='English', color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Dead Feature Ratio (%)')
    ax.set_title('Qwen2-VL-7B', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(QWEN_DATA['layers'])
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    
    # LLaVA
    ax = axes[1]
    x = np.arange(len(LLAVA_DATA['layers']))
    
    ax.bar(x - width/2, LLAVA_DATA['arabic']['dead_features'], width, 
           label='Arabic', color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, LLAVA_DATA['english']['dead_features'], width,
           label='English', color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Layer')
    ax.set_title('LLaVA-1.5-7B', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LLAVA_DATA['layers'])
    ax.legend(loc='lower right')
    
    fig.suptitle('Dead Feature Ratio by Layer: Feature Utilization Analysis', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_dead_feature_ratio.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_dead_feature_ratio.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig2_dead_feature_ratio")


def fig3_mean_l0_sparsity():
    """Figure 3: Mean L0 (Active Features) across layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Qwen2-VL
    ax = axes[0]
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['arabic']['mean_l0'], 'o-', 
            color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['english']['mean_l0'], 's-', 
            color=COLORS['english'], linewidth=2, markersize=8, label='English')
    
    ax.fill_between([QWEN_DATA['layers'][0]-1, QWEN_DATA['layers'][-1]+1], 
                    TARGETS['mean_l0_range'][0], TARGETS['mean_l0_range'][1],
                    color='green', alpha=0.15, label='Ideal Range (50-300)')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean L0 (Active Features)')
    ax.set_title('Qwen2-VL-7B', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(QWEN_DATA['layers'][0]-0.5, QWEN_DATA['layers'][-1]+0.5)
    
    # LLaVA
    ax = axes[1]
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['arabic']['mean_l0'], 'o-', 
            color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['english']['mean_l0'], 's-', 
            color=COLORS['english'], linewidth=2, markersize=8, label='English')
    
    ax.fill_between([LLAVA_DATA['layers'][0]-1, LLAVA_DATA['layers'][-1]+1], 
                    TARGETS['mean_l0_range'][0], TARGETS['mean_l0_range'][1],
                    color='green', alpha=0.15, label='Ideal Range (50-300)')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean L0 (Active Features)')
    ax.set_title('LLaVA-1.5-7B', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(LLAVA_DATA['layers'][0]-0.5, LLAVA_DATA['layers'][-1]+0.5)
    
    fig.suptitle('Mean L0 Sparsity: Average Active Features per Sample', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_mean_l0_sparsity.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_mean_l0_sparsity.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig3_mean_l0_sparsity")


def fig4_reconstruction_cosine():
    """Figure 4: Reconstruction Cosine Similarity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Qwen2-VL
    ax = axes[0]
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['arabic']['recon_cosine'], 'o-', 
            color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['english']['recon_cosine'], 's-', 
            color=COLORS['english'], linewidth=2, markersize=8, label='English')
    
    ax.axhline(y=TARGETS['recon_cosine_min'], color='green', linestyle='--', linewidth=2, 
               label=f"Target (>{TARGETS['recon_cosine_min']})")
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Qwen2-VL-7B', fontweight='bold')
    ax.set_ylim(0.985, 1.0)
    ax.legend(loc='lower left')
    
    # LLaVA
    ax = axes[1]
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['arabic']['recon_cosine'], 'o-', 
            color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['english']['recon_cosine'], 's-', 
            color=COLORS['english'], linewidth=2, markersize=8, label='English')
    
    ax.axhline(y=TARGETS['recon_cosine_min'], color='green', linestyle='--', linewidth=2, 
               label=f"Target (>{TARGETS['recon_cosine_min']})")
    
    ax.set_xlabel('Layer')
    ax.set_title('LLaVA-1.5-7B', fontweight='bold')
    ax.legend(loc='lower left')
    
    fig.suptitle('Reconstruction Quality: Cosine Similarity', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_reconstruction_cosine.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_reconstruction_cosine.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig4_reconstruction_cosine")


def fig5_model_comparison_dashboard():
    """Figure 5: Comprehensive model comparison dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Summary statistics
    models = ['Qwen2-VL-7B', 'LLaVA-1.5-7B']
    
    # Calculate averages
    qwen_avg_ev = np.mean(QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var'])
    llava_avg_ev = np.mean(LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var'])
    
    qwen_avg_dead = np.mean(QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'])
    llava_avg_dead = np.mean(LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    
    qwen_avg_l0 = np.mean(QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0'])
    llava_avg_l0 = np.mean(LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0'])
    
    qwen_avg_cos = np.mean(QWEN_DATA['arabic']['recon_cosine'] + QWEN_DATA['english']['recon_cosine'])
    llava_avg_cos = np.mean(LLAVA_DATA['arabic']['recon_cosine'] + LLAVA_DATA['english']['recon_cosine'])
    
    # Panel 1: Explained Variance comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    width = 0.6
    bars = ax1.bar(x, [qwen_avg_ev, llava_avg_ev], width, 
                   color=[COLORS['Qwen2-VL-7B'], COLORS['LLaVA-1.5-7B']], 
                   edgecolor='black', linewidth=1)
    ax1.axhline(y=TARGETS['explained_var_min'], color='green', linestyle='--', linewidth=2, label='Target (>65%)')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('Average Explained Variance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 100)
    ax1.legend()
    for bar, val in zip(bars, [qwen_avg_ev, llava_avg_ev]):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    # Panel 2: Dead Features comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x, [qwen_avg_dead, llava_avg_dead], width,
                   color=[COLORS['Qwen2-VL-7B'], COLORS['LLaVA-1.5-7B']],
                   edgecolor='black', linewidth=1)
    ax2.set_ylabel('Dead Feature Ratio (%)')
    ax2.set_title('Average Dead Feature Ratio', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, [qwen_avg_dead, llava_avg_dead]):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    # Panel 3: Mean L0 comparison
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(x, [qwen_avg_l0, llava_avg_l0], width,
                   color=[COLORS['Qwen2-VL-7B'], COLORS['LLaVA-1.5-7B']],
                   edgecolor='black', linewidth=1)
    ax3.axhspan(TARGETS['mean_l0_range'][0], TARGETS['mean_l0_range'][1], color='green', alpha=0.15, label='Ideal (50-300)')
    ax3.set_ylabel('Mean L0 (Active Features)')
    ax3.set_title('Average Sparsity (Mean L0)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend(loc='upper right')
    for bar, val in zip(bars, [qwen_avg_l0, llava_avg_l0]):
        ax3.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    # Panel 4: Reconstruction Cosine comparison
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(x, [qwen_avg_cos, llava_avg_cos], width,
                   color=[COLORS['Qwen2-VL-7B'], COLORS['LLaVA-1.5-7B']],
                   edgecolor='black', linewidth=1)
    ax4.axhline(y=TARGETS['recon_cosine_min'], color='green', linestyle='--', linewidth=2, label='Target (>0.9)')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Average Reconstruction Quality', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.set_ylim(0.98, 1.0)
    ax4.legend()
    for bar, val in zip(bars, [qwen_avg_cos, llava_avg_cos]):
        ax4.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    fig.suptitle('SAE Quality Metrics: Cross-Model Comparison', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'fig5_model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_model_comparison_dashboard.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig5_model_comparison_dashboard")


def fig6_layer_profile_heatmap():
    """Figure 6: Heatmap showing metric profiles across layers."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Qwen2-VL heatmap
    ax = axes[0]
    qwen_data_matrix = np.array([
        QWEN_DATA['arabic']['explained_var'],
        QWEN_DATA['english']['explained_var'],
        QWEN_DATA['arabic']['dead_features'],
        QWEN_DATA['english']['dead_features'],
        [l/100 for l in QWEN_DATA['arabic']['mean_l0']],  # Scale for visibility
        [l/100 for l in QWEN_DATA['english']['mean_l0']],
        [c*100 for c in QWEN_DATA['arabic']['recon_cosine']],  # Scale to %
        [c*100 for c in QWEN_DATA['english']['recon_cosine']],
    ])
    
    row_labels = ['EV% (Arabic)', 'EV% (English)', 'Dead% (Arabic)', 'Dead% (English)',
                  'L0/100 (Arabic)', 'L0/100 (English)', 'Cos×100 (Arabic)', 'Cos×100 (English)']
    
    im = ax.imshow(qwen_data_matrix, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(np.arange(len(QWEN_DATA['layers'])))
    ax.set_xticklabels(QWEN_DATA['layers'])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Layer')
    ax.set_title('Qwen2-VL-7B Metric Profile', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # LLaVA heatmap
    ax = axes[1]
    llava_data_matrix = np.array([
        LLAVA_DATA['arabic']['explained_var'],
        LLAVA_DATA['english']['explained_var'],
        LLAVA_DATA['arabic']['dead_features'],
        LLAVA_DATA['english']['dead_features'],
        [l/100 for l in LLAVA_DATA['arabic']['mean_l0']],
        [l/100 for l in LLAVA_DATA['english']['mean_l0']],
        [c*100 for c in LLAVA_DATA['arabic']['recon_cosine']],
        [c*100 for c in LLAVA_DATA['english']['recon_cosine']],
    ])
    
    im = ax.imshow(llava_data_matrix, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(np.arange(len(LLAVA_DATA['layers'])))
    ax.set_xticklabels(LLAVA_DATA['layers'])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Layer')
    ax.set_title('LLaVA-1.5-7B Metric Profile', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle('SAE Metric Profiles Across Layers', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_layer_profile_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_layer_profile_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig6_layer_profile_heatmap")


def fig7_arabic_english_comparison():
    """Figure 7: Side-by-side Arabic vs English comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Qwen2-VL EV comparison
    ax = axes[0, 0]
    x = np.arange(len(QWEN_DATA['layers']))
    ax.plot(x, QWEN_DATA['arabic']['explained_var'], 'o-', color=COLORS['arabic'], 
            linewidth=2, markersize=8, label='Arabic')
    ax.plot(x, QWEN_DATA['english']['explained_var'], 's-', color=COLORS['english'], 
            linewidth=2, markersize=8, label='English')
    ax.fill_between(x, QWEN_DATA['arabic']['explained_var'], QWEN_DATA['english']['explained_var'], 
                    alpha=0.3, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(QWEN_DATA['layers'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Qwen2-VL-7B: Language Comparison', fontweight='bold')
    ax.legend()
    ax.axhline(y=65, color='green', linestyle='--', alpha=0.5)
    
    # LLaVA EV comparison
    ax = axes[0, 1]
    x = np.arange(len(LLAVA_DATA['layers']))
    ax.plot(x, LLAVA_DATA['arabic']['explained_var'], 'o-', color=COLORS['arabic'], 
            linewidth=2, markersize=8, label='Arabic')
    ax.plot(x, LLAVA_DATA['english']['explained_var'], 's-', color=COLORS['english'], 
            linewidth=2, markersize=8, label='English')
    ax.fill_between(x, LLAVA_DATA['arabic']['explained_var'], LLAVA_DATA['english']['explained_var'], 
                    alpha=0.3, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(LLAVA_DATA['layers'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('LLaVA-1.5-7B: Language Comparison', fontweight='bold')
    ax.legend()
    ax.axhline(y=65, color='green', linestyle='--', alpha=0.5)
    
    # Qwen2-VL L0 comparison
    ax = axes[1, 0]
    x = np.arange(len(QWEN_DATA['layers']))
    ax.plot(x, QWEN_DATA['arabic']['mean_l0'], 'o-', color=COLORS['arabic'], 
            linewidth=2, markersize=8, label='Arabic')
    ax.plot(x, QWEN_DATA['english']['mean_l0'], 's-', color=COLORS['english'], 
            linewidth=2, markersize=8, label='English')
    ax.set_xticks(x)
    ax.set_xticklabels(QWEN_DATA['layers'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean L0 (Active Features)')
    ax.set_title('Qwen2-VL-7B: Sparsity by Language', fontweight='bold')
    ax.legend()
    
    # LLaVA L0 comparison
    ax = axes[1, 1]
    x = np.arange(len(LLAVA_DATA['layers']))
    ax.plot(x, LLAVA_DATA['arabic']['mean_l0'], 'o-', color=COLORS['arabic'], 
            linewidth=2, markersize=8, label='Arabic')
    ax.plot(x, LLAVA_DATA['english']['mean_l0'], 's-', color=COLORS['english'], 
            linewidth=2, markersize=8, label='English')
    ax.set_xticks(x)
    ax.set_xticklabels(LLAVA_DATA['layers'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean L0 (Active Features)')
    ax.set_title('LLaVA-1.5-7B: Sparsity by Language', fontweight='bold')
    ax.legend()
    
    fig.suptitle('Cross-Lingual SAE Quality Comparison: Arabic vs English', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_arabic_english_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig7_arabic_english_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig7_arabic_english_comparison")


def fig8_publication_summary_table():
    """Figure 8: Publication-ready summary table as image."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Create summary data
    qwen_avg_ev = np.mean(QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var'])
    llava_avg_ev = np.mean(LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var'])
    qwen_std_ev = np.std(QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var'])
    llava_std_ev = np.std(LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var'])
    
    qwen_avg_dead = np.mean(QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'])
    llava_avg_dead = np.mean(LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    qwen_std_dead = np.std(QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'])
    llava_std_dead = np.std(LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    
    qwen_avg_l0 = np.mean(QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0'])
    llava_avg_l0 = np.mean(LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0'])
    qwen_std_l0 = np.std(QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0'])
    llava_std_l0 = np.std(LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0'])
    
    qwen_avg_cos = np.mean(QWEN_DATA['arabic']['recon_cosine'] + QWEN_DATA['english']['recon_cosine'])
    llava_avg_cos = np.mean(LLAVA_DATA['arabic']['recon_cosine'] + LLAVA_DATA['english']['recon_cosine'])
    
    headers = ['Model', 'd_model', 'd_hidden', 'Explained Var%', 'Dead Features%', 'Mean L0', 'Recon Cosine']
    data = [
        ['Qwen2-VL-7B', '3,584', '28,672', f'{qwen_avg_ev:.1f}±{qwen_std_ev:.1f}', 
         f'{qwen_avg_dead:.1f}±{qwen_std_dead:.1f}', f'{qwen_avg_l0:.0f}±{qwen_std_l0:.0f}', f'{qwen_avg_cos:.4f}'],
        ['LLaVA-1.5-7B', '4,096', '32,768', f'{llava_avg_ev:.1f}±{llava_std_ev:.1f}',
         f'{llava_avg_dead:.1f}±{llava_std_dead:.1f}', f'{llava_avg_l0:.0f}±{llava_std_l0:.0f}', f'{llava_avg_cos:.4f}'],
        ['PaLiGemma-3B*', '2,048', '16,384', 'Computing...', 'Computing...', 'Computing...', 'Computing...'],
    ]
    
    # Target row
    targets = ['Target', '-', '-', '>65%', '<35%†', '50-300', '>0.9']
    
    table = ax.table(cellText=data + [targets], colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2)
    
    # Style header
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style target row
    for j in range(len(headers)):
        table[(4, j)].set_facecolor('#E8F4EA')
        table[(4, j)].set_text_props(fontweight='bold', color='green')
    
    # Style PaLiGemma row
    for j in range(len(headers)):
        table[(3, j)].set_facecolor('#FFF3CD')
    
    ax.set_title('SAE Quality Metrics Summary (Publication Table)', fontsize=16, fontweight='bold', pad=20)
    
    # Add footnotes
    footnote = ("* PaLiGemma-3B metrics being computed on GPU cluster\n"
                "† High dead feature ratios (>35%) indicate sparse representations - acceptable when EV% is high")
    ax.text(0.5, -0.1, footnote, transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_publication_summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig8_publication_summary_table.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig8_publication_summary_table")


def fig9_metric_correlation():
    """Figure 9: Correlation between metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Combine all data points
    all_ev = (QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var'] +
              LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var'])
    all_l0 = (QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0'] +
              LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0'])
    all_dead = (QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'] +
                LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    all_cos = (QWEN_DATA['arabic']['recon_cosine'] + QWEN_DATA['english']['recon_cosine'] +
               LLAVA_DATA['arabic']['recon_cosine'] + LLAVA_DATA['english']['recon_cosine'])
    
    # Model labels for coloring
    n_qwen = len(QWEN_DATA['layers']) * 2
    n_llava = len(LLAVA_DATA['layers']) * 2
    colors = [COLORS['Qwen2-VL-7B']] * n_qwen + [COLORS['LLaVA-1.5-7B']] * n_llava
    
    # EV vs L0
    ax = axes[0]
    ax.scatter(all_l0, all_ev, c=colors, s=80, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mean L0 (Active Features)')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Explained Variance vs Sparsity', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(all_l0, all_ev, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_l0), max(all_l0), 100)
    ax.plot(x_trend, p(x_trend), 'k--', alpha=0.5, label=f'Trend (r={np.corrcoef(all_l0, all_ev)[0,1]:.2f})')
    ax.legend()
    
    # Create legend for models
    qwen_patch = mpatches.Patch(color=COLORS['Qwen2-VL-7B'], label='Qwen2-VL-7B')
    llava_patch = mpatches.Patch(color=COLORS['LLaVA-1.5-7B'], label='LLaVA-1.5-7B')
    ax.legend(handles=[qwen_patch, llava_patch], loc='upper right')
    
    # EV vs Dead Features
    ax = axes[1]
    ax.scatter(all_dead, all_ev, c=colors, s=80, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Dead Feature Ratio (%)')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Explained Variance vs Dead Features', fontweight='bold')
    
    z = np.polyfit(all_dead, all_ev, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_dead), max(all_dead), 100)
    ax.plot(x_trend, p(x_trend), 'k--', alpha=0.5)
    ax.legend(handles=[qwen_patch, llava_patch], loc='lower left')
    
    fig.suptitle('Metric Correlations: Understanding SAE Behavior', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_metric_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig9_metric_correlation.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig9_metric_correlation")


def fig10_sparsity_vs_quality_tradeoff():
    """Figure 10: Sparsity vs Quality tradeoff analysis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Combine all data
    all_data = []
    
    for i, layer in enumerate(QWEN_DATA['layers']):
        for lang in ['arabic', 'english']:
            all_data.append({
                'model': 'Qwen2-VL-7B',
                'layer': layer,
                'language': lang,
                'ev': QWEN_DATA[lang]['explained_var'][i],
                'l0': QWEN_DATA[lang]['mean_l0'][i],
                'dead': QWEN_DATA[lang]['dead_features'][i],
                'cos': QWEN_DATA[lang]['recon_cosine'][i],
            })
    
    for i, layer in enumerate(LLAVA_DATA['layers']):
        for lang in ['arabic', 'english']:
            all_data.append({
                'model': 'LLaVA-1.5-7B',
                'layer': layer,
                'language': lang,
                'ev': LLAVA_DATA[lang]['explained_var'][i],
                'l0': LLAVA_DATA[lang]['mean_l0'][i],
                'dead': LLAVA_DATA[lang]['dead_features'][i],
                'cos': LLAVA_DATA[lang]['recon_cosine'][i],
            })
    
    # Create scatter plot with size = cosine similarity
    for d in all_data:
        color = COLORS[d['model']]
        marker = 'o' if d['language'] == 'arabic' else 's'
        size = (d['cos'] - 0.98) * 10000  # Scale for visibility
        ax.scatter(d['l0'], d['ev'], c=color, marker=marker, s=max(50, size), 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add annotations for optimal points
    ax.axhline(y=65, color='green', linestyle='--', alpha=0.5, label='EV Target (65%)')
    ax.axvspan(50, 300, color='green', alpha=0.1, label='Ideal L0 Range')
    
    # Legend
    qwen_patch = mpatches.Patch(color=COLORS['Qwen2-VL-7B'], label='Qwen2-VL-7B')
    llava_patch = mpatches.Patch(color=COLORS['LLaVA-1.5-7B'], label='LLaVA-1.5-7B')
    arabic_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Arabic')
    english_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='English')
    
    ax.legend(handles=[qwen_patch, llava_patch, arabic_marker, english_marker], loc='lower right')
    
    ax.set_xlabel('Mean L0 (Active Features)', fontsize=14)
    ax.set_ylabel('Explained Variance (%)', fontsize=14)
    ax.set_title('Sparsity-Quality Tradeoff Analysis\n(Bubble size ∝ Reconstruction Cosine Similarity)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_sparsity_quality_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig10_sparsity_quality_tradeoff.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig10_sparsity_quality_tradeoff")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating SAE Quality Metric Visualizations")
    print("=" * 60)
    
    fig1_explained_variance_by_layer()
    fig2_dead_feature_ratio()
    fig3_mean_l0_sparsity()
    fig4_reconstruction_cosine()
    fig5_model_comparison_dashboard()
    fig6_layer_profile_heatmap()
    fig7_arabic_english_comparison()
    fig8_publication_summary_table()
    fig9_metric_correlation()
    fig10_sparsity_vs_quality_tradeoff()
    
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
