#!/usr/bin/env python3
"""
Generate Publication-Quality Visualizations for SAE Quality Metrics
All 3 Models: PaLiGemma-3B, Qwen2-VL-7B, LLaVA-1.5-7B
====================================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

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

OUTPUT_DIR = Path("/home2/jmsk62/mechanistic_intrep/sae_captioning_project/visualizations/sae_quality_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'PaLiGemma-3B': '#F18F01',
    'Qwen2-VL-7B': '#2E86AB',
    'LLaVA-1.5-7B': '#A23B72',
    'arabic': '#E63946',
    'english': '#457B9D',
}

# ============================================================
# DATA
# ============================================================

PALIGEMMA_DATA = {
    'layers': [3, 6, 9, 12, 15, 17],
    'arabic': {
        'explained_var': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'dead_features': [50.1, 49.6, 50.9, 58.6, 56.6, 63.7],
        'mean_l0': [8184, 8264, 8053, 6782, 7115, 5953],
        'recon_cosine': [0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999],
    },
    'english': {
        'explained_var': [100.0, 37.5, 100.0, 100.0, 100.0, 100.0],
        'dead_features': [50.2, 43.0, 50.5, 54.8, 53.8, 60.5],
        'mean_l0': [8159, 7262, 7992, 7412, 7567, 6472],
        'recon_cosine': [0.9999, 0.9960, 0.9999, 0.9990, 1.0000, 0.9998],
    },
    'd_model': 2048, 'd_hidden': 16384,
}

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
    'd_model': 3584, 'd_hidden': 28672,
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
    'd_model': 4096, 'd_hidden': 32768,
}

# Normalized layer positions (0-1 range for comparable x-axis)
def normalize_layers(layers, max_layer):
    return [l / max_layer for l in layers]


# ============================================================
# Figure 1: Explained Variance - All 3 Models
# ============================================================
def fig1_explained_variance_all():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # PaLiGemma
    ax = axes[0]
    x = np.arange(len(PALIGEMMA_DATA['layers']))
    w = 0.35
    ax.bar(x - w/2, PALIGEMMA_DATA['arabic']['explained_var'], w, color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Arabic')
    ax.bar(x + w/2, PALIGEMMA_DATA['english']['explained_var'], w, color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5, label='English')
    ax.axhline(y=65, color='green', linestyle='--', linewidth=2, label='Target (>65%)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PaLiGemma-3B (2048-d)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(PALIGEMMA_DATA['layers'])
    ax.set_ylim(0, 105); ax.legend(loc='lower right', fontsize=10)
    # Annotate Layer 6 English anomaly
    ax.annotate('*retrained\nSAE', xy=(1, 37.5), xytext=(1.5, 50), textcoords='data',
                fontsize=8, color='red', ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # Qwen2-VL
    ax = axes[1]
    x = np.arange(len(QWEN_DATA['layers']))
    w = 0.35
    ax.bar(x - w/2, QWEN_DATA['arabic']['explained_var'], w, color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Arabic')
    ax.bar(x + w/2, QWEN_DATA['english']['explained_var'], w, color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5, label='English')
    ax.axhline(y=65, color='green', linestyle='--', linewidth=2, label='Target (>65%)')
    ax.set_xlabel('Layer')
    ax.set_title('Qwen2-VL-7B (3584-d)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(QWEN_DATA['layers'])
    ax.legend(loc='lower right', fontsize=10)
    
    # LLaVA
    ax = axes[2]
    x = np.arange(len(LLAVA_DATA['layers']))
    ax.bar(x - w/2, LLAVA_DATA['arabic']['explained_var'], w, color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Arabic')
    ax.bar(x + w/2, LLAVA_DATA['english']['explained_var'], w, color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5, label='English')
    ax.axhline(y=65, color='green', linestyle='--', linewidth=2, label='Target (>65%)')
    ax.set_xlabel('Layer')
    ax.set_title('LLaVA-1.5-7B (4096-d)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(LLAVA_DATA['layers'])
    ax.legend(loc='lower right', fontsize=10)
    
    fig.suptitle('Explained Variance by Layer: SAE Reconstruction Quality', fontsize=20, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_explained_variance_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig1_explained_variance_all_models")


# ============================================================
# Figure 2: Dead Feature Ratio - All 3 Models
# ============================================================
def fig2_dead_features_all():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    ax = axes[0]
    x = np.arange(len(PALIGEMMA_DATA['layers'])); w = 0.35
    ax.bar(x - w/2, PALIGEMMA_DATA['arabic']['dead_features'], w, color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Arabic')
    ax.bar(x + w/2, PALIGEMMA_DATA['english']['dead_features'], w, color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5, label='English')
    ax.set_xlabel('Layer'); ax.set_ylabel('Dead Feature Ratio (%)')
    ax.set_title('PaLiGemma-3B', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(PALIGEMMA_DATA['layers'])
    ax.set_ylim(0, 100); ax.legend(fontsize=10)
    
    ax = axes[1]
    x = np.arange(len(QWEN_DATA['layers'])); w = 0.35
    ax.bar(x - w/2, QWEN_DATA['arabic']['dead_features'], w, color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Arabic')
    ax.bar(x + w/2, QWEN_DATA['english']['dead_features'], w, color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5, label='English')
    ax.set_xlabel('Layer'); ax.set_title('Qwen2-VL-7B', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(QWEN_DATA['layers']); ax.legend(fontsize=10)
    
    ax = axes[2]
    x = np.arange(len(LLAVA_DATA['layers']))
    ax.bar(x - w/2, LLAVA_DATA['arabic']['dead_features'], w, color=COLORS['arabic'], alpha=0.8, edgecolor='black', linewidth=0.5, label='Arabic')
    ax.bar(x + w/2, LLAVA_DATA['english']['dead_features'], w, color=COLORS['english'], alpha=0.8, edgecolor='black', linewidth=0.5, label='English')
    ax.set_xlabel('Layer'); ax.set_title('LLaVA-1.5-7B', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(LLAVA_DATA['layers']); ax.legend(fontsize=10)
    
    fig.suptitle('Dead Feature Ratio by Layer: Feature Utilization', fontsize=20, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_dead_features_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig2_dead_features_all_models")


# ============================================================
# Figure 3: Mean L0 Sparsity - All 3 Models
# ============================================================
def fig3_mean_l0_all():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    ax.plot(PALIGEMMA_DATA['layers'], PALIGEMMA_DATA['arabic']['mean_l0'], 'o-', color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(PALIGEMMA_DATA['layers'], PALIGEMMA_DATA['english']['mean_l0'], 's-', color=COLORS['english'], linewidth=2, markersize=8, label='English')
    ax.fill_between([min(PALIGEMMA_DATA['layers'])-1, max(PALIGEMMA_DATA['layers'])+1], 50, 300,
                    color='green', alpha=0.12, label='Ideal (50-300)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean L0 (Active Features)')
    ax.set_title('PaLiGemma-3B', fontweight='bold'); ax.legend(fontsize=10)
    ax.set_xlim(min(PALIGEMMA_DATA['layers'])-0.5, max(PALIGEMMA_DATA['layers'])+0.5)
    
    ax = axes[1]
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['arabic']['mean_l0'], 'o-', color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['english']['mean_l0'], 's-', color=COLORS['english'], linewidth=2, markersize=8, label='English')
    ax.fill_between([min(QWEN_DATA['layers'])-1, max(QWEN_DATA['layers'])+1], 50, 300, color='green', alpha=0.12, label='Ideal (50-300)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean L0')
    ax.set_title('Qwen2-VL-7B', fontweight='bold'); ax.legend(fontsize=10)
    ax.set_xlim(min(QWEN_DATA['layers'])-0.5, max(QWEN_DATA['layers'])+0.5)
    
    ax = axes[2]
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['arabic']['mean_l0'], 'o-', color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['english']['mean_l0'], 's-', color=COLORS['english'], linewidth=2, markersize=8, label='English')
    ax.fill_between([min(LLAVA_DATA['layers'])-1, max(LLAVA_DATA['layers'])+1], 50, 300, color='green', alpha=0.12, label='Ideal (50-300)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean L0')
    ax.set_title('LLaVA-1.5-7B', fontweight='bold'); ax.legend(fontsize=10)
    ax.set_xlim(min(LLAVA_DATA['layers'])-0.5, max(LLAVA_DATA['layers'])+0.5)
    
    fig.suptitle('Mean L0 Sparsity: Active Features per Sample', fontsize=20, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_mean_l0_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig3_mean_l0_all_models")


# ============================================================
# Figure 4: Reconstruction Cosine - All 3 Models
# ============================================================
def fig4_recon_cosine_all():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    ax = axes[0]
    ax.plot(PALIGEMMA_DATA['layers'], PALIGEMMA_DATA['arabic']['recon_cosine'], 'o-', color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(PALIGEMMA_DATA['layers'], PALIGEMMA_DATA['english']['recon_cosine'], 's-', color=COLORS['english'], linewidth=2, markersize=8, label='English')
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='Target (>0.9)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Cosine Similarity')
    ax.set_title('PaLiGemma-3B', fontweight='bold')
    ax.set_ylim(0.985, 1.002); ax.legend(fontsize=10)
    
    ax = axes[1]
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['arabic']['recon_cosine'], 'o-', color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(QWEN_DATA['layers'], QWEN_DATA['english']['recon_cosine'], 's-', color=COLORS['english'], linewidth=2, markersize=8, label='English')
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='Target (>0.9)')
    ax.set_xlabel('Layer'); ax.set_title('Qwen2-VL-7B', fontweight='bold'); ax.legend(fontsize=10)
    
    ax = axes[2]
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['arabic']['recon_cosine'], 'o-', color=COLORS['arabic'], linewidth=2, markersize=8, label='Arabic')
    ax.plot(LLAVA_DATA['layers'], LLAVA_DATA['english']['recon_cosine'], 's-', color=COLORS['english'], linewidth=2, markersize=8, label='English')
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='Target (>0.9)')
    ax.set_xlabel('Layer'); ax.set_title('LLaVA-1.5-7B', fontweight='bold'); ax.legend(fontsize=10)
    
    fig.suptitle('Reconstruction Quality: Cosine Similarity', fontsize=20, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_recon_cosine_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig4_recon_cosine_all_models")


# ============================================================
# Figure 5: Cross-Model Comparison Dashboard (4 panels)
# ============================================================
def fig5_dashboard():
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    models = ['PaLiGemma-3B', 'Qwen2-VL-7B', 'LLaVA-1.5-7B']
    model_colors = [COLORS[m] for m in models]
    
    # Averages (all models bilingual)
    pali_ev = np.mean(PALIGEMMA_DATA['arabic']['explained_var'] + PALIGEMMA_DATA['english']['explained_var'])
    qwen_ev = np.mean(QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var'])
    llava_ev = np.mean(LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var'])
    
    pali_dead = np.mean(PALIGEMMA_DATA['arabic']['dead_features'] + PALIGEMMA_DATA['english']['dead_features'])
    qwen_dead = np.mean(QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'])
    llava_dead = np.mean(LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    
    pali_l0 = np.mean(PALIGEMMA_DATA['arabic']['mean_l0'] + PALIGEMMA_DATA['english']['mean_l0'])
    qwen_l0 = np.mean(QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0'])
    llava_l0 = np.mean(LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0'])
    
    pali_cos = np.mean(PALIGEMMA_DATA['arabic']['recon_cosine'] + PALIGEMMA_DATA['english']['recon_cosine'])
    qwen_cos = np.mean(QWEN_DATA['arabic']['recon_cosine'] + QWEN_DATA['english']['recon_cosine'])
    llava_cos = np.mean(LLAVA_DATA['arabic']['recon_cosine'] + LLAVA_DATA['english']['recon_cosine'])
    
    x = np.arange(3); w = 0.55
    
    # EV
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(x, [pali_ev, qwen_ev, llava_ev], w, color=model_colors, edgecolor='black', linewidth=1)
    ax.axhline(y=65, color='green', linestyle='--', linewidth=2, label='Target (>65%)')
    ax.set_ylabel('Explained Variance (%)'); ax.set_title('Explained Variance', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10); ax.set_ylim(0, 110); ax.legend(fontsize=10)
    for bar, val in zip(bars, [pali_ev, qwen_ev, llava_ev]):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    # Dead
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(x, [pali_dead, qwen_dead, llava_dead], w, color=model_colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Dead Feature Ratio (%)'); ax.set_title('Dead Feature Ratio', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10); ax.set_ylim(0, 100)
    for bar, val in zip(bars, [pali_dead, qwen_dead, llava_dead]):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    # L0
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(x, [pali_l0, qwen_l0, llava_l0], w, color=model_colors, edgecolor='black', linewidth=1)
    ax.axhspan(50, 300, color='green', alpha=0.12, label='Ideal (50-300)')
    ax.set_ylabel('Mean L0'); ax.set_title('Sparsity (Mean L0)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10); ax.legend(fontsize=10)
    for bar, val in zip(bars, [pali_l0, qwen_l0, llava_l0]):
        ax.annotate(f'{val:.0f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    # Cosine
    ax = fig.add_subplot(gs[1, 1])
    bars = ax.bar(x, [pali_cos, qwen_cos, llava_cos], w, color=model_colors, edgecolor='black', linewidth=1)
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='Target (>0.9)')
    ax.set_ylabel('Cosine Similarity'); ax.set_title('Reconstruction Quality', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10); ax.set_ylim(0.98, 1.002); ax.legend(fontsize=10)
    for bar, val in zip(bars, [pali_cos, qwen_cos, llava_cos]):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    fig.suptitle('SAE Quality Metrics: Three-Model Comparison', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'fig5_three_model_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig5_three_model_dashboard")


# ============================================================
# Figure 6: Sparsity vs Quality Tradeoff (all models)
# ============================================================
def fig6_tradeoff():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # PaLiGemma (Arabic + English)
    for i, layer in enumerate(PALIGEMMA_DATA['layers']):
        ax.scatter(PALIGEMMA_DATA['arabic']['mean_l0'][i], PALIGEMMA_DATA['arabic']['explained_var'][i],
                   c=COLORS['PaLiGemma-3B'], s=120, marker='o', alpha=0.7, edgecolor='black', linewidth=0.5, zorder=5)
        ax.scatter(PALIGEMMA_DATA['english']['mean_l0'][i], PALIGEMMA_DATA['english']['explained_var'][i],
                   c=COLORS['PaLiGemma-3B'], s=120, marker='s', alpha=0.8, edgecolor='black', linewidth=0.5, zorder=5)
    
    # Qwen2-VL
    for i in range(len(QWEN_DATA['layers'])):
        for lang, marker in [('arabic', 'o'), ('english', 's')]:
            ax.scatter(QWEN_DATA[lang]['mean_l0'][i], QWEN_DATA[lang]['explained_var'][i],
                       c=COLORS['Qwen2-VL-7B'], s=80, marker=marker, alpha=0.7, edgecolor='black', linewidth=0.5, zorder=4)
    
    # LLaVA
    for i in range(len(LLAVA_DATA['layers'])):
        for lang, marker in [('arabic', 'o'), ('english', 's')]:
            ax.scatter(LLAVA_DATA[lang]['mean_l0'][i], LLAVA_DATA[lang]['explained_var'][i],
                       c=COLORS['LLaVA-1.5-7B'], s=80, marker=marker, alpha=0.7, edgecolor='black', linewidth=0.5, zorder=4)
    
    ax.axhline(y=65, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='EV Target (65%)')
    ax.axvspan(50, 300, color='green', alpha=0.08, label='Ideal L0 Range (50-300)')
    
    # Legend
    handles = [
        mpatches.Patch(color=COLORS['PaLiGemma-3B'], label='PaLiGemma-3B'),
        mpatches.Patch(color=COLORS['Qwen2-VL-7B'], label='Qwen2-VL-7B'),
        mpatches.Patch(color=COLORS['LLaVA-1.5-7B'], label='LLaVA-1.5-7B'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Arabic'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='English'),
    ]
    ax.legend(handles=handles, loc='center left', fontsize=11, framealpha=0.9)
    
    ax.set_xlabel('Mean L0 (Active Features)', fontsize=14)
    ax.set_ylabel('Explained Variance (%)', fontsize=14)
    ax.set_title('Sparsity-Quality Tradeoff: All Models\nPaLiGemma achieves near-perfect EV but with lower sparsity',
                 fontsize=16, fontweight='bold')
    
    # Annotate clusters
    ax.annotate('PaLiGemma:\nHigh fidelity,\nlow sparsity', xy=(7500, 99.5), fontsize=11,
                fontweight='bold', color=COLORS['PaLiGemma-3B'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.annotate('LLaVA:\nBalanced', xy=(600, 79), fontsize=11,
                fontweight='bold', color=COLORS['LLaVA-1.5-7B'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.annotate('Qwen2-VL:\nModerate sparsity', xy=(1700, 55), fontsize=11,
                fontweight='bold', color=COLORS['Qwen2-VL-7B'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_sparsity_quality_tradeoff_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig6_sparsity_quality_tradeoff_all")


# ============================================================
# Figure 7: L0 as % of Features (normalized comparison)
# ============================================================
def fig7_normalized_sparsity():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute L0 as % of d_hidden
    pali_ar_pct = [l0 / PALIGEMMA_DATA['d_hidden'] * 100 for l0 in PALIGEMMA_DATA['arabic']['mean_l0']]
    pali_en_pct = [l0 / PALIGEMMA_DATA['d_hidden'] * 100 for l0 in PALIGEMMA_DATA['english']['mean_l0']]
    qwen_ar_pct = [l0 / QWEN_DATA['d_hidden'] * 100 for l0 in QWEN_DATA['arabic']['mean_l0']]
    qwen_en_pct = [l0 / QWEN_DATA['d_hidden'] * 100 for l0 in QWEN_DATA['english']['mean_l0']]
    llava_ar_pct = [l0 / LLAVA_DATA['d_hidden'] * 100 for l0 in LLAVA_DATA['arabic']['mean_l0']]
    llava_en_pct = [l0 / LLAVA_DATA['d_hidden'] * 100 for l0 in LLAVA_DATA['english']['mean_l0']]
    
    # Normalize layers to 0-1
    pali_x = normalize_layers(PALIGEMMA_DATA['layers'], 17)
    qwen_x = normalize_layers(QWEN_DATA['layers'], 27)
    llava_x = normalize_layers(LLAVA_DATA['layers'], 31)
    
    ax.plot(pali_x, pali_en_pct, 'D-', color=COLORS['PaLiGemma-3B'], linewidth=2.5, markersize=9, label='PaLiGemma-3B (English)')
    ax.plot(pali_x, pali_ar_pct, 'D--', color=COLORS['PaLiGemma-3B'], linewidth=1.5, markersize=7, alpha=0.6, label='PaLiGemma-3B (Arabic)')
    ax.plot(qwen_x, qwen_en_pct, 's-', color=COLORS['Qwen2-VL-7B'], linewidth=2, markersize=8, label='Qwen2-VL-7B (English)')
    ax.plot(qwen_x, qwen_ar_pct, 's--', color=COLORS['Qwen2-VL-7B'], linewidth=1.5, markersize=6, alpha=0.6, label='Qwen2-VL-7B (Arabic)')
    ax.plot(llava_x, llava_en_pct, 'o-', color=COLORS['LLaVA-1.5-7B'], linewidth=2, markersize=8, label='LLaVA-1.5-7B (English)')
    ax.plot(llava_x, llava_ar_pct, 'o--', color=COLORS['LLaVA-1.5-7B'], linewidth=1.5, markersize=6, alpha=0.6, label='LLaVA-1.5-7B (Arabic)')
    
    ax.axhspan(0, 2, color='green', alpha=0.1, label='Ideal (<2%)')
    
    ax.set_xlabel('Normalized Layer Position (0=early, 1=late)', fontsize=14)
    ax.set_ylabel('Active Features (% of d_hidden)', fontsize=14)
    ax.set_title('Normalized Sparsity: Active Features as % of Dictionary Size', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_normalized_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig7_normalized_sparsity")


# ============================================================
# Figure 8: Publication Summary Table (as image)
# ============================================================
def fig8_summary_table():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')
    
    pali_ev = np.mean(PALIGEMMA_DATA['arabic']['explained_var'] + PALIGEMMA_DATA['english']['explained_var'])
    pali_std_ev = np.std(PALIGEMMA_DATA['arabic']['explained_var'] + PALIGEMMA_DATA['english']['explained_var'])
    pali_dead = np.mean(PALIGEMMA_DATA['arabic']['dead_features'] + PALIGEMMA_DATA['english']['dead_features'])
    pali_std_dead = np.std(PALIGEMMA_DATA['arabic']['dead_features'] + PALIGEMMA_DATA['english']['dead_features'])
    pali_l0 = np.mean(PALIGEMMA_DATA['arabic']['mean_l0'] + PALIGEMMA_DATA['english']['mean_l0'])
    pali_std_l0 = np.std(PALIGEMMA_DATA['arabic']['mean_l0'] + PALIGEMMA_DATA['english']['mean_l0'])
    pali_cos = np.mean(PALIGEMMA_DATA['arabic']['recon_cosine'] + PALIGEMMA_DATA['english']['recon_cosine'])
    
    qwen_ev = np.mean(QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var'])
    qwen_std_ev = np.std(QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var'])
    qwen_dead = np.mean(QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'])
    qwen_std_dead = np.std(QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'])
    qwen_l0 = np.mean(QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0'])
    qwen_std_l0 = np.std(QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0'])
    qwen_cos = np.mean(QWEN_DATA['arabic']['recon_cosine'] + QWEN_DATA['english']['recon_cosine'])
    
    llava_ev = np.mean(LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var'])
    llava_std_ev = np.std(LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var'])
    llava_dead = np.mean(LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    llava_std_dead = np.std(LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    llava_l0 = np.mean(LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0'])
    llava_std_l0 = np.std(LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0'])
    llava_cos = np.mean(LLAVA_DATA['arabic']['recon_cosine'] + LLAVA_DATA['english']['recon_cosine'])
    
    headers = ['Model', 'd_model', 'Features', 'Explained Var%', 'Dead Features%', 'Mean L0', 'L0/d_hidden%', 'Recon Cosine']
    data = [
        ['PaLiGemma-3B', '2,048', '16,384', f'{pali_ev:.1f}±{pali_std_ev:.1f}',
         f'{pali_dead:.1f}±{pali_std_dead:.1f}', f'{pali_l0:.0f}±{pali_std_l0:.0f}',
         f'{pali_l0/16384*100:.1f}%', f'{pali_cos:.4f}'],
        ['Qwen2-VL-7B', '3,584', '28,672', f'{qwen_ev:.1f}±{qwen_std_ev:.1f}',
         f'{qwen_dead:.1f}±{qwen_std_dead:.1f}', f'{qwen_l0:.0f}±{qwen_std_l0:.0f}',
         f'{qwen_l0/28672*100:.1f}%', f'{qwen_cos:.4f}'],
        ['LLaVA-1.5-7B', '4,096', '32,768', f'{llava_ev:.1f}±{llava_std_ev:.1f}',
         f'{llava_dead:.1f}±{llava_std_dead:.1f}', f'{llava_l0:.0f}±{llava_std_l0:.0f}',
         f'{llava_l0/32768*100:.1f}%', f'{llava_cos:.4f}'],
        ['Target', '-', '-', '>65%', '-', '50-300', '<2%', '>0.9'],
    ]
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.0)
    
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#003366'); table[(0, j)].set_text_props(color='white', fontweight='bold')
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#FFF3E0')  # PaLiGemma orange tint
        table[(4, j)].set_facecolor('#E8F5E9'); table[(4, j)].set_text_props(fontweight='bold', color='green')
    
    ax.set_title('SAE Quality Metrics: Complete Three-Model Summary', fontsize=16, fontweight='bold', pad=20)
    footnote = "L0/d_hidden% shows the fraction of active features, allowing fair comparison across different dictionary sizes"
    ax.text(0.5, -0.05, footnote, transform=ax.transAxes, ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_summary_table_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig8_summary_table_all")


# ============================================================
# Figure 9: Radar / Spider Chart
# ============================================================
def fig9_radar():
    categories = ['Explained\nVariance', 'Sparsity\n(inverse L0%)', 'Alive\nFeatures', 'Reconstruction\nCosine']
    N = len(categories)
    
    pali_l0_pct = np.mean(PALIGEMMA_DATA['arabic']['mean_l0'] + PALIGEMMA_DATA['english']['mean_l0']) / PALIGEMMA_DATA['d_hidden'] * 100
    qwen_l0_pct = np.mean(QWEN_DATA['arabic']['mean_l0'] + QWEN_DATA['english']['mean_l0']) / QWEN_DATA['d_hidden'] * 100
    llava_l0_pct = np.mean(LLAVA_DATA['arabic']['mean_l0'] + LLAVA_DATA['english']['mean_l0']) / LLAVA_DATA['d_hidden'] * 100
    
    pali_alive = 100 - np.mean(PALIGEMMA_DATA['arabic']['dead_features'] + PALIGEMMA_DATA['english']['dead_features'])
    qwen_alive = 100 - np.mean(QWEN_DATA['arabic']['dead_features'] + QWEN_DATA['english']['dead_features'])
    llava_alive = 100 - np.mean(LLAVA_DATA['arabic']['dead_features'] + LLAVA_DATA['english']['dead_features'])
    
    # Normalize all to 0-100 scale
    values_pali = [
        min(np.mean(PALIGEMMA_DATA['arabic']['explained_var'] + PALIGEMMA_DATA['english']['explained_var']), 100),
        max(0, 100 - pali_l0_pct * 2),  # invert: lower L0% = better
        min(pali_alive * 2, 100),  # alive features
        np.mean(PALIGEMMA_DATA['arabic']['recon_cosine'] + PALIGEMMA_DATA['english']['recon_cosine']) * 100,
    ]
    values_qwen = [
        np.mean(QWEN_DATA['arabic']['explained_var'] + QWEN_DATA['english']['explained_var']),
        max(0, 100 - qwen_l0_pct * 2),
        min(qwen_alive * 2, 100),
        np.mean(QWEN_DATA['arabic']['recon_cosine'] + QWEN_DATA['english']['recon_cosine']) * 100,
    ]
    values_llava = [
        np.mean(LLAVA_DATA['arabic']['explained_var'] + LLAVA_DATA['english']['explained_var']),
        max(0, 100 - llava_l0_pct * 2),
        min(llava_alive * 2, 100),
        np.mean(LLAVA_DATA['arabic']['recon_cosine'] + LLAVA_DATA['english']['recon_cosine']) * 100,
    ]
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for values, name, color in [(values_pali, 'PaLiGemma-3B', COLORS['PaLiGemma-3B']),
                                  (values_qwen, 'Qwen2-VL-7B', COLORS['Qwen2-VL-7B']),
                                  (values_llava, 'LLaVA-1.5-7B', COLORS['LLaVA-1.5-7B'])]:
        vals = values + values[:1]
        ax.plot(angles, vals, 'o-', linewidth=2.5, markersize=8, label=name, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 105); ax.set_title('SAE Quality Profile Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig9_radar_comparison")


# ============================================================
# Figure 10: Key Insight - EV vs Sparsity Tradeoff with Annotations
# ============================================================
def fig10_key_insight():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # L0 as % of features
    pali_ar_l0_pct = [l0/PALIGEMMA_DATA['d_hidden']*100 for l0 in PALIGEMMA_DATA['arabic']['mean_l0']]
    pali_en_l0_pct = [l0/PALIGEMMA_DATA['d_hidden']*100 for l0 in PALIGEMMA_DATA['english']['mean_l0']]
    qwen_ar_l0_pct = [l0/QWEN_DATA['d_hidden']*100 for l0 in QWEN_DATA['arabic']['mean_l0']]
    qwen_en_l0_pct = [l0/QWEN_DATA['d_hidden']*100 for l0 in QWEN_DATA['english']['mean_l0']]
    llava_ar_l0_pct = [l0/LLAVA_DATA['d_hidden']*100 for l0 in LLAVA_DATA['arabic']['mean_l0']]
    llava_en_l0_pct = [l0/LLAVA_DATA['d_hidden']*100 for l0 in LLAVA_DATA['english']['mean_l0']]
    
    ax.scatter(pali_en_l0_pct, PALIGEMMA_DATA['english']['explained_var'],
               c=COLORS['PaLiGemma-3B'], s=180, marker='s', alpha=0.9, edgecolor='black', linewidth=1, zorder=5, label='PaLiGemma-3B (En)')
    ax.scatter(pali_ar_l0_pct, PALIGEMMA_DATA['arabic']['explained_var'],
               c=COLORS['PaLiGemma-3B'], s=120, marker='o', alpha=0.6, edgecolor='black', linewidth=0.5, zorder=5, label='PaLiGemma-3B (Ar)')
    ax.scatter(qwen_en_l0_pct, QWEN_DATA['english']['explained_var'],
               c=COLORS['Qwen2-VL-7B'], s=120, marker='s', alpha=0.8, edgecolor='black', linewidth=0.5, zorder=4, label='Qwen2-VL-7B (En)')
    ax.scatter(qwen_ar_l0_pct, QWEN_DATA['arabic']['explained_var'],
               c=COLORS['Qwen2-VL-7B'], s=80, marker='o', alpha=0.5, edgecolor='black', linewidth=0.5, zorder=3, label='Qwen2-VL-7B (Ar)')
    ax.scatter(llava_en_l0_pct, LLAVA_DATA['english']['explained_var'],
               c=COLORS['LLaVA-1.5-7B'], s=120, marker='s', alpha=0.8, edgecolor='black', linewidth=0.5, zorder=4, label='LLaVA-1.5-7B (En)')
    ax.scatter(llava_ar_l0_pct, LLAVA_DATA['arabic']['explained_var'],
               c=COLORS['LLaVA-1.5-7B'], s=80, marker='o', alpha=0.5, edgecolor='black', linewidth=0.5, zorder=3, label='LLaVA-1.5-7B (Ar)')
    
    ax.axhline(y=65, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvspan(0, 2, color='green', alpha=0.08)
    
    # Quadrant labels
    ax.text(1, 102, '✓ IDEAL\nHigh EV + Low L0%', fontsize=13, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), ha='center')
    ax.text(45, 102, '⚠ HIGH FIDELITY\nHigh EV + High L0%', fontsize=13, fontweight='bold', color=COLORS['PaLiGemma-3B'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), ha='center')
    
    ax.set_xlabel('Active Features (% of dictionary size)', fontsize=14)
    ax.set_ylabel('Explained Variance (%)', fontsize=14)
    ax.set_title('Key Finding: Reconstruction-Sparsity Tradeoff Across Models', fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_key_insight.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig10_key_insight")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating 3-Model SAE Quality Visualizations")
    print("=" * 60)
    
    fig1_explained_variance_all()
    fig2_dead_features_all()
    fig3_mean_l0_all()
    fig4_recon_cosine_all()
    fig5_dashboard()
    fig6_tradeoff()
    fig7_normalized_sparsity()
    fig8_summary_table()
    fig9_radar()
    fig10_key_insight()
    
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
