#!/usr/bin/env python3
"""
Generate all publication-quality figures for the CLMB research presentation.
Produces 10 figures covering SAE quality, probes, CLBAS, intervention, and more.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────
COLORS = {
    'paligemma': '#2196F3',   # blue
    'qwen':      '#FF9800',   # orange
    'llava':     '#4CAF50',   # green
    'llama':     '#9C27B0',   # purple
    'arabic':    '#E53935',   # red
    'english':   '#1E88E5',   # blue
    'targeted':  '#D32F2F',   # dark red
    'random':    '#9E9E9E',   # grey
    'baseline':  '#424242',   # dark grey
}

MODEL_NAMES = ['PaLiGemma-3B', 'Qwen2-VL-7B', 'LLaVA-1.5-7B', 'Llama-3.2-Vision']
MODEL_KEYS = ['paligemma', 'qwen', 'llava', 'llama']
MODEL_COLORS = [COLORS[k] for k in MODEL_KEYS]

# =====================================================================
# FIGURE 1: SAE Quality — Explained Variance Comparison
# =====================================================================
def fig1_sae_quality():
    ev_means = [94.8, 71.7, 83.5, 99.9]
    ev_stds  = [17.3, 12.6, 3.2,  0.1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(MODEL_NAMES, ev_means, color=MODEL_COLORS, edgecolor='white',
                  linewidth=1.2, width=0.6, yerr=ev_stds, capsize=6,
                  error_kw={'linewidth': 1.5})

    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('SAE Reconstruction Quality Across 4 VLMs')
    ax.set_ylim(0, 115)
    ax.axhline(y=65, color='grey', linestyle='--', alpha=0.5, label='Anthropic baseline (≥65%)')

    for bar, val, std in zip(bars, ev_means, ev_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_sae_quality.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig1_sae_quality.pdf'))
    plt.close(fig)
    print("✓ Figure 1: SAE Quality")


# =====================================================================
# FIGURE 2: SAE Quality — Multi-metric Dashboard
# =====================================================================
def fig2_sae_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # a) Explained Variance
    ev = [94.8, 71.7, 83.5, 99.9]
    axes[0, 0].bar(MODEL_NAMES, ev, color=MODEL_COLORS, width=0.6)
    axes[0, 0].set_title('(a) Explained Variance (%)')
    axes[0, 0].set_ylim(0, 110)
    for i, v in enumerate(ev):
        axes[0, 0].text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold')

    # b) Dead Features
    df = [53.5, 78.0, 95.0, 2.9]
    axes[0, 1].bar(MODEL_NAMES, df, color=MODEL_COLORS, width=0.6)
    axes[0, 1].set_title('(b) Dead Features (%)')
    axes[0, 1].set_ylim(0, 105)
    for i, v in enumerate(df):
        axes[0, 1].text(i, v + 1.5, f'{v:.1f}', ha='center', fontweight='bold')

    # c) Mean L0 Sparsity
    l0 = [7435, 1633, 1025, 14182]
    axes[1, 0].bar(MODEL_NAMES, l0, color=MODEL_COLORS, width=0.6)
    axes[1, 0].set_title('(c) Mean L0 (Active Features)')
    for i, v in enumerate(l0):
        axes[1, 0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold', fontsize=9)

    # d) Reconstruction Cosine
    cos = [0.9995, 0.9950, 0.9945, 0.9998]
    axes[1, 1].bar(MODEL_NAMES, cos, color=MODEL_COLORS, width=0.6)
    axes[1, 1].set_title('(d) Reconstruction Cosine Similarity')
    axes[1, 1].set_ylim(0.990, 1.001)
    for i, v in enumerate(cos):
        axes[1, 1].text(i, v + 0.0002, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)

    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15)

    fig.suptitle('SAE Quality Metrics Across 4 Vision-Language Models', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_sae_dashboard.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig2_sae_dashboard.pdf'))
    plt.close(fig)
    print("✓ Figure 2: SAE Dashboard")


# =====================================================================
# FIGURE 3: Cross-Lingual Probe Accuracy — Grouped Bar Chart
# =====================================================================
def fig3_probe_accuracy():
    arabic  = [88.6, 90.3, 89.9, 98.5]
    english = [85.3, 91.8, 96.3, 99.4]
    gap     = [e - a for a, e in zip(arabic, english)]

    x = np.arange(len(MODEL_NAMES))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_ar = ax.bar(x - width/2, arabic, width, color=COLORS['arabic'],
                     label='Arabic', edgecolor='white', linewidth=1)
    bars_en = ax.bar(x + width/2, english, width, color=COLORS['english'],
                     label='English', edgecolor='white', linewidth=1)

    # Gap annotations
    for i, g in enumerate(gap):
        color = '#D32F2F' if abs(g) > 3 else '#FF9800' if abs(g) > 1 else '#4CAF50'
        arrow = '↑' if g > 0 else '↓'
        ax.annotate(f'{arrow} {g:+.1f}%', xy=(x[i], max(arabic[i], english[i]) + 1),
                    ha='center', fontsize=10, fontweight='bold', color=color)

    ax.set_ylabel('Probe Accuracy (%)')
    ax.set_title('Cross-Lingual Gender Probe Accuracy (Arabic vs. English)')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylim(80, 105)
    ax.legend(loc='lower right')

    # Training regime labels
    regimes = ['Translation', 'Native Multi', 'EN-only', 'Native Multi']
    for i, r in enumerate(regimes):
        ax.text(i, 81, r, ha='center', fontsize=8, fontstyle='italic', color='grey')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_probe_accuracy.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig3_probe_accuracy.pdf'))
    plt.close(fig)
    print("✓ Figure 3: Probe Accuracy")


# =====================================================================
# FIGURE 4: Cross-Lingual Feature Overlap Heatmap
# =====================================================================
def fig4_feature_overlap():
    overlap = np.array([
        [0.5, 0.1, 0.1, 0.7],   # Jaccard (%)
    ])
    cosine = np.array([
        [-0.003, 0.000, 0.001, 0.003],
    ])
    clbas = np.array([
        [0.1083, 0.0040, 0.0150, 0.0039],
    ])

    fig, axes = plt.subplots(3, 1, figsize=(8, 7))

    # Jaccard overlap
    im0 = axes[0].imshow(overlap, cmap='Reds', aspect='auto', vmin=0, vmax=2)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(MODEL_NAMES)
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['Jaccard\nOverlap (%)'])
    axes[0].set_title('Cross-Lingual Gender Feature Metrics')
    for j in range(4):
        axes[0].text(j, 0, f'{overlap[0, j]:.1f}%', ha='center', va='center',
                     fontweight='bold', fontsize=12, color='white' if overlap[0, j] > 0.5 else 'black')
    plt.colorbar(im0, ax=axes[0], shrink=0.6)

    # Cosine
    im1 = axes[1].imshow(cosine, cmap='RdBu_r', aspect='auto', vmin=-0.05, vmax=0.05)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(MODEL_NAMES)
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(['Cosine\nSimilarity'])
    for j in range(4):
        axes[1].text(j, 0, f'{cosine[0, j]:.3f}', ha='center', va='center',
                     fontweight='bold', fontsize=12)
    plt.colorbar(im1, ax=axes[1], shrink=0.6)

    # CLBAS
    im2 = axes[2].imshow(clbas, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.15)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(MODEL_NAMES)
    axes[2].set_yticks([0])
    axes[2].set_yticklabels(['CLBAS\nScore'])
    for j in range(4):
        axes[2].text(j, 0, f'{clbas[0, j]:.4f}', ha='center', va='center',
                     fontweight='bold', fontsize=12, color='white' if clbas[0, j] > 0.06 else 'black')
    plt.colorbar(im2, ax=axes[2], shrink=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_feature_overlap.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig4_feature_overlap.pdf'))
    plt.close(fig)
    print("✓ Figure 4: Feature Overlap")


# =====================================================================
# FIGURE 5: CLBAS Summary — Radar-style or Bar
# =====================================================================
def fig5_clbas_summary():
    clbas = [0.1083, 0.0040, 0.0150, 0.0039]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(MODEL_NAMES, clbas, color=MODEL_COLORS, width=0.6, edgecolor='white', linewidth=1.2)

    ax.set_ylabel('CLBAS Score')
    ax.set_title('Cross-Lingual Bias Alignment Score (Lower = More Shared Bias)')

    for bar, val in zip(bars, clbas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add interpretation zone
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Threshold: Moderate alignment')
    ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Threshold: Very low alignment')
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig5_clbas_summary.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig5_clbas_summary.pdf'))
    plt.close(fig)
    print("✓ Figure 5: CLBAS Summary")


# =====================================================================
# FIGURE 6: Intervention — Before vs After Ablation
# =====================================================================
def fig6_intervention():
    categories = ['Total\nGender\nTerms', 'he/his/him', 'she/her', 'man/woman', 'boy/girl']
    baseline   = [83, 15, 10, 39, 12]
    ablated    = [58, 0,  4,  40, 8]
    change_pct = [((a - b)/b)*100 if b > 0 else 0 for b, a in zip(baseline, ablated)]

    x = np.arange(len(categories))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Left: grouped bars
    bars_b = ax1.bar(x - width/2, baseline, width, color=COLORS['baseline'],
                     label='Baseline', edgecolor='white')
    bars_a = ax1.bar(x + width/2, ablated, width, color=COLORS['targeted'],
                     label='After Ablation (k=100)', edgecolor='white')

    ax1.set_ylabel('Count')
    ax1.set_title('Gender Term Counts: Baseline vs. Targeted Ablation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper right')

    for i in range(len(categories)):
        ax1.text(i - width/2, baseline[i] + 0.8, str(baseline[i]), ha='center', fontweight='bold', fontsize=10)
        ax1.text(i + width/2, ablated[i] + 0.8, str(ablated[i]), ha='center', fontweight='bold', fontsize=10)

    # Right: percentage change waterfall
    colors_change = ['#D32F2F' if c < -10 else '#FF9800' if c < 0 else '#4CAF50' for c in change_pct]
    bars_c = ax2.barh(categories, change_pct, color=colors_change, edgecolor='white', height=0.5)
    ax2.set_xlabel('Change (%)')
    ax2.set_title('Percentage Change')
    ax2.axvline(x=0, color='black', linewidth=0.8)

    for i, v in enumerate(change_pct):
        ax2.text(v - 3 if v < 0 else v + 1, i, f'{v:+.0f}%', va='center',
                 fontweight='bold', fontsize=10)

    fig.suptitle('Causal Intervention: SAE Feature Ablation on PaLiGemma-3B (Layer 9)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig6_intervention.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig6_intervention.pdf'))
    plt.close(fig)
    print("✓ Figure 6: Intervention Results")


# =====================================================================
# FIGURE 7: Targeted vs Random Ablation Control
# =====================================================================
def fig7_ablation_control():
    conditions = ['Baseline', 'Targeted\n(k=100)', 'Random\nRun 1', 'Random\nRun 2', 'Random\nRun 3']
    values = [318, 257, 309, 284, 289]
    colors = [COLORS['baseline'], COLORS['targeted'],
              COLORS['random'], COLORS['random'], COLORS['random']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart
    bars = ax1.bar(conditions, values, color=colors, edgecolor='white', width=0.6)
    ax1.set_ylabel('Total Gender Terms')
    ax1.set_title('Targeted vs. Random Feature Ablation')
    ax1.set_ylim(200, 340)

    for bar, val in zip(bars, values):
        pct = ((val - 318) / 318) * 100
        label = f'{val}\n({pct:+.1f}%)' if val != 318 else f'{val}\n(baseline)'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 label, ha='center', fontweight='bold', fontsize=10)

    # Mean random line
    random_mean = np.mean([309, 284, 289])
    ax1.axhline(y=random_mean, color=COLORS['random'], linestyle='--', alpha=0.7,
                label=f'Random mean = {random_mean:.0f}')
    ax1.legend()

    # Right: effect comparison
    conditions2 = ['Targeted\nEffect', 'Random Mean\nEffect', 'Specificity\nΔ']
    values2 = [19.2, 7.5, 11.6]
    colors2 = [COLORS['targeted'], COLORS['random'], '#1565C0']

    bars2 = ax2.bar(conditions2, values2, color=colors2, edgecolor='white', width=0.5)
    ax2.set_ylabel('Effect Size (percentage points)')
    ax2.set_title('Effect Specificity Analysis')
    ax2.set_ylim(0, 28)

    for bar, val in zip(bars2, values2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f} pp', ha='center', fontweight='bold', fontsize=12)

    # Error bar on random
    ax2.errorbar(1, 7.5, yerr=3.4, fmt='none', ecolor='black', capsize=8, capthick=2)

    # Annotation: 2.5x
    ax2.annotate('2.5× more\neffective', xy=(0, 19.2), xytext=(0.5, 24),
                 fontsize=11, fontweight='bold', color='#D32F2F',
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2))

    fig.suptitle('Matched-Baseline Random Ablation Control Experiment',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig7_ablation_control.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig7_ablation_control.pdf'))
    plt.close(fig)
    print("✓ Figure 7: Ablation Control")


# =====================================================================
# FIGURE 8: Translation Amplification
# =====================================================================
def fig8_translation_amplification():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: gender word counts
    langs = ['English', 'Arabic']
    counts = [5518, 7932]
    lang_colors = [COLORS['english'], COLORS['arabic']]

    bars = ax1.bar(langs, counts, color=lang_colors, width=0.5, edgecolor='white')
    ax1.set_ylabel('Gender-Marked Words in Captions')
    ax1.set_title('Translation Amplification Effect')
    ax1.set_ylim(0, 9500)

    for bar, val in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{val:,}', ha='center', fontweight='bold', fontsize=13)

    # Arrow showing 1.44x
    ax1.annotate('1.44×', xy=(1, 7932), xytext=(0.5, 8500),
                 fontsize=16, fontweight='bold', color='#D32F2F',
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5))

    # Right: probe accuracy inversion
    models = ['PaLiGemma', 'Qwen2-VL', 'LLaVA', 'Llama']
    ar_acc = [88.6, 90.3, 89.9, 98.5]
    en_acc = [85.3, 91.8, 96.3, 99.4]
    gap = [e - a for a, e in zip(ar_acc, en_acc)]

    x = np.arange(len(models))
    ax2.bar(x, gap, color=[('#D32F2F' if g < 0 else '#1E88E5') for g in gap],
            width=0.5, edgecolor='white')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('EN − AR Accuracy Gap (%)')
    ax2.set_title('Cross-Lingual Accuracy Gap by Model')

    for i, g in enumerate(gap):
        ax2.text(i, g + (0.2 if g >= 0 else -0.5), f'{g:+.1f}%',
                 ha='center', fontweight='bold', fontsize=11,
                 color='#D32F2F' if g < 0 else '#1E88E5')

    # Highlight PaLiGemma
    ax2.annotate('Translation\nAmplification', xy=(0, gap[0]), xytext=(0.8, -5),
                 fontsize=10, fontweight='bold', color='#D32F2F',
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))

    fig.suptitle('Translation Pipeline Amplifies Gender Signal in Arabic',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig8_translation_amplification.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig8_translation_amplification.pdf'))
    plt.close(fig)
    print("✓ Figure 8: Translation Amplification")


# =====================================================================
# FIGURE 9: Pronoun vs Noun Differential Effect
# =====================================================================
def fig9_pronoun_noun():
    categories = ['he/his/him', 'she/her', 'boy/girl', 'man/woman']
    baseline   = [15, 10, 12, 39]
    ablated    = [0,  4,  8,  40]
    reduction  = [100, 60, 33, -2.6]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Horizontal bar chart of reduction percentage
    colors = ['#B71C1C' if r > 50 else '#E53935' if r > 20 else '#FF9800' if r > 0 else '#4CAF50'
              for r in reduction]

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, reduction, color=colors, height=0.5, edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlabel('Reduction in Gender Terms (%)')
    ax.set_title('Differential Effect: Pronouns vs. Nouns After SAE Ablation')
    ax.axvline(x=0, color='black', linewidth=0.8)

    for i, (r, b, a) in enumerate(zip(reduction, baseline, ablated)):
        label = f'{r:+.0f}%  ({b}→{a})'
        ax.text(r + 2 if r >= 0 else r - 2, i, label, va='center',
                fontweight='bold', fontsize=11, ha='left' if r >= 0 else 'right')

    # Divider line between pronouns and nouns
    ax.axhline(y=1.5, color='grey', linestyle=':', alpha=0.5)
    ax.text(80, 0.5, 'PRONOUNS', fontsize=10, fontweight='bold', color='grey',
            ha='center', va='center', style='italic')
    ax.text(80, 2.5, 'NOUNS', fontsize=10, fontweight='bold', color='grey',
            ha='center', va='center', style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig9_pronoun_noun.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig9_pronoun_noun.pdf'))
    plt.close(fig)
    print("✓ Figure 9: Pronoun vs Noun Effect")


# =====================================================================
# FIGURE 10: Literature Positioning — Novelty Map
# =====================================================================
def fig10_novelty_map():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define papers as (x=cross_lingual, y=mechanistic_depth, size, label, color)
    papers = [
        (0.2, 0.9, 400, 'Templeton+24\n(Anthropic SAEs)', '#90CAF9'),
        (0.2, 0.8, 350, 'Gao+24\n(OpenAI SAEs)', '#90CAF9'),
        (0.15, 0.7, 300, 'Cunningham+23\n(SAE foundations)', '#90CAF9'),
        (0.1, 0.6, 250, 'Marks+24\n(Sparse Circuits)', '#90CAF9'),
        (0.8, 0.15, 300, 'Bolukbasi+16\n(Word2Vec bias)', '#A5D6A7'),
        (0.6, 0.2, 250, 'Cho+22\n(DALL-Eval)', '#A5D6A7'),
        (0.7, 0.1, 200, 'Birhane+21\n(Multimodal bias)', '#A5D6A7'),
        (0.3, 0.3, 200, 'Tigges+23\n(Linear bias dirs)', '#FFF59D'),
        (0.4, 0.5, 200, 'Olson+25\n(Vision SAEs)', '#FFF59D'),
        # Our work
        (0.85, 0.88, 600, 'OUR WORK\n(CLMB)', '#FF5252'),
    ]

    for x, y, s, label, color in papers:
        ax.scatter(x, y, s=s, c=color, alpha=0.85, edgecolors='black', linewidth=1.2, zorder=3)
        offset_y = 0.05 if 'OUR' not in label else -0.08
        ax.text(x, y + offset_y, label, ha='center', va='bottom', fontsize=8,
                fontweight='bold' if 'OUR' in label else 'normal')

    ax.set_xlabel('Cross-Lingual Scope  →', fontsize=12)
    ax.set_ylabel('Mechanistic Interpretability Depth  →', fontsize=12)
    ax.set_title('Research Positioning: CLMB in the Literature Landscape', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#90CAF9', label='SAE / Mechanistic Interp.'),
        mpatches.Patch(color='#A5D6A7', label='Bias / Fairness'),
        mpatches.Patch(color='#FFF59D', label='Hybrid / Bridge'),
        mpatches.Patch(color='#FF5252', label='Our Work (CLMB)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

    # Quadrant labels
    ax.text(0.1, 1.0, 'Deep Mechanistic\n(English only)', fontsize=9, color='grey',
            ha='center', style='italic', alpha=0.7)
    ax.text(0.9, 0.05, 'Cross-lingual\n(No mechanistic)', fontsize=9, color='grey',
            ha='center', style='italic', alpha=0.7)
    ax.text(0.9, 1.0, 'NOVEL REGION\n(Mechanistic +\nCross-lingual)', fontsize=10,
            color='#D32F2F', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCDD2', alpha=0.8))

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig10_novelty_map.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig10_novelty_map.pdf'))
    plt.close(fig)
    print("✓ Figure 10: Novelty Map")


# =====================================================================
# FIGURE 11: CLMB Pipeline Diagram
# =====================================================================
def fig11_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Pipeline stages
    stages = [
        (1.0, 3, 'Data\nPreparation', '40K pairs\nbilingual', '#E3F2FD'),
        (3.5, 3, 'Activation\nExtraction', '4 VLMs\nall layers', '#E8F5E9'),
        (6.0, 3, 'SAE\nTraining', '8× expansion\n71-99.9% EV', '#FFF3E0'),
        (8.5, 3, 'Feature\nAnalysis', "Cohen's d\ntop-k features", '#F3E5F5'),
        (11.0, 3, 'Cross-Lingual\nAnalysis', 'CLBAS\n<1% overlap', '#FFEBEE'),
        (13.5, 3, 'Causal\nIntervention', '-19.2%\ngender terms', '#E8EAF6'),
    ]

    for x, y, title, detail, color in stages:
        rect = mpatches.FancyBboxPatch((x - 1, y - 1), 2, 2,
                                        boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y + 0.3, title, ha='center', va='center', fontweight='bold', fontsize=10)
        ax.text(x, y - 0.4, detail, ha='center', va='center', fontsize=8, color='#555')

    # Arrows
    for i in range(len(stages) - 1):
        ax.annotate('', xy=(stages[i+1][0] - 1.1, 3),
                    xytext=(stages[i][0] + 1.1, 3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    ax.set_title('CLMB Research Pipeline: 7-Stage Methodology', fontsize=15, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig11_pipeline.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig11_pipeline.pdf'))
    plt.close(fig)
    print("✓ Figure 11: Pipeline Diagram")


# =====================================================================
# FIGURE 12: Summary — Key Numbers at a Glance
# =====================================================================
def fig12_summary():
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.3)

    metrics = [
        ('4', 'VLMs\nAnalyzed', '#2196F3'),
        ('40K', 'Image-Caption\nPairs', '#4CAF50'),
        ('<1%', 'Feature\nOverlap', '#FF9800'),
        ('99.9%', 'Best SAE\nExplained Var.', '#9C27B0'),
        ('2.25', "Cohen's d\nEffect Size", '#E53935'),
        ('−19.2%', 'Targeted\nAblation', '#D32F2F'),
        ('100%', 'Pronoun\nElimination', '#00BCD4'),
        ('1.44×', 'Translation\nAmplification', '#FF5722'),
    ]

    for idx, (number, label, color) in enumerate(metrics):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Background circle
        circle = plt.Circle((0.5, 0.55), 0.35, color=color, alpha=0.15)
        ax.add_patch(circle)

        ax.text(0.5, 0.6, number, ha='center', va='center',
                fontsize=28, fontweight='bold', color=color)
        ax.text(0.5, 0.15, label, ha='center', va='center',
                fontsize=10, color='#333')

    fig.suptitle('CLMB Research: Key Numbers at a Glance', fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(OUT_DIR, 'fig12_summary.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig12_summary.pdf'))
    plt.close(fig)
    print("✓ Figure 12: Summary Dashboard")


# =====================================================================
# RUN ALL
# =====================================================================
if __name__ == '__main__':
    print(f"Generating figures in: {OUT_DIR}")
    print("=" * 60)

    fig1_sae_quality()
    fig2_sae_dashboard()
    fig3_probe_accuracy()
    fig4_feature_overlap()
    fig5_clbas_summary()
    fig6_intervention()
    fig7_ablation_control()
    fig8_translation_amplification()
    fig9_pronoun_noun()
    fig10_novelty_map()
    fig11_pipeline()
    fig12_summary()

    print("=" * 60)
    print(f"✅ All 12 figures saved to: {OUT_DIR}")
    print("   Formats: PNG (300 DPI) + PDF (vector)")
