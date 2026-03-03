#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for SAE Gender Bias Paper
==============================================================

Creates all main and supplementary figures from final intervention results.
All 5 experiments are complete (PaLiGemma L9/L17/L9+17, Qwen2-VL L12, Llama L20).

Justification for each figure:
- Fig 1: Cross-model intervention comparison (main result, answers RQ2+RQ4)
- Fig 2: Paired delta distributions (shows statistical rigor per reviewer request)
- Fig 3: Per-term heatmap (reveals mechanistic patterns in word-choice effects)
- Fig 4: Layer specificity (PaLiGemma L9 vs L17, answers "is it layer-specific?")
- Fig 5: SAE quality vs intervention effect (controls for SAE quality confound)
- Fig 6: Random ablation distributions (25-run control validity)

Created: 3 March 2026
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')
RESULTS_DIR = BASE_DIR / 'results' / 'improved_intervention'
OUTPUT_DIR = BASE_DIR / 'publication' / 'figures' / 'main'
SUPP_DIR = BASE_DIR / 'publication' / 'figures' / 'supplementary'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (colorblind-friendly)
COLORS = {
    'paligemma': '#2196F3',   # blue
    'qwen2vl': '#FF9800',     # orange
    'llama': '#4CAF50',       # green
    'baseline': '#757575',    # gray
    'targeted': '#E53935',    # red
    'random': '#9E9E9E',      # light gray
}


def load_results():
    """Load all result files."""
    results = {}
    for name, folder in [
        ('PaLiGemma L9', 'paligemma_L9'),
        ('PaLiGemma L17', 'paligemma_L17'),
        ('PaLiGemma L9+17', 'paligemma_L9_17'),
        ('Qwen2-VL L12', 'qwen2vl_L12'),
        ('Llama L20', 'llama32vision_L20'),
    ]:
        path = RESULTS_DIR / folder / 'summary_only.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results[name] = data['summary']
    return results


# ============================================================
# Figure 1: Cross-Model Intervention Comparison (Bar Chart)
# ============================================================
def fig1_cross_model_comparison(results):
    """
    Main result figure showing targeted vs random ablation across 3 models.
    Justification: This is the paper's central finding — causal effect of
    SAE-identified gender features across architectures (RQ2 + RQ4).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    models = ['PaLiGemma L9', 'Qwen2-VL L12', 'Llama L20']
    labels = ['PaLiGemma-3B\n(L9)', 'Qwen2-VL-7B\n(L12)', 'Llama-3.2-11B\n(L20)']
    colors = [COLORS['paligemma'], COLORS['qwen2vl'], COLORS['llama']]

    # Panel A: Targeted vs Random change %
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35

    targeted_vals = [results[m]['targeted_change_pct'] for m in models]
    random_vals = [results[m]['random_change_mean_pct'] for m in models]
    random_errs = [results[m]['random_change_std_pct'] for m in models]

    bars_t = ax.bar(x - width/2, targeted_vals, width, color=colors, edgecolor='black',
                     linewidth=0.8, label='Targeted ablation', alpha=0.9)
    bars_r = ax.bar(x + width/2, random_vals, width, yerr=random_errs,
                     color=[COLORS['random']]*3, edgecolor='black', linewidth=0.8,
                     label='Random ablation (mean±std)', capsize=4, alpha=0.7)

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_ylabel('Gender Term Change (%)')
    ax.set_title('(a) Targeted vs. Random Ablation Effect')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', framealpha=0.9)

    # Add significance markers
    for i, m in enumerate(models):
        ps = results[m].get('paired_statistics', {})
        diff = ps.get('difference_targeted_minus_random', {})
        ci = diff.get('ci_95', [0, 0])
        if ci[0] > 0 or ci[1] < 0:  # CI excludes zero
            y_pos = max(targeted_vals[i], random_vals[i] + random_errs[i]) + 1.5
            if targeted_vals[i] < 0:
                y_pos = min(targeted_vals[i], random_vals[i] - random_errs[i]) - 2.5
            ax.annotate('**', xy=(x[i], y_pos), ha='center', fontsize=14, fontweight='bold')

    # Panel B: Effect specificity (ratio)
    ax = axes[1]
    specificity = [results[m]['effect_specificity_pct'] for m in models]
    ratios = [results[m]['ratio_targeted_vs_random'] for m in models]

    bars = ax.bar(x, [abs(r) for r in ratios], 0.5, color=colors, edgecolor='black',
                   linewidth=0.8, alpha=0.9)

    ax.axhline(y=1, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Ratio = 1 (no specificity)')
    ax.set_ylabel('|Targeted / Random| Effect Ratio')
    ax.set_title('(b) Ablation Specificity Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', framealpha=0.9)

    # Add ratio labels
    for i, (bar, r) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{abs(r):.1f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_cross_model_intervention.png')
    plt.savefig(OUTPUT_DIR / 'fig1_cross_model_intervention.pdf')
    plt.close()
    print(f"✓ Fig 1 saved: {OUTPUT_DIR / 'fig1_cross_model_intervention.png'}")


# ============================================================
# Figure 2: Paired Delta Distributions (Box/Violin Plot)
# ============================================================
def fig2_paired_deltas(results):
    """
    Per-image paired delta distributions with bootstrap CIs.
    Justification: Reviewer requested per-image paired statistics to address
    the concern that aggregate counts hide per-image variance. Shows the
    distribution of per-image effects, not just means.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['PaLiGemma L9', 'Qwen2-VL L12', 'Llama L20']
    labels = ['PaLiGemma-3B\n(L9)', 'Qwen2-VL-7B\n(L12)', 'Llama-3.2-11B\n(L20)']
    colors = [COLORS['paligemma'], COLORS['qwen2vl'], COLORS['llama']]

    x_positions = np.arange(len(models))

    for i, m in enumerate(models):
        ps = results[m].get('paired_statistics', {})
        td = ps.get('targeted_delta', {})
        dd = ps.get('difference_targeted_minus_random', {})

        # Targeted delta (point + CI)
        mean_t = td.get('mean', 0)
        ci_t = td.get('ci_95', [0, 0])

        # Difference delta (point + CI)
        mean_d = dd.get('mean', 0)
        ci_d = dd.get('ci_95', [0, 0])

        # Plot targeted delta
        ax.errorbar(x_positions[i] - 0.15, mean_t,
                    yerr=[[mean_t - ci_t[0]], [ci_t[1] - mean_t]],
                    fmt='o', color=colors[i], markersize=8, capsize=6,
                    capthick=2, linewidth=2, label=f'{labels[i].split(chr(10))[0]} (targeted Δ)' if i == 0 else '')

        # Plot difference (targeted - random)
        ax.errorbar(x_positions[i] + 0.15, mean_d,
                    yerr=[[mean_d - ci_d[0]], [ci_d[1] - mean_d]],
                    fmt='s', color=colors[i], markersize=8, capsize=6,
                    capthick=2, linewidth=2, markerfacecolor='white', markeredgewidth=2)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Custom legend
    targeted_marker = plt.Line2D([0], [0], marker='o', color='gray', markersize=8,
                                  linestyle='none', label='Targeted Δ (vs baseline)')
    diff_marker = plt.Line2D([0], [0], marker='s', color='gray', markersize=8,
                              markerfacecolor='white', markeredgewidth=2,
                              linestyle='none', label='Targeted−Random Δ')
    color_patches = [mpatches.Patch(color=c, label=l.replace('\n', ' '))
                     for c, l in zip(colors, labels)]
    ax.legend(handles=color_patches + [targeted_marker, diff_marker],
              loc='lower left', framealpha=0.9, ncol=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Per-Image Gender Term Δ (mean ± 95% CI)')
    ax.set_title('Per-Image Paired Statistics: Targeted Ablation Effect')
    ax.set_xlim(-0.5, len(models) - 0.5)

    # Annotate CIs that exclude zero
    for i, m in enumerate(models):
        ps = results[m].get('paired_statistics', {})
        dd = ps.get('difference_targeted_minus_random', {})
        ci = dd.get('ci_95', [0, 0])
        if ci[0] > 0 or ci[1] < 0:
            y_pos = max(dd.get('mean', 0), ci[1]) + 0.03
            ax.annotate('CI excl. 0', xy=(x_positions[i] + 0.15, y_pos),
                       ha='center', fontsize=8, fontstyle='italic', color=colors[i])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_paired_deltas.png')
    plt.savefig(OUTPUT_DIR / 'fig2_paired_deltas.pdf')
    plt.close()
    print(f"✓ Fig 2 saved: {OUTPUT_DIR / 'fig2_paired_deltas.png'}")


# ============================================================
# Figure 3: Per-Term Heatmap (All 3 Models)
# ============================================================
def fig3_per_term_heatmap(results):
    """
    Heatmap of per-term % change across all 3 models.
    Justification: Reveals mechanistic patterns — which specific gender terms
    are affected by ablation and how the effect differs across architectures.
    Shows that PaLiGemma reduces pronouns while Qwen2-VL/Llama increase nouns.
    """
    models = ['PaLiGemma L9', 'Qwen2-VL L12', 'Llama L20']
    model_labels = ['PaLiGemma-3B', 'Qwen2-VL-7B', 'Llama-3.2-11B']

    # Collect all terms across models
    all_terms = set()
    for m in models:
        all_terms.update(results[m].get('baseline_per_term', {}).keys())
        all_terms.update(results[m].get('targeted_per_term', {}).keys())

    # Sort by total baseline frequency
    term_freqs = {}
    for t in all_terms:
        freq = sum(results[m].get('baseline_per_term', {}).get(t, 0) for m in models)
        term_freqs[t] = freq
    terms = sorted(term_freqs.keys(), key=lambda x: term_freqs[x], reverse=True)
    terms = [t for t in terms if term_freqs[t] >= 10]  # filter rare terms

    # Build matrix
    matrix = np.zeros((len(models), len(terms)))
    for i, m in enumerate(models):
        base = results[m].get('baseline_per_term', {})
        targ = results[m].get('targeted_per_term', {})
        for j, t in enumerate(terms):
            b = base.get(t, 0)
            a = targ.get(t, 0)
            if b > 0:
                matrix[i, j] = (a - b) / b * 100
            elif a > 0:
                matrix[i, j] = 100  # appeared from zero

    fig, ax = plt.subplots(figsize=(12, 3.5))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 50)
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title('Per-Term Gender Change (%) Under Targeted Ablation')

    # Add value labels
    for i in range(len(models)):
        for j in range(len(terms)):
            val = matrix[i, j]
            if abs(val) > 5:
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:+.0f}%', ha='center', va='center',
                       fontsize=7, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Change (%)', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_per_term_heatmap.png')
    plt.savefig(OUTPUT_DIR / 'fig3_per_term_heatmap.pdf')
    plt.close()
    print(f"✓ Fig 3 saved: {OUTPUT_DIR / 'fig3_per_term_heatmap.png'}")


# ============================================================
# Figure 4: Layer Specificity (PaLiGemma L9 vs L17)
# ============================================================
def fig4_layer_specificity(results):
    """
    PaLiGemma layer specificity: L9 (effect) vs L17 (no effect).
    Justification: Demonstrates that gender features are localized in middle
    layers, not distributed across all layers. L17 serves as negative control.
    This addresses the concern: 'maybe any layer ablation would work.'
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    configs = ['PaLiGemma L9', 'PaLiGemma L17', 'PaLiGemma L9+17']
    labels = ['L9\n(mid-layer)', 'L17\n(late layer)', 'L9 + L17\n(combined)']
    colors_config = [COLORS['paligemma'], '#90CAF9', '#1565C0']

    targeted = [results[c]['targeted_change_pct'] for c in configs]
    random_mean = [results[c]['random_change_mean_pct'] for c in configs]
    random_std = [results[c]['random_change_std_pct'] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width/2, targeted, width, color=colors_config, edgecolor='black',
           linewidth=0.8, label='Targeted ablation')
    ax.bar(x + width/2, random_mean, width, yerr=random_std,
           color=[COLORS['random']]*3, edgecolor='black', linewidth=0.8,
           label='Random (mean±std)', capsize=4, alpha=0.7)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Gender Term Change (%)')
    ax.set_title('PaLiGemma-3B: Layer Specificity of Gender Features')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower left', framealpha=0.9)

    # Annotate key finding
    ax.annotate('Gender effect\nconcentrated here',
               xy=(0, targeted[0]), xytext=(0.8, targeted[0] - 3),
               arrowprops=dict(arrowstyle='->', color=COLORS['paligemma']),
               fontsize=9, color=COLORS['paligemma'], fontweight='bold')
    ax.annotate('No effect\n(negative control)',
               xy=(1, targeted[1]), xytext=(1.8, targeted[1] + 4),
               arrowprops=dict(arrowstyle='->', color='#90CAF9'),
               fontsize=9, color='#90CAF9', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_layer_specificity.png')
    plt.savefig(OUTPUT_DIR / 'fig4_layer_specificity.pdf')
    plt.close()
    print(f"✓ Fig 4 saved: {OUTPUT_DIR / 'fig4_layer_specificity.png'}")


# ============================================================
# Figure 5: SAE Quality vs Intervention Effect (Supplementary)
# ============================================================
def fig5_sae_quality_vs_effect(results):
    """
    Scatter: SAE explained variance vs magnitude of intervention effect.
    Justification: Controls for the confound that SAE quality differences
    might explain the direction reversal. Shows that even the lowest-quality
    SAE (Llama, 36.6% EV) produces a significant effect, and the direction
    reversal is not predicted by SAE quality alone.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # SAE quality data (from §11 and §14.5)
    sae_data = {
        'PaLiGemma L9': {'ev': 99.8, 'cos': 0.9999, 'dead': 51.2, 'effect': -16.1},
        'Qwen2-VL L12': {'ev': 66.4, 'cos': 0.9965, 'dead': 71.6, 'effect': +3.95},
        'Llama L20': {'ev': 36.6, 'cos': 0.9956, 'dead': 98.6, 'effect': +5.02},
    }
    model_colors = {
        'PaLiGemma L9': COLORS['paligemma'],
        'Qwen2-VL L12': COLORS['qwen2vl'],
        'Llama L20': COLORS['llama'],
    }

    for name, d in sae_data.items():
        ax.scatter(d['ev'], abs(d['effect']), s=200, c=model_colors[name],
                   edgecolors='black', linewidth=1, zorder=5)
        # Label with direction arrow
        direction = '↓' if d['effect'] < 0 else '↑'
        ax.annotate(f"{name}\n({direction}{abs(d['effect']):.1f}%)",
                   xy=(d['ev'], abs(d['effect'])),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   color=model_colors[name])

    ax.set_xlabel('SAE Explained Variance (%)')
    ax.set_ylabel('|Targeted Change| (%)')
    ax.set_title('SAE Quality vs. Intervention Effect Magnitude')
    ax.set_xlim(20, 105)
    ax.set_ylim(0, 20)

    # Add note about direction
    ax.text(0.02, 0.98, '↓ = decrease, ↑ = increase in gender terms',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(SUPP_DIR / 'fig5_sae_quality_vs_effect.png')
    plt.savefig(SUPP_DIR / 'fig5_sae_quality_vs_effect.pdf')
    plt.close()
    print(f"✓ Fig 5 saved: {SUPP_DIR / 'fig5_sae_quality_vs_effect.png'}")


# ============================================================
# Figure 6: Random Ablation Distributions (Supplementary)
# ============================================================
def fig6_random_distributions(results):
    """
    Distribution of all 25 random ablation runs per model.
    Justification: Validates the random control — shows random ablation
    effects are normally distributed around zero, while targeted ablation
    falls far outside this distribution for all models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    models = ['PaLiGemma L9', 'Qwen2-VL L12', 'Llama L20']
    labels = ['PaLiGemma-3B (L9)', 'Qwen2-VL-7B (L12)', 'Llama-3.2-11B (L20)']
    colors = [COLORS['paligemma'], COLORS['qwen2vl'], COLORS['llama']]

    for i, (m, label, color) in enumerate(zip(models, labels, colors)):
        ax = axes[i]
        random_changes = results[m].get('random_changes_all', [])
        targeted_change = results[m]['targeted_change_pct']

        if random_changes:
            ax.hist(random_changes, bins=12, color=COLORS['random'], edgecolor='black',
                    linewidth=0.5, alpha=0.7, label='Random runs (n=25)')

            # Mark targeted
            ax.axvline(targeted_change, color=color, linewidth=2.5,
                       linestyle='--', label=f'Targeted ({targeted_change:+.1f}%)')

            # Mark random mean
            rm = np.mean(random_changes)
            ax.axvline(rm, color='black', linewidth=1, linestyle=':',
                       label=f'Random mean ({rm:+.1f}%)')

            # Shade 2σ region
            rs = np.std(random_changes)
            ax.axvspan(rm - 2*rs, rm + 2*rs, alpha=0.1, color='gray',
                       label='±2σ region')

        ax.set_xlabel('Gender Term Change (%)')
        if i == 0:
            ax.set_ylabel('Count (of 25 random runs)')
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc='upper right' if targeted_change < 0 else 'upper left')

    plt.suptitle('Random Ablation Control: 25 Runs per Model', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(SUPP_DIR / 'fig6_random_distributions.png')
    plt.savefig(SUPP_DIR / 'fig6_random_distributions.pdf')
    plt.close()
    print(f"✓ Fig 6 saved: {SUPP_DIR / 'fig6_random_distributions.png'}")


# ============================================================
# Figure 7: Comprehensive Summary Figure (main)
# ============================================================
def fig7_summary_panel(results):
    """
    Four-panel summary figure combining key results.
    Justification: Single figure that captures all main findings for
    presentations and paper abstract/overview.
    """
    fig = plt.figure(figsize=(14, 10))

    # Panel (a): Cross-model bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    models = ['PaLiGemma L9', 'Qwen2-VL L12', 'Llama L20']
    labels = ['PaLiGemma', 'Qwen2-VL', 'Llama']
    colors = [COLORS['paligemma'], COLORS['qwen2vl'], COLORS['llama']]
    x = np.arange(len(models))

    targeted = [results[m]['targeted_change_pct'] for m in models]
    random_m = [results[m]['random_change_mean_pct'] for m in models]
    random_s = [results[m]['random_change_std_pct'] for m in models]

    ax1.bar(x - 0.17, targeted, 0.34, color=colors, edgecolor='black', linewidth=0.6, label='Targeted')
    ax1.bar(x + 0.17, random_m, 0.34, yerr=random_s, color=[COLORS['random']]*3,
            edgecolor='black', linewidth=0.6, label='Random', capsize=3, alpha=0.7)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Gender Term Change (%)')
    ax1.set_title('(a) Intervention Effect Across Models')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8)

    # Panel (b): Bootstrap CIs
    ax2 = fig.add_subplot(2, 2, 2)
    for i, m in enumerate(models):
        ps = results[m].get('paired_statistics', {})
        dd = ps.get('difference_targeted_minus_random', {})
        mean = dd.get('mean', 0)
        ci = dd.get('ci_95', [0, 0])
        ax2.errorbar(i, mean, yerr=[[mean - ci[0]], [ci[1] - mean]],
                     fmt='D', color=colors[i], markersize=10, capsize=8,
                     capthick=2, linewidth=2, markeredgecolor='black')
    ax2.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Δ (Targeted − Random) per image')
    ax2.set_title('(b) Bootstrap 95% CI (Targeted − Random)')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(labels)
    ax2.text(0.5, 0.05, 'CI excludes 0 = significant', transform=ax2.transAxes,
             ha='center', fontsize=8, fontstyle='italic', color='gray')

    # Panel (c): Effect specificity ratios
    ax3 = fig.add_subplot(2, 2, 3)
    ratios = [abs(results[m]['ratio_targeted_vs_random']) for m in models]
    bars = ax3.bar(range(len(models)), ratios, color=colors, edgecolor='black', linewidth=0.6)
    ax3.axhline(1, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax3.set_ylabel('|Targeted / Random| Ratio')
    ax3.set_title('(c) Ablation Specificity')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(labels)
    for bar, r in zip(bars, ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{r:.1f}×', ha='center', fontsize=10, fontweight='bold')

    # Panel (d): Layer specificity (PaLiGemma)
    ax4 = fig.add_subplot(2, 2, 4)
    layer_configs = ['PaLiGemma L9', 'PaLiGemma L17']
    layer_labels = ['L9 (mid)', 'L17 (late)']
    layer_targeted = [results[c]['targeted_change_pct'] for c in layer_configs]
    layer_random = [results[c]['random_change_mean_pct'] for c in layer_configs]
    layer_std = [results[c]['random_change_std_pct'] for c in layer_configs]

    x_l = np.arange(2)
    ax4.bar(x_l - 0.17, layer_targeted, 0.34, color=[COLORS['paligemma'], '#90CAF9'],
            edgecolor='black', linewidth=0.6, label='Targeted')
    ax4.bar(x_l + 0.17, layer_random, 0.34, yerr=layer_std,
            color=[COLORS['random']]*2, edgecolor='black', linewidth=0.6,
            label='Random', capsize=3, alpha=0.7)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_ylabel('Gender Term Change (%)')
    ax4.set_title('(d) Layer Specificity (PaLiGemma)')
    ax4.set_xticks(x_l)
    ax4.set_xticklabels(layer_labels)
    ax4.legend(fontsize=8)
    ax4.annotate('Effect here only', xy=(0, layer_targeted[0]),
                 xytext=(0.5, layer_targeted[0] + 3),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_summary_panel.png')
    plt.savefig(OUTPUT_DIR / 'fig7_summary_panel.pdf')
    plt.close()
    print(f"✓ Fig 7 saved: {OUTPUT_DIR / 'fig7_summary_panel.png'}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} result sets: {list(results.keys())}")

    print("\nGenerating figures...")
    fig1_cross_model_comparison(results)
    fig2_paired_deltas(results)
    fig3_per_term_heatmap(results)
    fig4_layer_specificity(results)
    fig5_sae_quality_vs_effect(results)
    fig6_random_distributions(results)
    fig7_summary_panel(results)

    print(f"\n✅ All figures generated!")
    print(f"   Main figures:         {OUTPUT_DIR}")
    print(f"   Supplementary figures: {SUPP_DIR}")
