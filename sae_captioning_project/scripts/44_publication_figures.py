#!/usr/bin/env python3
"""
Generate publication-quality figures for the cross-lingual VLM bias paper.

Main Paper Figures:
1. Feature overlap comparison across models (<1%)
2. Cross-lingual ablation (SBI) results
3. Probe accuracy comparison (training regime insight)
4. SAE quality comparison plot

Appendix Figures:
A1. Full SAE metrics table (EV, L0, dead features)
A2. CLBAS table by layer
A3. Additional stability checks
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette for models
MODEL_COLORS = {
    'PaLiGemma-3B': '#E64B35',      # Red
    'Qwen2-VL-7B': '#4DBBD5',        # Cyan
    'LLaVA-1.5-7B': '#00A087',       # Teal
    'Llama-3.2-Vision': '#3C5488',   # Blue
}

RESULTS_DIR = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results')
OUTPUT_DIR = RESULTS_DIR / 'publication_figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_model_data():
    """Load data from all models."""
    data = {}
    
    # PaLiGemma
    try:
        with open(RESULTS_DIR / 'proper_cross_lingual/cross_lingual_results.json') as f:
            pali_data = json.load(f)
        data['PaLiGemma-3B'] = {
            'layers': [],
            'ar_probe': [],
            'en_probe': [],
            'feature_overlap': [],
            'clbas': [],
        }
        for layer, vals in pali_data.items():
            if isinstance(vals, dict) and 'arabic' in vals:
                data['PaLiGemma-3B']['layers'].append(int(layer))
                data['PaLiGemma-3B']['ar_probe'].append(vals['arabic']['probe_accuracy'])
                data['PaLiGemma-3B']['en_probe'].append(vals['english']['probe_accuracy'])
                # Feature overlap
                if 'alignment' in vals:
                    overlap = vals['alignment'].get('feature_overlap', {}).get('top_100', {}).get('overlap_percentage', 0)
                    data['PaLiGemma-3B']['feature_overlap'].append(overlap)
                    data['PaLiGemma-3B']['clbas'].append(vals['alignment'].get('clbas', 0))
    except Exception as e:
        print(f"Error loading PaLiGemma: {e}")
    
    # LLaVA
    try:
        with open(RESULTS_DIR / 'llava_analysis/cross_lingual_results.json') as f:
            llava_data = json.load(f)
        data['LLaVA-1.5-7B'] = {
            'layers': [],
            'ar_probe': [],
            'en_probe': [],
            'feature_overlap': [],
            'clbas': [],
        }
        for item in llava_data['layer_results']:
            data['LLaVA-1.5-7B']['layers'].append(item['layer'])
            data['LLaVA-1.5-7B']['ar_probe'].append(item['probe_accuracy']['arabic']['mean'])
            data['LLaVA-1.5-7B']['en_probe'].append(item['probe_accuracy']['english']['mean'])
            overlap = item.get('feature_overlap', {}).get('top_100', {}).get('overlap_percentage', 0)
            data['LLaVA-1.5-7B']['feature_overlap'].append(overlap)
            data['LLaVA-1.5-7B']['clbas'].append(item.get('clbas', {}).get('clbas_score', 0))
    except Exception as e:
        print(f"Error loading LLaVA: {e}")
    
    # Qwen2-VL
    try:
        with open(RESULTS_DIR / 'qwen2vl_analysis/model_comparison_results.json') as f:
            qwen_data = json.load(f)['qwen2vl']
        data['Qwen2-VL-7B'] = {
            'layers': list(map(int, qwen_data['arabic_probe_acc'].keys())),
            'ar_probe': list(qwen_data['arabic_probe_acc'].values()),
            'en_probe': list(qwen_data['english_probe_acc'].values()),
            'feature_overlap': [qwen_data['overlap_counts'].get(str(l), 0) / 100 for l in qwen_data['arabic_probe_acc'].keys()],
            'clbas': list(qwen_data['clbas_scores'].values()),
        }
    except Exception as e:
        print(f"Error loading Qwen: {e}")
    
    # Llama - has dict structure with 'layer_X' keys
    try:
        with open(RESULTS_DIR / 'llama32vision_analysis/cross_lingual_results.json') as f:
            llama_data = json.load(f)
        data['Llama-3.2-Vision'] = {
            'layers': [],
            'ar_probe': [],
            'en_probe': [],
            'feature_overlap': [],
            'clbas': [],
        }
        # Handle dict structure with layer_X keys
        for layer_key, layer_vals in llama_data['layer_results'].items():
            layer_num = int(layer_key.replace('layer_', ''))
            data['Llama-3.2-Vision']['layers'].append(layer_num)
            data['Llama-3.2-Vision']['ar_probe'].append(layer_vals['ar_probe_acc'])
            data['Llama-3.2-Vision']['en_probe'].append(layer_vals['en_probe_acc'])
            data['Llama-3.2-Vision']['feature_overlap'].append(layer_vals.get('overlap_pct', 0))
            data['Llama-3.2-Vision']['clbas'].append(layer_vals.get('clbas', 0))
    except Exception as e:
        print(f"Error loading Llama: {e}")
    
    return data


def fig1_feature_overlap_comparison(data):
    """
    Figure 1: Feature overlap is <1% across all models.
    Bar chart showing max overlap per model.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = []
    max_overlaps = []
    mean_overlaps = []
    
    for model in ['PaLiGemma-3B', 'Qwen2-VL-7B', 'LLaVA-1.5-7B', 'Llama-3.2-Vision']:
        if model in data and data[model]['feature_overlap']:
            models.append(model)
            max_overlaps.append(max(data[model]['feature_overlap']))
            mean_overlaps.append(np.mean(data[model]['feature_overlap']))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, max_overlaps, width, label='Max Overlap', 
                   color=[MODEL_COLORS[m] for m in models], alpha=0.8)
    bars2 = ax.bar(x + width/2, mean_overlaps, width, label='Mean Overlap',
                   color=[MODEL_COLORS[m] for m in models], alpha=0.4)
    
    # Add 1% threshold line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='1% Threshold')
    
    ax.set_ylabel('Feature Overlap (%)')
    ax.set_title('Cross-Lingual Gender Feature Overlap\n(Top 100 Features per Language)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-', '\n') for m in models])
    ax.legend()
    ax.set_ylim(0, 2)
    
    # Add value labels
    for bar, val in zip(bars1, max_overlaps):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_feature_overlap.pdf')
    plt.savefig(OUTPUT_DIR / 'fig1_feature_overlap.png')
    plt.close()
    print("✓ Figure 1: Feature overlap comparison saved")


def fig2_probe_accuracy_comparison(data):
    """
    Figure 2: Probe accuracy reveals training regime effects.
    Grouped bar chart with Arabic vs English per model.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['PaLiGemma-3B', 'Qwen2-VL-7B', 'LLaVA-1.5-7B', 'Llama-3.2-Vision']
    training_regimes = ['Translation\nonly', 'Native\nmultilingual', 'English\nonly', 'Native\nmultilingual']
    
    ar_means = []
    ar_stds = []
    en_means = []
    en_stds = []
    
    for model in models:
        if model in data:
            ar_means.append(np.mean(data[model]['ar_probe']) * 100)
            ar_stds.append(np.std(data[model]['ar_probe']) * 100)
            en_means.append(np.mean(data[model]['en_probe']) * 100)
            en_stds.append(np.std(data[model]['en_probe']) * 100)
        else:
            ar_means.append(0)
            ar_stds.append(0)
            en_means.append(0)
            en_stds.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ar_means, width, yerr=ar_stds, label='Arabic',
                   color='#E64B35', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, en_means, width, yerr=en_stds, label='English',
                   color='#4DBBD5', alpha=0.8, capsize=3)
    
    ax.set_ylabel('Probe Accuracy (%)')
    ax.set_title('Gender Probe Accuracy by Model and Language\n(Training Regime Effects)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m.replace("-", chr(10))}\n({tr})' for m, tr in zip(models, training_regimes)], fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(80, 102)
    
    # Add gap annotations
    for i, (ar, en) in enumerate(zip(ar_means, en_means)):
        gap = en - ar
        color = 'green' if gap > 0 else 'red'
        ax.annotate(f'Δ={gap:+.1f}%', xy=(i, max(ar, en) + 2),
                   ha='center', fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_probe_accuracy.pdf')
    plt.savefig(OUTPUT_DIR / 'fig2_probe_accuracy.png')
    plt.close()
    print("✓ Figure 2: Probe accuracy comparison saved")


def fig3_sae_quality_comparison():
    """
    Figure 3: SAE quality comparison (EV vs L0 trade-off).
    """
    # Load SAE quality data
    try:
        with open(RESULTS_DIR / 'four_model_comparison/four_model_comparison_data.json') as f:
            comp_data = json.load(f)
    except:
        print("Could not load comparison data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models_info = []
    for model_name, model_data in comp_data.get('models', {}).items():
        sae = model_data.get('sae_quality', {})
        ev = sae.get('explained_variance_pct', {}).get('mean', 0)
        l0 = sae.get('mean_l0', {}).get('mean', 0)
        
        # Map model name to color key
        color_key = model_name
        if 'Llama' in model_name:
            color_key = 'Llama-3.2-Vision'
        
        models_info.append({
            'name': model_name,
            'ev': ev,
            'l0': l0,
            'color': MODEL_COLORS.get(color_key, '#888888')
        })
    
    for m in models_info:
        ax.scatter(m['l0'], m['ev'], s=200, c=m['color'], alpha=0.8, 
                   edgecolors='black', linewidths=1.5, zorder=5)
        # Add label
        offset = (300, 2) if m['l0'] < 5000 else (-500, -3)
        ax.annotate(m['name'].replace('-', '\n'), xy=(m['l0'], m['ev']),
                   xytext=(m['l0'] + offset[0], m['ev'] + offset[1]),
                   fontsize=9, ha='center')
    
    ax.set_xlabel('Mean L0 (Active Features per Sample)')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('SAE Quality: Reconstruction vs Sparsity Trade-off')
    
    # Add annotation for ideal region
    ax.axhspan(80, 100, alpha=0.1, color='green', label='Good EV (>80%)')
    ax.axvspan(0, 2000, alpha=0.1, color='blue', label='Good Sparsity (L0<2k)')
    
    ax.set_xlim(-500, 16000)
    ax.set_ylim(30, 105)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_sae_quality.pdf')
    plt.savefig(OUTPUT_DIR / 'fig3_sae_quality.png')
    plt.close()
    print("✓ Figure 3: SAE quality comparison saved")


def fig4_clbas_heatmap(data):
    """
    Figure 4 (Appendix): CLBAS scores heatmap across layers.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    for idx, (model, ax) in enumerate(zip(['PaLiGemma-3B', 'Qwen2-VL-7B', 'LLaVA-1.5-7B', 'Llama-3.2-Vision'], axes)):
        if model in data and data[model]['clbas']:
            layers = data[model]['layers']
            clbas = data[model]['clbas']
            
            # Create heatmap data
            heatmap_data = np.array(clbas).reshape(1, -1)
            
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.15)
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, fontsize=8)
            ax.set_yticks([])
            ax.set_xlabel('Layer')
            ax.set_title(model.replace('-', '\n'), fontsize=10)
            
            # Add text annotations
            for j, val in enumerate(clbas):
                ax.text(j, 0, f'{val:.3f}', ha='center', va='center', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model)
    
    fig.colorbar(im, ax=axes, label='CLBAS Score', shrink=0.8)
    fig.suptitle('Cross-Lingual Bias Alignment Score (CLBAS) by Layer', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figA1_clbas_heatmap.pdf')
    plt.savefig(OUTPUT_DIR / 'figA1_clbas_heatmap.png')
    plt.close()
    print("✓ Figure A1: CLBAS heatmap saved")


def create_sae_metrics_table():
    """
    Create appendix table with full SAE metrics.
    """
    try:
        with open(RESULTS_DIR / 'four_model_comparison/four_model_comparison_data.json') as f:
            comp_data = json.load(f)
    except:
        print("Could not load comparison data for table")
        return
    
    table_md = """
## Appendix A: Full SAE Quality Metrics

| Model | Explained Variance | Dead Features | Mean L0 | Recon. Cosine |
|-------|-------------------|---------------|---------|---------------|
"""
    
    for model_name, model_data in comp_data.get('models', {}).items():
        sae = model_data.get('sae_quality', {})
        ev = sae.get('explained_variance_pct', {})
        dead = sae.get('dead_feature_pct', {})
        l0 = sae.get('mean_l0', {})
        cos = sae.get('reconstruction_cosine', {})
        
        table_md += f"| {model_name} | {ev.get('mean', 0):.1f} ± {ev.get('std', 0):.1f}% | "
        table_md += f"{dead.get('mean', 0):.1f} ± {dead.get('std', 0):.1f}% | "
        table_md += f"{l0.get('mean', 0):,.0f} ± {l0.get('std', 0):,.0f} | "
        table_md += f"{cos.get('mean', 0):.4f} |\n"
    
    with open(OUTPUT_DIR / 'appendix_sae_metrics.md', 'w') as f:
        f.write(table_md)
    
    print("✓ Appendix: SAE metrics table saved")


def create_paper_structure():
    """
    Create markdown file outlining paper figure/table structure.
    """
    structure = """# Publication Figure and Table Structure

## Main Paper

### Figure 1: Cross-Lingual Feature Overlap
- **File:** `fig1_feature_overlap.pdf`
- **Message:** Gender features are encoded independently per language (<1% overlap)
- **Key insight:** Challenges assumption of universal bias representations

### Figure 2: Probe Accuracy by Training Regime  
- **File:** `fig2_probe_accuracy.pdf`
- **Message:** Training approach determines cross-lingual bias encoding
- **Key patterns:**
  - Translation-based (PaLiGemma): Inverse bias (Arabic > English)
  - English-only (LLaVA): Largest EN-AR gap (+6.4%)
  - Native multilingual (Qwen, Llama): Balanced accuracy

### Figure 3: SAE Quality Trade-off
- **File:** `fig3_sae_quality.pdf`
- **Message:** Reconstruction vs sparsity trade-off across architectures
- **Key insight:** LLaVA achieves best balance for interpretability

### Table 1: Cross-Lingual Bias Summary
(In main text - see FOUR_MODEL_COMPARISON_REPORT.md)

---

## Appendix

### Figure A1: CLBAS Heatmap by Layer
- **File:** `figA1_clbas_heatmap.pdf`
- **Message:** CLBAS scores are consistently low across all layers

### Table A1: Full SAE Quality Metrics
- **File:** `appendix_sae_metrics.md`
- **Content:** EV, L0, dead features, reconstruction cosine for all models

### Table A2: Statistical Significance Tests
(In FOUR_MODEL_COMPARISON_REPORT.md Section 4.3)

---

## Figure Locations

All publication figures saved to:
`results/publication_figures/`

"""
    
    with open(OUTPUT_DIR / 'PAPER_STRUCTURE.md', 'w') as f:
        f.write(structure)
    
    print("✓ Paper structure guide saved")


def main():
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    
    # Load data
    print("\nLoading model data...")
    data = load_all_model_data()
    print(f"Loaded data for: {list(data.keys())}")
    
    # Generate main paper figures
    print("\n--- Main Paper Figures ---")
    fig1_feature_overlap_comparison(data)
    fig2_probe_accuracy_comparison(data)
    fig3_sae_quality_comparison()
    
    # Generate appendix figures
    print("\n--- Appendix Figures ---")
    fig4_clbas_heatmap(data)
    create_sae_metrics_table()
    
    # Create paper structure guide
    print("\n--- Documentation ---")
    create_paper_structure()
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
