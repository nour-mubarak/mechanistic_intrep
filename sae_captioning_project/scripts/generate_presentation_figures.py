#!/usr/bin/env python3
"""
Generate Presentation Figures and Summary
==========================================

Creates publication-ready figures for supervisor presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Paths
PROJECT_DIR = Path("/home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project")
RESULTS_DIR = PROJECT_DIR / "results"
PRESENTATION_DIR = PROJECT_DIR / "presentation"
PRESENTATION_DIR.mkdir(exist_ok=True)

def load_results():
    """Load all result files."""
    results = {}
    
    # Model comparison
    model_comp_path = RESULTS_DIR / "qwen2vl_analysis" / "model_comparison_results.json"
    if model_comp_path.exists():
        with open(model_comp_path) as f:
            results['model_comparison'] = json.load(f)
    
    # Cross-lingual overlap
    overlap_path = RESULTS_DIR / "cross_lingual_overlap" / "cross_lingual_overlap_results.json"
    if overlap_path.exists():
        with open(overlap_path) as f:
            results['overlap'] = json.load(f)
    
    # SBI results
    sbi_path = RESULTS_DIR / "sbi_analysis" / "sbi_results.json"
    if sbi_path.exists():
        with open(sbi_path) as f:
            results['sbi'] = json.load(f)
    
    # Feature interpretation
    feat_path = RESULTS_DIR / "feature_interpretation" / "feature_interpretation_results.json"
    if feat_path.exists():
        with open(feat_path) as f:
            results['features'] = json.load(f)
    
    return results


def create_main_comparison_figure(results):
    """Create the main model comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Data
    if 'model_comparison' in results:
        mc = results['model_comparison']
        
        # Plot 1: CLBAS comparison
        ax1 = axes[0, 0]
        models = ['PaLiGemma-3B', 'Qwen2-VL-7B']
        clbas_means = [
            mc['paligemma']['summary']['mean_clbas'],
            mc['qwen2vl']['summary']['mean_clbas']
        ]
        colors = ['#3498db', '#e74c3c']
        bars = ax1.bar(models, clbas_means, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Mean CLBAS Score')
        ax1.set_title('Cross-Lingual Alignment (Lower = More Specific)')
        ax1.set_ylim(0, 0.05)
        for bar, val in zip(bars, clbas_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 2: Feature overlap
        ax2 = axes[0, 1]
        overlaps = [
            mc['paligemma']['summary']['total_overlap'],
            mc['qwen2vl']['summary']['total_overlap']
        ]
        bars = ax2.bar(models, overlaps, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Total Feature Overlap Count')
        ax2.set_title('Shared Gender Features Across Languages')
        ax2.set_ylim(0, 5)
        for bar, val in zip(bars, overlaps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(int(val)), ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Plot 3: CLBAS by layer (PaLiGemma)
        ax3 = axes[1, 0]
        pg_clbas = mc['paligemma']['clbas_scores']
        layers_pg = [int(k) for k in pg_clbas.keys()]
        values_pg = list(pg_clbas.values())
        ax3.plot(layers_pg, values_pg, 'o-', color='#3498db', linewidth=2, markersize=8, label='PaLiGemma-3B')
        
        qw_clbas = mc['qwen2vl']['clbas_scores']
        layers_qw = [int(k) for k in qw_clbas.keys()]
        values_qw = list(qw_clbas.values())
        ax3.plot(layers_qw, values_qw, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Qwen2-VL-7B')
        
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('CLBAS Score')
        ax3.set_title('CLBAS by Layer')
        ax3.legend()
        ax3.set_ylim(0, 0.05)
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Alignment threshold')
        
        # Plot 4: Probe accuracy comparison
        ax4 = axes[1, 1]
        x = np.arange(2)
        width = 0.35
        
        ar_acc = [0.865, mc['qwen2vl']['summary']['mean_ar_probe']]
        en_acc = [0.930, mc['qwen2vl']['summary']['mean_en_probe']]
        
        bars1 = ax4.bar(x - width/2, ar_acc, width, label='Arabic', color='#27ae60', edgecolor='black')
        bars2 = ax4.bar(x + width/2, en_acc, width, label='English', color='#9b59b6', edgecolor='black')
        
        ax4.set_ylabel('Probe Accuracy')
        ax4.set_title('Gender Probe Accuracy by Language')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.set_ylim(0.8, 1.0)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Cross-Lingual Gender Bias Analysis: Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PRESENTATION_DIR / 'main_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created main_comparison.png")


def create_key_findings_figure(results):
    """Create a summary figure of key findings."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Key Findings: Cross-Lingual Gender Bias in Vision-Language Models', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Finding 1: Near-zero alignment
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.7, 'Finding 1', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.5, 'CLBAS = 0.027', fontsize=24, fontweight='bold', ha='center', 
             transform=ax1.transAxes, color='#e74c3c')
    ax1.text(0.5, 0.3, '(PaLiGemma-3B)', fontsize=12, ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.1, '~0.4% feature overlap', fontsize=11, ha='center', transform=ax1.transAxes, style='italic')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Near-Zero Cross-Lingual Alignment', fontsize=12, pad=10)
    
    # Finding 2: Scaling effect
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.7, 'Finding 2', fontsize=14, fontweight='bold', ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.5, '6.7×', fontsize=32, fontweight='bold', ha='center', 
             transform=ax2.transAxes, color='#3498db')
    ax2.text(0.5, 0.3, 'lower alignment in 7B vs 3B', fontsize=11, ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.1, 'Larger = More specific', fontsize=11, ha='center', transform=ax2.transAxes, style='italic')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Scaling Increases Specificity', fontsize=12, pad=10)
    
    # Finding 3: Distributed encoding
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.7, 'Finding 3', fontsize=14, fontweight='bold', ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.5, '0%', fontsize=32, fontweight='bold', ha='center', 
             transform=ax3.transAxes, color='#27ae60')
    ax3.text(0.5, 0.3, 'accuracy drop at k=200', fontsize=11, ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.1, 'Gender is distributed', fontsize=11, ha='center', transform=ax3.transAxes, style='italic')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Distributed Encoding', fontsize=12, pad=10)
    
    # CLBAS by layer plot
    ax4 = fig.add_subplot(gs[1, :2])
    if 'model_comparison' in results:
        mc = results['model_comparison']
        
        pg_clbas = mc['paligemma']['clbas_scores']
        layers_pg = [int(k) for k in pg_clbas.keys()]
        values_pg = list(pg_clbas.values())
        ax4.plot(layers_pg, values_pg, 'o-', color='#3498db', linewidth=2.5, markersize=10, label='PaLiGemma-3B')
        
        qw_clbas = mc['qwen2vl']['clbas_scores']
        layers_qw = [int(k) for k in qw_clbas.keys()]
        values_qw = list(qw_clbas.values())
        ax4.plot(layers_qw, values_qw, 's-', color='#e74c3c', linewidth=2.5, markersize=10, label='Qwen2-VL-7B')
        
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        ax4.text(max(layers_qw), 0.52, 'Alignment threshold (0.5)', fontsize=10, alpha=0.7)
        
        ax4.set_xlabel('Layer', fontsize=12)
        ax4.set_ylabel('CLBAS Score', fontsize=12)
        ax4.set_title('Cross-Lingual Bias Alignment Score by Layer', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11, loc='upper left')
        ax4.set_ylim(0, 0.6)
        ax4.fill_between(layers_pg, 0, 0.1, alpha=0.2, color='green', label='Language-specific zone')
    
    # SBI plot
    ax5 = fig.add_subplot(gs[1, 2])
    k_values = [10, 25, 50, 100, 200]
    ar_drops = [0.05, -0.05, -0.04, 0.02, -0.02]
    en_drops = [0.13, -0.24, 0.03, 0.13, 0.29]
    
    ax5.plot(k_values, ar_drops, 'o-', color='#27ae60', linewidth=2, markersize=8, label='Arabic')
    ax5.plot(k_values, en_drops, 's-', color='#9b59b6', linewidth=2, markersize=8, label='English')
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax5.axhline(y=5, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Features Ablated (k)', fontsize=11)
    ax5.set_ylabel('Accuracy Drop (%)', fontsize=11)
    ax5.set_title('SBI: Ablation Effect', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.set_ylim(-1, 1)
    
    # Probe accuracy comparison
    ax6 = fig.add_subplot(gs[2, :])
    
    # PaLiGemma data
    pg_layers = [0, 3, 6, 9, 12, 15, 17]
    pg_ar = [86.3, 85.4, 87.4, 88.1, 86.7, 86.3, 85.5]
    pg_en = [92.1, 94.1, 91.4, 95.0, 92.5, 93.3, 92.7]
    
    x = np.arange(len(pg_layers))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, pg_ar, width, label='Arabic', color='#27ae60', alpha=0.8)
    bars2 = ax6.bar(x + width/2, pg_en, width, label='English', color='#9b59b6', alpha=0.8)
    
    ax6.set_xlabel('Layer', fontsize=12)
    ax6.set_ylabel('Probe Accuracy (%)', fontsize=12)
    ax6.set_title('Gender Probe Accuracy by Layer (PaLiGemma-3B)', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(pg_layers)
    ax6.legend(fontsize=11)
    ax6.set_ylim(80, 100)
    ax6.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    
    # Highlight best layer
    ax6.annotate('Best layer', xy=(3, 95), xytext=(4, 97),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.savefig(PRESENTATION_DIR / 'key_findings_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created key_findings_summary.png")


def create_methodology_figure():
    """Create methodology overview figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Pipeline boxes
    boxes = [
        {'x': 0.05, 'y': 0.6, 'w': 0.18, 'h': 0.25, 'text': 'VLM\n(PaLiGemma/\nQwen2-VL)', 'color': '#3498db'},
        {'x': 0.28, 'y': 0.6, 'w': 0.18, 'h': 0.25, 'text': 'Activation\nExtraction\n(Hooks)', 'color': '#2ecc71'},
        {'x': 0.51, 'y': 0.6, 'w': 0.18, 'h': 0.25, 'text': 'SAE\nTraining\n(16K features)', 'color': '#e74c3c'},
        {'x': 0.74, 'y': 0.6, 'w': 0.18, 'h': 0.25, 'text': 'Cross-Lingual\nAnalysis\n(CLBAS)', 'color': '#9b59b6'},
    ]
    
    for box in boxes:
        rect = mpatches.FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                                        boxstyle="round,pad=0.02", 
                                        facecolor=box['color'], alpha=0.8,
                                        edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, box['text'],
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Arrows
    for i in range(3):
        ax.annotate('', xy=(boxes[i+1]['x']-0.02, 0.725), xytext=(boxes[i]['x']+boxes[i]['w']+0.02, 0.725),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Bottom: Analysis types
    analyses = [
        {'x': 0.08, 'y': 0.15, 'text': 'CLBAS\nMetric', 'desc': 'Feature alignment'},
        {'x': 0.28, 'y': 0.15, 'text': 'Linear\nProbes', 'desc': 'Gender classification'},
        {'x': 0.48, 'y': 0.15, 'text': 'SBI\nAnalysis', 'desc': 'Feature ablation'},
        {'x': 0.68, 'y': 0.15, 'text': 'Statistical\nTests', 'desc': 'Bootstrap/Permutation'},
    ]
    
    for a in analyses:
        rect = mpatches.FancyBboxPatch((a['x'], a['y']), 0.15, 0.2,
                                        boxstyle="round,pad=0.01", 
                                        facecolor='#f39c12', alpha=0.7,
                                        edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(a['x'] + 0.075, a['y'] + 0.13, a['text'],
               ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(a['x'] + 0.075, a['y'] + 0.05, a['desc'],
               ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow from main pipeline to analyses
    ax.annotate('', xy=(0.46, 0.4), xytext=(0.46, 0.55),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Methodology Overview: Cross-Lingual SAE Analysis Pipeline', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(PRESENTATION_DIR / 'methodology_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created methodology_overview.png")


def create_conclusion_figure():
    """Create conclusion summary figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Main conclusion box
    main_rect = mpatches.FancyBboxPatch((0.1, 0.55), 0.8, 0.35,
                                         boxstyle="round,pad=0.02", 
                                         facecolor='#2c3e50', alpha=0.9,
                                         edgecolor='white', linewidth=3)
    ax.add_patch(main_rect)
    
    conclusion_text = """Cross-lingual gender bias in VLMs is encoded through
LANGUAGE-SPECIFIC FEATURE CIRCUITS
rather than shared representations.

Larger models show 6.7× more language-specific processing."""
    
    ax.text(0.5, 0.725, conclusion_text, ha='center', va='center', 
           fontsize=14, color='white', fontweight='bold', linespacing=1.5)
    
    # Implication boxes
    implications = [
        {'x': 0.05, 'y': 0.1, 'text': 'Multilingual bias analysis\nrequires language-specific\nevaluation', 'color': '#e74c3c'},
        {'x': 0.35, 'y': 0.1, 'text': 'Simple ablation is\nINSUFFICIENT for\ndebiasing', 'color': '#f39c12'},
        {'x': 0.65, 'y': 0.1, 'text': 'Scaling may increase\nlanguage-specific\nprocessing', 'color': '#27ae60'},
    ]
    
    for imp in implications:
        rect = mpatches.FancyBboxPatch((imp['x'], imp['y']), 0.28, 0.3,
                                        boxstyle="round,pad=0.02", 
                                        facecolor=imp['color'], alpha=0.8,
                                        edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(imp['x'] + 0.14, imp['y'] + 0.15, imp['text'],
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Key Conclusions & Implications', fontsize=18, fontweight='bold', pad=20)
    
    plt.savefig(PRESENTATION_DIR / 'conclusions.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created conclusions.png")


def copy_key_figures():
    """Copy key figures to presentation directory."""
    import shutil
    
    figures_to_copy = [
        RESULTS_DIR / "qwen2vl_analysis" / "publication_summary.png",
        RESULTS_DIR / "qwen2vl_analysis" / "final_model_comparison.png",
        RESULTS_DIR / "qwen2vl_analysis" / "qwen2vl_vs_paligemma_comparison.png",
        RESULTS_DIR / "cross_lingual_overlap" / "visualizations" / "cross_lingual_summary_heatmap.png",
        RESULTS_DIR / "cross_lingual_overlap" / "visualizations" / "feature_overlap_by_layer.png",
        RESULTS_DIR / "sbi_analysis" / "visualizations" / "sbi_accuracy_vs_k.png",
        RESULTS_DIR / "sbi_analysis" / "visualizations" / "sbi_cross_lingual_specificity.png",
        PROJECT_DIR / "visualizations" / "layer_comparison_arabic.png",
        PROJECT_DIR / "visualizations" / "layer_comparison_english.png",
        PROJECT_DIR / "visualizations" / "layer_9_arabic" / "tsne_gender.png",
    ]
    
    for src in figures_to_copy:
        if src.exists():
            dst = PRESENTATION_DIR / src.name
            shutil.copy(src, dst)
            print(f"✓ Copied {src.name}")
        else:
            print(f"✗ Not found: {src}")


def main():
    print("="*60)
    print("Generating Presentation Figures")
    print("="*60)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"Loaded: {list(results.keys())}")
    
    # Create figures
    print("\nCreating figures...")
    create_main_comparison_figure(results)
    create_key_findings_figure(results)
    create_methodology_figure()
    create_conclusion_figure()
    
    # Copy existing figures
    print("\nCopying existing figures...")
    copy_key_figures()
    
    print("\n" + "="*60)
    print(f"All figures saved to: {PRESENTATION_DIR}")
    print("="*60)
    
    # List all files
    print("\nGenerated files:")
    for f in sorted(PRESENTATION_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
