#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Publication-Quality Pipeline Diagram
==============================================

Creates a comprehensive pipeline diagram for the SAE Gender Bias VLM paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np

# Set up figure with high DPI for publication
fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
COLORS = {
    'data': '#E3F2FD',           # Light blue
    'data_border': '#1565C0',    # Dark blue
    'model': '#FFF3E0',          # Light orange
    'model_border': '#E65100',   # Dark orange
    'sae': '#E8F5E9',            # Light green
    'sae_border': '#2E7D32',     # Dark green
    'analysis': '#F3E5F5',       # Light purple
    'analysis_border': '#7B1FA2', # Dark purple
    'intervention': '#FFEBEE',   # Light red
    'intervention_border': '#C62828', # Dark red
    'crosslingual': '#FFF8E1',   # Light amber
    'crosslingual_border': '#FF8F00', # Dark amber
    'arrow': '#424242',          # Dark gray
    'text': '#212121',           # Almost black
}

def add_box(ax, x, y, width, height, text, subtext=None, color='#E3F2FD', border_color='#1565C0', fontsize=11):
    """Add a rounded box with text."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(box)
    
    # Main text
    if subtext:
        ax.text(x + width/2, y + height*0.65, text, 
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color=COLORS['text'])
        ax.text(x + width/2, y + height*0.3, subtext, 
                ha='center', va='center', fontsize=fontsize-2, style='italic',
                color='#616161')
    else:
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color=COLORS['text'], wrap=True)
    return box

def add_arrow(ax, start, end, color='#424242', style='->'):
    """Add an arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle=style,
                            mutation_scale=15,
                            lw=2,
                            color=color,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    return arrow

def add_curved_arrow(ax, start, end, rad=0.2, color='#424242'):
    """Add a curved arrow."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='->',
                            mutation_scale=15,
                            lw=2,
                            color=color,
                            connectionstyle="arc3,rad=" + str(rad))
    ax.add_patch(arrow)
    return arrow

# ============================================================================
# STAGE LABELS (Top)
# ============================================================================
stages = [
    (1.5, "Stage 1", "Data"),
    (4.5, "Stage 2", "Extraction"),
    (7.5, "Stage 3", "SAE"),
    (10.5, "Stage 4", "Intervention"),
]

for x, stage, name in stages:
    ax.text(x, 9.5, stage, ha='center', va='center', fontsize=10, color='#757575')
    ax.text(x, 9.2, name, ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['text'])

# ============================================================================
# STAGE 1: DATA
# ============================================================================
# Dataset box
add_box(ax, 0.3, 7.2, 2.4, 1.2, "Flickr8K Dataset", "8,092 images",
        color=COLORS['data'], border_color=COLORS['data_border'])

# Bilingual captions
add_box(ax, 0.3, 5.6, 2.4, 1.2, "Bilingual Captions", "English + Arabic",
        color=COLORS['data'], border_color=COLORS['data_border'])

# Gender labels
add_box(ax, 0.3, 4.0, 2.4, 1.2, "Gender Labels", "Male: 5,562 | Female: 2,047",
        color=COLORS['data'], border_color=COLORS['data_border'])

# Arrows within Stage 1
add_arrow(ax, (1.5, 7.2), (1.5, 6.9))
add_arrow(ax, (1.5, 5.6), (1.5, 5.3))

# ============================================================================
# STAGE 2: MODEL & EXTRACTION
# ============================================================================
# VLM Models box
add_box(ax, 3.3, 7.2, 2.4, 1.2, "Vision-Language Models", 
        "4 architectures",
        color=COLORS['model'], border_color=COLORS['model_border'])

# Model list (smaller boxes)
models_y = 5.4
model_height = 0.6
model_width = 2.4
model_names = ["PaLiGemma-3B", "Qwen2-VL-7B", "LLaVA-1.5-7B", "Llama-3.2-11B"]
for i, name in enumerate(model_names):
    y = models_y - i * 0.7
    box = FancyBboxPatch((3.3, y), model_width, model_height,
                         boxstyle="round,pad=0.01,rounding_size=0.1",
                         facecolor=COLORS['model'], edgecolor=COLORS['model_border'], 
                         linewidth=1.5, alpha=0.8)
    ax.add_patch(box)
    ax.text(3.3 + model_width/2, y + model_height/2, name,
            ha='center', va='center', fontsize=9, color=COLORS['text'])

# Activation Extraction
add_box(ax, 3.3, 2.0, 2.4, 1.2, "Activation Extraction", 
        "PyTorch hooks · Mean-pool",
        color=COLORS['model'], border_color=COLORS['model_border'])

# Arrows
add_arrow(ax, (1.5, 4.0), (1.5, 3.2))  # Down from gender labels
add_curved_arrow(ax, (2.7, 3.2), (3.3, 2.6), rad=0.3)  # Curve to extraction
add_arrow(ax, (4.5, 5.4), (4.5, 3.3))  # Models to extraction

# ============================================================================
# STAGE 3: SAE TRAINING & ANALYSIS
# ============================================================================
# SAE Architecture box
add_box(ax, 6.3, 7.2, 2.4, 1.5, "SAE Training", 
        "ReLU + L1 · 8× expansion",
        color=COLORS['sae'], border_color=COLORS['sae_border'])

# SAE equations (small text box)
eq_box = FancyBboxPatch((6.3, 5.3), 2.4, 1.5,
                        boxstyle="round,pad=0.02,rounding_size=0.1",
                        facecolor='white', edgecolor=COLORS['sae_border'], 
                        linewidth=1.5, linestyle='--')
ax.add_patch(eq_box)
ax.text(7.5, 6.4, "h = ReLU(Wenc·x + b)", ha='center', va='center', 
        fontsize=8, family='monospace')
ax.text(7.5, 6.0, "x̂ = Wdec·h + b", ha='center', va='center', 
        fontsize=8, family='monospace')
ax.text(7.5, 5.6, "L = MSE + λ||h||₁", ha='center', va='center', 
        fontsize=8, family='monospace')

# Feature Identification
add_box(ax, 6.3, 3.5, 2.4, 1.4, "Gender Feature ID", 
        "Differential activation\n|μmale - μfemale|",
        color=COLORS['sae'], border_color=COLORS['sae_border'])

# Top-k selection
add_box(ax, 6.3, 2.0, 2.4, 1.0, "Top-k Features", 
        "k=100 (0.6%)",
        color=COLORS['sae'], border_color=COLORS['sae_border'])

# Arrows
add_arrow(ax, (5.7, 2.6), (6.3, 2.6))  # Extraction to SAE
add_arrow(ax, (7.5, 7.2), (7.5, 6.9))
add_arrow(ax, (7.5, 5.3), (7.5, 5.0))
add_arrow(ax, (7.5, 3.5), (7.5, 3.1))

# ============================================================================
# STAGE 4: INTERVENTION & RESULTS
# ============================================================================
# Causal Intervention
add_box(ax, 9.3, 7.2, 2.4, 1.5, "Causal Intervention", 
        "Targeted ablation\n25 random controls",
        color=COLORS['intervention'], border_color=COLORS['intervention_border'])

# Results box
add_box(ax, 9.3, 5.2, 2.4, 1.5, "Main Results", 
        "Effect ratios: 1.8-7.1×\nCIs exclude zero",
        color=COLORS['intervention'], border_color=COLORS['intervention_border'])

# Direction Discovery
dir_box = FancyBboxPatch((9.3, 3.3), 2.4, 1.5,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['intervention'], 
                         edgecolor=COLORS['intervention_border'], linewidth=2)
ax.add_patch(dir_box)
ax.text(10.5, 4.5, "Direction Discovery", ha='center', va='center', 
        fontsize=11, fontweight='bold', color=COLORS['text'])
ax.text(10.5, 4.0, "PaLiGemma: −16.1%", ha='center', va='center', 
        fontsize=9, color='#C62828')
ax.text(10.5, 3.65, "Qwen/Llama: +4-5%", ha='center', va='center', 
        fontsize=9, color='#2E7D32')

# Arrows
add_arrow(ax, (8.7, 2.5), (9.3, 3.8))  # Features to intervention
add_arrow(ax, (10.5, 7.2), (10.5, 6.8))
add_arrow(ax, (10.5, 5.2), (10.5, 4.9))

# ============================================================================
# CROSS-LINGUAL ANALYSIS (Bottom section)
# ============================================================================
# Cross-lingual box (spanning bottom)
cross_box = FancyBboxPatch((0.3, 0.3), 13.4, 1.4,
                           boxstyle="round,pad=0.02,rounding_size=0.2",
                           facecolor=COLORS['crosslingual'], 
                           edgecolor=COLORS['crosslingual_border'], linewidth=2)
ax.add_patch(cross_box)

ax.text(7, 1.3, "Cross-Lingual Analysis: 2×2 Causal Ablation", 
        ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['text'])

# Sub-boxes for cross-lingual
sub_width = 3.0
sub_height = 0.6
sub_y = 0.5

# EN→EN
ax.add_patch(FancyBboxPatch((0.8, sub_y), sub_width, sub_height,
             boxstyle="round,pad=0.01,rounding_size=0.1",
             facecolor='white', edgecolor=COLORS['crosslingual_border'], linewidth=1.5))
ax.text(0.8 + sub_width/2, sub_y + sub_height/2, "EN→EN: ✓ Significant", 
        ha='center', va='center', fontsize=9, color='#2E7D32', fontweight='bold')

# EN→AR
ax.add_patch(FancyBboxPatch((4.2, sub_y), sub_width, sub_height,
             boxstyle="round,pad=0.01,rounding_size=0.1",
             facecolor='white', edgecolor=COLORS['crosslingual_border'], linewidth=1.5))
ax.text(4.2 + sub_width/2, sub_y + sub_height/2, "EN→AR: ✗ No transfer", 
        ha='center', va='center', fontsize=9, color='#C62828', fontweight='bold')

# AR→EN
ax.add_patch(FancyBboxPatch((7.6, sub_y), sub_width, sub_height,
             boxstyle="round,pad=0.01,rounding_size=0.1",
             facecolor='white', edgecolor=COLORS['crosslingual_border'], linewidth=1.5))
ax.text(7.6 + sub_width/2, sub_y + sub_height/2, "AR→EN: ✗ No transfer*", 
        ha='center', va='center', fontsize=9, color='#C62828', fontweight='bold')

# Insight
ax.add_patch(FancyBboxPatch((11.0, sub_y), 2.4, sub_height,
             boxstyle="round,pad=0.01,rounding_size=0.1",
             facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5))
ax.text(11.0 + 1.2, sub_y + sub_height/2, "Overlap → Transfer", 
        ha='center', va='center', fontsize=9, color='#2E7D32', fontweight='bold')

# Arrow from intervention to cross-lingual
add_arrow(ax, (10.5, 3.3), (10.5, 1.8))

# ============================================================================
# KEY INSIGHT CALLOUT (Right side)
# ============================================================================
insight_box = FancyBboxPatch((11.9, 5.8), 1.8, 2.8,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor='#FFFDE7', edgecolor='#F57F17', 
                             linewidth=2, linestyle='-')
ax.add_patch(insight_box)
ax.text(12.8, 8.3, "Key Insight", ha='center', va='center', 
        fontsize=10, fontweight='bold', color='#F57F17')
ax.text(12.8, 7.7, "Excitatory", ha='center', va='center', fontsize=9, color='#C62828')
ax.text(12.8, 7.4, "vs", ha='center', va='center', fontsize=8, color='#757575')
ax.text(12.8, 7.1, "Inhibitory", ha='center', va='center', fontsize=9, color='#2E7D32')
ax.text(12.8, 6.5, "Same ablation", ha='center', va='center', fontsize=8, color='#424242')
ax.text(12.8, 6.2, "→ opposite", ha='center', va='center', fontsize=8, color='#424242')
ax.text(12.8, 5.9, "effects!", ha='center', va='center', fontsize=8, color='#424242')

# Connect to direction discovery
add_curved_arrow(ax, (11.7, 3.9), (11.9, 6.5), rad=-0.3)

# ============================================================================
# LEGEND
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=COLORS['data'], edgecolor=COLORS['data_border'], 
                   linewidth=2, label='Data'),
    mpatches.Patch(facecolor=COLORS['model'], edgecolor=COLORS['model_border'], 
                   linewidth=2, label='Model/Extraction'),
    mpatches.Patch(facecolor=COLORS['sae'], edgecolor=COLORS['sae_border'], 
                   linewidth=2, label='SAE Training'),
    mpatches.Patch(facecolor=COLORS['intervention'], edgecolor=COLORS['intervention_border'], 
                   linewidth=2, label='Intervention'),
    mpatches.Patch(facecolor=COLORS['crosslingual'], edgecolor=COLORS['crosslingual_border'], 
                   linewidth=2, label='Cross-Lingual'),
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.99),
          framealpha=0.9, fontsize=9, ncol=5)

# ============================================================================
# FOOTNOTE
# ============================================================================
ax.text(7, -0.1, "*PaLiGemma (57% overlap) shows AR→EN transfer; Qwen (0% overlap) shows none",
        ha='center', va='center', fontsize=8, color='#757575', style='italic')

# ============================================================================
# SAVE
# ============================================================================
plt.tight_layout()
output_dir = "/home2/jmsk62/mechanistic_intrep/sae_captioning_project/publication/figures/main"
plt.savefig(output_dir + "/fig1_pipeline_overview_new.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(output_dir + "/fig1_pipeline_overview_new.pdf", bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Pipeline diagram saved to:")
print("   " + output_dir + "/fig1_pipeline_overview_new.png")
print("   " + output_dir + "/fig1_pipeline_overview_new.pdf")

plt.close()
