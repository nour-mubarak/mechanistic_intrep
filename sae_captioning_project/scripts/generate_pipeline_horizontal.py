#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Simplified Pipeline Diagram - Horizontal Flow
=======================================================

Creates a cleaner, more publication-ready pipeline diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up figure with high DPI for publication
fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Color scheme - more muted for publication
COLORS = {
    'data': '#BBDEFB',           # Light blue
    'data_border': '#1976D2',    # Blue
    'model': '#FFE0B2',          # Light orange  
    'model_border': '#F57C00',   # Orange
    'sae': '#C8E6C9',            # Light green
    'sae_border': '#388E3C',     # Green
    'intervention': '#FFCDD2',   # Light red
    'intervention_border': '#D32F2F', # Red
    'crosslingual': '#FFF9C4',   # Light yellow
    'crosslingual_border': '#FBC02D', # Yellow/amber
    'arrow': '#424242',          # Dark gray
    'text': '#212121',           # Almost black
}

def add_box(ax, x, y, width, height, title, subtitle=None, items=None,
            color='#E3F2FD', border_color='#1565C0'):
    """Add a styled box with title and optional items."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(box)
    
    # Title at top
    title_y = y + height - 0.3
    ax.text(x + width/2, title_y, title, 
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=COLORS['text'])
    
    # Optional subtitle
    if subtitle:
        ax.text(x + width/2, title_y - 0.35, subtitle, 
                ha='center', va='center', fontsize=9, style='italic',
                color='#616161')
    
    # Optional items list
    if items:
        start_y = title_y - (0.7 if subtitle else 0.45)
        for i, item in enumerate(items):
            ax.text(x + 0.15, start_y - i * 0.35, "• " + item, 
                    ha='left', va='center', fontsize=8, color='#424242')
    
    return box

def add_arrow(ax, start, end, color='#424242'):
    """Add an arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='-|>',
                            mutation_scale=20,
                            lw=2.5,
                            color=color,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    return arrow

# ============================================================================
# MAIN PIPELINE BOXES (Horizontal flow)
# ============================================================================
box_width = 2.3
box_height = 2.8
y_main = 2.8
gap = 0.6

# Stage 1: Data
add_box(ax, 0.3, y_main, box_width, box_height,
        "Data", "Flickr8K",
        items=["8,092 images", "EN + AR captions", "Gender labels", "Male: 5,562", "Female: 2,047"],
        color=COLORS['data'], border_color=COLORS['data_border'])

# Stage 2: Models & Extraction
add_box(ax, 0.3 + box_width + gap, y_main, box_width, box_height,
        "VLM Extraction", "4 architectures",
        items=["PaLiGemma-3B", "Qwen2-VL-7B", "LLaVA-1.5-7B", "Llama-3.2-11B", "Hook decoder layers"],
        color=COLORS['model'], border_color=COLORS['model_border'])

# Stage 3: SAE
add_box(ax, 0.3 + 2*(box_width + gap), y_main, box_width, box_height,
        "SAE Training", "ReLU + L1",
        items=["8x expansion", "Activation dim: 2k-4k", "Latent dim: 16k-32k", "Gender feature ID", "Top-100 (0.6%)"],
        color=COLORS['sae'], border_color=COLORS['sae_border'])

# Stage 4: Intervention
add_box(ax, 0.3 + 3*(box_width + gap), y_main, box_width, box_height,
        "Causal Ablation", "Intervention",
        items=["Zero-out features", "25 random controls", "Bootstrap CIs", "Effect ratios", "1.8-7.1x"],
        color=COLORS['intervention'], border_color=COLORS['intervention_border'])

# Arrows between main boxes
x_positions = [0.3 + box_width, 0.3 + 2*box_width + gap, 0.3 + 3*box_width + 2*gap]
arrow_y = y_main + box_height/2
for x in x_positions:
    add_arrow(ax, (x, arrow_y), (x + gap, arrow_y))

# ============================================================================
# CROSS-LINGUAL SECTION (Below)
# ============================================================================
cross_y = 0.4
cross_height = 2.0
cross_width = 13.4

# Background box
cross_box = FancyBboxPatch((0.3, cross_y), cross_width, cross_height,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=COLORS['crosslingual'], 
                           edgecolor=COLORS['crosslingual_border'], linewidth=2)
ax.add_patch(cross_box)

ax.text(7, cross_y + cross_height - 0.25, "Cross-Lingual Analysis: 2x2 Causal Ablation", 
        ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['text'])

# Sub-boxes for the 2x2 design
sub_width = 2.8
sub_height = 1.2
sub_y = cross_y + 0.25

# Box 1: EN features -> EN captions
sub_box1 = FancyBboxPatch((0.6, sub_y), sub_width, sub_height,
                          boxstyle="round,pad=0.01,rounding_size=0.08",
                          facecolor='white', edgecolor=COLORS['sae_border'], linewidth=1.5)
ax.add_patch(sub_box1)
ax.text(0.6 + sub_width/2, sub_y + sub_height - 0.25, "EN Features -> EN", 
        ha='center', va='center', fontsize=9, fontweight='bold', color=COLORS['text'])
ax.text(0.6 + sub_width/2, sub_y + 0.35, "Significant effect", 
        ha='center', va='center', fontsize=8, color='#388E3C')

# Box 2: EN features -> AR captions
sub_box2 = FancyBboxPatch((3.7, sub_y), sub_width, sub_height,
                          boxstyle="round,pad=0.01,rounding_size=0.08",
                          facecolor='white', edgecolor=COLORS['intervention_border'], linewidth=1.5)
ax.add_patch(sub_box2)
ax.text(3.7 + sub_width/2, sub_y + sub_height - 0.25, "EN Features -> AR", 
        ha='center', va='center', fontsize=9, fontweight='bold', color=COLORS['text'])
ax.text(3.7 + sub_width/2, sub_y + 0.35, "No transfer", 
        ha='center', va='center', fontsize=8, color='#D32F2F')

# Box 3: AR features -> EN captions
sub_box3 = FancyBboxPatch((6.8, sub_y), sub_width, sub_height,
                          boxstyle="round,pad=0.01,rounding_size=0.08",
                          facecolor='white', edgecolor=COLORS['intervention_border'], linewidth=1.5)
ax.add_patch(sub_box3)
ax.text(6.8 + sub_width/2, sub_y + sub_height - 0.25, "AR Features -> EN", 
        ha='center', va='center', fontsize=9, fontweight='bold', color=COLORS['text'])
ax.text(6.8 + sub_width/2, sub_y + 0.35, "Overlap-dependent*", 
        ha='center', va='center', fontsize=8, color='#FF8F00')

# Box 4: Key Finding
finding_box = FancyBboxPatch((9.9, sub_y), 3.5, sub_height,
                             boxstyle="round,pad=0.01,rounding_size=0.08",
                             facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(finding_box)
ax.text(9.9 + 3.5/2, sub_y + sub_height - 0.25, "Key Insight", 
        ha='center', va='center', fontsize=9, fontweight='bold', color='#2E7D32')
ax.text(9.9 + 3.5/2, sub_y + 0.6, "Feature overlap predicts", 
        ha='center', va='center', fontsize=8, color='#424242')
ax.text(9.9 + 3.5/2, sub_y + 0.3, "cross-lingual transfer", 
        ha='center', va='center', fontsize=8, color='#424242')

# Arrow from intervention to cross-lingual
add_arrow(ax, (0.3 + 3*(box_width + gap) + box_width/2, y_main), 
          (0.3 + 3*(box_width + gap) + box_width/2, cross_y + cross_height))

# ============================================================================
# KEY FINDING CALLOUT (Top right)
# ============================================================================
insight_x = 12.0
insight_y = 4.0
insight_width = 1.8
insight_height = 1.6

insight_box = FancyBboxPatch((insight_x, insight_y), insight_width, insight_height,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#FFF8E1', edgecolor='#FF6F00', linewidth=2)
ax.add_patch(insight_box)
ax.text(insight_x + insight_width/2, insight_y + insight_height - 0.25, 
        "Direction Effect", ha='center', va='center', fontsize=9, fontweight='bold', color='#E65100')
ax.text(insight_x + insight_width/2, insight_y + 0.9, 
        "Excitatory vs", ha='center', va='center', fontsize=8, color='#424242')
ax.text(insight_x + insight_width/2, insight_y + 0.6, 
        "Inhibitory", ha='center', va='center', fontsize=8, color='#424242')
ax.text(insight_x + insight_width/2, insight_y + 0.25, 
        "features", ha='center', va='center', fontsize=8, color='#424242')

# Arrow from intervention to insight
arrow = FancyArrowPatch((0.3 + 3*(box_width + gap) + box_width, y_main + box_height - 0.5),
                        (insight_x, insight_y + insight_height/2),
                        arrowstyle='-|>',
                        mutation_scale=15,
                        lw=1.5,
                        color='#FF6F00',
                        connectionstyle="arc3,rad=-0.2")
ax.add_patch(arrow)

# ============================================================================
# FOOTNOTE
# ============================================================================
ax.text(7, 0.1, "*PaLiGemma (57% overlap) shows transfer; Qwen2-VL (0% overlap) shows none",
        ha='center', va='center', fontsize=7, color='#757575', style='italic')

# ============================================================================
# LEGEND
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=COLORS['data'], edgecolor=COLORS['data_border'], 
                   linewidth=2, label='Data'),
    mpatches.Patch(facecolor=COLORS['model'], edgecolor=COLORS['model_border'], 
                   linewidth=2, label='Extraction'),
    mpatches.Patch(facecolor=COLORS['sae'], edgecolor=COLORS['sae_border'], 
                   linewidth=2, label='SAE'),
    mpatches.Patch(facecolor=COLORS['intervention'], edgecolor=COLORS['intervention_border'], 
                   linewidth=2, label='Intervention'),
    mpatches.Patch(facecolor=COLORS['crosslingual'], edgecolor=COLORS['crosslingual_border'], 
                   linewidth=2, label='Cross-Lingual'),
]

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
          framealpha=0.95, fontsize=8, ncol=5)

# ============================================================================
# SAVE
# ============================================================================
plt.tight_layout()
output_dir = "/home2/jmsk62/mechanistic_intrep/sae_captioning_project/publication/figures/main"
plt.savefig(output_dir + "/fig1_pipeline_horizontal.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(output_dir + "/fig1_pipeline_horizontal.pdf", bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Horizontal pipeline diagram saved to:")
print("   " + output_dir + "/fig1_pipeline_horizontal.png")
print("   " + output_dir + "/fig1_pipeline_horizontal.pdf")

plt.close()
