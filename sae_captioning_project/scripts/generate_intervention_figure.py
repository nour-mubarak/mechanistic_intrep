#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Intervention Experiment Design Figure
===============================================

Creates a publication-quality figure explaining the 3-phase intervention experiment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set up figure
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Color scheme
COLORS = {
    'input': '#E3F2FD',          # Light blue
    'input_border': '#1565C0',   # Dark blue
    'baseline': '#E8F5E9',       # Light green
    'baseline_border': '#2E7D32', # Dark green
    'targeted': '#FFCDD2',       # Light red
    'targeted_border': '#C62828', # Dark red
    'random': '#FFF3E0',         # Light orange
    'random_border': '#E65100',  # Dark orange
    'compare': '#F3E5F5',        # Light purple
    'compare_border': '#7B1FA2', # Dark purple
    'arrow': '#424242',          # Dark gray
    'text': '#212121',           # Almost black
}

def add_box(ax, x, y, width, height, title, lines=None, 
            color='#E3F2FD', border_color='#1565C0', title_size=11):
    """Add a styled box with title and optional bullet points."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         facecolor=color, edgecolor=border_color, linewidth=2.5)
    ax.add_patch(box)
    
    # Title
    title_y = y + height - 0.35
    ax.text(x + width/2, title_y, title, 
            ha='center', va='center', fontsize=title_size, fontweight='bold',
            color=COLORS['text'])
    
    # Content lines
    if lines:
        start_y = title_y - 0.45
        for i, line in enumerate(lines):
            ax.text(x + 0.15, start_y - i * 0.35, line, 
                    ha='left', va='center', fontsize=9, color='#424242')
    
    return box

def add_arrow(ax, start, end, color='#424242', style='-|>', lw=2.5):
    """Add an arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle=style,
                            mutation_scale=20,
                            lw=lw,
                            color=color,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    return arrow

def add_curved_arrow(ax, start, end, rad=0.2, color='#424242'):
    """Add a curved arrow."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='-|>',
                            mutation_scale=18,
                            lw=2,
                            color=color,
                            connectionstyle="arc3,rad=" + str(rad))
    ax.add_patch(arrow)
    return arrow

# ============================================================================
# TITLE
# ============================================================================
ax.text(7, 7.7, "Intervention Experiment Design", 
        ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['text'])

# ============================================================================
# INPUT: 500 IMAGES (Left side)
# ============================================================================
add_box(ax, 0.3, 3.0, 2.0, 2.5, "Input", 
        lines=["500 images", "Random sample", "(seed=42)", "From Flickr8K"],
        color=COLORS['input'], border_color=COLORS['input_border'])

# Image stack visual
for i in range(3):
    rect = FancyBboxPatch((0.6 + i*0.08, 5.8 - i*0.12), 1.4, 1.0,
                          boxstyle="round,pad=0.01,rounding_size=0.05",
                          facecolor='white', edgecolor='#90A4AE', linewidth=1)
    ax.add_patch(rect)
ax.text(1.45, 6.15, "500 images", ha='center', va='center', fontsize=8, color='#546E7A')

# ============================================================================
# PHASE 1: BASELINE (Top branch)
# ============================================================================
phase1_x = 3.5
phase1_y = 5.5
add_box(ax, phase1_x, phase1_y, 3.0, 1.8, "Phase 1: BASELINE",
        lines=["No modification", "Standard VLM inference", "Generate captions"],
        color=COLORS['baseline'], border_color=COLORS['baseline_border'])

# Phase 1 label
ax.text(phase1_x + 1.5, phase1_y + 1.95, "Control condition",
        ha='center', va='center', fontsize=8, style='italic', color='#2E7D32')

# ============================================================================
# PHASE 2: TARGETED ABLATION (Middle branch)
# ============================================================================
phase2_x = 3.5
phase2_y = 3.0
add_box(ax, phase2_x, phase2_y, 3.0, 1.8, "Phase 2: TARGETED",
        lines=["Zero gender features", "k=100 (top differential)", "0.6% of SAE features"],
        color=COLORS['targeted'], border_color=COLORS['targeted_border'])

# Phase 2 label
ax.text(phase2_x + 1.5, phase2_y + 1.95, "Experimental condition",
        ha='center', va='center', fontsize=8, style='italic', color='#C62828')

# ============================================================================
# PHASE 3: RANDOM ABLATION (Bottom branch)
# ============================================================================
phase3_x = 3.5
phase3_y = 0.5
add_box(ax, phase3_x, phase3_y, 3.0, 1.8, "Phase 3: RANDOM",
        lines=["Zero random features", "k=100 (randomly selected)", "Repeat 25 times"],
        color=COLORS['random'], border_color=COLORS['random_border'])

# Phase 3 label  
ax.text(phase3_x + 1.5, phase3_y + 1.95, "Ablation control",
        ha='center', va='center', fontsize=8, style='italic', color='#E65100')

# 25x indicator
ax.text(phase3_x + 2.7, phase3_y + 0.3, "x25",
        ha='center', va='center', fontsize=14, fontweight='bold', color='#E65100')

# ============================================================================
# ARROWS FROM INPUT TO PHASES
# ============================================================================
# Input to Phase 1
add_arrow(ax, (2.3, 5.5), (3.5, 6.2))
# Input to Phase 2
add_arrow(ax, (2.3, 4.25), (3.5, 4.0))
# Input to Phase 3
add_arrow(ax, (2.3, 3.5), (3.5, 1.8), color='#424242')
add_curved_arrow(ax, (2.3, 3.2), (3.5, 1.4), rad=0.3)

# ============================================================================
# OUTPUT BOXES
# ============================================================================
out_x = 7.5
out_width = 2.2
out_height = 1.4

# Output 1: Baseline captions
out1_y = 5.7
box1 = FancyBboxPatch((out_x, out1_y), out_width, out_height,
                      boxstyle="round,pad=0.01,rounding_size=0.1",
                      facecolor='white', edgecolor=COLORS['baseline_border'], linewidth=2)
ax.add_patch(box1)
ax.text(out_x + out_width/2, out1_y + out_height - 0.3, "Baseline Output",
        ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['text'])
ax.text(out_x + out_width/2, out1_y + 0.5, "1,522 gender terms",
        ha='center', va='center', fontsize=9, color='#2E7D32')
ax.text(out_x + out_width/2, out1_y + 0.2, "(PaLiGemma example)",
        ha='center', va='center', fontsize=7, color='#757575')

# Output 2: Targeted captions
out2_y = 3.2
box2 = FancyBboxPatch((out_x, out2_y), out_width, out_height,
                      boxstyle="round,pad=0.01,rounding_size=0.1",
                      facecolor='white', edgecolor=COLORS['targeted_border'], linewidth=2)
ax.add_patch(box2)
ax.text(out_x + out_width/2, out2_y + out_height - 0.3, "Targeted Output",
        ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['text'])
ax.text(out_x + out_width/2, out2_y + 0.5, "1,277 gender terms",
        ha='center', va='center', fontsize=9, color='#C62828')
ax.text(out_x + out_width/2, out2_y + 0.2, "(-16.1% change)",
        ha='center', va='center', fontsize=8, fontweight='bold', color='#C62828')

# Output 3: Random captions (distribution)
out3_y = 0.7
box3 = FancyBboxPatch((out_x, out3_y), out_width, out_height,
                      boxstyle="round,pad=0.01,rounding_size=0.1",
                      facecolor='white', edgecolor=COLORS['random_border'], linewidth=2)
ax.add_patch(box3)
ax.text(out_x + out_width/2, out3_y + out_height - 0.3, "Random Output",
        ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['text'])
ax.text(out_x + out_width/2, out3_y + 0.55, "Mean: -8.7%",
        ha='center', va='center', fontsize=9, color='#E65100')
ax.text(out_x + out_width/2, out3_y + 0.25, "Std: +/-3.9%",
        ha='center', va='center', fontsize=8, color='#E65100')

# Arrows from phases to outputs
add_arrow(ax, (6.5, 6.4), (7.5, 6.4))
add_arrow(ax, (6.5, 3.9), (7.5, 3.9))
add_arrow(ax, (6.5, 1.4), (7.5, 1.4))

# ============================================================================
# COMPARISON BOX (Right side)
# ============================================================================
comp_x = 10.5
comp_y = 2.5
comp_width = 3.2
comp_height = 3.5

comp_box = FancyBboxPatch((comp_x, comp_y), comp_width, comp_height,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor=COLORS['compare'], edgecolor=COLORS['compare_border'], 
                          linewidth=2.5)
ax.add_patch(comp_box)

ax.text(comp_x + comp_width/2, comp_y + comp_height - 0.35, "Statistical Comparison",
        ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['text'])

# Comparison content
comp_lines = [
    "Per-image paired analysis:",
    "",
    "delta_i = targeted_i - baseline_i",
    "",
    "Tests:",
    "  - Bootstrap 95% CI",
    "  - Wilcoxon signed-rank",
    "",
    "Key metric:",
    "  Targeted vs Random"
]

start_y = comp_y + comp_height - 0.7
for i, line in enumerate(comp_lines):
    if line.startswith("delta"):
        ax.text(comp_x + 0.2, start_y - i * 0.28, line,
                ha='left', va='center', fontsize=8, family='monospace', color='#7B1FA2')
    elif line.startswith("  "):
        ax.text(comp_x + 0.2, start_y - i * 0.28, line,
                ha='left', va='center', fontsize=8, color='#424242')
    else:
        ax.text(comp_x + 0.2, start_y - i * 0.28, line,
                ha='left', va='center', fontsize=9, fontweight='bold' if ":" in line else 'normal',
                color=COLORS['text'])

# Arrows to comparison
add_curved_arrow(ax, (out_x + out_width, out1_y + out_height/2), (comp_x, comp_y + comp_height - 0.8), rad=-0.2)
add_arrow(ax, (out_x + out_width, out2_y + out_height/2), (comp_x, comp_y + comp_height/2))
add_curved_arrow(ax, (out_x + out_width, out3_y + out_height/2), (comp_x, comp_y + 0.8), rad=0.2)

# ============================================================================
# RESULT CALLOUT
# ============================================================================
result_x = 10.7
result_y = 0.5
result_box = FancyBboxPatch((result_x, result_y), 2.8, 1.6,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(result_box)
ax.text(result_x + 1.4, result_y + 1.35, "Result",
        ha='center', va='center', fontsize=10, fontweight='bold', color='#2E7D32')
ax.text(result_x + 1.4, result_y + 0.95, "Effect ratio: 1.8-7.1x",
        ha='center', va='center', fontsize=9, color=COLORS['text'])
ax.text(result_x + 1.4, result_y + 0.65, "CI excludes zero",
        ha='center', va='center', fontsize=9, color=COLORS['text'])
ax.text(result_x + 1.4, result_y + 0.3, "Targeted > Random",
        ha='center', va='center', fontsize=10, fontweight='bold', color='#2E7D32')

add_arrow(ax, (comp_x + comp_width/2, comp_y), (result_x + 1.4, result_y + 1.6))

# ============================================================================
# LEGEND
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=COLORS['baseline'], edgecolor=COLORS['baseline_border'], 
                   linewidth=2, label='Phase 1: Baseline'),
    mpatches.Patch(facecolor=COLORS['targeted'], edgecolor=COLORS['targeted_border'], 
                   linewidth=2, label='Phase 2: Targeted'),
    mpatches.Patch(facecolor=COLORS['random'], edgecolor=COLORS['random_border'], 
                   linewidth=2, label='Phase 3: Random (x25)'),
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
          framealpha=0.95, fontsize=9)

# ============================================================================
# SAVE
# ============================================================================
plt.tight_layout()
output_dir = "/home2/jmsk62/mechanistic_intrep/sae_captioning_project/publication/figures/main"
plt.savefig(output_dir + "/fig_intervention_design.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(output_dir + "/fig_intervention_design.pdf", bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Intervention design figure saved to:")
print("   " + output_dir + "/fig_intervention_design.png")
print("   " + output_dir + "/fig_intervention_design.pdf")

plt.close()
