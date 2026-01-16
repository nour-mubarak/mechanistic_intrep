# Cross-Lingual SAE Analysis for Vision-Language Model Gender Bias

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project uses **Sparse Autoencoders (SAEs)** to perform mechanistic interpretability analysis on **PaLiGemma-3B** for understanding cross-lingual gender bias in Arabic-English image captioning.

### Key Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Feature Overlap** | 0.4% | Arabic and English use almost entirely separate gender features |
| **CLBAS Score** | 0.025 | Very low cross-lingual bias alignment |
| **Arabic Probe Accuracy** | 88.5% | Gender is linearly encoded in SAE features |
| **English Probe Accuracy** | 85.3% | Slightly lower than Arabic |

**Novel Finding**: The model develops **language-specific gender circuits** rather than a shared universal gender representation.

## Research Questions

| # | Question | Status |
|---|----------|--------|
| RQ1 | Where do gender representations diverge between Arabic and English? | ✅ All layers show near-complete divergence |
| RQ2 | Are there language-specific gender features? | ✅ 99.6% of features are language-specific |
| RQ3 | Can we steer the model to reduce bias? | 🔄 SBI experiments in progress |
| RQ4 | Grammatical vs semantic gender differences? | ✅ Arabic shows stronger encoding (88.5% vs 85.3%) |

## Project Structure

\`\`\`
sae_captioning_project/
├── README.md                 # This file
├── RESEARCH_PLAN.md          # Detailed research methodology
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
│
├── configs/                  # Configuration files
│   ├── config.yaml          # Main configuration
│   └── clmb_config.yaml     # CLMB framework settings
│
├── scripts/                  # Pipeline scripts (numbered)
│   ├── 01_prepare_data.py   # Dataset preparation
│   ├── 02_extract_activations.py
│   ├── 03_train_sae.py      # SAE training
│   ├── 24_cross_lingual_overlap.py    # Feature overlap analysis
│   ├── 25_cross_lingual_feature_interpretation.py
│   ├── 26_surgical_bias_intervention.py  # SBI experiments
│   └── slurm_*.sh           # SLURM job scripts
│
├── src/                      # Source code
│   ├── models/
│   │   ├── sae.py           # SAE architecture (2048 → 16384)
│   │   └── hooks.py         # Activation hooks
│   ├── clmb/                # Novel CLMB framework
│   │   ├── hbl.py           # Hierarchical Bias Localization
│   │   ├── clfa.py          # Cross-Lingual Feature Alignment
│   │   └── sbi.py           # Surgical Bias Intervention
│   └── analysis/            # Analysis utilities
│
├── docs/                     # Documentation
│   ├── guides/              # User guides
│   ├── status/              # Pipeline status reports
│   └── CLMB_FRAMEWORK.md    # Framework documentation
│
├── results/                  # Analysis outputs
│   ├── cross_lingual_overlap/
│   ├── feature_interpretation/
│   └── sbi_analysis/
│
└── visualizations/           # Generated plots
\`\`\`

## Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/nour-mubarak/mechanistic_intrep.git
cd mechanistic_intrep/sae_captioning_project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
\`\`\`

## Quick Start

### On NCC Cluster (Durham)

\`\`\`bash
# Full pipeline with SLURM
sbatch scripts/slurm_00_full_pipeline.sh

# Or run individual analysis
sbatch scripts/slurm_24_cross_lingual_overlap.sh
\`\`\`

## Key Results

### Cross-Lingual Feature Overlap

| Layer | Overlap % | CLBAS Score |
|-------|-----------|-------------|
| 0 | 0.0% | 0.013 |
| 3 | 0.0% | 0.011 |
| 6 | 0.0% | 0.015 |
| 9 | 2.0% | 0.028 |
| 12 | 1.0% | 0.039 |
| 15 | 0.0% | 0.028 |
| 17 | 0.0% | 0.041 |

## CLMB Framework

Our novel **Cross-Lingual Mechanistic Bias (CLMB)** framework:

1. **HBL**: Hierarchical Bias Localization
2. **CLFA**: Cross-Lingual Feature Alignment
3. **SBI**: Surgical Bias Intervention
4. **CLBAS**: Cross-Lingual Bias Alignment Score

## License

MIT License
