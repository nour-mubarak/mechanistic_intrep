# Cross-Lingual SAE Analysis for Vision-Language Model Gender Bias

## Project Overview

This project uses **Sparse Autoencoders (SAEs)** to perform mechanistic interpretability analysis on **Gemma-3-4B** for understanding cross-lingual gender bias in Arabic-English image captioning.

### Research Questions

1. **Where do gender representations diverge?** Identify layers where Arabic and English gender feature overlap changes
2. **Are there language-specific gender features?** Find features unique to each language's gender encoding
3. **Can we steer to reduce bias?** Test whether suppressing features reduces gendered outputs
4. **Grammatical vs semantic gender:** How does Arabic morphological gender differ from semantic associations?

## Project Structure

```
sae_captioning_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── configs/
│   ├── config.yaml             # Main configuration
│   └── slurm_config.yaml       # SLURM job parameters
├── scripts/
│   ├── 01_prepare_data.py      # Dataset preparation
│   ├── 02_extract_activations.py # Activation extraction
│   ├── 03_train_sae.py         # SAE training
│   ├── 04_analyze_features.py  # Feature analysis
│   ├── 05_steering_experiments.py # Intervention experiments
│   ├── 06_generate_visualizations.py # Create plots
│   └── run_full_pipeline.py    # End-to-end pipeline
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sae.py              # SAE architecture
│   │   └── hooks.py            # Activation hooks
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Data loading utilities
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── features.py         # Feature analysis
│   │   └── metrics.py          # Evaluation metrics
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # Visualization functions
├── slurm/
│   ├── submit_extraction.sh    # SLURM job for activation extraction
│   ├── submit_training.sh      # SLURM job for SAE training
│   └── submit_analysis.sh      # SLURM job for analysis
├── data/                       # Data directory (gitignored)
├── checkpoints/                # Model checkpoints
├── results/                    # Analysis results
├── visualizations/             # Generated plots
└── logs/                       # Log files
```

## Installation

```bash
# Clone and setup
cd sae_captioning_project
pip install -e . --break-system-packages

# Or install requirements directly
pip install -r requirements.txt --break-system-packages
```

## Quick Start

### 1. Prepare your data

Place your image-caption dataset in `data/raw/` with the following structure:
- `images/`: Directory containing images
- `captions.csv`: CSV with columns `image_id`, `english_prompt`, `arabic_prompt`, `ground_truth_gender`

```bash
python scripts/01_prepare_data.py --config configs/config.yaml
```

### 2. Run the full pipeline

```bash
python scripts/run_full_pipeline.py --config configs/config.yaml
```

### 3. Or run steps individually

```bash
# Extract activations
python scripts/02_extract_activations.py --config configs/config.yaml

# Train SAEs
python scripts/03_train_sae.py --config configs/config.yaml

# Analyze features
python scripts/04_analyze_features.py --config configs/config.yaml

# Run steering experiments
python scripts/05_steering_experiments.py --config configs/config.yaml

# Generate visualizations
python scripts/06_generate_visualizations.py --config configs/config.yaml
```

## SLURM Submission (NCC/HPC)

```bash
# Submit all jobs in sequence
sbatch slurm/submit_extraction.sh
sbatch --dependency=afterok:$EXTRACTION_JOB_ID slurm/submit_training.sh
sbatch --dependency=afterok:$TRAINING_JOB_ID slurm/submit_analysis.sh
```

## Key Outputs

1. **Trained SAEs**: `checkpoints/sae_layer_{N}.pt`
2. **Feature Analysis**: `results/feature_analysis.json`
3. **Cross-lingual Comparison**: `results/cross_lingual_analysis.json`
4. **Steering Results**: `results/steering_experiments.json`
5. **Visualizations**: `visualizations/*.png` and `visualizations/*.html`

## Citation

If you use this code, please cite:
```bibtex
@misc{sae_captioning_2025,
  title={Cross-Lingual SAE Analysis for Vision-Language Gender Bias},
  author={Your Name},
  year={2025}
}
```

## License

MIT License
