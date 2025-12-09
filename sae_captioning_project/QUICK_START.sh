#!/bin/bash
# SAE Captioning Project - Quick Start Guide

# Activate environment
cd /home/nour/mchanistic\ project/mechanistic_intrep
source env/bin/activate
cd sae_captioning_project

# Option 1: Run full 7-stage pipeline
echo "Running full pipeline (all stages including mechanistic analysis)..."
nohup python scripts/run_full_pipeline.py --config configs/config.yaml > pipeline.log 2>&1 &
echo "Pipeline started. Monitor with: tail -f pipeline.log"

# Option 2: Run mechanistic analysis only (Stage 7)
echo "Running mechanistic analysis stage..."
python scripts/07_integrated_mechanistic_analysis.py \
  --config configs/config.yaml \
  --features_dir results/features \
  --output_dir results/mechanistic_analysis \
  --enable_multilingual \
  --languages en ar

# Option 3: Validate integration
echo "Validating mechanistic interpretability integration..."
python validate_integration.py

# Monitor W&B experiments
echo "View experiments at: https://wandb.ai/noureddine-lounici/sae-captioning"
