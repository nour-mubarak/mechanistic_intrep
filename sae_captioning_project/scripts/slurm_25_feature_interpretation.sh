#!/bin/bash
#SBATCH --job-name=feature_interpret
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/feature_interpretation_%j.out
#SBATCH --error=logs/feature_interpretation_%j.err

# Cross-Lingual Feature Interpretation
# Analyzes what the language-specific gender features encode

echo "=============================================="
echo "Cross-Lingual Feature Interpretation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Navigate to project directory
cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

# Activate virtual environment
source venv/bin/activate

# Check Python
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Run the analysis
echo ""
echo "Starting feature interpretation analysis..."
python scripts/25_cross_lingual_feature_interpretation.py \
    --config configs/config.yaml \
    --output_dir results/feature_interpretation \
    --layers 0,3,6,9,12,15,17 \
    --device cpu

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "=============================================="
echo "Results saved to: results/feature_interpretation/"
