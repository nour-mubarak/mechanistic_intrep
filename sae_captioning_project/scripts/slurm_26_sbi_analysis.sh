#!/bin/bash
#SBATCH --job-name=sbi_analysis
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sbi_analysis_%j.out
#SBATCH --error=logs/sbi_analysis_%j.err

# Surgical Bias Intervention Analysis
# Tests if ablating gender features reduces bias

echo "=============================================="
echo "Surgical Bias Intervention (SBI) Analysis"
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
echo "Starting SBI analysis..."
python scripts/26_surgical_bias_intervention.py \
    --config configs/config.yaml \
    --output_dir results/sbi_analysis \
    --layers 0,3,6,9,12,15,17 \
    --device cpu \
    --wandb \
    --wandb_project sae-cross-lingual-bias

echo ""
echo "=============================================="
echo "SBI Analysis complete!"
echo "=============================================="
echo "Results saved to: results/sbi_analysis/"
