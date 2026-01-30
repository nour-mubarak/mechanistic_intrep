#!/bin/bash
#SBATCH --job-name=cross_lingual_overlap
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cross_lingual_overlap_%j.out
#SBATCH --error=logs/cross_lingual_overlap_%j.err

# Cross-Lingual Feature Overlap Analysis
# Computes the key novel metric: feature overlap between Arabic/English

echo "=============================================="
echo "Cross-Lingual Feature Overlap Analysis"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Navigate to project directory
cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project

# Activate virtual environment
source venv/bin/activate

# Check Python
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Run the analysis
echo ""
echo "Starting analysis..."
python scripts/24_cross_lingual_overlap.py \
    --config configs/config.yaml \
    --output_dir results/cross_lingual_overlap \
    --layers 0,3,6,9,12,15,17 \
    --device cpu

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "=============================================="
echo "Results saved to: results/cross_lingual_overlap/"
