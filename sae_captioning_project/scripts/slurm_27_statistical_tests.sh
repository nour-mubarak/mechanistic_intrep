#!/bin/bash
#SBATCH --job-name=stat_sig
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/stat_significance_%j.out
#SBATCH --error=logs/stat_significance_%j.err

# Statistical Significance Tests for Cross-Lingual SAE Analysis

echo "=============================================="
echo "Statistical Significance Tests"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

echo "Python: $(which python)"
echo "Memory limit: 180G"

# Set Python to use unbuffered output
export PYTHONUNBUFFERED=1

python scripts/27_statistical_significance.py \
    --config configs/config.yaml \
    --layers 0,3,6,9,12,15,17 \
    --device cpu \
    --n_bootstrap 1000 \
    --output_dir results/statistical_tests

echo ""
echo "Statistical tests complete!"
echo "Date: $(date)"
