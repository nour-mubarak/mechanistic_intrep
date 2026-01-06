#!/bin/bash
#SBATCH --job-name=01_prepare_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/step1_prepare_data_%j.out
#SBATCH --error=logs/step1_prepare_data_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jmsk62@durham.ac.uk

# Step 1: Data Preparation
# =========================
# Prepares the full dataset for activation extraction

echo "=========================================="
echo "Step 1: Data Preparation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

# Load modules
module purge

# Activate environment
source venv/bin/activate

# Run data preparation
python scripts/01_prepare_data.py \
    --config configs/config.yaml

EXIT_CODE=$?

echo ""
echo "Step 1 completed at $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
