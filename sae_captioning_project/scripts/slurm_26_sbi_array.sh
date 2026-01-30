#!/bin/bash
#SBATCH --job-name=sbi_layer
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/sbi_layer_%A_%a.out
#SBATCH --error=logs/sbi_layer_%A_%a.err
#SBATCH --array=0,3,6,9,12,15,17

# Surgical Bias Intervention - Single Layer Analysis
# Uses array jobs for parallelism

echo "=============================================="
echo "SBI Analysis - Layer ${SLURM_ARRAY_TASK_ID}"
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

# Optional: Enable W&B logging (set to true for remote monitoring)
USE_WANDB=true

if [ "$USE_WANDB" = true ]; then
    python scripts/26_surgical_bias_intervention.py \
        --config configs/config.yaml \
        --output_dir results/sbi_analysis \
        --layers ${SLURM_ARRAY_TASK_ID} \
        --device cpu \
        --wandb
else
    python scripts/26_surgical_bias_intervention.py \
        --config configs/config.yaml \
        --output_dir results/sbi_analysis \
        --layers ${SLURM_ARRAY_TASK_ID} \
        --device cpu
fi

echo ""
echo "Layer ${SLURM_ARRAY_TASK_ID} complete!"
echo "Date: $(date)"
