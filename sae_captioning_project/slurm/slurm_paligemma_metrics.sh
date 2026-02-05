#!/bin/bash
#SBATCH --job-name=paligemma_metrics
#SBATCH --output=logs/paligemma_metrics_%j.out
#SBATCH --error=logs/paligemma_metrics_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@durham.ac.uk

# ============================================================
# SLURM Job: Compute PaLiGemma-3B SAE Quality Metrics
# ============================================================
# This job computes publication-ready SAE metrics:
# - Explained Variance %
# - Dead Feature Ratio  
# - Mean L0 (Sparsity)
# - Reconstruction Cosine Similarity
# ============================================================

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "============================================================"

# Set up environment
cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project

# Check for GPU
nvidia-smi

# Run the computation
echo ""
echo "Running PaLiGemma-3B metrics computation..."
echo ""

python3 scripts/compute_paligemma_metrics.py \
    --device cuda \
    --batch_size 512 \
    --max_samples 20000

echo ""
echo "============================================================"
echo "End Time: $(date)"
echo "============================================================"
