#!/bin/bash
#SBATCH --job-name=llama_sae_retrain
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:hopper:1
#SBATCH --time=8:00:00
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/llama_sae_retrain_%j.out
#SBATCH --error=logs/llama_sae_retrain_%j.err

# =============================================================================
# Llama 3.2 Vision SAE Retraining with Tuned Hyperparameters
# =============================================================================
# Original SAEs: EV=36.6%, Dead=98.6%
# Target: EV>65%, Dead<35%
# Changes: L1=5e-5 (was 1e-4), epochs=100, L1 warmup

echo "=============================================="
echo "Llama 3.2 Vision SAE Retraining"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

PYTHON=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/venv/bin/python3

export PYTHONUNBUFFERED=1

# Run retraining for all layers and languages
$PYTHON scripts/41_llama_sae_retrain_tuned.py \
    --all \
    --l1 5e-5 \
    --epochs 100 \
    --input_dir checkpoints/llama32vision/layer_checkpoints \
    --output_dir checkpoints/llama32vision/saes_tuned \
    --device cuda

echo ""
echo "=============================================="
echo "Retraining Complete!"
echo "=============================================="
echo "End: $(date)"
