#!/bin/bash
#SBATCH --job-name=train_arabic_saes
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_sae_arabic_L%a_%j.out
#SBATCH --error=logs/train_sae_arabic_L%a_%j.err
#SBATCH --array=0-2

# Train Arabic SAEs for layers that have data: 0, 3, 6
# (9, 12, 15, 17 need extraction to complete first)

echo "=========================================="
echo "Training Arabic SAE for Target Layer"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

# Activate virtual environment with Python 3.10
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "Python: $(which python) - $(python --version)"
export TRANSFORMERS_CACHE="/home2/jmsk62/.cache/huggingface"
export HF_HOME="/home2/jmsk62/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Map array index to layer
LAYERS=(0 3 6)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Training Arabic SAE for Layer $LAYER"

# Check if layer checkpoints exist for Arabic (chunk files OR merged file)
CHUNK_COUNT=$(ls checkpoints/full_layers_ncc/layer_checkpoints/layer_${LAYER}_arabic_chunk_*.pt 2>/dev/null | wc -l)
MERGED_FILE="checkpoints/full_layers_ncc/layer_checkpoints/layer_${LAYER}_arabic.pt"
echo "Found $CHUNK_COUNT Arabic chunks for layer $LAYER"

if [ $CHUNK_COUNT -lt 50 ] && [ ! -f "$MERGED_FILE" ]; then
    echo "ERROR: Not enough Arabic data for layer $LAYER (need at least 50 chunks or merged file)"
    exit 1
fi

if [ -f "$MERGED_FILE" ]; then
    echo "Found merged file: $MERGED_FILE"
fi

# Train SAE for this layer using NCC format (Arabic)
# Using 15% sampling with 48GB memory on Ampere GPU (~4M tokens per layer)
python scripts/train_sae_ncc.py \
    --layer $LAYER \
    --language arabic \
    --config configs/config.yaml \
    --checkpoint-dir checkpoints/full_layers_ncc/layer_checkpoints \
    --output-dir checkpoints/saes \
    --sample-ratio 0.15 \
    --device cuda

echo "SAE training for Arabic Layer $LAYER complete"
echo "Date: $(date)"
