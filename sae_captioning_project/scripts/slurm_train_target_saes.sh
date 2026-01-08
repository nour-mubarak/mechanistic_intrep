#!/bin/bash
#SBATCH --job-name=train_target_saes
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_sae_L%a_%j.out
#SBATCH --error=logs/train_sae_L%a_%j.err
#SBATCH --array=0-6

# Train SAEs for target layers only: 0, 3, 6, 9, 12, 15, 17

echo "=========================================="
echo "Training SAE for Target Layer"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

# Activate environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate sae_env

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TRANSFORMERS_CACHE="/home2/jmsk62/.cache/huggingface"
export HF_HOME="/home2/jmsk62/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Map array index to layer
LAYERS=(0 3 6 9 12 15 17)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Training SAE for Layer $LAYER"

# Check if layer checkpoints exist
CHUNK_COUNT=$(ls checkpoints/full_layers_ncc/layer_checkpoints/layer_${LAYER}_*.pt 2>/dev/null | wc -l)
echo "Found $CHUNK_COUNT chunks for layer $LAYER"

if [ $CHUNK_COUNT -lt 50 ]; then
    echo "ERROR: Not enough chunks for layer $LAYER (need at least 50, have $CHUNK_COUNT)"
    exit 1
fi

# Train SAE for this layer
python scripts/03_train_sae.py \
    --layer $LAYER \
    --activations-dir checkpoints/full_layers_ncc/layer_checkpoints \
    --output-dir checkpoints/saes \
    --expansion-factor 8 \
    --l1-coefficient 5e-4 \
    --learning-rate 1e-4 \
    --num-epochs 10 \
    --batch-size 32

echo "SAE training for Layer $LAYER complete"
echo "Date: $(date)"
