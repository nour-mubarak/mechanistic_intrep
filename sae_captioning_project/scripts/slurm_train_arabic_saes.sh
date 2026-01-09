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

echo "=========================================="
echo "Training Arabic SAE for Target Layer"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TRANSFORMERS_CACHE="/home2/jmsk62/.cache/huggingface"
export HF_HOME="/home2/jmsk62/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

LAYERS=(0 3 6)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Training Arabic SAE for Layer $LAYER"

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
