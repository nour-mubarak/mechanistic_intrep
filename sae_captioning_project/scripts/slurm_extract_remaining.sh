#!/bin/bash
#SBATCH --job-name=extract_remaining
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=4:00:00
#SBATCH --output=logs/extract_L%a_%j.out
#SBATCH --error=logs/extract_L%a_%j.err
#SBATCH --array=0-3

# Extract remaining target layers: 9, 12, 15, 17

echo "=========================================="
echo "Extracting Remaining Target Layers"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

# Activate virtual environment with Python 3.10
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TRANSFORMERS_CACHE="/home2/jmsk62/.cache/huggingface"
export HF_HOME="/home2/jmsk62/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Python: $(which python) - $(python --version)"

# Map array index to layer
LAYERS=(9 12 15 17)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Extracting Layer $LAYER"

# Run extraction for this layer
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --output-dir checkpoints/full_layers_ncc \
    --languages english \
    --layer-ranges $LAYER \
    --batch-size 1 \
    --checkpoint-interval 100

echo "Layer $LAYER extraction complete"
echo "Date: $(date)"
