#!/bin/bash
#SBATCH --job-name=extract_arabic
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/extract_arabic_L%a_%j.out
#SBATCH --error=logs/extract_arabic_L%a_%j.err
#SBATCH --array=0-6

# Extract Arabic NCC activations for target layers: 0, 3, 6, 9, 12, 15, 17
# These match the English extractions for cross-lingual comparison

echo "=========================================="
echo "Arabic NCC Extraction"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

# Activate virtual environment
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TRANSFORMERS_CACHE="/home2/jmsk62/.cache/huggingface"
export HF_HOME="/home2/jmsk62/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Python: $(which python) - $(python --version)"

# Map array index to layer
LAYERS=(0 3 6 9 12 15 17)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Extracting Arabic activations for Layer $LAYER"

# Run extraction for this layer with Arabic language only
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges $LAYER \
    --languages arabic \
    --batch-size 1 \
    --checkpoint-interval 100 \
    --output-dir checkpoints/full_layers_ncc

echo "Arabic extraction for Layer $LAYER complete"
echo "Date: $(date)"
