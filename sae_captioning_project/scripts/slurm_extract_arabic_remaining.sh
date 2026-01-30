#!/bin/bash
#SBATCH --job-name=extract_ar_rem
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_arabic_remaining_L%a_%j.out
#SBATCH --error=logs/extract_arabic_remaining_L%a_%j.err
#SBATCH --array=0-3

# Extract Arabic NCC activations for REMAINING layers: 9, 12, 15, 17
# Layers 0, 3, 6 are already complete

echo "=========================================="
echo "Arabic NCC Extraction - Remaining Layers"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project

# Activate virtual environment
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TRANSFORMERS_CACHE="/home2/jmsk62/.cache/huggingface"
export HF_HOME="/home2/jmsk62/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Python: $(which python) - $(python --version)"

# Map array index to remaining layers
LAYERS=(9 12 15 17)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Extracting Arabic activations for Layer $LAYER"

# Check if layer already has sufficient chunks (100+)
CHUNK_COUNT=$(ls checkpoints/full_layers_ncc/layer_checkpoints/layer_${LAYER}_arabic_chunk_*.pt 2>/dev/null | wc -l)
MERGED_FILE="checkpoints/full_layers_ncc/layer_checkpoints/layer_${LAYER}_arabic.pt"

if [ -f "$MERGED_FILE" ]; then
    echo "Layer $LAYER already has merged file, skipping..."
    exit 0
fi

if [ $CHUNK_COUNT -ge 100 ]; then
    echo "Layer $LAYER already has $CHUNK_COUNT chunks (>=100), skipping..."
    exit 0
fi

echo "Found $CHUNK_COUNT existing chunks for layer $LAYER, will resume extraction..."

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
