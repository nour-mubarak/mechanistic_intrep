#!/bin/bash
#SBATCH --job-name=extract_layer_ranges
#SBATCH --partition=res-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_extraction_ranges_%j.out
#SBATCH --error=logs/slurm_extraction_ranges_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmsk62@durham.ac.uk

# SLURM Job Script for Layer Range Extraction on Durham NCC
# ===========================================================
#
# This script extracts activations from specific layer ranges.
# Useful for splitting extraction across multiple jobs or GPUs.
#
# Submit with: sbatch scripts/slurm_extract_layer_ranges.sh

echo "======================================"
echo "Layer Range Activation Extraction"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# Load required modules (adjust based on NCC configuration)
module purge
module load cuda/12.1
module load gcc/11.2.0

echo "Loaded modules:"
module list
echo ""

# Activate virtual environment
source venv/bin/activate

# Verify GPU availability
echo "GPU Information:"
nvidia-smi
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Configuration - MODIFY THESE LAYER RANGES AS NEEDED
CONFIG_FILE="configs/config.yaml"
BATCH_SIZE=1
CHECKPOINT_INTERVAL=50
OUTPUT_DIR="checkpoints/full_layers_ncc"

# Define layer ranges to extract
# Option 1: Extract first half (layers 0-16)
LAYER_RANGES="0-16"

# Option 2: Extract second half (layers 17-33)
# LAYER_RANGES="17-33"

# Option 3: Extract specific important layers
# LAYER_RANGES="0 2 6 10 14 18 22 26 30 33"

# Option 4: Extract every other layer
# LAYER_RANGES="0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Layer ranges: $LAYER_RANGES"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run extraction for specified layer ranges
echo "Starting extraction at $(date)"
python scripts/18_extract_full_activations_ncc.py \
    --config "$CONFIG_FILE" \
    --layer-ranges $LAYER_RANGES \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "logs/extraction_ranges_${SLURM_JOB_ID}.log"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Extraction completed successfully for layers: $LAYER_RANGES"
else
    echo "ERROR: Extraction failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
