#!/bin/bash
#SBATCH --job-name=full_activation_extraction
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_extraction_%j.out
#SBATCH --error=logs/slurm_extraction_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@durham.ac.uk

# SLURM Job Script for Full Layer Activation Extraction on Durham NCC
# ====================================================================
#
# This script extracts activations from all 34 layers of Gemma-3-4B
# on the full 2000-sample dataset using Durham's NCC GPU cluster.
#
# Submit with: sbatch scripts/slurm_extract_full_activations.sh

echo "======================================"
echo "Full Layer Activation Extraction (NCC)"
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
module load python/3.10
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
export CUDA_LAUNCH_BLOCKING=0

# Configuration
CONFIG_FILE="configs/config.yaml"
BATCH_SIZE=1
CHECKPOINT_INTERVAL=50
OUTPUT_DIR="checkpoints/full_layers_ncc"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run extraction for ALL layers (0-33)
echo "Starting extraction at $(date)"
python scripts/18_extract_full_activations_ncc.py \
    --config "$CONFIG_FILE" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "logs/extraction_${SLURM_JOB_ID}.log"

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Extraction completed successfully"
    echo ""
    echo "Outputs saved to: $OUTPUT_DIR"
    echo "  - Layer-wise checkpoints: $OUTPUT_DIR/layer_checkpoints/"
    echo "  - Combined files: $OUTPUT_DIR/activations_*_all_layers.pt"
    echo "  - Metadata: $OUTPUT_DIR/extraction_metadata.json"
else
    echo "ERROR: Extraction failed with exit code $EXIT_CODE"
    echo "Check logs for details: logs/extraction_${SLURM_JOB_ID}.log"
fi

exit $EXIT_CODE
