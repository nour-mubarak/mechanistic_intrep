#!/bin/bash
# Parallel Layer Extraction on Durham NCC
# ========================================
#
# This script submits multiple SLURM jobs to extract different layer ranges
# in parallel across multiple GPUs, significantly reducing total extraction time.
#
# Usage: bash scripts/slurm_parallel_extraction.sh

set -e

echo "======================================"
echo "Parallel Layer Extraction Submission"
echo "======================================"
echo "Submitting jobs to Durham NCC cluster"
echo ""

# Create logs directory
mkdir -p logs

# Job configuration
CONFIG_FILE="configs/config.yaml"
BATCH_SIZE=1
CHECKPOINT_INTERVAL=50
OUTPUT_DIR="checkpoints/full_layers_ncc"
TIME_LIMIT="6:00:00"  # 6 hours per job
MEMORY="64G"
CPUS=16

# Email notification (MODIFY THIS)
EMAIL="jmsk62@durham.ac.uk"

# Function to submit extraction job for a layer range
submit_job() {
    local layer_range=$1
    local job_name=$2

    echo "Submitting job: $job_name (layers: $layer_range)"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=res-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gres=gpu:1
#SBATCH --mem=$MEMORY
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$EMAIL

# Load modules
module purge
module load cuda/12.1
module load gcc/11.2.0

# Activate environment
source venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "Job: $job_name"
echo "Layers: $layer_range"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start: \$(date)"
echo "======================================"
echo ""

# Run extraction
python scripts/18_extract_full_activations_ncc.py \
    --config "$CONFIG_FILE" \
    --layer-ranges $layer_range \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "logs/${job_name}_\${SLURM_JOB_ID}.log"

EXIT_CODE=\$?

echo ""
echo "======================================"
echo "Job $job_name completed at \$(date)"
echo "Exit code: \$EXIT_CODE"
echo "======================================"

exit \$EXIT_CODE
EOF

    # Store job ID
    sleep 1  # Small delay to avoid overwhelming scheduler
}

echo "Strategy: Split 34 layers into 4 parallel jobs"
echo ""

# Submit 4 parallel jobs, each handling ~8-9 layers
# This reduces total time from ~2 hours to ~30-40 minutes

submit_job "0-8" "extract_layers_00_08"
submit_job "9-16" "extract_layers_09_16"
submit_job "17-24" "extract_layers_17_24"
submit_job "25-33" "extract_layers_25_33"

echo ""
echo "======================================"
echo "All jobs submitted!"
echo "======================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  logs/extract_layers_*.out"
echo ""
echo "When all jobs complete, outputs will be in:"
echo "  $OUTPUT_DIR/layer_checkpoints/"
echo ""
echo "Expected completion time: ~40-60 minutes"
echo "(vs ~2 hours for sequential extraction)"
