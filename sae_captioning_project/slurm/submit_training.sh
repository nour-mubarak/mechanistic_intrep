#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@university.edu

# ============================================================================
# SAE Training Job
# ============================================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Load required modules
module purge
module load cuda/12.1
module load python/3.10

# Activate virtual environment
source ~/envs/sae_env/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create logs directory
mkdir -p logs

# Run SAE training
echo "Starting SAE training..."
python scripts/03_train_sae.py --config configs/config.yaml

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "SAE training completed successfully!"
else
    echo "SAE training failed with exit code $EXIT_CODE"
fi

echo ""
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE
