#!/bin/bash
#SBATCH --job-name=sae_extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/extraction_%j.out
#SBATCH --error=logs/extraction_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@university.edu

# ============================================================================
# SAE Activation Extraction Job
# ============================================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Load required modules (adjust for your HPC environment)
# module purge
# module load cuda/12.1
# module load python/3.10

# Activate virtual environment if needed
# source ~/envs/sae_env/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
# Or use absolute path:
# cd /path/to/sae_captioning_project

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=~/.cache/huggingface  # Adjust cache location

# Create logs directory if needed
mkdir -p logs

# Run activation extraction
echo "Starting activation extraction..."
python3 scripts/02_extract_activations.py --config configs/config.yaml

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Activation extraction completed successfully!"
else
    echo "Activation extraction failed with exit code $EXIT_CODE"
fi

echo ""
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE
