#!/bin/bash
#SBATCH --job-name=sae_analyze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@university.edu

# ============================================================================
# SAE Analysis Job
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

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Create directories
mkdir -p logs results visualizations

# Step 1: Feature Analysis
echo "Starting feature analysis..."
python3 scripts/04_analyze_features.py --config configs/config.yaml

if [ $? -ne 0 ]; then
    echo "Feature analysis failed!"
    exit 1
fi

# Step 2: Steering Experiments (optional - can be slow)
echo ""
echo "Starting steering experiments..."
python3 scripts/05_steering_experiments.py --config configs/config.yaml

if [ $? -ne 0 ]; then
    echo "Steering experiments failed (continuing anyway)..."
fi

# Step 3: Generate Visualizations
echo ""
echo "Generating visualizations..."
python3 scripts/06_generate_visualizations.py --config configs/config.yaml

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Analysis pipeline completed successfully!"
else
    echo "Analysis pipeline finished with some errors"
fi

echo ""
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE
