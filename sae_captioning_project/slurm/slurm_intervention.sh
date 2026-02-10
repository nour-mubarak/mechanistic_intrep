#!/bin/bash
#SBATCH --job-name=caption_intervention
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/intervention_%j.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/intervention_%j.err
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

echo "=========================================="
echo "Caption Generation Intervention Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Activate environment
cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source /home2/jmsk62/mechanistic_intrep/sae_captioning_project/venv/bin/activate
export PATH="/home2/jmsk62/mechanistic_intrep/sae_captioning_project/venv/bin:$PATH"

# Verify Python environment
PYTHON=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/venv/bin/python3
echo "Python: $PYTHON"
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Run intervention experiment
echo "Running intervention experiment..."
$PYTHON scripts/45_caption_intervention.py \
    --layer 9 \
    --n_images 100 \
    --k_values 200 500 1000 \
    --device cuda

echo ""
echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
