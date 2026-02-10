#!/bin/bash
#SBATCH --job-name=random_ablation
#SBATCH --partition=res-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:turing:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/random_ablation_%j.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/random_ablation_%j.err

echo "=========================================="
echo "Random Ablation Control Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Use explicit python path from venv
PYTHON=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/venv/bin/python3

echo "Python: $PYTHON"
echo "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project

echo "Running random ablation control experiment..."
$PYTHON scripts/46_random_ablation_control.py

echo ""
echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
