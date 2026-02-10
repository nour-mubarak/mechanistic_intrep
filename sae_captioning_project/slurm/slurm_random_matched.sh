#!/bin/bash
#SBATCH --job-name=random_matched
#SBATCH --partition=res-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:turing:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/random_matched_%j.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/random_matched_%j.err

echo "=========================================="
echo "Random Ablation (Matched Baseline)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

PYTHON=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/venv/bin/python3

echo "Python: $PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
$PYTHON scripts/47_random_ablation_matched.py

echo "Completed: $(date)"
