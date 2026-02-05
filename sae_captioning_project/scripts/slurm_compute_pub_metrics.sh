#!/bin/bash
#SBATCH --job-name=pub_metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pub_metrics_%j.out
#SBATCH --error=logs/pub_metrics_%j.err

# Compute publication-ready SAE metrics for all models

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project

# Load required modules
module load python/3.9
module load cuda/11.8

# Activate environment if needed
# source venv/bin/activate

echo "Starting publication metrics computation..."
echo "Date: $(date)"
echo "Node: $(hostname)"

python3 scripts/compute_publication_metrics.py --device cuda 2>&1

echo "Completed at: $(date)"
