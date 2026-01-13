#!/bin/bash
#SBATCH --job-name=train_L6_ar
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_sae_layer6_arabic_%j.out
#SBATCH --error=logs/train_sae_layer6_arabic_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jmsk62@durham.ac.uk

# Train Layer 6 Arabic SAE
echo "======================================"
echo "Training Layer 6 Arabic SAE"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd ~/sae_project/sae_captioning_project

module purge
module load cuda/12.1

source venv/bin/activate 2>/dev/null || echo "Using system Python"

nvidia-smi --query-gpu=name,memory.total --format=csv

# Backup old SAE
mv checkpoints/saes/sae_arabic_layer_6.pt checkpoints/saes/sae_arabic_layer_6_old.pt 2>/dev/null

python3 scripts/03_train_sae.py \
    --config configs/config.yaml \
    --layer 6 \
    --language arabic

echo "Completed at: $(date)"
