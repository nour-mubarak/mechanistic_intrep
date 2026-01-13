#!/bin/bash
#SBATCH --job-name=extract_L6_fix
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/extract_layer6_fix_%j.out
#SBATCH --error=logs/extract_layer6_fix_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmsk62@durham.ac.uk

# Re-extract Layer 6 activations for both Arabic and English
# ===========================================================

echo "======================================"
echo "Layer 6 Re-extraction (Fix)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

cd ~/sae_project/sae_captioning_project

# Load modules
module purge
module load cuda/12.1
module load gcc/11.2.0

# Activate environment
source venv/bin/activate 2>/dev/null || source ~/venv/bin/activate 2>/dev/null || echo "Using system Python"

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Backup old corrupt files
mkdir -p checkpoints/full_layers_ncc/layer_checkpoints/backup_layer6
mv checkpoints/full_layers_ncc/layer_checkpoints/layer_6_* checkpoints/full_layers_ncc/layer_checkpoints/backup_layer6/ 2>/dev/null

echo "=== Extracting Layer 6 Arabic ==="
python3 scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 6 \
    --languages arabic \
    --batch-size 1 \
    --checkpoint-interval 100 \
    --output-dir checkpoints/full_layers_ncc

echo ""
echo "=== Extracting Layer 6 English ==="
python3 scripts/22_extract_english_activations.py \
    --config configs/config.yaml \
    --layer 6 \
    --output-dir checkpoints/full_layers_ncc/layer_checkpoints

echo ""
echo "=== Verification ==="
ls -lh checkpoints/full_layers_ncc/layer_checkpoints/layer_6_*

echo ""
echo "Completed at: $(date)"
