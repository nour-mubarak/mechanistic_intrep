#!/bin/bash
#SBATCH --job-name=extract_L6_en
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/extract_layer6_english_%j.out
#SBATCH --error=logs/extract_layer6_english_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jmsk62@durham.ac.uk

# Extract Layer 6 English activations (parallel with Arabic)
# ===========================================================

echo "======================================"
echo "Layer 6 English Extraction (Parallel)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

cd ~/sae_project/sae_captioning_project

# Load modules
module purge
module load cuda/12.1

# Activate environment
source venv/bin/activate 2>/dev/null || echo "Using system Python"

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

echo "=== Extracting Layer 6 English ==="
python3 scripts/22_extract_english_activations.py \
    --config configs/config.yaml \
    --layer 6 \
    --output-dir checkpoints/full_layers_ncc/layer_checkpoints

echo ""
echo "=== Verification ==="
ls -lh checkpoints/full_layers_ncc/layer_checkpoints/layer_6_english*

echo ""
echo "Completed at: $(date)"
