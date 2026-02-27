#!/bin/bash
#SBATCH --job-name=improved_intervention_paligemma_multilayer
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/improved_intervention_paligemma_multilayer_%j.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/improved_intervention_paligemma_multilayer_%j.err
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00

echo "=========================================="
echo "Improved Intervention: PaLiGemma Multi-Layer"
echo "L9 + L17 combined ablation"
echo "500 images, 25 random runs"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate
PYTHON=venv/bin/python3

echo "Python: $PYTHON"
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Multi-layer: ablate L9 and L17 simultaneously
echo ""
echo "=== PaLiGemma L9+L17 Combined ==="
$PYTHON scripts/improved/48_comprehensive_intervention.py \
    --model paligemma \
    --layers 9 17 \
    --multi_layer \
    --k 100 \
    --n_images 500 \
    --n_random_runs 25 \
    --device cuda

echo ""
echo "Completed: $(date)"
echo "=========================================="
