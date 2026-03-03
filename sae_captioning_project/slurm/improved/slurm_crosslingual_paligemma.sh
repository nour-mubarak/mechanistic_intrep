#!/bin/bash
#SBATCH --job-name=crosslingual_paligemma
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/crosslingual_paligemma_%j.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/crosslingual_paligemma_%j.err
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

echo "=========================================="
echo "Cross-Lingual Ablation: PaLiGemma-3B"
echo "4 conditions: EN→EN, EN→AR, AR→EN, AR→AR"
echo "500 images, 10 random runs per condition"
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

# PaLiGemma Layer 9: the primary intervention layer
echo ""
echo "=== Cross-Lingual Ablation: PaLiGemma L9 ==="
$PYTHON scripts/improved/52_cross_lingual_ablation.py \
    --model paligemma \
    --layer 9 \
    --k 100 \
    --n_images 500 \
    --n_random_runs 10 \
    --device cuda

echo ""
echo "End time: $(date)"
echo "=========================================="
