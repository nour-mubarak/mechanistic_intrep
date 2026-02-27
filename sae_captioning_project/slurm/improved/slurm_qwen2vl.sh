#!/bin/bash
#SBATCH --job-name=improved_intervention_qwen2vl
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/improved_intervention_qwen2vl_%j.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/improved_intervention_qwen2vl_%j.err
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

echo "=========================================="
echo "Improved Intervention: Qwen2-VL-7B"
echo "Cross-model replication"
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

# Qwen2-VL: Layer 12 is approximately equivalent to PaLiGemma L9
# (middle layer of 28-layer model vs middle of 18-layer)
echo ""
echo "=== Qwen2-VL Layer 12 ==="
$PYTHON scripts/improved/48_comprehensive_intervention.py \
    --model qwen2vl \
    --layers 12 \
    --k 100 \
    --n_images 500 \
    --n_random_runs 25 \
    --device cuda

echo ""
echo "Completed: $(date)"
echo "=========================================="
