#!/bin/bash
#SBATCH --job-name=improved_intervention_llama32vision
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/improved_intervention_llama32vision_%j.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/improved_intervention_llama32vision_%j.err
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

echo "=========================================="
echo "Improved Intervention: Llama-3.2-Vision-11B"
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

# Llama-3.2-Vision: Layer 20 is approximately equivalent to PaLiGemma L9
# (middle layer of 40-layer model)
echo ""
echo "=== Llama-3.2-Vision Layer 20 ==="
$PYTHON scripts/improved/48_comprehensive_intervention.py \
    --model llama32vision \
    --layers 20 \
    --k 100 \
    --n_images 500 \
    --n_random_runs 25 \
    --device cuda

echo ""
echo "Completed: $(date)"
echo "=========================================="
