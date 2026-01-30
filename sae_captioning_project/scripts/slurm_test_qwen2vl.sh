#!/bin/bash
#SBATCH --job-name=qwen2vl_test
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:30:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/qwen2vl_test_%j.out
#SBATCH --error=logs/qwen2vl_test_%j.err

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

echo "Starting Qwen2-VL test..."
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python scripts/test_qwen2vl.py
