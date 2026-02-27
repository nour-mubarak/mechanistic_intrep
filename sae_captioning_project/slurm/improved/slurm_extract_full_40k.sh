#!/bin/bash
#SBATCH --job-name=extract_full_40k
#SBATCH --output=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/extract_full_40k_%j_%a.out
#SBATCH --error=/home2/jmsk62/mechanistic_intrep/sae_captioning_project/logs/extract_full_40k_%j_%a.err
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-1

# Array job: index 0 = english, index 1 = arabic
LANGUAGES=("english" "arabic")
LANGUAGE=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Full 40K Activation Extraction"
echo "Language: $LANGUAGE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID, Array: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate
PYTHON=venv/bin/python3

echo "Python: $PYTHON"
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "=== Extracting $LANGUAGE activations (layers 9, 17) ==="
$PYTHON scripts/improved/49_extract_full_40k_activations.py \
    --language $LANGUAGE \
    --layers 9 17 \
    --batch_size 4 \
    --chunk_size 1000

echo ""
echo "Completed: $(date)"
echo "=========================================="
