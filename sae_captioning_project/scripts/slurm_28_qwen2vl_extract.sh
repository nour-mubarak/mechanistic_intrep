#!/bin/bash
#SBATCH --job-name=qwen2vl_extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/qwen2vl_extract_%A_%a.out
#SBATCH --error=logs/qwen2vl_extract_%A_%a.err
#SBATCH --array=0-1

# Qwen2-VL-7B-Instruct Activation Extraction
# Array job: 0=arabic, 1=english

echo "=============================================="
echo "Qwen2-VL Activation Extraction"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

# Set language based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    LANGUAGE="arabic"
else
    LANGUAGE="english"
fi

echo "Language: $LANGUAGE"
echo "Python: $(which python)"
echo ""

# Set Python to use unbuffered output
export PYTHONUNBUFFERED=1

# Install Qwen dependencies if needed
pip install qwen-vl-utils --quiet 2>/dev/null || true

# Run extraction
python scripts/28_extract_qwen2vl_activations.py \
    --language $LANGUAGE \
    --layers 0,4,8,12,16,20,24,27 \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/qwen2vl \
    --device cuda

echo ""
echo "$LANGUAGE extraction complete!"
echo "Date: $(date)"
