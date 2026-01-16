#!/bin/bash
#SBATCH --job-name=qwen2vl_sae
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/qwen2vl_sae_%A_%a.out
#SBATCH --error=logs/qwen2vl_sae_%A_%a.err
#SBATCH --array=0-15

# Qwen2-VL SAE Training
# Array: 8 layers × 2 languages = 16 jobs
# Index mapping: 0-7 = arabic layers, 8-15 = english layers

echo "=============================================="
echo "Qwen2-VL SAE Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

# Layer mapping: 0,4,8,12,16,20,24,27
LAYERS=(0 4 8 12 16 20 24 27)

# Determine language and layer from array index
if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
    LANGUAGE="arabic"
    LAYER_IDX=$SLURM_ARRAY_TASK_ID
else
    LANGUAGE="english"
    LAYER_IDX=$((SLURM_ARRAY_TASK_ID - 8))
fi

LAYER=${LAYERS[$LAYER_IDX]}

echo "Language: $LANGUAGE"
echo "Layer: $LAYER"
echo "Python: $(which python)"
echo ""

export PYTHONUNBUFFERED=1

python scripts/29_train_qwen2vl_sae.py \
    --language $LANGUAGE \
    --layer $LAYER \
    --input_dir checkpoints/qwen2vl/layer_checkpoints \
    --output_dir checkpoints/qwen2vl/saes \
    --expansion_factor 8 \
    --epochs 50 \
    --device cpu

echo ""
echo "Layer $LAYER $LANGUAGE complete!"
echo "Date: $(date)"
