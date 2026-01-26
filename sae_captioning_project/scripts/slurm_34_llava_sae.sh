#!/bin/bash
#SBATCH --job-name=llava_sae
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/llava_sae_%A_%a.out
#SBATCH --error=logs/llava_sae_%A_%a.err
#SBATCH --array=0-17

# LLaVA-1.5-7B SAE Training
# Array: 9 layers × 2 languages = 18 jobs
# Index mapping: 0-8 = arabic layers, 9-17 = english layers
#
# LLaVA hidden_size: 4096 → SAE latent: 32,768 (8× expansion)

echo "=============================================="
echo "LLaVA-1.5-7B SAE Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

# Layer mapping: 0,4,8,12,16,20,24,28,31 (9 layers)
LAYERS=(0 4 8 12 16 20 24 28 31)

# Determine language and layer from array index
if [ $SLURM_ARRAY_TASK_ID -lt 9 ]; then
    LANGUAGE="arabic"
    LAYER_IDX=$SLURM_ARRAY_TASK_ID
else
    LANGUAGE="english"
    LAYER_IDX=$((SLURM_ARRAY_TASK_ID - 9))
fi

LAYER=${LAYERS[$LAYER_IDX]}

echo "Language: $LANGUAGE"
echo "Layer: $LAYER"
echo "Model d_model: 4096"
echo "SAE expansion: 8× → 32,768 features"
echo "Python: $(which python)"
echo ""

export PYTHONUNBUFFERED=1

python scripts/34_llava_train_sae.py \
    --language $LANGUAGE \
    --layer $LAYER \
    --input_dir checkpoints/llava/layer_checkpoints \
    --output_dir checkpoints/llava/saes \
    --expansion_factor 8 \
    --epochs 50 \
    --device cpu \
    --wandb \
    --wandb_project llava-sae-analysis

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Layer $LAYER $LANGUAGE SAE training complete!"
else
    echo "✗ Layer $LAYER $LANGUAGE SAE training FAILED with code $EXIT_CODE"
fi
echo "Date: $(date)"

exit $EXIT_CODE
