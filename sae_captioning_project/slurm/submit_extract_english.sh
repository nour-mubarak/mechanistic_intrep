#!/bin/bash
#SBATCH --job-name=en_extract
#SBATCH --partition=gpu-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/extract_english_%A_%a.out
#SBATCH --error=logs/extract_english_%A_%a.err
#SBATCH --array=0,3,9,12,15,17

echo "=========================================="
echo "English Activation Extraction"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Layer: $SLURM_ARRAY_TASK_ID"
echo "Started: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

mkdir -p logs checkpoints/full_layers_ncc/layer_checkpoints

source venv/bin/activate 2>/dev/null || source ../venv/bin/activate

echo "Python: $(which python)"
echo ""

# Extract for the layer specified by array task
python scripts/22_extract_english_activations.py \
    --config configs/config.yaml \
    --layer $SLURM_ARRAY_TASK_ID \
    --output-dir checkpoints/full_layers_ncc \
    --checkpoint-interval 100 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Layer $SLURM_ARRAY_TASK_ID extraction completed!"
    echo "Finished: $(date)"
    echo "=========================================="
else
    echo "Extraction FAILED for layer $SLURM_ARRAY_TASK_ID"
    exit 1
fi
