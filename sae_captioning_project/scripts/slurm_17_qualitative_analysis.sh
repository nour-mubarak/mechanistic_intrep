#!/bin/bash
#SBATCH --job-name=17_qualitative
#SBATCH --partition=res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=4:00:00
#SBATCH --output=logs/step10_qualitative_%j.out
#SBATCH --error=logs/step10_qualitative_%j.err

# Step 10: Qualitative Visual Analysis
# =====================================
# Deep qualitative analysis of top activating images with caption themes

echo "=========================================="
echo "Step 10: Qualitative Visual Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
source venv/bin/activate

# Run qualitative analysis for key layers
for layer in 10 14 18 22; do
    echo ""
    echo "Qualitative analysis for layer $layer..."

    python scripts/17_qualitative_visual_analysis.py \
        --config configs/config.yaml \
        --layer $layer \
        --num-features 3 \
        --top-k 30 \
        --output visualizations/qualitative_layer_${layer}/

    echo "Layer $layer completed"
done

echo ""
echo "Qualitative visual analysis completed at $(date)"
