#!/bin/bash
#SBATCH --job-name=13_feature_ablation
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/step6_feature_ablation_%j.out
#SBATCH --error=logs/step6_feature_ablation_%j.err

# Step 6: Feature Ablation Analysis
# ==================================
# Tests causal effects of gender-biased features

echo "=========================================="
echo "Step 6: Feature Ablation Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
source venv/bin/activate

# Run ablation analysis for key layers
for layer in 10 14 18 22; do
    echo ""
    echo "Ablation analysis for layer $layer..."

    python scripts/13_feature_ablation_analysis.py \
        --config configs/config.yaml \
        --layer $layer \
        --num-features 3 \
        --output visualizations/feature_ablation/layer_${layer}/

    echo "Layer $layer completed"
done

echo ""
echo "Feature ablation analysis completed at $(date)"
