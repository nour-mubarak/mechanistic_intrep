#!/bin/bash
#SBATCH --job-name=15_feature_amplification
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/step8_feature_amplification_%j.out
#SBATCH --error=logs/step8_feature_amplification_%j.err

# Step 8: Feature Amplification Analysis
# =======================================
# Tests dose-response effects of amplifying gender-biased features

echo "=========================================="
echo "Step 8: Feature Amplification Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
module load python/3.10 cuda/12.1
source venv/bin/activate

# Run amplification analysis for key layers
for layer in 10 14 18 22; do
    echo ""
    echo "Amplification analysis for layer $layer..."

    python scripts/15_feature_amplification.py \
        --config configs/config.yaml \
        --layer $layer \
        --num-features 3 \
        --amplification-factors 1.5 2.0 3.0 \
        --output visualizations/feature_amplification_layer_${layer}/

    echo "Layer $layer completed"
done

echo ""
echo "Feature amplification analysis completed at $(date)"
