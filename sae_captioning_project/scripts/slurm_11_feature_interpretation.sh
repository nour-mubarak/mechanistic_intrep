#!/bin/bash
#SBATCH --job-name=11_feature_interpretation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/step5_feature_interpretation_%j.out
#SBATCH --error=logs/step5_feature_interpretation_%j.err

# Step 5: Feature Interpretation
# ===============================
# Interprets SAE features across all layers

echo "=========================================="
echo "Step 5: Feature Interpretation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
module load python/3.10 cuda/12.1
source venv/bin/activate

# Run feature interpretation for key layers
for layer in 0 6 10 14 18 22 26 30 33; do
    echo ""
    echo "Analyzing layer $layer..."

    python scripts/11_feature_interpretation.py \
        --config configs/config.yaml \
        --layer $layer \
        --num-features 100 \
        --output visualizations/feature_interpretation/layer_${layer}/

    echo "Layer $layer completed"
done

echo ""
echo "Feature interpretation completed at $(date)"
