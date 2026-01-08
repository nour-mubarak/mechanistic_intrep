#!/bin/bash
#SBATCH --job-name=14_visual_patterns
#SBATCH --partition=res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=3:00:00
#SBATCH --output=logs/step7_visual_patterns_%j.out
#SBATCH --error=logs/step7_visual_patterns_%j.err

# Step 7: Visual Pattern Analysis
# ================================
# Identifies visual patterns that activate gender-biased features

echo "=========================================="
echo "Step 7: Visual Pattern Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
source venv/bin/activate

# Run visual pattern analysis for key layers
for layer in 10 14 18 22; do
    echo ""
    echo "Visual pattern analysis for layer $layer..."

    python scripts/14_visual_pattern_analysis.py \
        --config configs/config.yaml \
        --layer $layer \
        --num-features 3 \
        --output visualizations/visual_patterns_layer_${layer}/

    echo "Layer $layer completed"
done

echo ""
echo "Visual pattern analysis completed at $(date)"
