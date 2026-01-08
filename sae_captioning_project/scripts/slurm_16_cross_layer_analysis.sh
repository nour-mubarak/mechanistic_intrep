#!/bin/bash
#SBATCH --job-name=16_cross_layer
#SBATCH --partition=res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=6:00:00
#SBATCH --output=logs/step9_cross_layer_%j.out
#SBATCH --error=logs/step9_cross_layer_%j.err

# Step 9: Cross-Layer Analysis
# =============================
# Compares gender bias evolution across all layers

echo "=========================================="
echo "Step 9: Cross-Layer Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing all 34 layers"
echo "Start: $(date)"
echo "=========================================="

module purge
source venv/bin/activate

# Run cross-layer analysis with fine granularity
# Compare every 2nd layer for detailed evolution tracking
python scripts/16_cross_layer_analysis.py \
    --config configs/config.yaml \
    --layers $(seq 0 2 33) \
    --num-features 3 \
    --output visualizations/cross_layer_analysis_full/

EXIT_CODE=$?

echo ""
echo "Cross-layer analysis completed at $(date)"
echo "Exit code: $EXIT_CODE"

# Also run analysis for key layers only (more detailed)
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Running detailed analysis for key layers..."

    python scripts/16_cross_layer_analysis.py \
        --config configs/config.yaml \
        --layers 0 6 10 14 18 22 26 30 33 \
        --num-features 5 \
        --output visualizations/cross_layer_analysis_key/

    echo "Key layer analysis completed"
fi

exit $EXIT_CODE
