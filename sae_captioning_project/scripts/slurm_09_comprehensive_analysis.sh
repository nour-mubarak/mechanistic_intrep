#!/bin/bash
#SBATCH --job-name=09_comprehensive_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/step4_comprehensive_analysis_%j.out
#SBATCH --error=logs/step4_comprehensive_analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@durham.ac.uk

# Step 4-9: Comprehensive Analysis
# =================================
# Runs comprehensive analysis across all layers

echo "=========================================="
echo "Step 4-9: Comprehensive Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Analyzing all 34 layers"
echo "Start: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.10 cuda/12.1
source venv/bin/activate

# Run comprehensive analysis for all layers
python scripts/09_comprehensive_analysis.py \
    --config configs/config.yaml \
    --layers $(seq 0 33) \
    --output visualizations/comprehensive_all_layers/

EXIT_CODE=$?

echo ""
echo "Comprehensive analysis completed at $(date)"
echo "Exit code: $EXIT_CODE"

# Generate summary
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis Results Summary"
    echo "=========================================="
    echo "Output directory: visualizations/comprehensive_all_layers/"
    echo ""
    echo "Generated files:"
    ls -lh visualizations/comprehensive_all_layers/
fi

exit $EXIT_CODE
