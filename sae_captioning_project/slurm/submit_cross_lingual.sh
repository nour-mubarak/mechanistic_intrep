#!/bin/bash
#SBATCH --job-name=cross_ling
#SBATCH --partition=gpu-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cross_lingual_%j.out
#SBATCH --error=logs/cross_lingual_%j.err

echo "=========================================="
echo "Cross-Lingual Analysis with Image Visualization"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

mkdir -p logs visualizations/cross_lingual visualizations/sample_predictions

source venv/bin/activate 2>/dev/null || source ../venv/bin/activate

echo "Python: $(which python)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

python scripts/21_cross_lingual_image_analysis.py \
    --config configs/config.yaml \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "Finished: $(date)"
    echo "=========================================="
    
    echo ""
    echo "Generated visualizations:"
    ls -la visualizations/cross_lingual/
    ls -la visualizations/sample_predictions/
else
    echo "Analysis FAILED"
    exit 1
fi
