#!/bin/bash
#SBATCH --job-name=qwen2vl_analysis
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/qwen2vl_analysis_%j.out
#SBATCH --error=logs/qwen2vl_analysis_%j.err

# Qwen2-VL Cross-Lingual Analysis
# Run after SAE training is complete

echo "=============================================="
echo "Qwen2-VL Cross-Lingual Analysis"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

export PYTHONUNBUFFERED=1

# Run cross-lingual analysis
python scripts/30_qwen2vl_cross_lingual_analysis.py \
    --layers 0,4,8,12,16,20,24,27 \
    --checkpoints_dir checkpoints/qwen2vl \
    --output_dir results/qwen2vl_analysis \
    --paligemma_results results/cross_lingual_overlap/cross_lingual_overlap_results.json \
    --device cpu \
    --wandb \
    --wandb_project qwen2vl-sae-analysis

echo ""
echo "Analysis complete!"
echo "Results saved to: results/qwen2vl_analysis/"
echo "Date: $(date)"
