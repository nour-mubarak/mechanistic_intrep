#!/bin/bash
#SBATCH --job-name=qwen2vl_gpu
#SBATCH --partition=res-gpu-small
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ampere:1
#SBATCH --output=logs/qwen2vl_analysis_%j.out
#SBATCH --error=logs/qwen2vl_analysis_%j.err

# Qwen2-VL Cross-Lingual Analysis - GPU Version (Extended Time)
echo "=============================================="
echo "Qwen2-VL Cross-Lingual Analysis (GPU)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python scripts/30_qwen2vl_cross_lingual_analysis.py \
    --layers 0,4,8,12,16,20,24,27 \
    --checkpoints_dir checkpoints/qwen2vl \
    --output_dir results/qwen2vl_analysis \
    --paligemma_results results/cross_lingual_overlap/cross_lingual_overlap_results.json \
    --device cuda \
    --wandb \
    --wandb_project qwen2vl-sae-analysis

echo "Analysis complete! Date: $(date)"
