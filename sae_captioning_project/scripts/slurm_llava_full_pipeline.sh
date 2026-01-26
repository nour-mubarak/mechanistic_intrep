#!/bin/bash
#SBATCH --job-name=llava_full
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/llava_full_pipeline_%j.out
#SBATCH --error=logs/llava_full_pipeline_%j.err

# =============================================================================
# LLaVA-1.5-7B Full Pipeline
# =============================================================================
# Runs the complete LLaVA pipeline sequentially:
# 1. Extract activations (both languages)
# 2. Train SAEs (all layers, both languages)
# 3. Run cross-lingual analysis
#
# For parallel execution, use the individual SLURM array jobs instead:
# - slurm_33_llava_extract.sh (array job for extraction)
# - slurm_34_llava_sae.sh (array job for SAE training)
# - slurm_35_llava_analysis.sh (analysis job)

echo "=============================================="
echo "LLaVA-1.5-7B Full Pipeline"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME=~/.cache/huggingface

LAYERS="0,4,8,12,16,20,24,28,31"

# Create output directories
mkdir -p checkpoints/llava/layer_checkpoints
mkdir -p checkpoints/llava/saes
mkdir -p results/llava_analysis
mkdir -p logs

# =============================================================================
# Stage 1: Extract Arabic Activations
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 1/5: Extracting Arabic Activations"
echo "=============================================="
echo "Start: $(date)"

python scripts/33_llava_extract_activations.py \
    --language arabic \
    --layers $LAYERS \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/llava \
    --device cuda \
    --wandb \
    --wandb_project llava-sae-analysis

if [ $? -ne 0 ]; then
    echo "ERROR: Arabic extraction failed!"
    exit 1
fi
echo "Arabic extraction complete: $(date)"

# =============================================================================
# Stage 2: Extract English Activations
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 2/5: Extracting English Activations"
echo "=============================================="
echo "Start: $(date)"

python scripts/33_llava_extract_activations.py \
    --language english \
    --layers $LAYERS \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/llava \
    --device cuda \
    --wandb \
    --wandb_project llava-sae-analysis

if [ $? -ne 0 ]; then
    echo "ERROR: English extraction failed!"
    exit 1
fi
echo "English extraction complete: $(date)"

# =============================================================================
# Stage 3: Train SAEs (Arabic)
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 3/5: Training Arabic SAEs"
echo "=============================================="
echo "Start: $(date)"

for LAYER in 0 4 8 12 16 20 24 28 31; do
    echo "Training SAE for Arabic layer $LAYER..."
    python scripts/34_llava_train_sae.py \
        --language arabic \
        --layer $LAYER \
        --input_dir checkpoints/llava/layer_checkpoints \
        --output_dir checkpoints/llava/saes \
        --expansion_factor 8 \
        --epochs 50 \
        --device cuda \
        --wandb \
        --wandb_project llava-sae-analysis
    
    if [ $? -ne 0 ]; then
        echo "WARNING: Arabic layer $LAYER SAE training failed, continuing..."
    fi
done
echo "Arabic SAE training complete: $(date)"

# =============================================================================
# Stage 4: Train SAEs (English)
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 4/5: Training English SAEs"
echo "=============================================="
echo "Start: $(date)"

for LAYER in 0 4 8 12 16 20 24 28 31; do
    echo "Training SAE for English layer $LAYER..."
    python scripts/34_llava_train_sae.py \
        --language english \
        --layer $LAYER \
        --input_dir checkpoints/llava/layer_checkpoints \
        --output_dir checkpoints/llava/saes \
        --expansion_factor 8 \
        --epochs 50 \
        --device cuda \
        --wandb \
        --wandb_project llava-sae-analysis
    
    if [ $? -ne 0 ]; then
        echo "WARNING: English layer $LAYER SAE training failed, continuing..."
    fi
done
echo "English SAE training complete: $(date)"

# =============================================================================
# Stage 5: Cross-Lingual Analysis
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 5/5: Cross-Lingual Analysis"
echo "=============================================="
echo "Start: $(date)"

python scripts/35_llava_cross_lingual_analysis.py \
    --layers $LAYERS \
    --checkpoints_dir checkpoints/llava \
    --output_dir results/llava_analysis \
    --paligemma_results results/cross_lingual_overlap/cross_lingual_overlap_results.json \
    --qwen2vl_results results/qwen2vl_analysis/cross_lingual_results.json \
    --device cuda \
    --wandb \
    --wandb_project llava-sae-analysis

if [ $? -ne 0 ]; then
    echo "WARNING: Cross-lingual analysis failed!"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "End: $(date)"
echo ""
echo "Output locations:"
echo "  Activations: checkpoints/llava/layer_checkpoints/"
echo "  SAE models:  checkpoints/llava/saes/"
echo "  Results:     results/llava_analysis/"
echo ""
echo "Files:"
ls -la checkpoints/llava/ 2>/dev/null | head -20
echo ""
ls -la results/llava_analysis/ 2>/dev/null | head -20
