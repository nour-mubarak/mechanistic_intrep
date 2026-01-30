#!/bin/bash
#SBATCH --job-name=llama32v_full
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/llama32vision_full_pipeline_%j.out
#SBATCH --error=logs/llama32vision_full_pipeline_%j.err

# =============================================================================
# Llama 3.2 Vision (11B) Full Pipeline
# =============================================================================
# Runs the complete Llama 3.2 Vision pipeline sequentially:
# 1. Extract activations (both languages)
# 2. Train SAEs (all layers, both languages)
# 3. Run cross-lingual analysis

echo "=============================================="
echo "Llama 3.2 Vision (11B) Full Pipeline"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

# Fixed: Use parent directory which has all required data files
cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME=~/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:-""}  # Set your HuggingFace token for Llama access
export WANDB_PROJECT="llama32vision-sae-analysis"
export WANDB_ENTITY="nourmubarak"

# Ensure wandb is logged in
echo "Checking W&B login..."
wandb login --relogin 2>/dev/null || echo "W&B login check complete"

# Layers to analyze (Llama 3.2 Vision has 40 layers)
LAYERS="0,5,10,15,20,25,30,35,39"

# Create output directories
mkdir -p checkpoints/llama32vision/layer_checkpoints
mkdir -p checkpoints/llama32vision/saes
mkdir -p results/llama32vision_analysis
mkdir -p logs

# =============================================================================
# Stage 1: Extract Arabic Activations
# =============================================================================
echo ""
echo "=============================================="
echo "Stage 1/5: Extracting Arabic Activations"
echo "=============================================="
echo "Start: $(date)"

python scripts/38_llama32vision_extract_activations.py \
    --language arabic \
    --layers $LAYERS \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/llama32vision \
    --device cuda \
    --wandb \
    --wandb_project llama32vision-sae-analysis

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

python scripts/38_llama32vision_extract_activations.py \
    --language english \
    --layers $LAYERS \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/llama32vision \
    --device cuda \
    --wandb \
    --wandb_project llama32vision-sae-analysis

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

for LAYER in 0 5 10 15 20 25 30 35 39; do
    echo "Training SAE for Arabic layer $LAYER..."
    python scripts/39_llama32vision_train_sae.py \
        --language arabic \
        --layer $LAYER \
        --input_dir checkpoints/llama32vision/layer_checkpoints \
        --output_dir checkpoints/llama32vision/saes \
        --expansion_factor 8 \
        --epochs 50 \
        --device cuda \
        --wandb \
        --wandb_project llama32vision-sae-analysis

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

for LAYER in 0 5 10 15 20 25 30 35 39; do
    echo "Training SAE for English layer $LAYER..."
    python scripts/39_llama32vision_train_sae.py \
        --language english \
        --layer $LAYER \
        --input_dir checkpoints/llama32vision/layer_checkpoints \
        --output_dir checkpoints/llama32vision/saes \
        --expansion_factor 8 \
        --epochs 50 \
        --device cuda \
        --wandb \
        --wandb_project llama32vision-sae-analysis

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

python scripts/40_llama32vision_cross_lingual_analysis.py \
    --layers $LAYERS \
    --checkpoints_dir checkpoints/llama32vision \
    --output_dir results/llama32vision_analysis \
    --device cuda \
    --wandb \
    --wandb_project llama32vision-sae-analysis

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
echo "  Activations: checkpoints/llama32vision/layer_checkpoints/"
echo "  SAE models:  checkpoints/llama32vision/saes/"
echo "  Results:     results/llama32vision_analysis/"
echo ""
echo "Files:"
ls -la checkpoints/llama32vision/ 2>/dev/null | head -20
echo ""
ls -la results/llama32vision_analysis/ 2>/dev/null | head -20
