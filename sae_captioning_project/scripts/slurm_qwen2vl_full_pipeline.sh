#!/bin/bash
#SBATCH --job-name=qwen2vl_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/qwen2vl_full_%j.out
#SBATCH --error=logs/qwen2vl_full_%j.err

# ============================================
# Qwen2-VL-7B Full Pipeline
# ============================================
# This script runs the complete pipeline:
# 1. Extract Arabic activations
# 2. Extract English activations  
# 3. Train SAEs (submitted as separate jobs)
# 4. Run cross-lingual analysis

echo "=============================================="
echo "Qwen2-VL-7B Full Pipeline"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

export PYTHONUNBUFFERED=1

# Create directories
mkdir -p checkpoints/qwen2vl/layer_checkpoints
mkdir -p checkpoints/qwen2vl/saes
mkdir -p results/qwen2vl_analysis

# Install Qwen dependencies
echo "Installing Qwen dependencies..."
pip install qwen-vl-utils transformers>=4.40.0 --quiet 2>/dev/null || true

# ============================================
# STEP 1: Extract Arabic Activations
# ============================================
echo ""
echo "=============================================="
echo "STEP 1: Extracting Arabic Activations"
echo "=============================================="

python scripts/28_extract_qwen2vl_activations.py \
    --language arabic \
    --layers 0,4,8,12,16,20,24,27 \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/qwen2vl \
    --device cuda

# ============================================
# STEP 2: Extract English Activations
# ============================================
echo ""
echo "=============================================="
echo "STEP 2: Extracting English Activations"
echo "=============================================="

python scripts/28_extract_qwen2vl_activations.py \
    --language english \
    --layers 0,4,8,12,16,20,24,27 \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/qwen2vl \
    --device cuda

# ============================================
# STEP 3: Train SAEs (on CPU to free GPU)
# ============================================
echo ""
echo "=============================================="
echo "STEP 3: Training SAEs"
echo "=============================================="

# Train SAEs for each layer and language
for LAYER in 0 4 8 12 16 20 24 27; do
    echo ""
    echo "--- Training Arabic Layer $LAYER ---"
    python scripts/29_train_qwen2vl_sae.py \
        --language arabic \
        --layer $LAYER \
        --epochs 50 \
        --device cpu &
    
    echo "--- Training English Layer $LAYER ---"
    python scripts/29_train_qwen2vl_sae.py \
        --language english \
        --layer $LAYER \
        --epochs 50 \
        --device cpu &
    
    # Wait for both to finish before next layer
    wait
done

# ============================================
# STEP 4: Cross-Lingual Analysis
# ============================================
echo ""
echo "=============================================="
echo "STEP 4: Cross-Lingual Analysis"
echo "=============================================="

python scripts/30_qwen2vl_cross_lingual_analysis.py \
    --layers 0,4,8,12,16,20,24,27 \
    --checkpoints_dir checkpoints/qwen2vl \
    --output_dir results/qwen2vl_analysis \
    --paligemma_results results/cross_lingual_overlap/cross_lingual_overlap_results.json \
    --device cpu

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
echo "Date: $(date)"
echo ""
echo "Results saved to:"
echo "  - checkpoints/qwen2vl/layer_checkpoints/  (activations)"
echo "  - checkpoints/qwen2vl/saes/               (SAE models)"
echo "  - results/qwen2vl_analysis/               (analysis results)"
echo ""
echo "To view results:"
echo "  cat results/qwen2vl_analysis/qwen2vl_cross_lingual_results.json | python -m json.tool"
