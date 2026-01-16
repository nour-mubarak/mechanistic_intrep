#!/bin/bash
# ============================================================================
# Local Pipeline Runner for 24GB GPU
# ============================================================================
#
# Run pipeline stages individually on local GPU
# Usage: ./run_local.sh <stage_number>
#   stage_number: 2 (extraction), 3 (training), 4 (analysis)
#
# ============================================================================

set -e  # Exit on error

PROJECT_DIR="/home/nour/mchanistic project/mechanistic_intrep/sae_captioning_project"
cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p logs checkpoints results visualizations

# Set environment variables for memory efficiency
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=~/.cache/huggingface

# Function to check GPU memory
check_gpu_memory() {
    echo "Current GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
    echo ""
}

# Function to run a stage
run_stage() {
    local stage=$1
    local script=$2
    local name=$3

    echo "========================================"
    echo "STAGE $stage: $name"
    echo "Started: $(date)"
    echo "========================================"

    check_gpu_memory

    echo "Running: python3 scripts/$script --config configs/config.yaml"
    echo ""

    # Run with output to both console and log file
    python3 "scripts/$script" --config configs/config.yaml 2>&1 | tee "logs/${script%.py}_$(date +%Y%m%d_%H%M%S).log"

    EXIT_CODE=${PIPESTATUS[0]}

    echo ""
    echo "========================================"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Stage $stage completed successfully!"
    else
        echo "❌ Stage $stage failed with exit code $EXIT_CODE"
    fi
    echo "Finished: $(date)"
    echo "========================================"
    echo ""

    check_gpu_memory

    return $EXIT_CODE
}

# Main execution
STAGE=${1:-2}

case $STAGE in
    2)
        run_stage 2 "02_extract_activations.py" "Activation Extraction"
        ;;
    3)
        run_stage 3 "03_train_sae.py" "SAE Training"
        ;;
    4)
        echo "Running Analysis Pipeline (stages 4-6)..."
        run_stage 4 "04_analyze_features.py" "Feature Analysis" && \
        run_stage 5 "05_steering_experiments.py" "Steering Experiments" && \
        run_stage 6 "06_generate_visualizations.py" "Visualization Generation"
        ;;
    all)
        echo "Running FULL pipeline (stages 2-6)..."
        run_stage 2 "02_extract_activations.py" "Activation Extraction" && \
        run_stage 3 "03_train_sae.py" "SAE Training" && \
        run_stage 4 "04_analyze_features.py" "Feature Analysis" && \
        run_stage 5 "05_steering_experiments.py" "Steering Experiments" && \
        run_stage 6 "06_generate_visualizations.py" "Visualization Generation"
        ;;
    *)
        echo "Usage: $0 <stage>"
        echo "  Stages:"
        echo "    2   - Activation Extraction"
        echo "    3   - SAE Training"
        echo "    4   - Analysis (features + steering + visualizations)"
        echo "    all - Run all stages"
        exit 1
        ;;
esac

echo ""
echo "🎉 Pipeline completed!"
echo "View results in wandb: https://wandb.ai/nourmubarak/sae-captioning-bias"
