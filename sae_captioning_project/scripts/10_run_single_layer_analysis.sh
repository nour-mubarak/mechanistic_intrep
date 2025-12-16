#!/bin/bash
# Run comprehensive analysis one layer at a time to avoid OOM

cd "/home/nour/mchanistic project/mechanistic_intrep/sae_captioning_project"

LAYERS=(10 14 18 22)

for LAYER in "${LAYERS[@]}"; do
    echo "========================================"
    echo "Analyzing Layer $LAYER"
    echo "========================================"

    python3 scripts/09_comprehensive_analysis.py \
        --config configs/config.yaml \
        --layers $LAYER \
        --use-wandb \
        2>&1 | tee "logs/analysis_layer_${LAYER}_optimized.log"

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Layer $LAYER failed with exit code $EXIT_CODE"
    else
        echo "SUCCESS: Layer $LAYER completed"
    fi

    # Clear memory between layers
    sleep 5

    echo ""
done

echo "========================================"
echo "All layers completed!"
echo "========================================"
