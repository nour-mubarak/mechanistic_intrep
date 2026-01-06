#!/bin/bash
# Step 3: Train SAEs for All Layers
# ==================================
# Trains Sparse Autoencoders for all 34 layers in parallel

echo "=========================================="
echo "Step 3: Training SAEs for All Layers"
echo "Submitting 34 parallel training jobs"
echo "=========================================="

# Train SAEs for all layers (0-33)
for layer in {0..33}; do
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=03_train_sae_${layer}
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step3_train_sae_layer${layer}_%j.out
#SBATCH --error=logs/step3_train_sae_layer${layer}_%j.err

module purge
source venv/bin/activate

echo "Training SAE for layer ${layer}"

python scripts/03_train_sae.py \
    --config configs/config.yaml \
    --layer ${layer} \
    --activations checkpoints/full_layers_ncc/layer_checkpoints/layer_${layer}_english.pt \
    --epochs 50 \
    --batch-size 256 \
    --output checkpoints/saes/layer_${layer}/

EXIT_CODE=\$?
echo "Layer ${layer} SAE training completed with exit code: \$EXIT_CODE"
exit \$EXIT_CODE
EOF

    echo "Submitted SAE training for layer $layer"
    sleep 0.5  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "All 34 SAE training jobs submitted"
echo "Expected completion: 4-8 hours (parallel execution)"
