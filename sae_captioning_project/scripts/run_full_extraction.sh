#!/bin/bash
# Full Layer Activation Extraction Script
# ========================================
#
# Extracts activations from all 34 layers of Gemma-3-4B on the full 2000-sample dataset
# using Neural Corpus Compilation (NCC) methodology.

set -e  # Exit on error

echo "======================================"
echo "Full Layer Activation Extraction (NCC)"
echo "======================================"
echo ""

# Configuration
CONFIG_FILE="configs/config.yaml"
BATCH_SIZE=1  # Conservative for 24GB GPU
CHECKPOINT_INTERVAL=50
OUTPUT_DIR="checkpoints/full_layers_ncc"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Option 1: Extract ALL 34 layers (0-33)
echo "Starting extraction for ALL layers (0-33)..."
python scripts/18_extract_full_activations_ncc.py \
    --config "$CONFIG_FILE" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "======================================"
echo "Extraction Complete!"
echo "======================================"
echo ""
echo "Outputs saved to: $OUTPUT_DIR"
echo "  - Layer-wise checkpoints: $OUTPUT_DIR/layer_checkpoints/"
echo "  - Combined files: $OUTPUT_DIR/activations_*_all_layers.pt"
echo "  - Metadata: $OUTPUT_DIR/extraction_metadata.json"
echo ""
echo "Next steps:"
echo "  1. Train SAEs on extracted activations"
echo "  2. Run comprehensive analysis across all layers"
echo "  3. Identify layer-specific gender bias patterns"
