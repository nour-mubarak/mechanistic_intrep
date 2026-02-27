#!/bin/bash
# ==========================================================
# Master Submission Script: All Improved Experiments
# ==========================================================
#
# Addresses ALL reviewer feedback:
#   1. 500+ images (up from 100)
#   2. Multi-model (PaLiGemma, Qwen2-VL, Llama-3.2-Vision)
#   3. Multi-layer (L9, L17, L9+L17 combined)
#   4. 25 random runs (up from 3)
#   5. Length normalization (gender terms / total tokens)
#   6. Per-image paired stats (bootstrap CI, Wilcoxon)
#   7. Non-binary term tracking
#   8. Full 40K SAE training
#
# Usage:
#   bash slurm/improved/submit_all.sh           # Submit everything
#   bash slurm/improved/submit_all.sh --dry-run  # Show what would be submitted
# ==========================================================

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

CD=/home2/jmsk62/mechanistic_intrep/sae_captioning_project
SLURM_DIR=$CD/slurm/improved

submit() {
    local script="$1"
    local desc="$2"
    if $DRY_RUN; then
        echo "[DRY] Would submit: $desc"
        echo "      Script: $script"
    else
        local jobid
        jobid=$(sbatch --parsable "$script")
        echo "[SUBMITTED] $desc → Job $jobid"
    fi
}

echo "=========================================="
echo "Submitting All Improved Experiments"
echo "=========================================="
echo ""

# ---- PHASE 1: Intervention Experiments ----
echo "--- PHASE 1: Intervention Experiments ---"
echo ""

# 1a. PaLiGemma L9 (primary experiment, 500 images, 25 random runs)
submit "$SLURM_DIR/slurm_paligemma_L9.sh" "PaLiGemma L9 (500 img, 25 random)"

# 1b. PaLiGemma L17 (layer comparison)
submit "$SLURM_DIR/slurm_paligemma_L17.sh" "PaLiGemma L17 (layer comparison)"

# 1c. PaLiGemma L9+L17 combined (multi-layer)
submit "$SLURM_DIR/slurm_paligemma_multilayer.sh" "PaLiGemma L9+L17 multi-layer"

# 1d. Qwen2-VL (cross-model replication)
submit "$SLURM_DIR/slurm_qwen2vl.sh" "Qwen2-VL L12 (cross-model)"

# 1e. Llama-3.2-Vision (cross-model replication)
submit "$SLURM_DIR/slurm_llama32vision.sh" "Llama-3.2-Vision L20 (cross-model)"

echo ""

# ---- PHASE 2: Full 40K SAE Training Pipeline ----
echo "--- PHASE 2: Full 40K SAE Training ---"
echo ""

# 2a. Extract full 40K activations (array job: english + arabic)
submit "$SLURM_DIR/slurm_extract_full_40k.sh" "Extract full 40K activations (en+ar)"

# 2b. Train SAEs on full 40K (submit separately, run after extraction completes)
submit "$SLURM_DIR/slurm_train_sae_full_40k.sh" "Train SAE full 40K (run after extraction)"

echo ""
echo "=========================================="
echo "Submission complete!"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: results/improved_intervention/"
echo "=========================================="
