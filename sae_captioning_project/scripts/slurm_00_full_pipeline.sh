#!/bin/bash
# Complete End-to-End Pipeline on Durham NCC
# ===========================================
#
# This master script orchestrates the entire mechanistic interpretability
# pipeline from data preparation through final analysis on the full dataset.
#
# Pipeline Steps:
#   01. Data Preparation
#   02. Activation Extraction (all 34 layers)
#   03. SAE Training (all layers)
#   04-09. Comprehensive Analysis
#   10-17. Advanced Mechanistic Analysis
#
# Usage: bash scripts/slurm_00_full_pipeline.sh

set -e

echo "=========================================="
echo "Full Mechanistic Interpretability Pipeline"
echo "=========================================="
echo "Durham NCC Cluster"
echo "Start time: $(date)"
echo ""

# Configuration
EMAIL="jmsk62@durham.ac.uk"
PROJECT_DIR="/home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project"
CONFIG_FILE="configs/config.yaml"

# Job dependency tracking
JOB_IDS=()

# Create logs directory
mkdir -p logs pipeline_status

echo "Pipeline configuration:"
echo "  Project directory: $PROJECT_DIR"
echo "  Config file: $CONFIG_FILE"
echo "  Email: $EMAIL"
echo ""

# Function to submit job and track dependency
submit_step() {
    local script=$1
    local name=$2
    local dependency=$3
    local job_id

    echo "Submitting: $name" >&2

    if [ -z "$dependency" ]; then
        # No dependency
        job_id=$(sbatch --parsable "$script")
    else
        # With dependency
        job_id=$(sbatch --parsable --dependency=afterok:$dependency "$script")
    fi

    echo "  Job ID: $job_id" >&2
    echo "$job_id" >> "pipeline_status/${name}.jobid"

    JOB_IDS+=($job_id)
    echo $job_id
}

# ==========================================
# STEP 1: Data Preparation
# ==========================================
echo ""
echo "=========================================="
echo "STEP 1: Data Preparation"
echo "=========================================="

STEP1_JOB=$(submit_step "scripts/slurm_01_prepare_data.sh" "step1_prepare_data" "")

# ==========================================
# STEP 2: Activation Extraction (All Layers)
# ==========================================
echo ""
echo "=========================================="
echo "STEP 2: Activation Extraction (All Layers)"
echo "=========================================="
echo "Extracting from all 34 layers on full dataset"

# Use parallel extraction for speed
STEP2_JOB=$(submit_step "scripts/slurm_02_parallel_extraction.sh" "step2_extract_activations" "$STEP1_JOB")

# ==========================================
# STEP 3: SAE Training (All Layers)
# ==========================================
echo ""
echo "=========================================="
echo "STEP 3: SAE Training (All Layers)"
echo "=========================================="
echo "Training SAEs for all 34 layers"

STEP3_JOB=$(submit_step "scripts/slurm_03_train_all_saes.sh" "step3_train_saes" "$STEP2_JOB")

# ==========================================
# STEP 4-9: Comprehensive Analysis
# ==========================================
echo ""
echo "=========================================="
echo "STEP 4-9: Comprehensive Analysis"
echo "=========================================="

STEP4_JOB=$(submit_step "scripts/slurm_09_comprehensive_analysis.sh" "step4_comprehensive_analysis" "$STEP3_JOB")

# ==========================================
# STEP 10-17: Advanced Mechanistic Analysis
# ==========================================
echo ""
echo "=========================================="
echo "STEP 10-17: Advanced Analysis"
echo "=========================================="

# Feature interpretation (parallel for all layers)
STEP5_JOB=$(submit_step "scripts/slurm_11_feature_interpretation.sh" "step5_feature_interpretation" "$STEP4_JOB")

# Feature ablation analysis
STEP6_JOB=$(submit_step "scripts/slurm_13_feature_ablation.sh" "step6_feature_ablation" "$STEP4_JOB")

# Visual pattern analysis
STEP7_JOB=$(submit_step "scripts/slurm_14_visual_pattern_analysis.sh" "step7_visual_patterns" "$STEP4_JOB")

# Feature amplification
STEP8_JOB=$(submit_step "scripts/slurm_15_feature_amplification.sh" "step8_feature_amplification" "$STEP4_JOB")

# Cross-layer analysis
STEP9_JOB=$(submit_step "scripts/slurm_16_cross_layer_analysis.sh" "step9_cross_layer" "$STEP4_JOB")

# Qualitative visual analysis
STEP10_JOB=$(submit_step "scripts/slurm_17_qualitative_analysis.sh" "step10_qualitative" "$STEP4_JOB")

# ==========================================
# STEP FINAL: Generate Final Report
# ==========================================
echo ""
echo "=========================================="
echo "STEP FINAL: Generate Final Report"
echo "=========================================="

# Wait for all analysis steps to complete
DEPENDENCIES="${STEP5_JOB}:${STEP6_JOB}:${STEP7_JOB}:${STEP8_JOB}:${STEP9_JOB}:${STEP10_JOB}"
FINAL_JOB=$(submit_step "scripts/slurm_generate_final_report.sh" "step_final_report" "$DEPENDENCIES")

# ==========================================
# Pipeline Submission Complete
# ==========================================
echo ""
echo "=========================================="
echo "Pipeline Submitted Successfully!"
echo "=========================================="
echo ""
echo "Submitted jobs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  Step $((i+1)): ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor pipeline:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER'"
echo ""
echo "Check job details:"
echo "  scontrol show job <job_id>"
echo ""
echo "View logs:"
echo "  tail -f logs/step*.out"
echo ""
echo "Pipeline status files:"
echo "  ls -lh pipeline_status/*.jobid"
echo ""
echo "Expected total time: 8-12 hours"
echo "(Most steps run in parallel after SAE training)"
echo ""
echo "You will receive email notifications at:"
echo "  $EMAIL"
echo ""
