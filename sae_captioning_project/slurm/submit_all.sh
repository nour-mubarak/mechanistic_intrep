#!/bin/bash
# ============================================================================
# Master SLURM Submission Script
# ============================================================================
# 
# This script submits all pipeline jobs with proper dependencies.
# Jobs will run sequentially: extraction -> training -> analysis
#
# Usage:
#   ./slurm/submit_all.sh
#
# Or from project root:
#   bash slurm/submit_all.sh
#
# ============================================================================

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "SAE Cross-Lingual Analysis Pipeline"
echo "Project Root: $PROJECT_ROOT"
echo "========================================"

# Create necessary directories
mkdir -p logs checkpoints results visualizations

# Check if data is prepared
if [ ! -f "data/processed/samples.csv" ]; then
    echo ""
    echo "WARNING: Processed data not found!"
    echo "Please run data preparation first:"
    echo "  python scripts/01_prepare_data.py --config configs/config.yaml"
    echo ""
    echo "Or create sample data for testing:"
    echo "  python scripts/01_prepare_data.py --config configs/config.yaml --create-sample"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Submit jobs with dependencies
echo ""
echo "Submitting jobs..."

# Job 1: Activation Extraction
JOB1=$(sbatch --parsable slurm/submit_extraction.sh)
echo "Submitted extraction job: $JOB1"

# Job 2: SAE Training (depends on Job 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/submit_training.sh)
echo "Submitted training job: $JOB2 (depends on $JOB1)"

# Job 3: Analysis (depends on Job 2)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/submit_analysis.sh)
echo "Submitted analysis job: $JOB3 (depends on $JOB2)"

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "========================================"
echo ""
echo "Job chain: $JOB1 -> $JOB2 -> $JOB3"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/extraction_$JOB1.out"
echo ""
echo "Cancel all jobs with:"
echo "  scancel $JOB1 $JOB2 $JOB3"
echo ""

# Save job IDs for reference
echo "$JOB1 $JOB2 $JOB3" > logs/latest_jobs.txt
echo "Job IDs saved to logs/latest_jobs.txt"
