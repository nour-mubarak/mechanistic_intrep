#!/bin/bash
#SBATCH --job-name=sae_analysis
#SBATCH --partition=gpu-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# ==============================================================================
# SAE Feature Analysis Pipeline with W&B Tracking
# ==============================================================================

echo "=========================================="
echo "SAE Analysis Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Navigate to project directory
cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

# Create logs directory
mkdir -p logs results visualizations

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Check for W&B login
if python -c "import wandb; wandb.login()" 2>/dev/null; then
    echo "W&B: Logged in"
    WANDB_FLAG=""
else
    echo "W&B: Not logged in (will run offline)"
    WANDB_FLAG="--no-wandb"
fi
echo ""

# Run analysis pipeline
echo "Starting analysis pipeline..."
echo "=========================================="

python scripts/20_full_analysis_pipeline.py \
    --config configs/config.yaml \
    --device cuda \
    $WANDB_FLAG

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "Finished: $(date)"
    echo "=========================================="
    
    # List generated outputs
    echo ""
    echo "Generated files:"
    echo ""
    echo "Results:"
    ls -la results/ 2>/dev/null || echo "  No results directory"
    echo ""
    echo "Visualizations:"
    ls -la visualizations/ 2>/dev/null | head -20
    
    # Show report summary
    echo ""
    echo "Report preview:"
    head -50 results/ANALYSIS_REPORT.md 2>/dev/null || echo "No report generated"
else
    echo ""
    echo "=========================================="
    echo "Analysis FAILED with exit code $?"
    echo "Check logs/analysis_${SLURM_JOB_ID}.err for details"
    echo "=========================================="
    exit 1
fi
