#!/bin/bash
#SBATCH --job-name=clmb_analysis
#SBATCH --partition=res-gpu-small
#SBATCH --qos=res-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --output=logs/clmb_%j.out
#SBATCH --error=logs/clmb_%j.err

# CLMB (Cross-Lingual Multimodal Bias) Analysis
# ==============================================
# Novel methodology for bias analysis in multimodal models

echo "=========================================="
echo "CLMB Analysis Job Started"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Setup environment
cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source ~/.bashrc

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sae_env

# Set paths
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TRANSFORMERS_CACHE="/home2/jmsk62/.cache/huggingface"
export HF_HOME="/home2/jmsk62/.cache/huggingface"

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create output directories
mkdir -p logs
mkdir -p results/clmb

echo ""
echo "Running CLMB Analysis..."
echo ""

# Check if POT (optimal transport) is installed
python -c "import ot" 2>/dev/null || {
    echo "Installing POT library for optimal transport..."
    pip install POT --quiet
}

# Run the analysis
python scripts/19_clmb_analysis.py \
    --config configs/clmb_config.yaml \
    2>&1 | tee logs/clmb_${SLURM_JOB_ID}.log

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "CLMB Analysis Complete"
echo "Exit code: ${EXIT_CODE}"
echo "Date: $(date)"
echo "=========================================="

# Send notification
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: CLMB analysis completed" | mail -s "CLMB Analysis Complete" ${USER}@durham.ac.uk 2>/dev/null || true
else
    echo "FAILED: CLMB analysis failed with exit code ${EXIT_CODE}" | mail -s "CLMB Analysis Failed" ${USER}@durham.ac.uk 2>/dev/null || true
fi

exit $EXIT_CODE
