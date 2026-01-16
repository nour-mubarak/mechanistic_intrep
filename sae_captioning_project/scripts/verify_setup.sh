#!/bin/bash
# Durham NCC Setup Verification Script
# =====================================
# Run this script to verify everything is ready before launching the pipeline

set -e

echo "=========================================="
echo "Durham NCC Setup Verification"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check function
check_item() {
    local item=$1
    local status=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $item"
    else
        echo -e "${RED}✗${NC} $item"
    fi
}

# 1. Check hostname (should be on NCC)
echo "1. Checking cluster connection..."
if hostname | grep -q "ncc"; then
    check_item "Connected to Durham NCC cluster" "OK"
else
    check_item "NOT on Durham NCC cluster!" "FAIL"
    exit 1
fi
echo ""

# 2. Check SLURM availability
echo "2. Checking SLURM..."
if command -v sbatch &> /dev/null; then
    check_item "SLURM sbatch command available" "OK"
else
    check_item "SLURM not available!" "FAIL"
    exit 1
fi
echo ""

# 3. Check Python virtual environment
echo "3. Checking Python environment..."
if [ -d "venv" ]; then
    check_item "Virtual environment exists" "OK"

    # Activate and check Python version
    source venv/bin/activate
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "   Python version: ${PYTHON_VERSION}"

    # Check key packages
    echo "   Checking key packages..."
    if python -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        check_item "   PyTorch ${TORCH_VERSION}" "OK"
    else
        check_item "   PyTorch NOT installed" "FAIL"
    fi

    if python -c "import transformers" 2>/dev/null; then
        check_item "   Transformers installed" "OK"
    else
        check_item "   Transformers NOT installed" "FAIL"
    fi

    if python -c "import wandb" 2>/dev/null; then
        check_item "   Weights & Biases installed" "OK"
    else
        check_item "   Weights & Biases NOT installed" "FAIL"
    fi

else
    check_item "Virtual environment NOT found!" "FAIL"
    exit 1
fi
echo ""

# 4. Check required directories
echo "4. Checking required directories..."
for dir in logs checkpoints visualizations data; do
    if [ -d "$dir" ]; then
        check_item "Directory: $dir" "OK"
    else
        check_item "Directory: $dir MISSING" "FAIL"
    fi
done
echo ""

# 5. Check SLURM scripts
echo "5. Checking SLURM scripts..."
if [ -f "scripts/slurm_00_full_pipeline.sh" ]; then
    check_item "Master pipeline script exists" "OK"
else
    check_item "Master pipeline script MISSING!" "FAIL"
    exit 1
fi

# Count SLURM scripts
SLURM_SCRIPT_COUNT=$(ls scripts/slurm_*.sh 2>/dev/null | wc -l)
echo -e "   Found ${SLURM_SCRIPT_COUNT} SLURM scripts"
echo ""

# 6. Check email configuration
echo "6. Checking email configuration..."
if grep -q "jmsk62@durham.ac.uk" scripts/slurm_00_full_pipeline.sh; then
    check_item "Email configured in master script" "OK"
else
    echo -e "${YELLOW}⚠${NC} Email not configured properly"
fi
echo ""

# 7. Check available GPU resources
echo "7. Checking available resources..."
echo "   Available GPUs:"
sinfo -p res-gpu-small -o "   %10P %6D %6t %15N %8G" | head -5
echo ""

# 8. Check current job queue
echo "8. Checking job queue..."
JOBS=$(squeue -u $USER | wc -l)
if [ "$JOBS" -eq 1 ]; then
    check_item "No jobs currently running (ready to submit)" "OK"
else
    echo -e "${YELLOW}⚠${NC} You have $((JOBS-1)) job(s) in the queue"
    squeue -u $USER
fi
echo ""

# 9. Check disk space
echo "9. Checking disk space..."
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -gt 100 ]; then
    check_item "Sufficient disk space (~${AVAILABLE_GB}GB available)" "OK"
else
    echo -e "${YELLOW}⚠${NC} Low disk space: ${AVAILABLE_GB}GB available (need ~70GB for pipeline)"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Verification Complete!"
echo "=========================================="
echo ""
echo "If all checks passed, you're ready to launch:"
echo ""
echo -e "${GREEN}bash scripts/slurm_00_full_pipeline.sh${NC}"
echo ""
echo "This will submit the complete pipeline to SLURM."
echo "Expected runtime: 12-16 hours"
echo ""
echo "To monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER'"
echo ""
echo "=========================================="
