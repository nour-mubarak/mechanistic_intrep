#!/bin/bash
#SBATCH --job-name=llava_analysis
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/llava_analysis_%j.out
#SBATCH --error=logs/llava_analysis_%j.err

# LLaVA-1.5-7B Cross-Lingual Analysis
# Run after SAE training is complete
# Compares against PaLiGemma and Qwen2-VL results

echo "=============================================="
echo "LLaVA-1.5-7B Cross-Lingual Analysis"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

export PYTHONUNBUFFERED=1

echo "Running cross-lingual feature analysis..."
echo "- Computing CLBAS scores"
echo "- Training gender probes"
echo "- Measuring feature overlap"
echo "- Comparing with PaLiGemma and Qwen2-VL"
echo ""

# Run cross-lingual analysis
python scripts/35_llava_cross_lingual_analysis.py \
    --layers 0,4,8,12,16,20,24,28,31 \
    --checkpoints_dir checkpoints/llava \
    --output_dir results/llava_analysis \
    --paligemma_results results/cross_lingual_overlap/cross_lingual_overlap_results.json \
    --qwen2vl_results results/qwen2vl_analysis/cross_lingual_results.json \
    --device cpu \
    --wandb \
    --wandb_project llava-sae-analysis

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Analysis complete!"
    echo "Results saved to: results/llava_analysis/"
    echo ""
    echo "Output files:"
    ls -la results/llava_analysis/ 2>/dev/null || echo "  (directory may not exist yet)"
else
    echo "✗ Analysis FAILED with code $EXIT_CODE"
fi
echo "Date: $(date)"

exit $EXIT_CODE
