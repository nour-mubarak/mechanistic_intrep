#!/bin/bash
#SBATCH --job-name=02_parallel_extraction
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --output=logs/step2_parallel_extraction_%j.out
#SBATCH --error=logs/step2_parallel_extraction_%j.err

# Step 2: Parallel Activation Extraction
# =======================================
# Extract 1 layer per job to avoid OOM - 18 total jobs for PaLiGemma-3b

echo "=========================================="
echo "Step 2: Parallel Activation Extraction"
echo "Submitting 18 jobs (1 layer each)"
echo "Start: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project

# Clear previous job tracking
> pipeline_status/step2_extraction_jobs.txt

# Submit 1 job per layer (0-17 for PaLiGemma-3b)
for layer in {0..17}; do
    job_id=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=02_extract_L${layer}
#SBATCH --partition=res-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/step2_extract_layer${layer}_%j.out
#SBATCH --error=logs/step2_extract_layer${layer}_%j.err

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

echo "Extracting activations for layer ${layer}"
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges ${layer} \
    --batch-size 1 \
    --checkpoint-interval 100 \
    --output-dir checkpoints/full_layers_ncc

EXIT_CODE=\$?
echo "Layer ${layer} extraction completed with exit code: \$EXIT_CODE"
exit \$EXIT_CODE
EOF
)
    echo "Submitted extraction for layer $layer: Job $job_id"
    echo "$job_id" >> pipeline_status/step2_extraction_jobs.txt
    sleep 0.2
done

echo ""
echo "All 18 extraction jobs submitted"
echo "End: $(date)"
