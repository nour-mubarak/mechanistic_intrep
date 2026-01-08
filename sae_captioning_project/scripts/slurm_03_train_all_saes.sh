#!/bin/bash
#SBATCH --job-name=03_train_all_saes
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --output=logs/step3_train_all_saes_%j.out
#SBATCH --error=logs/step3_train_all_saes_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jmsk62@durham.ac.uk

# Step 3: Train SAEs for All Layers
# ==================================
# This is a launcher script that submits 18 parallel SAE training jobs (PaLiGemma-3b)

echo "=========================================="
echo "Step 3: Training SAEs for All Layers"
echo "Job ID: $SLURM_JOB_ID"
echo "Submitting 18 parallel training jobs"
echo "Start: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project

# Clear previous job tracking
> pipeline_status/step3_sae_jobs.txt

# Train SAEs for all layers (0-17 for PaLiGemma-3b)
for layer in {0..17}; do
    job_id=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=03_train_sae_${layer}
#SBATCH --partition=res-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/step3_train_sae_layer${layer}_%j.out
#SBATCH --error=logs/step3_train_sae_layer${layer}_%j.err

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
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
)
    echo "Submitted SAE training for layer $layer: Job $job_id"
    echo "$job_id" >> pipeline_status/step3_sae_jobs.txt
    sleep 0.2  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "All 18 SAE training jobs submitted"
echo "Expected completion: 4-8 hours (parallel execution)"
echo "Step 3 launcher completed at $(date)"
