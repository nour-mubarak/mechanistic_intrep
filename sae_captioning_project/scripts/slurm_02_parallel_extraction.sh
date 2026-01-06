#!/bin/bash
# Step 2: Parallel Activation Extraction
# =======================================
# Extracts activations from all 34 layers using 4 parallel jobs

echo "=========================================="
echo "Step 2: Parallel Activation Extraction"
echo "Submitting 4 parallel extraction jobs"
echo "=========================================="

# Submit 4 parallel jobs
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=02_extract_00_08
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/step2_extract_00_08_%j.out
#SBATCH --error=logs/step2_extract_00_08_%j.err

module purge
source venv/bin/activate

python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 0-8 \
    --batch-size 1 \
    --checkpoint-interval 50 \
    --output-dir checkpoints/full_layers_ncc
EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=02_extract_09_16
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/step2_extract_09_16_%j.out
#SBATCH --error=logs/step2_extract_09_16_%j.err

module purge
source venv/bin/activate

python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 9-16 \
    --batch-size 1 \
    --checkpoint-interval 50 \
    --output-dir checkpoints/full_layers_ncc
EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=02_extract_17_24
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/step2_extract_17_24_%j.out
#SBATCH --error=logs/step2_extract_17_24_%j.err

module purge
source venv/bin/activate

python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 17-24 \
    --batch-size 1 \
    --checkpoint-interval 50 \
    --output-dir checkpoints/full_layers_ncc
EOF

sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=02_extract_25_33
#SBATCH --partition=res-gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/step2_extract_25_33_%j.out
#SBATCH --error=logs/step2_extract_25_33_%j.err

module purge
source venv/bin/activate

python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 25-33 \
    --batch-size 1 \
    --checkpoint-interval 50 \
    --output-dir checkpoints/full_layers_ncc
EOF

echo "All extraction jobs submitted"
echo "Expected completion: 40-60 minutes"
