#!/bin/bash
#SBATCH --job-name=extract_test
#SBATCH --partition=res-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/extract_test_%j.out
#SBATCH --error=logs/extract_test_%j.err

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

echo "Testing Python..."
python --version
echo "Testing extraction for layer 0..."
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 0 \
    --batch-size 1 \
    --checkpoint-interval 50 \
    --output-dir checkpoints/test_layer0
echo "Done with exit code: $?"
