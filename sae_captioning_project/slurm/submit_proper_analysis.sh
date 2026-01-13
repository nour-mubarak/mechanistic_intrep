#!/bin/bash
#SBATCH --job-name=proper_xl
#SBATCH --partition=gpu-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/proper_cross_lingual_%j.out
#SBATCH --error=logs/proper_cross_lingual_%j.err

echo "=========================================="
echo "Proper Cross-Lingual Analysis"
echo "English SAE on English Data vs Arabic SAE on Arabic Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

cd /home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

python scripts/23_proper_cross_lingual_analysis.py

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "Finished: $(date)"
echo "=========================================="
