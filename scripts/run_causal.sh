#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 2
#SBATCH --mem 12G
#SBATCH --gres=gpu:pascal:1
#SBATCH -t 02:00:00
#SBATCH --qos short
#SBATCH --job-name mech_patch
#SBATCH --output output_%x_%j.log
#SBATCH --error  error_%x_%j.log
set -euo pipefail

PROJECT_ROOT="/home2/jmsk62/project/mechanistic_intrep"
PYTHON="/home2/jmsk62/project/mechanistic_intrep/mech_intrep/env/bin/python"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
srun "$PYTHON" -m mechanistic.causal.patch_layer --config mechanistic/config.yaml




# srun "$PYTHON" -m mechanistic.causal.patch_sae_latents --config mechanistic/config.yaml