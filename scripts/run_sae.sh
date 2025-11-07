<<<<<<< HEAD
#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH --gres=gpu:pascal:1
#SBATCH -t 08:00:00
#SBATCH --qos short
#SBATCH --job-name mech_sae
#SBATCH --output output_%x_%j.log
#SBATCH --error  error_%x_%j.log
set -euo pipefail

PROJECT_ROOT="/home2/jmsk62/project/mechanistic_intrep"
PYTHON="/home2/jmsk62/project/mechanistic_intrep/mech_intrep/env/bin/python"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
srun "$PYTHON" -m mechanistic.sae.train_sae --config mechanistic/config.yaml
srun "$PYTHON" -m mechanistic.sae.analyze_sae --config mechanistic/config.yaml
=======
#!/usr/bin/env bash
set -e
python -m mechanistic.sae.train_sae --config mechanistic/config.yaml
python -m mechanistic.sae.analyze_sae --config mechanistic/config.yaml
>>>>>>> 614d0f4 (Implement mechanistic interpretability pipeline for Arabic image captioning)
