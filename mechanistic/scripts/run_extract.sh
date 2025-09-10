#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2-00:00:00
#SBATCH --qos short
#SBATCH --gres=gpu:pascal:1
#SBATCH --job-name mech_extract
#SBATCH --output output_%x_%j.log
#SBATCH --error  error_%x_%j.log
set -euo pipefail

PROJECT_ROOT="/home2/jmsk62/project/mechanistic_intrep"
PYTHON="/home2/jmsk62/project/mechanistic_intrep/mech_intrep/env/bin/python"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.hf_cache"   # optional local cache
# If you added the T5 decoder helper, you can now run full extract:
srun "$PYTHON" -m mechanistic.extract.activation_extractor
