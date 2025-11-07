<<<<<<< HEAD
#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2-00:00:00
#SBATCH --qos short
#SBATCH --gres=gpu:pascal:1
#SBATCH --job-name mech_filter
#SBATCH --output output_%j.log
#SBATCH --error error_%j.log



set -euo pipefail

# Paths
PROJECT_ROOT="/home2/jmsk62/project/mechanistic_intrep"
PYTHON="/home2/jmsk62/project/mechanistic_intrep/mech_intrep/env/bin/python"

cd "$PROJECT_ROOT"

# Make sure mechanistic and subpackages are valid Python packages
touch mechanistic/__init__.py \
      mechanistic/extract/__init__.py \
      mechanistic/utils/__init__.py \
      mechanistic/sae/__init__.py \
      mechanistic/metrics/__init__.py \
      mechanistic/causal/__init__.py

# Point Python at your project
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Optional sanity check
"$PYTHON" -c "import mechanistic.extract.activation_extractor as m; print('Activation module file:', m.__file__)"

# Run the filter step (no GPU needed for this)
srun "$PYTHON" -m mechanistic.extract.activation_extractor --filter_only
=======
#!/usr/bin/env bash
set -e
python -m mechanistic.extract.activation_extractor --filter_only
>>>>>>> 614d0f4 (Implement mechanistic interpretability pipeline for Arabic image captioning)
