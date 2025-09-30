#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH --mem 28G
#SBATCH -t 2-00:00:00
#SBATCH --qos short
#SBATCH --gres=gpu:pascal:1
#SBATCH --job-name caption_peacock
#SBATCH --output output_%x_%j.log
#SBATCH --error  error_%x_%j.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python peacock_compatible.py