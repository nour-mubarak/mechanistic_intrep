#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --gres gpu:ampere:1
#SBATCH --job-name caption_AIN_full
#SBATCH --output output_%x_%j.log
#SBATCH --error  error_%x_%j.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python arabic_image_captioning.py --image /home2/jmsk62/project/mechanistic_intrep/dataset/images --language both
