#!/bin/bash
#SBATCH --job-name=llava_extract
#SBATCH --partition=gpu-bigmem
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/llava_extract_%A_%a.out
#SBATCH --error=logs/llava_extract_%A_%a.err
#SBATCH --array=0-1

# LLaVA-1.5-7B Activation Extraction
# Array job: 0=arabic, 1=english
# 
# LLaVA-1.5-7B uses SentencePiece with byte-fallback, allowing it to process
# Arabic via UTF-8 byte tokens even without native Arabic training.

echo "=============================================="
echo "LLaVA-1.5-7B Activation Extraction"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cd /home2/jmsk62/mechanistic_intrep/sae_captioning_project
source venv/bin/activate

# Set language based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    LANGUAGE="arabic"
else
    LANGUAGE="english"
fi

echo "Language: $LANGUAGE"
echo "Model: llava-hf/llava-1.5-7b-hf"
echo "Layers: 0,4,8,12,16,20,24,28,31 (9 layers across 32-layer model)"
echo "Python: $(which python)"
echo ""

# Set Python to use unbuffered output
export PYTHONUNBUFFERED=1

# Set HuggingFace cache
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

# Run extraction
python scripts/33_llava_extract_activations.py \
    --language $LANGUAGE \
    --layers 0,4,8,12,16,20,24,28,31 \
    --data_file data/processed/samples.csv \
    --images_dir data/raw/images \
    --output_dir checkpoints/llava \
    --device cuda \
    --wandb \
    --wandb_project llava-sae-analysis

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ $LANGUAGE extraction complete!"
else
    echo "✗ $LANGUAGE extraction FAILED with code $EXIT_CODE"
fi
echo "Date: $(date)"

exit $EXIT_CODE
