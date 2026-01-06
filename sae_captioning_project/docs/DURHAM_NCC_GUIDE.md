# Durham NCC Cluster Guide for Full Layer Extraction

**Durham NCC**: https://nccadmin.webspace.durham.ac.uk/
**Created**: January 5, 2026
**Task**: Extract activations from all 34 Gemma-3-4B layers on full 2000-sample dataset

---

## Overview

This guide provides step-by-step instructions for running the full layer activation extraction on Durham University's NCC (NVIDIA Cloud Computing) HPC cluster.

### What is Durham NCC?

Durham NCC is a GPU-accelerated HPC cluster that provides:
- NVIDIA A100/V100 GPUs (40GB VRAM)
- High-performance compute nodes
- SLURM job scheduling
- Shared storage for large datasets
- Module system for software dependencies

### Why Use NCC for This Task?

**Advantages**:
- **Powerful GPUs**: A100 GPUs with 40GB VRAM (vs local 24GB)
- **Parallel Processing**: Submit multiple jobs to extract layer ranges simultaneously
- **No Local Resource Usage**: Frees your local machine for other work
- **Faster Completion**: 40-60 minutes with parallel jobs vs 2+ hours locally
- **Recovery**: SLURM job management handles failures gracefully

**Task Requirements**:
- Extract 34 layers (0-33) from Gemma-3-4B
- Process 2000 samples for English and Arabic
- Total computational time: ~2 hours sequentially, ~40 minutes parallel

---

## Quick Start

### 1. Connect to NCC

```bash
ssh username@ncc.durham.ac.uk
```

Replace `username` with your Durham username.

### 2. Navigate to Project Directory

```bash
cd /path/to/mechanistic_intrep/sae_captioning_project
```

### 3. Setup Environment (First Time Only)

```bash
# Load Python module
module load python/3.10

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Update Email in SLURM Scripts

Edit the SLURM scripts and replace the email placeholder:

```bash
# Edit email address in scripts
sed -i 's/your.email@durham.ac.uk/your_actual_email@durham.ac.uk/g' scripts/slurm_*.sh
```

### 5. Submit Extraction Job

**Option A: Single Job (All Layers)**
```bash
sbatch scripts/slurm_extract_full_activations.sh
```

**Option B: Parallel Jobs (Faster - Recommended)**
```bash
bash scripts/slurm_parallel_extraction.sh
```

### 6. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <job_id>

# View live output
tail -f logs/slurm_extraction_<job_id>.out
```

### 7. Check Results

```bash
# List extracted layers
ls checkpoints/full_layers_ncc/layer_checkpoints/

# Check metadata
cat checkpoints/full_layers_ncc/extraction_metadata.json

# Verify file sizes
du -h checkpoints/full_layers_ncc/
```

---

## Detailed Instructions

### Environment Setup

#### Step 1: Initial Login

```bash
# SSH to NCC
ssh username@ncc.durham.ac.uk

# Check available resources
sinfo
```

#### Step 2: Load Required Modules

```bash
# Check available modules
module avail

# Load required modules (adjust versions as needed)
module load python/3.10
module load cuda/12.1
module load gcc/11.2.0

# Verify loaded modules
module list
```

#### Step 3: Create Virtual Environment

```bash
# Navigate to project
cd mechanistic_intrep/sae_captioning_project

# Create venv
python -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch torchvision transformers pillow pandas numpy scipy matplotlib seaborn tqdm pyyaml wandb
```

#### Step 4: Verify GPU Access (Interactive Session)

```bash
# Request interactive GPU session for testing
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash

# Inside session, check GPU
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Exit interactive session
exit
```

### Data Preparation

Ensure your data is accessible on NCC:

```bash
# Check data files
ls -lh data/processed/samples.csv

# Verify dataset size
wc -l data/processed/samples.csv
# Expected: 2001 lines (2000 samples + header)

# Check image files
ls data/processed/images/ | wc -l
# Expected: 2000 images
```

### Job Submission Options

#### Option 1: Sequential Extraction (All Layers in One Job)

**Use Case**: Simplest approach, single job handles everything

**Command**:
```bash
sbatch scripts/slurm_extract_full_activations.sh
```

**Resources**:
- 1 GPU
- 64GB RAM
- 16 CPUs
- 12 hours max time

**Expected Time**: ~2 hours

**Advantages**:
- Single job to monitor
- Guaranteed all layers in same output

**Disadvantages**:
- Slower than parallel
- If job fails, restart from beginning

#### Option 2: Parallel Extraction (Recommended)

**Use Case**: Fastest approach, multiple jobs run simultaneously

**Command**:
```bash
bash scripts/slurm_parallel_extraction.sh
```

**This submits 4 jobs**:
1. Layers 0-8 (Job 1)
2. Layers 9-16 (Job 2)
3. Layers 17-24 (Job 3)
4. Layers 25-33 (Job 4)

**Resources per job**:
- 1 GPU each (4 GPUs total)
- 64GB RAM each
- 16 CPUs each
- 6 hours max time each

**Expected Time**: ~40-60 minutes (all jobs run in parallel)

**Advantages**:
- 3-4× faster than sequential
- Each job is independent
- Failure in one job doesn't affect others

**Disadvantages**:
- Requires 4 available GPUs
- Slightly more complex monitoring

#### Option 3: Custom Layer Ranges

**Use Case**: Extract specific layers or retry failed ranges

**Edit** `scripts/slurm_extract_layer_ranges.sh`:
```bash
# Extract only early layers
LAYER_RANGES="0-10"

# Or specific important layers
LAYER_RANGES="0 10 14 18 22 26 30 33"
```

**Submit**:
```bash
sbatch scripts/slurm_extract_layer_ranges.sh
```

### Monitoring Jobs

#### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Example output:
# JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST
# 12345       gpu extract_  username  R       5:23      1 gpu01
```

**Status Codes**:
- `PD`: Pending (waiting for resources)
- `R`: Running
- `CG`: Completing
- `CD`: Completed
- `F`: Failed

#### View Job Details

```bash
# Show detailed job info
scontrol show job <job_id>

# Check job efficiency (after completion)
seff <job_id>
```

#### Monitor Job Output

```bash
# View output in real-time
tail -f logs/slurm_extraction_<job_id>.out

# View errors
tail -f logs/slurm_extraction_<job_id>.err

# Search for specific messages
grep "Layer" logs/slurm_extraction_<job_id>.out
grep "ERROR" logs/slurm_extraction_<job_id>.err
```

#### Check GPU Usage (While Job Running)

```bash
# SSH to compute node running job
ssh <nodelist>  # e.g., ssh gpu01

# Check GPU utilization
watch -n 1 nvidia-smi

# Exit node
exit
```

### Job Management

#### Cancel Job

```bash
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

#### Hold/Release Job

```bash
# Hold job (prevent from running)
scontrol hold <job_id>

# Release job
scontrol release <job_id>
```

#### Requeue Failed Job

```bash
scontrol requeue <job_id>
```

---

## Expected Outputs

### Directory Structure

After successful extraction:

```
checkpoints/full_layers_ncc/
├── extraction_metadata.json          # Configuration and stats
├── activations_english_all_layers.pt # Combined English (all 34 layers)
├── activations_arabic_all_layers.pt  # Combined Arabic (all 34 layers)
└── layer_checkpoints/                # Layer-wise storage
    ├── layer_0_english.pt   (~450 MB)
    ├── layer_0_arabic.pt    (~450 MB)
    ├── layer_1_english.pt
    ├── layer_1_arabic.pt
    ...
    ├── layer_33_english.pt
    └── layer_33_arabic.pt
```

### File Sizes

**Per Layer** (approximate):
- Single language: ~450 MB
- Both languages: ~900 MB

**Total Storage**:
- Layer checkpoints: ~30 GB (34 layers × 2 languages × 450 MB)
- Combined files: ~15 GB (2 files)
- **Total**: ~45 GB

**Verify Sufficient Space**:
```bash
# Check quota
quota -s

# Check available space in project directory
df -h .
```

### Validation

After extraction completes:

```bash
# Count layer files (should be 68: 34 layers × 2 languages)
ls checkpoints/full_layers_ncc/layer_checkpoints/ | wc -l

# Check metadata
cat checkpoints/full_layers_ncc/extraction_metadata.json | jq .

# Load and verify a layer in Python
python << EOF
import torch
data = torch.load('checkpoints/full_layers_ncc/layer_checkpoints/layer_10_english.pt')
print(f"Layer: {data['layer']}")
print(f"Shape: {data['activations'].shape}")
print(f"Samples: {len(data['genders'])}")
print(f"Hidden size: {data['hidden_size']}")
EOF
```

**Expected Output**:
```
Layer: 10
Shape: torch.Size([2000, 278, 2560])  # [samples, avg_seq_len, hidden_dim]
Samples: 2000
Hidden size: 2560
```

---

## Troubleshooting

### Common Issues

#### 1. Job Stays in PD (Pending) State

**Cause**: Waiting for available GPU

**Solutions**:
```bash
# Check queue position
squeue -u $USER --start

# Check partition availability
sinfo -p gpu

# Try different partition
#SBATCH --partition=gpu-short  # For shorter jobs
```

#### 2. Out of Memory (OOM) Error

**Symptoms**: Job fails with CUDA OOM in logs

**Solutions**:
```bash
# In SLURM script, reduce batch size
BATCH_SIZE=1  # Already at minimum

# Increase memory allocation
#SBATCH --mem=128G  # Instead of 64G

# Extract fewer layers at a time
LAYER_RANGES="0-8"  # Instead of 0-16
```

#### 3. Module Not Found Error

**Symptoms**: `ModuleNotFoundError: No module named 'torch'`

**Solutions**:
```bash
# Ensure venv activation in SLURM script
source venv/bin/activate

# Or use full path to Python
/path/to/venv/bin/python scripts/18_extract_full_activations_ncc.py
```

#### 4. Quota Exceeded

**Symptoms**: "Disk quota exceeded"

**Solutions**:
```bash
# Check quota
quota -s

# Clean up old checkpoints
rm -rf checkpoints/old_experiment/

# Use different output directory (if available)
--output-dir /scratch/$USER/activations/
```

#### 5. Job Timeout

**Symptoms**: Job killed after reaching time limit

**Solutions**:
```bash
# Increase time limit in SLURM script
#SBATCH --time=24:00:00  # 24 hours instead of 12

# Or reduce checkpoint interval for more frequent saves
CHECKPOINT_INTERVAL=25  # Save more frequently
```

#### 6. NaN Values in Activations

**Symptoms**: Logs show "NaN detected in layer X"

**Cause**: Numerical instability (rare with float32)

**Solutions**:
- Script automatically replaces NaN with zeros
- Check if specific layer consistently fails
- Verify model loaded correctly

### Recovery from Failed Jobs

If a job fails partway through:

```bash
# Check what was completed
ls checkpoints/full_layers_ncc/layer_checkpoints/

# Identify missing layers
python << EOF
import os
checkpoint_dir = 'checkpoints/full_layers_ncc/layer_checkpoints'
existing = {int(f.split('_')[1]) for f in os.listdir(checkpoint_dir) if f.startswith('layer_')}
all_layers = set(range(34))
missing = sorted(all_layers - existing)
print(f"Missing layers: {missing}")
EOF

# Re-run only missing layers
sbatch scripts/slurm_extract_layer_ranges.sh
# (Edit LAYER_RANGES in script first)
```

---

## Performance Optimization

### GPU Selection

NCC may have multiple GPU types. Request specific GPU if needed:

```bash
# Request A100 (40GB, faster)
#SBATCH --gres=gpu:a100:1

# Request V100 (32GB)
#SBATCH --gres=gpu:v100:1

# Any available GPU (default)
#SBATCH --gres=gpu:1
```

### Batch Size Tuning

With 40GB A100:

```bash
# Try larger batch size for faster extraction
BATCH_SIZE=2  # or even 4

# Monitor GPU memory during job
nvidia-smi
```

### Checkpoint Interval

```bash
# Less frequent saves = faster (but more data loss on failure)
CHECKPOINT_INTERVAL=100

# More frequent saves = slower but safer
CHECKPOINT_INTERVAL=25
```

### Parallel Job Optimization

For maximum speed with 8 GPUs available:

```bash
# Split into 8 jobs (4-5 layers each)
submit_job "0-4" "layers_00_04"
submit_job "5-8" "layers_05_08"
submit_job "9-12" "layers_09_12"
submit_job "13-16" "layers_13_16"
submit_job "17-20" "layers_17_20"
submit_job "21-24" "layers_21_24"
submit_job "25-28" "layers_25_28"
submit_job "29-33" "layers_29_33"
```

**Expected Time**: 20-30 minutes

---

## Best Practices

### 1. Test Before Full Run

Run a small test first:

```bash
# Interactive session
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash

# Inside session
source venv/bin/activate
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 10 \
    --languages english \
    --output-dir checkpoints/test/

# Verify output
ls checkpoints/test/layer_checkpoints/
```

### 2. Use Wandb for Monitoring

Enable Weights & Biases logging:

```yaml
# configs/config.yaml
logging:
  use_wandb: true
  wandb_project: "sae-captioning-bias"
  wandb_entity: "nourmubarak"
```

Monitor jobs remotely at wandb.ai while they run on NCC.

### 3. Organize Multiple Experiments

```bash
# Create experiment-specific directories
OUTPUT_DIR="checkpoints/exp_$(date +%Y%m%d)_full_layers"

# Or use descriptive names
OUTPUT_DIR="checkpoints/full_layers_gemma3_4b_2000samples"
```

### 4. Save SLURM Job IDs

```bash
# Submit and save job ID
JOB_ID=$(sbatch scripts/slurm_extract_full_activations.sh | awk '{print $4}')
echo $JOB_ID > logs/current_extraction_job.txt
echo "Submitted job: $JOB_ID"

# Later, check this job
cat logs/current_extraction_job.txt
squeue -j $(cat logs/current_extraction_job.txt)
```

### 5. Archive Logs

```bash
# After successful completion, archive logs
tar -czf logs/extraction_$(date +%Y%m%d).tar.gz logs/slurm_*.out logs/slurm_*.err

# Clean up individual log files
rm logs/slurm_*.out logs/slurm_*.err
```

---

## After Extraction: Next Steps

### 1. Download Results (If Working Locally)

```bash
# From local machine, download extracted activations
rsync -avz --progress \
    username@ncc.durham.ac.uk:/path/to/checkpoints/full_layers_ncc/ \
    ./checkpoints/full_layers_ncc/
```

### 2. Train SAEs on All Layers

```bash
# On NCC, submit SAE training jobs for each layer
for layer in {0..33}; do
    sbatch scripts/slurm_train_sae.sh --layer $layer
done
```

### 3. Run Comprehensive Analysis

```bash
# Analyze all 34 layers
sbatch scripts/slurm_comprehensive_analysis.sh --layers 0 1 2 3 ... 33
```

### 4. Fine-grained Cross-Layer Analysis

```bash
# Compare all adjacent layers
sbatch scripts/slurm_cross_layer_analysis.sh --layers $(seq 0 33)
```

---

## Cost and Resource Considerations

### Resource Usage

**Per Sequential Job** (All 34 layers):
- GPU hours: ~2 hours × 1 GPU = 2 GPU-hours
- CPU hours: ~2 hours × 16 CPUs = 32 CPU-hours
- Storage: ~45 GB

**Per Parallel Job** (4 jobs):
- GPU hours: ~0.5 hours × 4 GPUs = 2 GPU-hours (same total, but faster wall time)
- CPU hours: ~0.5 hours × 16 CPUs × 4 jobs = 32 CPU-hours
- Storage: ~45 GB

### Allocation Management

```bash
# Check your allocation usage
sreport cluster AccountUtilizationByUser Start=2026-01-01 End=2026-12-31 Users=$USER

# Check remaining allocation
sshare -u $USER
```

---

## Contact and Support

### NCC Support

- **Documentation**: https://nccadmin.webspace.durham.ac.uk/
- **Email**: ncc-support@durham.ac.uk
- **User Guide**: Check NCC website for latest documentation

### Project-Specific Issues

For issues with the extraction scripts:
- Check logs in `logs/`
- Review troubleshooting section above
- Verify data and environment setup

---

## Summary

### Recommended Workflow

1. **Setup** (First time only):
   ```bash
   ssh username@ncc.durham.ac.uk
   cd mechanistic_intrep/sae_captioning_project
   module load python/3.10 cuda/12.1
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Test** (Verify everything works):
   ```bash
   srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash
   # Run small test extraction for layer 10
   ```

3. **Submit** (Run full extraction):
   ```bash
   # Update email in scripts first!
   bash scripts/slurm_parallel_extraction.sh
   ```

4. **Monitor**:
   ```bash
   squeue -u $USER
   tail -f logs/extract_layers_*.out
   ```

5. **Verify**:
   ```bash
   ls checkpoints/full_layers_ncc/layer_checkpoints/ | wc -l
   # Should show 68 files (34 layers × 2 languages)
   ```

6. **Proceed** to SAE training and analysis

---

**Last Updated**: January 5, 2026
**Version**: 1.0
**Status**: Ready for deployment on Durham NCC
