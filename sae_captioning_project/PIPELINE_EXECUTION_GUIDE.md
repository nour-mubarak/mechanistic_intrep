# Complete Pipeline Execution Guide - Durham NCC

**Full mechanistic interpretability analysis on the complete dataset (2000 samples, all 34 layers)**

---

## Overview

This guide documents the complete end-to-end pipeline for analyzing gender bias in Gemma-3-4B vision-language model using Sparse Autoencoders (SAEs) on Durham's NCC cluster.

### Pipeline Steps

1. **Data Preparation** - Process full 2000-sample dataset
2. **Activation Extraction** - Extract all 34 layers (parallel, 40-60 min)
3. **SAE Training** - Train SAEs for all layers (parallel, 4-8 hours)
4. **Comprehensive Analysis** - Analyze all layers (8 hours)
5. **Advanced Analyses** - 6 specialized analyses (parallel, 2-6 hours each)
6. **Final Report** - Generate comprehensive summary

**Total Pipeline Time**: ~12-16 hours (most steps run in parallel)

---

## Quick Start

### Prerequisites

1. **Access to Durham NCC**:
   ```bash
   ssh username@ncc.durham.ac.uk
   ```

2. **Project Setup**:
   ```bash
   cd /path/to/mechanistic_intrep/sae_captioning_project

   # Create virtual environment (first time only)
   module load python/3.10
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Email Notifications**:
   ```bash
   # Update email in all SLURM scripts
   sed -i 's/your.email@durham.ac.uk/YOUR_ACTUAL_EMAIL@durham.ac.uk/g' scripts/slurm_*.sh
   ```

### Execute Complete Pipeline

**Option 1: Automated Full Pipeline (Recommended)**

```bash
# Submit entire pipeline with automatic job dependencies
bash scripts/slurm_00_full_pipeline.sh
```

This will:
- Submit all jobs with proper dependencies
- Each step waits for previous step to complete
- Runs parallel jobs where possible
- Sends email notifications at key stages

**Option 2: Manual Step-by-Step Execution**

See "Manual Execution" section below.

---

## Automated Pipeline Details

### What `slurm_00_full_pipeline.sh` Does

1. **Submits jobs in dependency chain**:
   ```
   Step 1 (Data Prep)
      ↓
   Step 2 (Activation Extraction - 4 parallel jobs)
      ↓
   Step 3 (SAE Training - 34 parallel jobs)
      ↓
   Step 4 (Comprehensive Analysis)
      ↓
   Steps 5-10 (Advanced Analyses - 6 parallel jobs)
      ↓
   Final Report
   ```

2. **Creates tracking files**:
   - Job IDs saved to `pipeline_status/*.jobid`
   - Logs saved to `logs/step*.out` and `logs/step*.err`

3. **Email notifications**:
   - Sent when each major step completes or fails
   - Final notification when entire pipeline completes

### Monitor Automated Pipeline

```bash
# Check all running jobs
squeue -u $USER

# Watch job status (updates every 5 seconds)
watch -n 5 'squeue -u $USER'

# Check specific step
cat pipeline_status/step1_prepare_data.jobid
scontrol show job $(cat pipeline_status/step1_prepare_data.jobid)

# View logs for specific step
tail -f logs/step1_prepare_data_*.out

# View all recent logs
ls -lt logs/ | head -20
```

---

## Manual Execution (Step-by-Step)

If you prefer to run steps manually or need to troubleshoot:

### Step 1: Data Preparation

```bash
sbatch scripts/slurm_01_prepare_data.sh
```

**What it does**: Prepares full 2000-sample dataset with English and Arabic prompts

**Time**: ~1 hour
**Resources**: 8 CPUs, 32GB RAM (no GPU)
**Output**: `data/processed/samples.csv`, `data/processed/images/`

**Verify**:
```bash
# Check dataset
wc -l data/processed/samples.csv  # Should show 2001 (2000 + header)
ls data/processed/images/ | wc -l  # Should show 2000
```

---

### Step 2: Activation Extraction (All 34 Layers)

```bash
bash scripts/slurm_02_parallel_extraction.sh
```

**What it does**: Submits 4 parallel jobs to extract layers 0-33

**Jobs submitted**:
- Job 1: Layers 0-8
- Job 2: Layers 9-16
- Job 3: Layers 17-24
- Job 4: Layers 25-33

**Time**: ~40-60 minutes (parallel)
**Resources**: 4 GPUs, 64GB RAM each
**Output**: `checkpoints/full_layers_ncc/layer_checkpoints/layer_*_{english,arabic}.pt`

**Monitor**:
```bash
# Watch all extraction jobs
squeue -u $USER | grep extract

# Check progress of specific job
tail -f logs/step2_extract_00_08_*.out
```

**Verify**:
```bash
# Should show 68 files (34 layers × 2 languages)
ls checkpoints/full_layers_ncc/layer_checkpoints/ | wc -l

# Check specific layer
python << EOF
import torch
data = torch.load('checkpoints/full_layers_ncc/layer_checkpoints/layer_10_english.pt')
print(f"Layer: {data['layer']}")
print(f"Shape: {data['activations'].shape}")
print(f"Samples: {len(data['genders'])}")
EOF
```

**Expected output**:
```
Layer: 10
Shape: torch.Size([2000, ~278, 2560])
Samples: 2000
```

---

### Step 3: SAE Training (All 34 Layers)

**IMPORTANT**: Wait for Step 2 to complete before starting this step.

```bash
bash scripts/slurm_03_train_all_saes.sh
```

**What it does**: Submits 34 parallel jobs to train SAE for each layer

**Time**: ~4-8 hours (parallel)
**Resources**: 34 GPUs total (1 per layer), 32GB RAM each
**Output**: `checkpoints/saes/layer_*/sae_final.pt`

**Monitor**:
```bash
# Count running SAE training jobs
squeue -u $USER | grep train_sae | wc -l

# Check progress of specific layer
tail -f logs/step3_train_sae_layer10_*.out

# Check which layers completed
ls checkpoints/saes/*/sae_final.pt | wc -l
```

**Verify**:
```bash
# Should show 34 SAE models
ls checkpoints/saes/*/sae_final.pt | wc -l

# Check specific SAE
python << EOF
import torch
sae = torch.load('checkpoints/saes/layer_10/sae_final.pt')
print(f"SAE state dict keys: {list(sae.keys())}")
EOF
```

---

### Step 4: Comprehensive Analysis (All Layers)

**IMPORTANT**: Wait for Step 3 to complete.

```bash
sbatch scripts/slurm_09_comprehensive_analysis.sh
```

**What it does**: Runs comprehensive analysis across all 34 layers
- Gender-biased feature identification
- Statistical significance testing
- Cross-lingual correlation analysis
- Feature activation distributions

**Time**: ~8 hours
**Resources**: 1 GPU, 64GB RAM
**Output**: `visualizations/comprehensive_all_layers/`

**Monitor**:
```bash
tail -f logs/step4_comprehensive_analysis_*.out
```

**Verify**:
```bash
# Check output files
ls visualizations/comprehensive_all_layers/

# Should include:
# - comprehensive_analysis_results.json
# - Various .png visualizations
```

---

### Steps 5-10: Advanced Analyses (Parallel)

**IMPORTANT**: Wait for Step 4 to complete.

Submit all advanced analyses in parallel:

```bash
# Feature Interpretation (Step 5)
sbatch scripts/slurm_11_feature_interpretation.sh

# Feature Ablation (Step 6)
sbatch scripts/slurm_13_feature_ablation.sh

# Visual Pattern Analysis (Step 7)
sbatch scripts/slurm_14_visual_pattern_analysis.sh

# Feature Amplification (Step 8)
sbatch scripts/slurm_15_feature_amplification.sh

# Cross-Layer Analysis (Step 9)
sbatch scripts/slurm_16_cross_layer_analysis.sh

# Qualitative Analysis (Step 10)
sbatch scripts/slurm_17_qualitative_analysis.sh
```

**Time**: 2-6 hours each (run in parallel)
**Resources**: 1 GPU each (6 GPUs total)

**Monitor all**:
```bash
squeue -u $USER | grep step
```

**Outputs**:
- Step 5: `visualizations/feature_interpretation/`
- Step 6: `visualizations/feature_ablation/`
- Step 7: `visualizations/visual_patterns_layer_*/`
- Step 8: `visualizations/feature_amplification_layer_*/`
- Step 9: `visualizations/cross_layer_analysis_full/`
- Step 10: `visualizations/qualitative_layer_*/`

---

### Step Final: Generate Report

**IMPORTANT**: Wait for Steps 5-10 to complete.

```bash
sbatch scripts/slurm_generate_final_report.sh
```

**What it does**:
- Consolidates all analysis results
- Generates comprehensive markdown report
- Creates summary statistics
- Lists all visualizations

**Time**: ~10 minutes
**Resources**: 8 CPUs, 16GB RAM (no GPU)
**Output**: `visualizations/FINAL_REPORT/COMPLETE_ANALYSIS_REPORT_*.md`

**View Report**:
```bash
cat visualizations/FINAL_REPORT/COMPLETE_ANALYSIS_REPORT_*.md
```

---

## Pipeline Outputs Summary

### Directory Structure

```
sae_captioning_project/
├── checkpoints/
│   ├── full_layers_ncc/
│   │   ├── layer_checkpoints/
│   │   │   ├── layer_0_english.pt (~450 MB)
│   │   │   ├── layer_0_arabic.pt (~450 MB)
│   │   │   └── ... (68 files total)
│   │   ├── activations_english_all_layers.pt (~15 GB)
│   │   └── activations_arabic_all_layers.pt (~15 GB)
│   └── saes/
│       ├── layer_0/sae_final.pt
│       ├── layer_1/sae_final.pt
│       └── ... (34 SAE models)
│
├── visualizations/
│   ├── comprehensive_all_layers/
│   │   ├── comprehensive_analysis_results.json
│   │   └── *.png (multiple visualizations)
│   ├── feature_interpretation/
│   ├── feature_ablation/
│   ├── visual_patterns_layer_*/
│   ├── feature_amplification_layer_*/
│   ├── cross_layer_analysis_full/
│   ├── qualitative_layer_*/
│   └── FINAL_REPORT/
│       └── COMPLETE_ANALYSIS_REPORT_*.md
│
└── logs/
    ├── step1_prepare_data_*.{out,err}
    ├── step2_extract_*_*.{out,err}
    ├── step3_train_sae_layer*_*.{out,err}
    └── ... (all job logs)
```

### Storage Requirements

| Component | Size |
|-----------|------|
| Raw activations (layer checkpoints) | ~30 GB |
| Combined activation files | ~30 GB |
| SAE models (34 layers) | ~5 GB |
| Visualizations | ~2 GB |
| Logs | ~500 MB |
| **Total** | **~68 GB** |

**Check your quota**:
```bash
quota -s
df -h .
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Jobs Stay Pending

**Symptom**: `squeue` shows `PD` status

**Solutions**:
```bash
# Check queue position
squeue -u $USER --start

# Check available resources
sinfo -p gpu

# If queue is very long, consider running fewer parallel jobs
```

#### 2. Extraction Jobs Fail with OOM

**Symptom**: "CUDA out of memory" in logs

**Solutions**:
- Already using minimum batch_size=1
- Increase memory: Edit scripts and change `--mem=64G` to `--mem=128G`
- Extract fewer layers per job (modify layer ranges)

#### 3. SAE Training Fails

**Symptom**: Training job exits with error

**Solutions**:
```bash
# Check specific layer log
cat logs/step3_train_sae_layer10_*.err

# Common issues:
# - Missing activation file: Verify Step 2 completed
# - NaN values: Already handled in code, check logs
# - OOM: Reduce batch size in slurm_03_train_all_saes.sh
```

#### 4. Dependencies Not Working

**Symptom**: Step starts before previous step completes

**Solutions**:
- Use automated pipeline (`slurm_00_full_pipeline.sh`) which handles dependencies
- If running manually, wait for previous step to complete before submitting next
- Check job status: `squeue -u $USER`

#### 5. Missing Visualizations

**Symptom**: Analysis completes but visualizations missing

**Solutions**:
```bash
# Check analysis logs for errors
grep -i error logs/step*_*.err

# Re-run specific analysis
sbatch scripts/slurm_XX_*.sh

# Verify input files exist
ls checkpoints/saes/layer_10/
```

---

## Recovery from Failures

### Partial Pipeline Failure

If a specific step fails:

1. **Identify failed step**:
   ```bash
   squeue -u $USER  # Check for failed jobs
   sacct -S 2026-01-05 -u $USER  # Check recent jobs
   ```

2. **Check logs**:
   ```bash
   grep -i error logs/step*_JOBID.err
   ```

3. **Re-run failed step**:
   ```bash
   sbatch scripts/slurm_XX_failed_step.sh
   ```

4. **Continue from next step**:
   - Automated pipeline handles this via dependencies
   - For manual execution, submit next step after fixing

### Complete Restart

If you need to restart entire pipeline:

```bash
# Cancel all running jobs
scancel -u $USER

# Clean up partial outputs (CAREFUL!)
rm -rf checkpoints/full_layers_ncc/*
rm -rf checkpoints/saes/*
rm -rf visualizations/*

# Restart from beginning
bash scripts/slurm_00_full_pipeline.sh
```

---

## Best Practices

### Before Submitting Pipeline

1. **Test with small subset** (optional):
   ```bash
   # Edit config.yaml temporarily
   data:
     num_samples: 100  # Instead of 2000

   # Run single layer extraction test
   python scripts/18_extract_full_activations_ncc.py \
       --config configs/config.yaml \
       --layer-ranges 10 \
       --languages english
   ```

2. **Verify data integrity**:
   ```bash
   # Check samples file
   head data/processed/samples.csv
   wc -l data/processed/samples.csv

   # Check images
   ls data/processed/images/ | head
   ```

3. **Check disk quota**:
   ```bash
   quota -s
   # Ensure you have >70 GB available
   ```

4. **Set correct email**:
   ```bash
   grep -r "your.email@durham.ac.uk" scripts/slurm_*.sh
   # Should return nothing if already updated
   ```

### During Pipeline Execution

1. **Monitor regularly**:
   ```bash
   # Add to crontab for automated monitoring
   */30 * * * * squeue -u $USER >> ~/pipeline_status_log.txt
   ```

2. **Check intermediate outputs**:
   ```bash
   # After extraction completes
   ls checkpoints/full_layers_ncc/layer_checkpoints/ | wc -l

   # After SAE training
   ls checkpoints/saes/*/sae_final.pt | wc -l
   ```

3. **Log resource usage**:
   ```bash
   # After job completes
   seff <job_id>
   ```

### After Pipeline Completion

1. **Verify all outputs**:
   ```bash
   python << 'EOF'
   from pathlib import Path

   # Check critical outputs
   checks = {
       "Layer checkpoints": len(list(Path("checkpoints/full_layers_ncc/layer_checkpoints").glob("*.pt"))),
       "SAE models": len(list(Path("checkpoints/saes").glob("*/sae_final.pt"))),
       "Comprehensive analysis": Path("visualizations/comprehensive_all_layers/comprehensive_analysis_results.json").exists(),
       "Final report": len(list(Path("visualizations/FINAL_REPORT").glob("*.md")))
   }

   for name, result in checks.items():
       print(f"{name}: {result}")
   EOF
   ```

2. **Archive logs**:
   ```bash
   tar -czf pipeline_logs_$(date +%Y%m%d).tar.gz logs/
   ```

3. **Backup critical results**:
   ```bash
   # Backup to your home directory or external storage
   rsync -av visualizations/ ~/backups/mechanistic_intrep_results/
   ```

---

## Expected Timeline

| Step | Name | Time | Can Run in Parallel? |
|------|------|------|---------------------|
| 1 | Data Preparation | 1 hour | No |
| 2 | Activation Extraction | 40-60 min | 4 parallel jobs |
| 3 | SAE Training | 4-8 hours | 34 parallel jobs |
| 4 | Comprehensive Analysis | 8 hours | No |
| 5-10 | Advanced Analyses | 2-6 hours each | 6 parallel jobs |
| Final | Report Generation | 10 min | No |

**Sequential Time**: ~30-40 hours
**With Parallelization**: ~12-16 hours
**Speedup**: ~2.5-3×

---

## Resource Allocation Strategy

### Conservative (Safer, Slower)

- Submit steps sequentially
- Only run 2-4 parallel jobs at a time
- Use when GPU queue is busy

### Aggressive (Faster, Requires More Resources)

- Use automated pipeline (parallel where possible)
- Up to 34 GPUs during SAE training
- Up to 6 GPUs during advanced analyses
- Use when cluster is relatively free

### Check Cluster Load

```bash
# Check current GPU utilization
sinfo -p gpu
squeue -p gpu

# Adjust parallelization based on availability
```

---

## Post-Pipeline: Using Results

### Load Analysis Results

```python
import torch
import json
from pathlib import Path

# Load comprehensive analysis
with open('visualizations/comprehensive_all_layers/comprehensive_analysis_results.json') as f:
    results = json.load(f)

# Get top gender-biased features for layer 10
layer_10 = results['layers']['10']
male_features = layer_10['male_biased_features'][:5]
female_features = layer_10['female_biased_features'][:5]

print("Top 5 male-biased features:", [f['feature'] for f in male_features])
print("Top 5 female-biased features:", [f['feature'] for f in female_features])

# Load cross-layer analysis
with open('visualizations/cross_layer_analysis_full/cross_layer_results.json') as f:
    cross_results = json.load(f)

# Compare layers 10 vs 22
layer_10_diff = cross_results['male_biased_ablation']['10']['differential']
layer_22_diff = cross_results['male_biased_ablation']['22']['differential']

print(f"Male bias differential - Layer 10: {layer_10_diff:.2f}, Layer 22: {layer_22_diff:.2f}")
```

### Generate Custom Visualizations

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot gender bias evolution across all layers
layers = sorted([int(k) for k in results['layers'].keys()])
male_diffs = [results['layers'][str(l)]['male_biased_features'][0]['male_mean'] -
              results['layers'][str(l)]['male_biased_features'][0]['female_mean']
              for l in layers]

plt.figure(figsize=(12, 6))
plt.plot(layers, male_diffs, marker='o')
plt.xlabel('Layer')
plt.ylabel('Gender Differential (Male - Female)')
plt.title('Male Bias Evolution Across Network Depth')
plt.grid(True)
plt.savefig('custom_analysis.png', dpi=150)
```

---

## Contact and Support

### Durham NCC Support

- **Portal**: https://nccadmin.webspace.durham.ac.uk/
- **Email**: ncc-support@durham.ac.uk
- **Documentation**: Check NCC website for cluster-specific information

### Pipeline Issues

- Review logs in `logs/` directory
- Check troubleshooting section above
- Verify all prerequisites completed

---

## Summary Checklist

- [ ] SSH access to Durham NCC configured
- [ ] Virtual environment created and activated
- [ ] Email addresses updated in SLURM scripts
- [ ] Data preparation completed
- [ ] Sufficient disk quota (>70 GB)
- [ ] Pipeline submitted (automated or manual)
- [ ] Jobs monitored regularly
- [ ] All outputs verified
- [ ] Results backed up
- [ ] Final report reviewed

---

**Last Updated**: January 5, 2026
**Version**: 1.0
**Status**: Production ready
