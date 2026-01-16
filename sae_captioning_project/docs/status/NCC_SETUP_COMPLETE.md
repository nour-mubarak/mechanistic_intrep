# Durham NCC Setup Complete ✅

**Setup Date**: January 6, 2026
**User**: jmsk62
**Cluster**: ncc.clients.dur.ac.uk

---

## 🎯 CRITICAL: Head Node vs Compute Nodes

### ✅ **You are CORRECTLY on the login/head node**

**Current hostname**: `ncc.clients.dur.ac.uk`

**What you SHOULD do on the head node:**
- ✅ Submit jobs with `sbatch`
- ✅ Check job status with `squeue -u $USER`
- ✅ Edit scripts and configuration files
- ✅ Light file operations (ls, cd, cat, etc.)
- ✅ View logs and outputs

**What you MUST NOT do on the head node:**
- ❌ **NEVER** run Python training scripts directly
- ❌ **NEVER** run activation extraction directly
- ❌ **NEVER** run any computationally intensive tasks
- ❌ **NEVER** load large models into memory
- ❌ **NEVER** run ML/DL workloads

### ✅ **All compute work runs on SLURM compute nodes automatically**

When you submit the pipeline with:
```bash
bash scripts/slurm_00_full_pipeline.sh
```

**SLURM automatically**:
- Allocates compute nodes (not head node)
- Allocates GPUs as needed
- Runs all intensive tasks on compute nodes
- Sends you email notifications
- Manages all dependencies

**You never touch compute nodes directly** - SLURM handles everything!

---

## 📋 Setup Completed

### ✅ Configuration Updates
- [x] Email address updated: `jmsk62@durham.ac.uk`
- [x] Project directory configured: `/home2/jmsk62/mechanistic_intrep/.../sae_captioning_project`
- [x] Partition names updated for NCC:
  - CPU jobs: `cpu` partition
  - GPU jobs: `res-gpu-small` partition
- [x] Module loading commands removed (using system Python 3.10)

### ✅ Environment Setup
- [x] Python virtual environment created: `venv/` (Python 3.10)
- [x] Dependencies installation: **IN PROGRESS** (running in background)
- [x] Required directories created:
  - `logs/` - Job output and error logs
  - `checkpoints/` - Model checkpoints and activations
  - `visualizations/` - Analysis outputs
  - `data/processed/` - Processed datasets

### ✅ Available Resources (as of setup)
- **GPUs**: 13 idle GPUs available
- **Partitions**:
  - `res-gpu-small` (13 GPUs available)
  - `res-gpu-large` (2 GPUs available)
  - `cpu` (9 nodes available)
  - `gpu-bigmem` (large memory GPUs)

---

## 🚀 Ready to Launch Pipeline

### **STEP 1: Verify Dependencies Installed**

```bash
# Check if background installation completed
source venv/bin/activate
python -c "import torch; import transformers; print('Dependencies OK!')"
```

### **STEP 2: Submit the Full Pipeline**

```bash
# This is the ONLY command you need to run!
bash scripts/slurm_00_full_pipeline.sh
```

This command:
- Submits ALL pipeline steps to SLURM
- Sets up automatic job dependencies
- Runs everything on compute nodes (NOT head node)
- Takes 12-16 hours total (mostly automated, parallel)
- Sends email notifications to: jmsk62@durham.ac.uk

### **STEP 3: Monitor Progress** (from head node - SAFE!)

```bash
# Check all your jobs
squeue -u $USER

# Watch job status (updates every 5 seconds)
watch -n 5 'squeue -u $USER'

# View real-time logs
tail -f logs/step1_prepare_data_*.out

# Check completed jobs today
sacct -u $USER --starttime=$(date +%Y-%m-%d)
```

---

## 📊 Pipeline Overview

| Step | Description | Time | Resources |
|------|-------------|------|-----------|
| 1 | Data Preparation | 1 hour | CPU only |
| 2 | Extract Activations (34 layers) | 40-60 min | 4 GPUs (parallel) |
| 3 | Train SAEs (34 layers) | 4-8 hours | 34 GPUs (parallel) |
| 4 | Comprehensive Analysis | 8 hours | 1 GPU |
| 5-10 | Advanced Analyses | 2-6 hours | 6 GPUs (parallel) |
| Final | Generate Report | 10 min | CPU only |

**Total**: ~12-16 hours (mostly running in parallel)

---

## 🔍 Checking Job Status

### Job States
- **PD** (Pending): Waiting for resources
- **R** (Running): Currently executing on compute node
- **CG** (Completing): Finishing up
- **CD** (Completed): Successfully finished
- **F** (Failed): Job failed - check error logs

### Useful Commands

```bash
# Detailed job info
scontrol show job <job_id>

# View job efficiency after completion
seff <job_id>

# Cancel a job if needed
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Check GPU usage on running jobs
squeue -u $USER -o "%.18i %.9P %.8T %.10M %.6D %R %b"
```

---

## 📁 Expected Outputs

After pipeline completion:

```
checkpoints/
├── full_layers_ncc/
│   └── layer_checkpoints/
│       ├── layer_0_english_activations.pt
│       ├── layer_0_arabic_activations.pt
│       ├── ... (68 files total: 34 layers × 2 languages)
│       └── layer_33_arabic_activations.pt
└── saes/
    ├── layer_0/sae_final.pt
    ├── layer_1/sae_final.pt
    ├── ... (34 SAE models)
    └── layer_33/sae_final.pt

visualizations/
├── comprehensive_all_layers/     # All layer analysis
├── cross_layer_analysis_full/    # Evolution across depth
├── feature_ablation/             # Causal effects
├── visual_patterns_*/            # What triggers features
└── FINAL_REPORT/                 # Executive summary
    └── COMPLETE_ANALYSIS_REPORT_*.md

logs/
├── step1_prepare_data_*.out/err
├── step2_extract_activations_*.out/err
├── step3_train_saes_*.out/err
└── ... (all pipeline step logs)
```

**Total storage needed**: ~70 GB

---

## ⚠️ Troubleshooting

### Issue: Jobs stuck in PENDING (PD)
**Solution**: GPU resources busy, wait for availability
```bash
sinfo -p res-gpu-small  # Check partition status
```

### Issue: Job FAILED
**Solution**: Check error logs
```bash
ls -lt logs/*.err | head -1  # Find latest error log
cat logs/step*_FAILED_JOB_ID.err
```

### Issue: Out of Memory (OOM)
**Solution**: Increase memory in script
```bash
# Edit the relevant slurm_*.sh script
#SBATCH --mem=128G  # Increase from default
```

### Issue: Need to restart from specific step
**Solution**: Submit individual step scripts
```bash
# Example: Restart from SAE training
sbatch scripts/slurm_03_train_all_saes.sh
```

---

## 📞 Support

- **NCC Documentation**: https://nccadmin.webspace.durham.ac.uk/
- **NCC Support**: ncc-support@durham.ac.uk
- **SLURM Documentation**: https://slurm.schedmd.com/

---

## 🎓 Key Reminders

1. **Always submit jobs via SLURM** - never run training/extraction on head node
2. **Monitor from head node** - checking status is safe and encouraged
3. **Email notifications** - you'll get alerts at major milestones
4. **Storage limits** - pipeline needs ~70GB, check quota with `quota -s`
5. **Job dependencies** - all handled automatically, don't manually submit steps
6. **Be patient** - 12-16 hours is normal for full pipeline

---

## ✅ Next Steps

1. **Wait for dependencies to install** (~5-10 minutes)
2. **Verify installation**: `source venv/bin/activate && python -c "import torch"`
3. **Submit pipeline**: `bash scripts/slurm_00_full_pipeline.sh`
4. **Monitor progress**: `squeue -u $USER`
5. **Check emails**: Job completion notifications
6. **Review results**: `cat visualizations/FINAL_REPORT/*.md`

---

**Ready to start? Run this command:**

```bash
bash scripts/slurm_00_full_pipeline.sh
```

Then relax - the cluster does the heavy lifting! ☕

---

**Last Updated**: January 6, 2026
**Status**: READY TO LAUNCH 🚀
