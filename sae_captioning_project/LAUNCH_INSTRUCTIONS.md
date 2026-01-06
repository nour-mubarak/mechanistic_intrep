# 🚀 Pipeline Launch Instructions

**Date**: January 6, 2026
**Cluster**: Durham NCC
**Status**: READY TO LAUNCH ✅

---

## ⚡ Quick Launch (3 Steps)

### Step 1: Verify Setup (Optional but Recommended)

```bash
./verify_setup.sh
```

This will check that everything is configured correctly.

### Step 2: Activate Python Environment

```bash
source venv/bin/activate
```

### Step 3: Launch Pipeline

```bash
bash scripts/slurm_00_full_pipeline.sh
```

**That's it!** The pipeline will now run automatically for 12-16 hours.

---

## 🔍 What Happens Next

After you run the pipeline script, SLURM will:

1. **Submit all jobs** to the cluster queue
2. **Allocate compute nodes** (NOT the head node you're on now)
3. **Execute each step** in the correct order with dependencies
4. **Send email notifications** to jmsk62@durham.ac.uk at key milestones
5. **Generate results** in the `visualizations/` directory

### Expected Output

```
==========================================
Full Mechanistic Interpretability Pipeline
==========================================
Durham NCC Cluster
Start time: Mon Jan  6 09:50:00 GMT 2026

Pipeline configuration:
  Project directory: /home2/jmsk62/.../sae_captioning_project
  Config file: configs/config.yaml
  Email: jmsk62@durham.ac.uk

==========================================
STEP 1: Data Preparation
==========================================
Submitting: step1_prepare_data
  Job ID: 123456

==========================================
STEP 2: Activation Extraction (All Layers)
==========================================
Extracting from all 34 layers on full dataset
Submitting: step2_extract_activations
  Job ID: 123457

[... more steps ...]

==========================================
Pipeline Submitted Successfully!
==========================================

Submitted jobs:
  Step 1: 123456
  Step 2: 123457
  Step 3: 123458
  [... etc ...]

Monitor pipeline:
  squeue -u $USER
  watch -n 5 'squeue -u $USER'

Expected total time: 8-12 hours
```

---

## 📊 Monitoring Your Jobs

### View All Your Jobs

```bash
squeue -u $USER
```

**Output Example:**
```
JOBID  PARTITION     NAME            USER   ST  TIME  NODES NODELIST(REASON)
123456 cpu           01_prepare_data jmsk62 R   0:05  1     cpu1
123457 res-gpu-small 02_extract_act  jmsk62 PD  0:00  4     (Dependency)
123458 res-gpu-small 03_train_saes   jmsk62 PD  0:00  34    (Dependency)
```

**Job States:**
- **PD** = Pending (waiting for dependencies or resources)
- **R** = Running (actively executing on compute node)
- **CG** = Completing (finishing up)
- **CD** = Completed (successfully finished)

### Watch Jobs in Real-Time

```bash
watch -n 5 'squeue -u $USER'
```

Press `Ctrl+C` to exit the watch command.

### View Job Details

```bash
scontrol show job <job_id>
```

### Check Logs While Job is Running

```bash
# View latest output log
tail -f logs/step1_prepare_data_*.out

# View latest error log
tail -f logs/step1_prepare_data_*.err
```

---

## 📧 Email Notifications

You'll receive emails at **jmsk62@durham.ac.uk** when:
- Jobs start running
- Jobs complete successfully
- Jobs fail (with error information)

---

## ⏱️ Pipeline Timeline

| Step | What It Does | Time | GPUs | Status |
|------|--------------|------|------|--------|
| 1 | Prepare dataset | ~1 hour | 0 | Waits for nothing |
| 2 | Extract activations (34 layers) | 40-60 min | 4 | Waits for Step 1 |
| 3 | Train SAEs (34 models) | 4-8 hours | 34 | Waits for Step 2 |
| 4 | Comprehensive analysis | ~8 hours | 1 | Waits for Step 3 |
| 5-10 | Advanced analyses (6 types) | 2-6 hours | 6 | Waits for Step 4 (run in parallel) |
| Final | Generate consolidated report | ~10 min | 0 | Waits for Steps 5-10 |

**Total Time**: 12-16 hours

All dependencies are automatic - no manual intervention needed!

---

## 🎯 Expected Outputs

After completion, you'll have:

### Activation Files
```
checkpoints/full_layers_ncc/layer_checkpoints/
├── layer_0_english_activations.pt
├── layer_0_arabic_activations.pt
├── layer_1_english_activations.pt
├── layer_1_arabic_activations.pt
├── ...
└── layer_33_arabic_activations.pt
```
**Total**: 68 files (~30 GB)

### Trained SAE Models
```
checkpoints/saes/
├── layer_0/sae_final.pt
├── layer_1/sae_final.pt
├── ...
└── layer_33/sae_final.pt
```
**Total**: 34 models (~5 GB)

### Analysis Results
```
visualizations/
├── comprehensive_all_layers/      # Feature analysis for all layers
├── cross_layer_analysis_full/     # How features evolve across depth
├── feature_ablation/              # Causal impact analysis
├── feature_amplification/         # Feature amplification experiments
├── visual_patterns_*/             # What triggers each feature
├── qualitative_visual/            # Qualitative pattern analysis
└── FINAL_REPORT/                  # Executive summary
    └── COMPLETE_ANALYSIS_REPORT_YYYYMMDD_HHMMSS.md
```

---

## 🛠️ Troubleshooting

### Jobs Stuck in Pending (PD) State

**Reason**: Either waiting for dependency or GPU resources
**Solution**: Check reason with:
```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R"
```

If reason is `(Dependency)` - this is normal, wait for previous job
If reason is `(Resources)` - cluster is busy, job will start when GPUs free

### Job Failed (F State)

**Solution**: Check error log
```bash
# Find the failed job's log
ls -lt logs/*.err | head -5

# View the error
cat logs/step*_JOBID.err
```

Common fixes:
- **Out of Memory**: Increase `--mem=` in the SLURM script
- **Missing file**: Check previous step completed successfully
- **Python error**: Activate venv and test the Python script manually

### Need to Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel ALL your jobs
scancel -u $USER
```

### Need to Restart from Specific Step

```bash
# Example: Restart from SAE training
sbatch scripts/slurm_03_train_all_saes.sh
```

---

## ⚠️ Important Reminders

### ✅ DO (from head node):
- ✅ Submit jobs with `sbatch`
- ✅ Monitor jobs with `squeue`
- ✅ View logs with `cat` or `tail`
- ✅ Edit configuration files
- ✅ Run the verification script

### ❌ DO NOT (from head node):
- ❌ **NEVER** run Python training scripts directly
- ❌ **NEVER** run activation extraction directly
- ❌ **NEVER** load large models into memory
- ❌ **NEVER** run computationally intensive tasks

**Why?** The head node is shared by all users. Heavy computation must run on compute nodes via SLURM!

---

## 📈 Checking Results

### After Pipeline Completes

1. **View Final Report**:
```bash
cat visualizations/FINAL_REPORT/COMPLETE_ANALYSIS_REPORT_*.md
```

2. **Check All Outputs Created**:
```bash
# Should show 68 activation files
ls checkpoints/full_layers_ncc/layer_checkpoints/*.pt | wc -l

# Should show 34 SAE models
ls checkpoints/saes/*/sae_final.pt | wc -l

# View all visualization directories
ls -lh visualizations/
```

3. **Review Specific Analysis**:
```bash
# Cross-layer evolution
ls visualizations/cross_layer_analysis_full/

# Feature ablation results
ls visualizations/feature_ablation/

# Visual pattern analysis
ls visualizations/visual_patterns_*/
```

---

## 💾 Backup Results (After Completion)

```bash
# Create backup of critical results
mkdir -p ~/backups/mech_intrep_results_$(date +%Y%m%d)

# Copy visualizations
rsync -av visualizations/ ~/backups/mech_intrep_results_$(date +%Y%m%d)/visualizations/

# Copy SAE models
rsync -av checkpoints/saes/ ~/backups/mech_intrep_results_$(date +%Y%m%d)/saes/

# Archive logs
tar -czf ~/backups/mech_intrep_results_$(date +%Y%m%d)/logs.tar.gz logs/

echo "Backup complete!"
```

---

## 📞 Get Help

### Durham NCC Support
- **Email**: ncc-support@durham.ac.uk
- **Portal**: https://nccadmin.webspace.durham.ac.uk/
- **Documentation**: Check docs/ directory

### Check Cluster Status
```bash
# Overall cluster status
sinfo

# GPU partition availability
sinfo -p res-gpu-small

# Your account limits
sacctmgr show user $USER withassoc
```

---

## 🎓 Advanced: Manual Step-by-Step Execution

If you prefer to run steps manually instead of the automated pipeline:

```bash
# Step 1: Data preparation
sbatch scripts/slurm_01_prepare_data.sh
# Wait for completion, then...

# Step 2: Activation extraction
bash scripts/slurm_02_parallel_extraction.sh
# Wait for all 4 parallel jobs to complete, then...

# Step 3: Train all SAEs
bash scripts/slurm_03_train_all_saes.sh
# Wait for all 34 models to train, then...

# Step 4: Comprehensive analysis
sbatch scripts/slurm_09_comprehensive_analysis.sh
# Wait for completion, then run steps 5-10 in parallel...

# Steps 5-10: Advanced analyses (can run in parallel)
sbatch scripts/slurm_11_feature_interpretation.sh &
sbatch scripts/slurm_13_feature_ablation.sh &
sbatch scripts/slurm_14_visual_pattern_analysis.sh &
sbatch scripts/slurm_15_feature_amplification.sh &
sbatch scripts/slurm_16_cross_layer_analysis.sh &
sbatch scripts/slurm_17_qualitative_analysis.sh &
wait
# Wait for all 6 to complete, then...

# Final: Generate report
sbatch scripts/slurm_generate_final_report.sh
```

**Note**: The automated pipeline (`slurm_00_full_pipeline.sh`) handles all this for you!

---

## ✅ Pre-Launch Checklist

Before running `bash scripts/slurm_00_full_pipeline.sh`, verify:

- [x] Connected to NCC cluster (hostname shows `ncc.clients.dur.ac.uk`)
- [x] Virtual environment created (`venv/` directory exists)
- [x] Dependencies installed (PyTorch, transformers, etc.)
- [x] Email configured in scripts (jmsk62@durham.ac.uk)
- [x] Partition names updated (res-gpu-small for GPUs, cpu for CPU)
- [x] Required directories created (logs/, checkpoints/, visualizations/)
- [x] Sufficient disk space (~70 GB needed)

Run `./verify_setup.sh` to check all items automatically.

---

## 🎯 Ready to Launch?

If all checks pass, execute:

```bash
source venv/bin/activate
bash scripts/slurm_00_full_pipeline.sh
```

Then relax and let the NCC cluster do the work! ☕

You'll receive email notifications, and the entire pipeline will run automatically over the next 12-16 hours.

---

**Good luck with your mechanistic interpretability research!** 🚀🧠

---

**Last Updated**: January 6, 2026
**Setup by**: Claude Code (Sonnet 4.5)
**User**: jmsk62
