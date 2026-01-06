# Quick Start - Full Pipeline on Durham NCC

**One-page reference for running the complete mechanistic interpretability pipeline**

---

## Prerequisites (5 minutes)

```bash
# 1. SSH to NCC
ssh username@ncc.durham.ac.uk

# 2. Navigate to project
cd /path/to/mechanistic_intrep/sae_captioning_project

# 3. Setup environment (first time only)
module load python/3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Update email in scripts
sed -i 's/your.email@durham.ac.uk/YOUR_EMAIL@durham.ac.uk/g' scripts/slurm_*.sh
```

---

## Execute Pipeline (1 command!)

```bash
# Submit complete pipeline with automatic dependencies
bash scripts/slurm_00_full_pipeline.sh
```

**That's it!** The pipeline will:
- Extract all 34 layers (40-60 min)
- Train 34 SAEs (4-8 hours, parallel)
- Run all analyses (2-8 hours)
- Generate final report

**Total time**: 12-16 hours (mostly automated)

---

## Monitor Progress

```bash
# Check all jobs
squeue -u $USER

# Watch job status (updates every 5 sec)
watch -n 5 'squeue -u $USER'

# View logs
tail -f logs/step*_*.out
```

---

## Verify Completion

```bash
# Check extraction (should show 68 files: 34 layers × 2 languages)
ls checkpoints/full_layers_ncc/layer_checkpoints/ | wc -l

# Check SAEs (should show 34 models)
ls checkpoints/saes/*/sae_final.pt | wc -l

# View final report
cat visualizations/FINAL_REPORT/COMPLETE_ANALYSIS_REPORT_*.md
```

---

## Pipeline Steps (Automatic)

| Step | What | Time | GPUs |
|------|------|------|------|
| 1 | Data prep | 1 hr | 0 |
| 2 | Extract activations (34 layers) | 40-60 min | 4 |
| 3 | Train SAEs (34 layers) | 4-8 hr | 34 |
| 4 | Comprehensive analysis | 8 hr | 1 |
| 5-10 | Advanced analyses (6 types) | 2-6 hr | 6 |
| Final | Generate report | 10 min | 0 |

All steps have automatic dependencies - no manual intervention needed!

---

## Outputs

```
checkpoints/
├── full_layers_ncc/layer_checkpoints/  (68 files, ~30 GB)
└── saes/layer_*/sae_final.pt           (34 models, ~5 GB)

visualizations/
├── comprehensive_all_layers/           (All layer analysis)
├── cross_layer_analysis_full/          (Evolution across depth)
├── feature_ablation/                   (Causal effects)
├── visual_patterns_*/                  (What triggers features)
└── FINAL_REPORT/                       (Summary)
```

---

## Manual Execution (Alternative)

If you prefer step-by-step:

```bash
# Step 1: Data
sbatch scripts/slurm_01_prepare_data.sh

# Step 2: Extract (wait for Step 1)
bash scripts/slurm_02_parallel_extraction.sh

# Step 3: Train SAEs (wait for Step 2)
bash scripts/slurm_03_train_all_saes.sh

# Step 4: Analysis (wait for Step 3)
sbatch scripts/slurm_09_comprehensive_analysis.sh

# Steps 5-10: Advanced (wait for Step 4, run in parallel)
sbatch scripts/slurm_11_feature_interpretation.sh
sbatch scripts/slurm_13_feature_ablation.sh
sbatch scripts/slurm_14_visual_pattern_analysis.sh
sbatch scripts/slurm_15_feature_amplification.sh
sbatch scripts/slurm_16_cross_layer_analysis.sh
sbatch scripts/slurm_17_qualitative_analysis.sh

# Final: Report (wait for Steps 5-10)
sbatch scripts/slurm_generate_final_report.sh
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Jobs pending | Wait for GPU availability: `sinfo -p gpu` |
| OOM errors | Increase memory: `--mem=128G` in scripts |
| Missing files | Verify previous step completed: `squeue -u $USER` |
| Need help | Check logs: `cat logs/step*_*.err` |

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/slurm_00_full_pipeline.sh` | **Main entry point** |
| `PIPELINE_EXECUTION_GUIDE.md` | Detailed documentation |
| `docs/DURHAM_NCC_GUIDE.md` | NCC-specific guide |
| `docs/NCC_EXTRACTION_GUIDE.md` | Extraction methodology |

---

## After Pipeline Completes

1. **Review Results**:
   ```bash
   cat visualizations/FINAL_REPORT/COMPLETE_ANALYSIS_REPORT_*.md
   ```

2. **Backup Critical Outputs**:
   ```bash
   rsync -av visualizations/ ~/backups/results/
   ```

3. **Archive Logs**:
   ```bash
   tar -czf logs_$(date +%Y%m%d).tar.gz logs/
   ```

---

## Resources Required

- **Storage**: ~70 GB
- **GPUs**: Up to 34 during SAE training
- **Time**: 12-16 hours total
- **Cluster**: Durham NCC with GPU partition access

---

## Questions?

- **Detailed guide**: `PIPELINE_EXECUTION_GUIDE.md`
- **NCC support**: ncc-support@durham.ac.uk
- **NCC portal**: https://nccadmin.webspace.durham.ac.uk/

---

**Ready to start?**

```bash
bash scripts/slurm_00_full_pipeline.sh
```

Then sit back and let the cluster do the work! You'll receive email notifications at each major milestone.

---

**Last Updated**: January 5, 2026
