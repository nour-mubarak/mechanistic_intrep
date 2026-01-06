# Pipeline Files Summary

**Complete list of files created for full pipeline execution on Durham NCC**

---

## Main Pipeline Scripts

### Master Pipeline Controller

- **`scripts/slurm_00_full_pipeline.sh`** ⭐
  - **Purpose**: Main entry point - orchestrates entire pipeline
  - **Usage**: `bash scripts/slurm_00_full_pipeline.sh`
  - **Function**: Submits all jobs with automatic dependencies
  - **Creates**: Job dependency chain, tracking files

### Individual Pipeline Steps

1. **`scripts/slurm_01_prepare_data.sh`**
   - Data preparation (2000 samples)
   - 1 hour, no GPU

2. **`scripts/slurm_02_parallel_extraction.sh`**
   - Extracts all 34 layers in 4 parallel jobs
   - 40-60 minutes, 4 GPUs

3. **`scripts/slurm_03_train_all_saes.sh`**
   - Trains 34 SAEs in parallel
   - 4-8 hours, 34 GPUs

4. **`scripts/slurm_09_comprehensive_analysis.sh`**
   - Comprehensive analysis across all layers
   - 8 hours, 1 GPU

5. **`scripts/slurm_11_feature_interpretation.sh`**
   - Step 5: Feature interpretation
   - 6 hours, 1 GPU

6. **`scripts/slurm_13_feature_ablation.sh`**
   - Step 6: Causal ablation analysis
   - 4 hours, 1 GPU

7. **`scripts/slurm_14_visual_pattern_analysis.sh`**
   - Step 7: Visual pattern identification
   - 3 hours, 1 GPU

8. **`scripts/slurm_15_feature_amplification.sh`**
   - Step 8: Dose-response analysis
   - 4 hours, 1 GPU

9. **`scripts/slurm_16_cross_layer_analysis.sh`**
   - Step 9: Cross-layer evolution
   - 6 hours, 1 GPU

10. **`scripts/slurm_17_qualitative_analysis.sh`**
    - Step 10: Qualitative visual analysis
    - 4 hours, 1 GPU

11. **`scripts/slurm_generate_final_report.sh`**
    - Final: Consolidate results and generate report
    - 10 minutes, no GPU

---

## Core Extraction Scripts

### Main Extraction Script

- **`scripts/18_extract_full_activations_ncc.py`** ⭐
  - **Purpose**: Extracts activations from all 34 layers using NCC methodology
  - **Features**:
    - Supports layer ranges: `--layer-ranges 0-10 20-33`
    - Efficient batching and memory management
    - Layer-wise checkpoint saves
    - NaN detection and handling
  - **Usage**:
    ```bash
    python scripts/18_extract_full_activations_ncc.py \
        --config configs/config.yaml \
        --layer-ranges 0-33 \
        --batch-size 1 \
        --checkpoint-interval 50
    ```

### Legacy Extraction Scripts (For Reference)

- **`scripts/slurm_extract_full_activations.sh`**
  - Sequential extraction (all layers, single job)
  - Use for small runs or if GPUs limited

- **`scripts/slurm_extract_layer_ranges.sh`**
  - Custom layer range extraction
  - Template for extracting specific layers

- **`scripts/slurm_parallel_extraction.sh`**
  - Parallel extraction template (4 jobs)
  - Alternative to automated pipeline

---

## Documentation

### Quick References

- **`QUICK_START.md`** ⭐
  - One-page quick reference
  - Fastest way to get started
  - Essential commands and monitoring

### Comprehensive Guides

- **`PIPELINE_EXECUTION_GUIDE.md`** ⭐⭐⭐
  - **Most important documentation**
  - Complete step-by-step guide
  - Troubleshooting section
  - Best practices
  - Expected timelines
  - Resource requirements

- **`docs/DURHAM_NCC_GUIDE.md`** ⭐⭐
  - Durham NCC cluster-specific guide
  - Environment setup
  - SLURM job management
  - Monitoring and troubleshooting
  - Performance optimization

- **`docs/NCC_EXTRACTION_GUIDE.md`** ⭐
  - Neural Corpus Compilation methodology
  - Technical specifications
  - Output formats
  - Usage examples
  - Future extensions

### Other Documentation

- **`README_NCC_EXTRACTION.md`**
  - Overview of NCC extraction
  - Quick validation examples

---

## Helper Scripts

- **`scripts/run_full_extraction.sh`**
  - Local execution script (not for NCC)
  - Reference implementation

---

## File Organization

```
sae_captioning_project/
│
├── QUICK_START.md                          ⭐ Start here!
├── PIPELINE_EXECUTION_GUIDE.md             ⭐⭐⭐ Main guide
├── README_NCC_EXTRACTION.md
├── PIPELINE_FILES_SUMMARY.md               (This file)
│
├── docs/
│   ├── DURHAM_NCC_GUIDE.md                 ⭐⭐ NCC-specific
│   └── NCC_EXTRACTION_GUIDE.md             ⭐ Methodology
│
└── scripts/
    ├── slurm_00_full_pipeline.sh           ⭐ Master script
    ├── slurm_01_prepare_data.sh
    ├── slurm_02_parallel_extraction.sh
    ├── slurm_03_train_all_saes.sh
    ├── slurm_09_comprehensive_analysis.sh
    ├── slurm_11_feature_interpretation.sh
    ├── slurm_13_feature_ablation.sh
    ├── slurm_14_visual_pattern_analysis.sh
    ├── slurm_15_feature_amplification.sh
    ├── slurm_16_cross_layer_analysis.sh
    ├── slurm_17_qualitative_analysis.sh
    ├── slurm_generate_final_report.sh
    │
    ├── 18_extract_full_activations_ncc.py  ⭐ Core extraction
    │
    ├── slurm_extract_full_activations.sh   (Alternative)
    ├── slurm_extract_layer_ranges.sh       (Template)
    └── slurm_parallel_extraction.sh        (Alternative)
```

---

## Usage Workflow

### For First-Time Users

1. **Read**: `QUICK_START.md` (5 minutes)
2. **Setup**: Follow prerequisites
3. **Execute**: `bash scripts/slurm_00_full_pipeline.sh`
4. **Monitor**: Use commands from quick start
5. **Review**: Check `PIPELINE_EXECUTION_GUIDE.md` for details

### For Detailed Understanding

1. **Read**: `PIPELINE_EXECUTION_GUIDE.md` (complete walkthrough)
2. **Reference**: `docs/DURHAM_NCC_GUIDE.md` (cluster-specific)
3. **Technical**: `docs/NCC_EXTRACTION_GUIDE.md` (methodology)

### For Troubleshooting

1. **Check**: `PIPELINE_EXECUTION_GUIDE.md` → Troubleshooting section
2. **Reference**: `docs/DURHAM_NCC_GUIDE.md` → Common Issues
3. **Logs**: Review `logs/step*_*.err`

---

## Key Features

### Automation
- ✅ Single command execution
- ✅ Automatic job dependencies
- ✅ Parallel processing where possible
- ✅ Email notifications at milestones

### Robustness
- ✅ Checkpointing for recovery
- ✅ Memory management (OOM prevention)
- ✅ NaN detection and handling
- ✅ Layer-wise storage for flexibility

### Monitoring
- ✅ SLURM job tracking
- ✅ Detailed logging
- ✅ Progress indicators
- ✅ Resource usage reporting

### Documentation
- ✅ Quick start guide
- ✅ Comprehensive pipeline guide
- ✅ NCC-specific instructions
- ✅ Troubleshooting sections

---

## Configuration

All scripts use the central configuration file:

**`configs/config.yaml`**

Key settings:
```yaml
model:
  name: "google/gemma-3-4b-it"

data:
  num_samples: 2000  # Full dataset

layers:
  extraction: [2, 6, 10, 14, 18, 22, 26, 30]  # Can be 0-33

sae:
  expansion_factor: 8
  epochs: 50
  batch_size: 256

logging:
  use_wandb: true
  wandb_project: "sae-captioning-bias"
```

---

## Outputs Generated

### Checkpoints
- **`checkpoints/full_layers_ncc/layer_checkpoints/`**
  - 68 files (34 layers × 2 languages)
  - ~450 MB per file
  - Total: ~30 GB

### SAE Models
- **`checkpoints/saes/layer_*/sae_final.pt`**
  - 34 trained SAE models
  - ~150 MB per model
  - Total: ~5 GB

### Visualizations
- **`visualizations/comprehensive_all_layers/`** - All layer analysis
- **`visualizations/cross_layer_analysis_full/`** - Evolution
- **`visualizations/feature_ablation/`** - Causal effects
- **`visualizations/visual_patterns_*/`** - Visual triggers
- **`visualizations/feature_amplification_*/`** - Dose-response
- **`visualizations/qualitative_*/`** - Qualitative analysis
- **`visualizations/FINAL_REPORT/`** - Summary report

### Logs
- **`logs/step*_*.{out,err}`** - All job logs
- **`pipeline_status/*.jobid`** - Job ID tracking

**Total Storage**: ~70 GB

---

## Customization

### Modify Layer Ranges

Edit extraction scripts:
```bash
# Extract only early layers
LAYER_RANGES="0-16"

# Extract specific layers
LAYER_RANGES="0 10 14 18 22 26 30 33"
```

### Adjust Resources

Edit SLURM directives in scripts:
```bash
#SBATCH --mem=128G      # More memory
#SBATCH --time=12:00:00 # More time
#SBATCH --gres=gpu:2    # Multiple GPUs
```

### Change Batch Sizes

Edit in scripts:
```bash
# For extraction
--batch-size 2

# For SAE training
--batch-size 512
```

---

## Version History

- **v1.0** (2026-01-05): Initial complete pipeline
  - All 34 layers extraction
  - Automated dependency management
  - Comprehensive documentation

---

## Next Steps

After pipeline completes:

1. **Review final report**:
   ```bash
   cat visualizations/FINAL_REPORT/COMPLETE_ANALYSIS_REPORT_*.md
   ```

2. **Analyze specific findings**:
   - Gender bias evolution across layers
   - Causal feature effects
   - Visual pattern triggers

3. **Prepare manuscript**:
   - Use generated visualizations
   - Reference quantitative results
   - Cite methodology from docs

4. **Design interventions**:
   - Target identified gender-biased features
   - Test debiasing strategies
   - Evaluate impact on downstream tasks

---

## Summary

**To run the complete pipeline**:

```bash
# 1. Setup (first time)
cd /path/to/sae_captioning_project
source venv/bin/activate
sed -i 's/your.email@durham.ac.uk/YOUR_EMAIL/g' scripts/slurm_*.sh

# 2. Execute
bash scripts/slurm_00_full_pipeline.sh

# 3. Monitor
squeue -u $USER

# 4. Review results
cat visualizations/FINAL_REPORT/COMPLETE_ANALYSIS_REPORT_*.md
```

**That's it!** The complete mechanistic interpretability analysis will run automatically.

---

**Created**: January 5, 2026
**Status**: Production ready
**Tested**: Ready for deployment on Durham NCC
