# Key Files Index
## Cross-Lingual SAE Mechanistic Interpretability Project

*Last Updated: January 16, 2026*

---

## 📂 Essential Files by Category

### 🔧 Configuration Files
| File | Purpose |
|------|---------|
| `configs/config.yaml` | Main configuration (model, SAE, layers) |
| `configs/clmb_config.yaml` | CLMB framework settings |
| `configs/config_layer6.yaml` | Layer-specific config example |

### 📊 Core Source Code

#### Models (`src/models/`)
| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/models/sae.py` | Sparse Autoencoder implementation | `SAEConfig`, `SparseAutoencoder` |
| `src/models/hooks.py` | Activation extraction hooks | `ActivationHook` |
| `src/models/__init__.py` | Module exports | - |

#### CLMB Framework (`src/clmb/`)
| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/clmb/hbl.py` | Hierarchical Bias Localization | `compute_bias_attribution()` |
| `src/clmb/clfa.py` | Cross-Lingual Feature Alignment | `align_features()`, `optimal_transport()` |
| `src/clmb/sbi.py` | Surgical Bias Intervention | `ablate()`, `neutralize()` |
| `src/clmb/extractors.py` | Model-agnostic extractors | `extract_activations()` |
| `src/clmb/models.py` | Model wrappers | `ModelWrapper` |

#### Analysis (`src/analysis/`)
| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/analysis/features.py` | Feature analysis utilities | `get_top_features()`, `compute_effect_size()` |
| `src/analysis/metrics.py` | Metrics computation | `compute_clbas()`, `compute_overlap()` |

#### Data (`src/data/`)
| File | Purpose |
|------|---------|
| `src/data/dataset.py` | Dataset loading and processing |

#### Visualization (`src/visualization/`)
| File | Purpose |
|------|---------|
| `src/visualization/plots.py` | Plotting utilities |

### 🚀 Pipeline Scripts (Ordered)

#### Stage 1: Data Preparation
| Script | Purpose | SLURM |
|--------|---------|-------|
| `scripts/01_prepare_data.py` | Prepare dataset with gender labels | `slurm_01_prepare_data.sh` |

#### Stage 2: Activation Extraction
| Script | Purpose | SLURM |
|--------|---------|-------|
| `scripts/02_extract_activations.py` | Generic extraction | `slurm_02_parallel_extraction.sh` |
| `scripts/18_extract_full_activations_ncc.py` | Arabic extraction (NCC) | `slurm_extract_arabic_ncc.sh` |
| `scripts/22_extract_english_activations.py` | English extraction | `slurm_extract_layer6_english.sh` |

#### Stage 3: SAE Training
| Script | Purpose | SLURM |
|--------|---------|-------|
| `scripts/03_train_sae.py` | Train SAE per layer/language | `slurm_03_train_all_saes.sh` |
| `scripts/train_sae_ncc.py` | NCC-optimized training | `slurm_train_arabic_saes.sh` |

#### Stage 4: Feature Analysis
| Script | Purpose | SLURM |
|--------|---------|-------|
| `scripts/04_analyze_features.py` | Compute feature statistics | - |
| `scripts/09_comprehensive_analysis.py` | Full analysis | `slurm_09_comprehensive_analysis.sh` |
| `scripts/11_feature_interpretation.py` | Interpret features | `slurm_11_feature_interpretation.sh` |

#### Stage 5: Cross-Lingual Analysis
| Script | Purpose | SLURM |
|--------|---------|-------|
| `scripts/24_cross_lingual_overlap.py` | **KEY**: CLBAS & overlap | `slurm_24_cross_lingual_overlap.sh` |
| `scripts/25_cross_lingual_feature_interpretation.py` | Feature interpretation | `slurm_25_feature_interpretation.sh` |
| `scripts/23_proper_cross_lingual_analysis.py` | Cross-lingual probes | - |

#### Stage 6: Intervention
| Script | Purpose | SLURM |
|--------|---------|-------|
| `scripts/26_surgical_bias_intervention.py` | **KEY**: Ablation tests | `slurm_26_sbi_array.sh` |
| `scripts/13_feature_ablation_analysis.py` | Feature ablation | `slurm_13_feature_ablation.sh` |
| `scripts/15_feature_amplification.py` | Feature amplification | `slurm_15_feature_amplification.sh` |

#### Stage 7: Statistical Validation
| Script | Purpose | SLURM |
|--------|---------|-------|
| `scripts/27_statistical_significance.py` | **KEY**: Bootstrap tests | `slurm_27_statistical_tests.sh` |

### 🔧 Utility Scripts
| Script | Purpose |
|--------|---------|
| `scripts/merge_checkpoints.py` | Merge chunked activation files |
| `scripts/consolidate_results.py` | Consolidate results |
| `scripts/run_full_pipeline.py` | Run full pipeline locally |
| `scripts/verify_setup.sh` | Verify installation |

### 📋 SLURM Job Scripts
| Script | Purpose | Array? |
|--------|---------|--------|
| `scripts/slurm_00_full_pipeline.sh` | Complete pipeline | No |
| `scripts/slurm_02_parallel_extraction.sh` | Parallel extraction | Yes |
| `scripts/slurm_03_train_all_saes.sh` | Train all SAEs | Yes |
| `scripts/slurm_26_sbi_array.sh` | SBI with W&B | Yes |
| `scripts/slurm_27_statistical_tests.sh` | Statistical tests | No |

---

## 📁 Data Files

### Input Data
| Path | Description |
|------|-------------|
| `data/raw/captions.csv` | Raw image captions (Arabic + English) |
| `data/raw/images/` | COCO images directory |
| `data/processed/samples.csv` | Processed samples with gender labels |
| `data/processed/data_summary.yaml` | Dataset statistics |

### Checkpoints
| Path | Description | Size |
|------|-------------|------|
| `checkpoints/full_layers_ncc/layer_checkpoints/layer_{N}_arabic.pt` | Arabic activations | ~22GB/layer |
| `checkpoints/full_layers_ncc/layer_checkpoints/layer_{N}_english.pt` | English activations | ~79MB/layer |
| `checkpoints/saes/sae_arabic_layer_{N}.pt` | Arabic SAE | ~256MB |
| `checkpoints/saes/sae_english_layer_{N}.pt` | English SAE | ~256MB |
| `checkpoints/saes/sae_*_history.json` | Training history | ~4KB |

### Results
| Path | Description |
|------|-------------|
| `results/cross_lingual_overlap/cross_lingual_overlap_results.json` | **KEY**: CLBAS results |
| `results/feature_interpretation/feature_interpretation_results.json` | Feature analysis |
| `results/sbi_analysis/sbi_results.json` | **KEY**: Ablation results |
| `results/sbi_analysis/visualizations/` | SBI plots |
| `results/statistical_tests/statistical_significance_results.json` | **KEY**: P-values |
| `results/feature_stats_layer_{N}_{lang}.csv` | Feature statistics |
| `results/TECHNICAL_REPORT.md` | Technical summary |
| `results/ANALYSIS_REPORT.md` | Analysis summary |

---

## 📚 Documentation

### Main Docs
| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `RESEARCH_PLAN.md` | Research methodology |
| `docs/COMPLETE_PIPELINE_DOCUMENTATION.md` | **THIS FILE** - Complete reference |
| `docs/CLMB_FRAMEWORK.md` | CLMB framework details |

### Guides (`docs/guides/`)
| File | Purpose |
|------|---------|
| `QUICK_START.md` | Getting started |
| `PIPELINE_EXECUTION_GUIDE.md` | Running the pipeline |
| `MECHANISTIC_INTERPRETABILITY_GUIDE.md` | MI concepts |
| `LAUNCH_INSTRUCTIONS.md` | Launch commands |
| `WANDB_TRACKING.md` | W&B integration |

### Cluster Docs
| File | Purpose |
|------|---------|
| `docs/DURHAM_NCC_GUIDE.md` | Durham NCC cluster guide |
| `docs/NCC_EXTRACTION_GUIDE.md` | Extraction on NCC |

### Status Docs (`docs/status/`)
| File | Purpose |
|------|---------|
| `COMPLETION_STATUS.md` | Pipeline completion status |
| `PIPELINE_RUN_STATUS.md` | Run status |

---

## 🎯 Key Scripts for New Model Testing

### Minimum Required Scripts
1. `scripts/02_extract_activations.py` - Extract activations (adapt hooks)
2. `scripts/03_train_sae.py` - Train SAE (adjust d_model)
3. `scripts/24_cross_lingual_overlap.py` - Compute CLBAS
4. `scripts/26_surgical_bias_intervention.py` - Run ablation tests
5. `scripts/27_statistical_significance.py` - Statistical validation

### Minimum Required Source Files
1. `src/models/sae.py` - SAE implementation
2. `src/models/hooks.py` - Activation hooks (MODIFY FOR NEW MODEL)
3. `src/analysis/metrics.py` - Metrics computation

### Configuration
1. `configs/config.yaml` - Update model name, d_model, layers

---

## 🔄 Model Adaptation Checklist

When testing a new model (e.g., LLaMA, Qwen, etc.):

- [ ] Update `configs/config.yaml` with new model settings
- [ ] Modify `src/models/hooks.py` for model architecture
- [ ] Adjust `d_model` in SAE config to match hidden size
- [ ] Update layer indices based on model depth
- [ ] Test extraction on single layer before full run
- [ ] Adjust SLURM memory/time based on model size

---

## 📊 Key Metrics Reference

| Metric | File | Function |
|--------|------|----------|
| CLBAS | `24_cross_lingual_overlap.py` | `compute_clbas()` |
| Feature Overlap | `24_cross_lingual_overlap.py` | `compute_feature_overlap()` |
| Cohen's d | `src/analysis/metrics.py` | `compute_effect_size()` |
| Probe Accuracy | `26_surgical_bias_intervention.py` | `train_gender_probe()` |
| Bootstrap CI | `27_statistical_significance.py` | `bootstrap_clbas()` |

---

## 📦 Dependencies

```txt
# Core
torch>=2.1.0
transformers>=4.40.0
numpy
scipy
scikit-learn
pandas

# Arabic NLP
camel-tools

# Visualization
matplotlib
seaborn

# Tracking
wandb

# Optimal Transport (for CLFA)
POT
```

---

*Generated: January 16, 2026*
