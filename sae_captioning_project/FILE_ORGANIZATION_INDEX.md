# Project File Organization Index
## Cross-Lingual Gender Bias Analysis in VLMs

**Last Updated**: January 2026

---

## Quick Navigation

| What You Need | Location |
|---------------|----------|
| **Three-Model Comparison** | `results/three_model_comparison/` |
| **Final Results Summary** | `results/qwen2vl_analysis/model_comparison_cosine.json` |
| **Main Comparison Figure** | `results/three_model_comparison/comprehensive_dashboard.png` |
| **Technical Report** | `results/TECHNICAL_REPORT.md` |
| **Probe Comparison** | `results/PROBE_COMPARISON_REPORT.md` |
| **Methodology Guide** | `docs/METHODOLOGY_VERIFICATION.md` |
| **Presentation** | `presentation/SUPERVISOR_PRESENTATION.md` |

---

## 1. Model Comparison Overview

### Models Analyzed
| Model | Parameters | Hidden Dim | Layers | SAE Features | Arabic Support |
|-------|------------|------------|--------|--------------|----------------|
| **PaLiGemma-3B** | 3B | 2,048 | 26 | 16,384 | Native multilingual |
| **Qwen2-VL-7B-Instruct** | 7B | 3,584 | 28 | 28,672 | Native Arabic tokens |
| **LLaVA-1.5-7B** | 7B | 4,096 | 32 | 32,768 | Byte-fallback (UTF-8) |

### Layers Analyzed
| PaLiGemma-3B | Qwen2-VL-7B | LLaVA-1.5-7B |
|--------------|-------------|--------------|
| 0, 3, 6, 9, 12, 15, 17 | 0, 4, 8, 12, 16, 20, 24, 27 | 0, 4, 8, 12, 16, 20, 24, 28, 31 |

---

## 2. File Organization by Model

### 📁 PaLiGemma-3B Files

#### Checkpoints (Activations)
```
checkpoints/full_layers_ncc/
├── layer_checkpoints/
│   ├── layer_0_arabic.pt
│   ├── layer_0_english.pt
│   ├── layer_3_arabic.pt
│   ├── layer_3_english.pt
│   ├── layer_6_arabic.pt
│   ├── layer_6_english.pt
│   ├── layer_9_arabic.pt
│   ├── layer_9_english.pt
│   ├── layer_12_arabic.pt
│   ├── layer_12_english.pt
│   ├── layer_15_arabic.pt
│   ├── layer_15_english.pt
│   ├── layer_17_arabic.pt
│   └── layer_17_english.pt
```

#### SAE Models
```
checkpoints/saes/
├── sae_layer_0.pt                    # Layer 0 (shared)
├── sae_arabic_layer_3.pt             # Arabic SAEs
├── sae_arabic_layer_6.pt
├── sae_arabic_layer_9.pt
├── sae_arabic_layer_12.pt
├── sae_arabic_layer_15.pt
├── sae_arabic_layer_17.pt
├── sae_english_layer_3.pt            # English SAEs
├── sae_english_layer_6.pt
├── sae_english_layer_9.pt
├── sae_english_layer_12.pt
├── sae_english_layer_15.pt
├── sae_english_layer_17.pt
└── *_history.json                    # Training histories
```

#### Results
```
results/
├── proper_cross_lingual/
│   └── cross_lingual_results.json    # ⭐ Main PaLiGemma results
├── cross_lingual_overlap/
├── sbi_analysis/
│   └── sbi_results.json              # Surgical Bias Intervention
├── feature_stats_layer_*_arabic.csv  # Feature statistics
├── feature_stats_layer_*_english.csv
├── analysis_report.json
└── ANALYSIS_REPORT.md
```

#### Visualizations
```
visualizations/
├── proper_cross_lingual/             # ⭐ Cross-lingual analysis
│   ├── summary.png                   # Overall comparison
│   ├── layer_0_analysis.png
│   ├── layer_3_analysis.png
│   ├── layer_9_analysis.png
│   ├── layer_12_analysis.png
│   ├── layer_15_analysis.png
│   └── layer_17_analysis.png
├── sample_predictions/               # Sample image predictions
│   ├── layer_*_arabic/
│   │   ├── sample_grid.png
│   │   └── misclassified_detail.png
│   └── layer_*_english/
│       ├── sample_grid.png
│       └── misclassified_detail.png
├── layer_*/                          # Per-layer analysis
│   ├── tsne_gender.png
│   ├── top_gender_features.png
│   └── feature_distributions.png
├── cross_lingual/                    # CLBAS visualizations
│   ├── layer_*_clbas.png
│   └── layer_*_comparison.png
├── layer_comparison.png
├── layer_comparison_arabic.png
├── layer_comparison_english.png
├── layer_heatmap.png
├── layer_heatmap_arabic.png
├── layer_heatmap_english.png
└── accuracy_progression.png
```

---

### 📁 Qwen2-VL-7B Files

#### Checkpoints (Activations)
```
checkpoints/qwen2vl/
├── layer_checkpoints/
│   ├── layer_0_arabic.pt
│   ├── layer_0_english.pt
│   ├── layer_4_arabic.pt
│   ├── layer_4_english.pt
│   ├── layer_8_arabic.pt
│   ├── layer_8_english.pt
│   ├── layer_12_arabic.pt
│   ├── layer_12_english.pt
│   ├── layer_16_arabic.pt
│   ├── layer_16_english.pt
│   ├── layer_20_arabic.pt
│   ├── layer_20_english.pt
│   ├── layer_24_arabic.pt
│   ├── layer_24_english.pt
│   ├── layer_27_arabic.pt
│   └── layer_27_english.pt
```

#### SAE Models
```
checkpoints/qwen2vl/saes/
├── qwen2vl_sae_arabic_layer_0.pt
├── qwen2vl_sae_arabic_layer_4.pt
├── qwen2vl_sae_arabic_layer_8.pt
├── qwen2vl_sae_arabic_layer_12.pt
├── qwen2vl_sae_arabic_layer_16.pt
├── qwen2vl_sae_arabic_layer_20.pt
├── qwen2vl_sae_arabic_layer_24.pt
├── qwen2vl_sae_arabic_layer_27.pt
├── qwen2vl_sae_english_layer_0.pt
├── qwen2vl_sae_english_layer_4.pt
├── qwen2vl_sae_english_layer_8.pt
├── qwen2vl_sae_english_layer_12.pt
├── qwen2vl_sae_english_layer_16.pt
├── qwen2vl_sae_english_layer_20.pt
├── qwen2vl_sae_english_layer_24.pt
├── qwen2vl_sae_english_layer_27.pt
└── *_history.json
```

#### Results
```
results/qwen2vl_analysis/
├── model_comparison_cosine.json      # ⭐ Main comparison results
├── model_comparison_results.json
├── qwen2vl_analysis_results.json
├── cosine_similarity_comparison.png  # ⭐ Key comparison figure
├── final_model_comparison.png
├── publication_summary.png
├── qwen2vl_detailed_analysis.png
├── qwen2vl_vs_paligemma_comparison.png
└── qwen2vl_vs_paligemma_comparison.pdf
```

---

### 📁 LLaVA-1.5-7B Files

#### Checkpoints (Activations)
```
checkpoints/llava/
├── layer_checkpoints/
│   ├── layer_0_arabic.pt
│   ├── layer_0_english.pt
│   ├── layer_4_arabic.pt
│   ├── layer_4_english.pt
│   ├── layer_8_arabic.pt
│   ├── layer_8_english.pt
│   ├── layer_12_arabic.pt
│   ├── layer_12_english.pt
│   ├── layer_16_arabic.pt
│   ├── layer_16_english.pt
│   ├── layer_20_arabic.pt
│   ├── layer_20_english.pt
│   ├── layer_24_arabic.pt
│   ├── layer_24_english.pt
│   ├── layer_28_arabic.pt
│   ├── layer_28_english.pt
│   ├── layer_31_arabic.pt
│   └── layer_31_english.pt
```

#### SAE Models
```
checkpoints/llava/saes/
├── llava_sae_arabic_layer_0.pt
├── llava_sae_arabic_layer_4.pt
├── llava_sae_arabic_layer_8.pt
├── llava_sae_arabic_layer_12.pt
├── llava_sae_arabic_layer_16.pt
├── llava_sae_arabic_layer_20.pt
├── llava_sae_arabic_layer_24.pt
├── llava_sae_arabic_layer_28.pt
├── llava_sae_arabic_layer_31.pt
├── llava_sae_english_layer_0.pt
├── llava_sae_english_layer_4.pt
├── llava_sae_english_layer_8.pt
├── llava_sae_english_layer_12.pt
├── llava_sae_english_layer_16.pt
├── llava_sae_english_layer_20.pt
├── llava_sae_english_layer_24.pt
├── llava_sae_english_layer_28.pt
├── llava_sae_english_layer_31.pt
└── *_history.json
```

#### Results
```
results/llava_analysis/
├── cross_lingual_results.json        # Main LLaVA results
├── feature_overlap_results.json
├── probe_results.json
├── llava_analysis_summary.png
└── layer_*_analysis.png
```

---

### 📁 Three-Model Comparison

#### Results
```
results/three_model_comparison/
├── comparison_report.md              # ⭐ Markdown report
├── combined_metrics.csv              # All metrics in CSV
├── summary_statistics.json           # Statistical summaries
├── comprehensive_dashboard.png       # ⭐ Main comparison figure
├── comprehensive_dashboard.pdf
├── clbas_comparison.png
├── clbas_comparison.pdf
├── probe_accuracy_comparison.png
├── probe_accuracy_comparison.pdf
├── layer_position_heatmap.png
└── layer_position_heatmap.pdf
```

---

## 3. Scripts by Model

### PaLiGemma-3B Pipeline
```
scripts/
├── 01_prepare_data.py               # Data preparation
├── 02_extract_activations.py        # Activation extraction
├── 03_train_sae.py                  # SAE training
├── 23_proper_cross_lingual_analysis.py  # Cross-lingual analysis
├── 24_cross_lingual_overlap.py      # Feature overlap
├── 26_surgical_bias_intervention.py # SBI analysis
└── 27_statistical_significance.py   # Statistical tests
```

### Qwen2-VL-7B Pipeline
```
scripts/
├── 28_extract_qwen2vl_activations.py    # Activation extraction
├── 29_train_qwen2vl_sae.py              # SAE training
├── 30_qwen2vl_cross_lingual_analysis.py # Cross-lingual analysis
├── 31_qwen2vl_comprehensive_analysis.py # Full analysis
└── 32_generate_qwen2vl_visualizations.py # Visualizations
```

### LLaVA-1.5-7B Pipeline
```
scripts/
├── 33_llava_extract_activations.py      # Activation extraction (Arabic via byte-fallback)
├── 34_llava_train_sae.py                # SAE training (d=4096 → 32,768 features)
├── 35_llava_cross_lingual_analysis.py   # Cross-lingual analysis
└── slurm_33_llava_extract.sh            # SLURM array job for extraction
├── slurm_34_llava_sae.sh                # SLURM array job for SAE training
├── slurm_35_llava_analysis.sh           # SLURM job for analysis
└── slurm_llava_full_pipeline.sh         # Full sequential pipeline
```

### Three-Model Comparison
```
scripts/
└── 37_three_model_comparison.py         # ⭐ Comprehensive 3-model analysis
```

---

## 4. Key Results Files

### ⭐ Most Important Files

| File | Description |
|------|-------------|
| `results/three_model_comparison/comprehensive_dashboard.png` | Three-model comparison figure |
| `results/three_model_comparison/comparison_report.md` | Full comparison report |
| `results/qwen2vl_analysis/model_comparison_cosine.json` | PaLiGemma vs Qwen2-VL data |
| `results/proper_cross_lingual/cross_lingual_results.json` | PaLiGemma detailed results |
| `results/llava_analysis/cross_lingual_results.json` | LLaVA detailed results |
| `results/sbi_analysis/sbi_results.json` | Surgical intervention results |
| `results/PROBE_COMPARISON_REPORT.md` | Probe accuracy comparison |
| `docs/METHODOLOGY_VERIFICATION.md` | Full methodology documentation |

### Key Figures

| Figure | Location | Description |
|--------|----------|-------------|
| 3-Model Dashboard | `results/three_model_comparison/comprehensive_dashboard.png` | ⭐ Main comparison |
| CLBAS Comparison | `results/three_model_comparison/clbas_comparison.png` | Cross-lingual scores |
| Probe Accuracy | `results/three_model_comparison/probe_accuracy_comparison.png` | Gender probes |
| Layer Heatmap | `results/three_model_comparison/layer_position_heatmap.png` | By layer depth |
| Model Comparison | `results/qwen2vl_analysis/cosine_similarity_comparison.png` | PaLiGemma vs Qwen2-VL |
| PaLiGemma Summary | `visualizations/proper_cross_lingual/summary.png` | Layer-wise comparison |
| Sample Predictions | `visualizations/sample_predictions/layer_3_arabic/sample_grid.png` | Example predictions |

---

## 5. Results Summary

### Three-Model Comparison

| Model | Arabic Support | Mean CLBAS | Probe Gap | SAE Features |
|-------|----------------|------------|-----------|--------------|
| PaLiGemma-3B | Native multilingual | ~0.027 | AR+3.3% | 16,384 |
| Qwen2-VL-7B | Native Arabic tokens | ~0.004 | EN+1.5% | 28,672 |
| LLaVA-1.5-7B | Byte-fallback (UTF-8) | TBD | TBD | 32,768 |

### Cosine Similarity (Cross-Lingual Alignment)

| Model | Mean | Max | Interpretation |
|-------|------|-----|----------------|
| PaLiGemma-3B | **0.027** | 0.041 | Low alignment |
| Qwen2-VL-7B | **0.004** | 0.008 | Very low alignment |
| LLaVA-1.5-7B | TBD | TBD | TBD |
| **Ratio** | 6.7× | - | Larger model = more specific |

### Probe Accuracy

| Model | Arabic | English | Higher |
|-------|--------|---------|--------|
| PaLiGemma-3B | **0.886** | 0.853 | Arabic +3.3% |
| Qwen2-VL-7B | 0.903 | **0.918** | English +1.5% |
| LLaVA-1.5-7B | TBD | TBD | TBD |

### Feature Overlap

| Model | Overlap Count | Jaccard |
|-------|---------------|---------|
| PaLiGemma-3B | 3 | ~0.015 |
| Qwen2-VL-7B | 1 | ~0.005 |
| LLaVA-1.5-7B | TBD | TBD |

---

## 6. Documentation Files

```
docs/
├── METHODOLOGY_VERIFICATION.md      # ⭐ Full methodology
├── CLMB_FRAMEWORK.md
├── DURHAM_NCC_GUIDE.md
└── NCC_EXTRACTION_GUIDE.md

presentation/
├── SUPERVISOR_PRESENTATION.md       # ⭐ Presentation slides
├── COSINE_SIMILARITY_DEFENSE.md     # Literature defense
└── figures/

results/
├── PROBE_COMPARISON_REPORT.md       # ⭐ Probe comparison
├── COSINE_SIMILARITY_ANALYSIS.md
├── TECHNICAL_REPORT.md
└── ANALYSIS_REPORT.md
```

---

## 7. Quick Commands

### View key results
```bash
# Three-model comparison
cat results/three_model_comparison/comparison_report.md

# PaLiGemma results
cat results/proper_cross_lingual/cross_lingual_results.json | python -m json.tool | head -50

# Model comparison
cat results/qwen2vl_analysis/model_comparison_cosine.json | python -m json.tool

# LLaVA results
cat results/llava_analysis/cross_lingual_results.json | python -m json.tool | head -50

# SBI results
cat results/sbi_analysis/sbi_results.json | python -m json.tool | head -50
```

### Check file sizes
```bash
# SAE model sizes
du -sh checkpoints/saes/*.pt
du -sh checkpoints/qwen2vl/saes/*.pt
du -sh checkpoints/llava/saes/*.pt
```

### Run pipelines
```bash
# LLaVA full pipeline
sbatch scripts/slurm_llava_full_pipeline.sh

# Or individual steps:
sbatch scripts/slurm_33_llava_extract.sh   # Array job: 0=arabic, 1=english
sbatch scripts/slurm_34_llava_sae.sh       # Array job: 0-17 (9 layers × 2 langs)
sbatch scripts/slurm_35_llava_analysis.sh  # Cross-lingual analysis

# Three-model comparison
python scripts/37_three_model_comparison.py
```

---

*This index provides a complete map of all files for PaLiGemma-3B, Qwen2-VL-7B, and LLaVA-1.5-7B analysis.*
