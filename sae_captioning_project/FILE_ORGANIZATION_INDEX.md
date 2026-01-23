# Project File Organization Index
## Cross-Lingual Gender Bias Analysis in VLMs

**Last Updated**: January 23, 2026

---

## Quick Navigation

| What You Need | Location |
|---------------|----------|
| **Final Results Summary** | `results/qwen2vl_analysis/model_comparison_cosine.json` |
| **Main Comparison Figure** | `results/qwen2vl_analysis/cosine_similarity_comparison.png` |
| **Technical Report** | `results/TECHNICAL_REPORT.md` |
| **Probe Comparison** | `results/PROBE_COMPARISON_REPORT.md` |
| **Methodology Guide** | `docs/METHODOLOGY_VERIFICATION.md` |
| **Presentation** | `presentation/SUPERVISOR_PRESENTATION.md` |

---

## 1. Model Comparison Overview

### Models Analyzed
| Model | Parameters | Hidden Dim | Layers | SAE Features |
|-------|------------|------------|--------|--------------|
| **PaLiGemma-3B** | 3B | 2,048 | 18 | 16,384 |
| **Qwen2-VL-7B-Instruct** | 7B | 3,584 | 28 | 28,672 |

### Layers Analyzed
| PaLiGemma-3B | Qwen2-VL-7B |
|--------------|-------------|
| 0, 3, 6, 9, 12, 15, 17 | 0, 4, 8, 12, 16, 20, 24, 27 |

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

---

## 4. Key Results Files

### ⭐ Most Important Files

| File | Description |
|------|-------------|
| `results/qwen2vl_analysis/model_comparison_cosine.json` | Final comparison data |
| `results/proper_cross_lingual/cross_lingual_results.json` | PaLiGemma detailed results |
| `results/sbi_analysis/sbi_results.json` | Surgical intervention results |
| `results/PROBE_COMPARISON_REPORT.md` | Probe accuracy comparison |
| `docs/METHODOLOGY_VERIFICATION.md` | Full methodology documentation |

### Key Figures

| Figure | Location | Description |
|--------|----------|-------------|
| Model Comparison | `results/qwen2vl_analysis/cosine_similarity_comparison.png` | Side-by-side cosine sim |
| Publication Summary | `results/qwen2vl_analysis/publication_summary.png` | Paper-ready figure |
| PaLiGemma Summary | `visualizations/proper_cross_lingual/summary.png` | Layer-wise comparison |
| Sample Predictions | `visualizations/sample_predictions/layer_3_arabic/sample_grid.png` | Example predictions |

---

## 5. Results Summary

### Cosine Similarity (Cross-Lingual Alignment)

| Model | Mean | Max | Interpretation |
|-------|------|-----|----------------|
| PaLiGemma-3B | **0.027** | 0.041 | Low alignment |
| Qwen2-VL-7B | **0.004** | 0.008 | Very low alignment |
| **Ratio** | 6.7× | - | Larger model = more specific |

### Probe Accuracy

| Model | Arabic | English | Higher |
|-------|--------|---------|--------|
| PaLiGemma-3B | **0.886** | 0.853 | Arabic +3.3% |
| Qwen2-VL-7B | 0.903 | **0.918** | English +1.5% |

### Feature Overlap

| Model | Overlap Count | Jaccard |
|-------|---------------|---------|
| PaLiGemma-3B | 3 | ~0.015 |
| Qwen2-VL-7B | 1 | ~0.005 |

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
# PaLiGemma results
cat results/proper_cross_lingual/cross_lingual_results.json | python -m json.tool | head -50

# Model comparison
cat results/qwen2vl_analysis/model_comparison_cosine.json | python -m json.tool

# SBI results
cat results/sbi_analysis/sbi_results.json | python -m json.tool | head -50
```

### Check file sizes
```bash
# SAE model sizes
du -sh checkpoints/saes/*.pt
du -sh checkpoints/qwen2vl/saes/*.pt
```

---

*This index provides a complete map of all files for both PaLiGemma-3B and Qwen2-VL-7B analysis.*
