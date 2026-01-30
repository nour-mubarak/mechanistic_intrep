# Publication Materials
## Cross-Lingual Gender Bias in Vision-Language Models: A Mechanistic Interpretability Study

**Last Updated**: January 2026

---

## Directory Structure

```
publication/
├── figures/
│   ├── main/           # Main paper figures (5-8 figures)
│   ├── supplementary/  # Supplementary figures
│   └── appendix/       # Appendix figures
├── tables/             # LaTeX tables
├── data/               # Processed data for reproducibility
└── latex/              # LaTeX source files
```

---

## Main Figures

### Figure 1: Model Comparison Overview
- **File**: `figures/main/fig1_model_comparison.png`
- **Source**: `results/qwen2vl_analysis/publication_summary.png`
- **Description**: Three-model comparison showing cosine similarity and probe accuracy

### Figure 2: Cross-Lingual Feature Alignment
- **File**: `figures/main/fig2_clbas_comparison.png`
- **Source**: `results/cross_lingual_overlap/visualizations/clbas_components_by_layer.png`
- **Description**: CLBAS scores across layers for all models

### Figure 3: Probe Accuracy by Layer
- **File**: `figures/main/fig3_probe_accuracy.png`
- **Source**: `results/qwen2vl_analysis/cosine_similarity_comparison.png`
- **Description**: Gender probe accuracy for Arabic vs English across layers

### Figure 4: Feature Overlap Analysis
- **File**: `figures/main/fig4_feature_overlap.png`
- **Source**: `results/cross_lingual_overlap/visualizations/feature_overlap_by_layer.png`
- **Description**: Shared vs language-specific features

### Figure 5: t-SNE Visualizations
- **File**: `figures/main/fig5_tsne_comparison.png`
- **Source**: `visualizations/layer_*/tsne_gender.png`
- **Description**: Feature space visualization showing gender clustering

### Figure 6: Surgical Bias Intervention
- **File**: `figures/main/fig6_sbi_results.png`
- **Source**: `results/sbi_analysis/visualizations/sbi_tradeoff.png`
- **Description**: Accuracy-fairness tradeoff with feature ablation

---

## Key Results Summary

### Table 1: Model Specifications

| Model | Parameters | Hidden Dim | Layers | SAE Features | Arabic Support |
|-------|------------|------------|--------|--------------|----------------|
| PaLiGemma-3B | 3B | 2,048 | 26 | 16,384 | Native multilingual |
| Qwen2-VL-7B | 7B | 3,584 | 28 | 28,672 | Native Arabic tokens |
| LLaVA-1.5-7B | 7B | 4,096 | 32 | 32,768 | Byte-fallback (UTF-8) |

### Table 2: Cross-Lingual Metrics

| Model | Mean Cosine Sim | Shared Features | Arabic Probe | English Probe | Probe Gap |
|-------|-----------------|-----------------|--------------|---------------|-----------|
| PaLiGemma-3B | 0.027 | 3 | 88.6% | 85.3% | AR+3.3% |
| Qwen2-VL-7B | 0.004 | 1 | 90.3% | 91.8% | EN+1.5% |
| LLaVA-1.5-7B | 0.001 | 1 | 89.9% | 96.3% | EN+6.4% |

### Table 3: Layer-wise Analysis (Best Layers)

| Model | Best CLBAS Layer | Best Probe Layer | Peak Cosine Sim |
|-------|------------------|------------------|-----------------|
| PaLiGemma-3B | Layer 15 | Layer 15 | 0.041 |
| Qwen2-VL-7B | Layer 20 | Layer 20 | 0.008 |
| LLaVA-1.5-7B | Layer 24 | Layer 28 | 0.016 |

---

## Data Files

### Raw Results
- `data/paligemma_results.json` - Full PaLiGemma analysis
- `data/qwen2vl_results.json` - Full Qwen2-VL analysis
- `data/llava_results.json` - Full LLaVA analysis
- `data/three_model_comparison.json` - Combined comparison

### Processed Metrics
- `data/cosine_similarity_all_layers.csv`
- `data/probe_accuracy_all_layers.csv`
- `data/feature_overlap_counts.csv`

---

## LaTeX Tables

### Usage
```latex
\input{tables/model_specs.tex}
\input{tables/cross_lingual_metrics.tex}
\input{tables/layer_analysis.tex}
```

---

## Figure Specifications

### Main Paper Figures
- **Format**: PDF (vector) or PNG (300 DPI minimum)
- **Width**: Single column (3.5") or double column (7")
- **Font**: Match paper font (typically Computer Modern or Times)
- **Colors**: Colorblind-friendly palette

### Color Scheme
- PaLiGemma: `#2ecc71` (Green)
- Qwen2-VL: `#3498db` (Blue)  
- LLaVA: `#e74c3c` (Red)
- Arabic: `#e74c3c` (Red)
- English: `#3498db` (Blue)

---

## Reproducibility

### Running the Analysis
```bash
# PaLiGemma pipeline
python scripts/23_proper_cross_lingual_analysis.py

# Qwen2-VL pipeline  
python scripts/30_qwen2vl_cross_lingual_analysis.py

# LLaVA pipeline
python scripts/35_llava_cross_lingual_analysis.py

# Three-model comparison
python scripts/37_three_model_comparison.py
```

### Requirements
- Python 3.10+
- PyTorch 2.1+
- transformers 4.40+
- See `requirements.txt` for full list

---

## Citation

```bibtex
@article{crosslingual_vlm_bias_2026,
  title={Cross-Lingual Gender Bias in Vision-Language Models: 
         A Mechanistic Interpretability Study},
  author={[Authors]},
  journal={[Venue]},
  year={2026}
}
```
