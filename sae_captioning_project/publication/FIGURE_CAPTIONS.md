# Figure Captions for Publication

## Main Figures

### Figure 1: Pipeline Overview (`fig1_pipeline_overview.png`)
**Caption:** Overview of the Cross-Lingual Mechanistic Bias (CLMB) analysis pipeline. (a) Vision-language models process paired images with Arabic and English prompts. (b) Sparse Autoencoders (SAEs) are trained on intermediate layer activations to decompose representations into interpretable features. (c) Gender-associated features are identified via linear probing and differential activation analysis. (d) Cross-lingual alignment is measured using cosine similarity and feature overlap metrics.

### Figure 2: CLBAS Components by Layer (`fig2_clbas_components.png`)
**Caption:** Cross-Lingual Bias Alignment Score (CLBAS) decomposition across layers. The CLBAS metric combines three components: (i) cosine similarity between mean gender-associated feature activations, (ii) overlap ratio of top-k gender features, and (iii) probe accuracy gap. Higher values indicate greater cross-lingual alignment of gender bias circuits. Peak alignment occurs in mid-to-late layers (layers 12-20), suggesting bias consolidation before output projection.

### Figure 3: Model Cosine Similarity Comparison (`fig3_cosine_similarity.png`)
**Caption:** Cross-lingual cosine similarity of gender-associated SAE features across vision-language models. PaLiGemma-3B exhibits significantly higher cross-lingual alignment (mean cos. sim. = 0.027) compared to Qwen2-VL-7B (mean cos. sim. = 0.004), suggesting that larger models with more sophisticated tokenization may develop more language-specific bias representations.

### Figure 4: Feature Overlap Analysis (`fig4_feature_overlap.png`)
**Caption:** Jaccard index of gender-associated SAE features (top-100) between Arabic and English across layers. Near-zero overlap indicates that the two languages utilize distinct feature subsets for encoding gender information, despite similar downstream task performance.

### Figure 5: Cross-Lingual Summary (`fig5_cross_lingual_summary.png`)
**Caption:** Summary of cross-lingual analysis across all analyzed layers. Left: Feature activation distributions for male vs. female associated features in both languages. Right: Layer-wise comparison of probe accuracy and cosine similarity metrics.

### Figure 6: SBI Tradeoff Analysis (`fig6_sbi_tradeoff.png`)
**Caption:** Surgical Bias Intervention (SBI) performance-fairness tradeoff. Ablating increasing numbers of gender-associated features (k=10 to k=200) has minimal impact on downstream task accuracy (< 0.3% drop) while maintaining high reconstruction quality (> 95%), demonstrating that gender bias can be surgically removed without compromising model utility.

---

## Supplementary Figures

### Figure S1: Qwen2-VL Detailed Analysis (`qwen2vl_detailed.png`)
**Caption:** Detailed layer-by-layer analysis for Qwen2-VL-7B showing (a) feature activation distributions, (b) probe accuracy progression, and (c) reconstruction quality metrics.

### Figure S2: Full Model Comparison (`model_comparison_full.png`)
**Caption:** Comprehensive comparison of PaLiGemma-3B and Qwen2-VL-7B across all metrics including cosine similarity, overlap counts, and probe accuracies.

### Figure S3: Cross-Lingual Heatmap (`cross_lingual_heatmap.png`)
**Caption:** Heatmap visualization of cross-lingual feature alignment. Rows represent Arabic features, columns represent English features. Color intensity indicates activation correlation.

### Figure S4: SBI Accuracy vs k (`sbi_accuracy_curve.png`)
**Caption:** Model accuracy as a function of the number of ablated gender-associated features. The flat curve demonstrates that gender features can be removed with negligible performance impact.

### Figure S5: SBI Cross-Lingual Specificity (`sbi_specificity.png`)
**Caption:** Language-specificity of surgical interventions. Ablating Arabic gender features has minimal cross-lingual impact on English performance and vice versa.

---

## Appendix Figures

### Figures A1-A6: Per-Layer Detailed Analysis
**Caption:** Detailed analysis for each analyzed layer showing: (a) SAE feature activation histogram, (b) top-k gender-associated features, (c) reconstruction error distribution, and (d) probe decision boundary visualization.

- Figure A1: Layer 0 (`layer_0_detailed.png`)
- Figure A2: Layer 3 (`layer_3_detailed.png`)
- Figure A3: Layer 9 (`layer_9_detailed.png`)
- Figure A4: Layer 12 (`layer_12_detailed.png`)
- Figure A5: Layer 15 (`layer_15_detailed.png`)
- Figure A6: Layer 17 (`layer_17_detailed.png`)
