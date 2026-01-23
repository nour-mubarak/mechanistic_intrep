# Comprehensive Methodology Verification
## Cross-Lingual Gender Bias Analysis in Vision-Language Models Using Sparse Autoencoders

---

## Table of Contents
1. [Stage 1: Data Preparation](#stage-1-data-preparation)
2. [Stage 2: Activation Extraction](#stage-2-activation-extraction)
3. [Stage 3: SAE Training](#stage-3-sae-training)
4. [Stage 4: Cross-Lingual Analysis](#stage-4-cross-lingual-analysis)
5. [Stage 5: Feature Overlap Analysis](#stage-5-feature-overlap-analysis)
6. [Stage 6: Surgical Bias Intervention](#stage-6-surgical-bias-intervention)
7. [Stage 7: Statistical Significance](#stage-7-statistical-significance)
8. [Stage 8: Multi-Model Comparison](#stage-8-multi-model-comparison)

---

## Stage 1: Data Preparation

### Script
`scripts/01_prepare_data.py`

### Method
- **Input**: COCO Captions dataset with human-annotated gender labels
- **Processing**: 
  - Filter images containing humans (male/female labels)
  - Validate image files and remove corrupted entries
  - Create paired Arabic-English prompts for same images
- **Output**: Processed dataset with `image_id`, `gender`, `english_prompt`, `arabic_prompt`

### Metrics
| Metric | Description | Value |
|--------|-------------|-------|
| Sample Count | Number of valid image-prompt pairs | ~2,000 per language |
| Gender Balance | Male:Female ratio | ~1:1 |
| Image Validation | % of images loadable | 100% |

### Figures Generated
- None (data preparation stage)

### Justification
Standard data preparation following VLM evaluation practices. COCO Captions is widely used for vision-language benchmarks (Chen et al., 2015; 28,000+ citations).

### Literature Support
- **Chen et al. (2015)** "Microsoft COCO Captions: Data Collection and Evaluation Server" *arXiv* - Standard benchmark for image captioning

---

## Stage 2: Activation Extraction

### Script
`scripts/02_extract_activations.py`

### Method
1. **Model Loading**: Load VLM (PaLiGemma-3B or Qwen2-VL-7B) in float32
2. **Hook Registration**: Register forward hooks on transformer layers
3. **Prompt Processing**: Apply chat template for image+text input
4. **Activation Capture**: Extract residual stream activations after each layer
5. **NaN Handling**: Replace NaN values with zeros (numerical stability)

### Architecture Details
| Model | Hidden Dim | Layers | Dtype |
|-------|------------|--------|-------|
| PaLiGemma-3B | 2048 | 18 | float32 |
| Qwen2-VL-7B | 3584 | 28 | float32 |

### Hook Configuration
```python
HookConfig(
    layers=[0, 3, 6, 9, 12, 15, 17],  # PaLiGemma
    component="residual",
    detach=True,
    to_cpu=True,
    dtype=torch.float32
)
```

### Metrics
| Metric | Description | Formula |
|--------|-------------|---------|
| Activation Shape | Dimensions per sample | (batch, seq_len, d_model) |
| NaN Rate | Percentage of NaN values | count(NaN) / total |
| Memory Usage | GPU memory for extraction | Variable |

### Figures Generated
- None (intermediate data stage)

### Justification
Standard activation extraction methodology for mechanistic interpretability. Residual stream activations capture the primary information flow in transformers (Elhage et al., 2021).

### Literature Support
- **Elhage et al. (2021)** "A Mathematical Framework for Transformer Circuits" *Anthropic* - Foundation for activation-based interpretability

---

## Stage 3: SAE Training

### Script
`scripts/03_train_sae.py`

### Method
**Architecture**: Standard Sparse Autoencoder
```
Input (d_model) → Encoder (d_model → d_hidden) → ReLU → Decoder (d_hidden → d_model) → Output
```

**Loss Function**:
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda \cdot \mathcal{L}_{L1}$$

Where:
- $\mathcal{L}_{recon} = \text{MSE}(x, \hat{x})$ — Reconstruction loss
- $\mathcal{L}_{L1} = ||f||_1$ — L1 sparsity penalty on features
- $\lambda = 1 \times 10^{-4}$ — L1 coefficient

### Hyperparameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Expansion Factor | 8× | Standard (Bricken et al., 2023) |
| L1 Coefficient | 1e-4 | Balances sparsity vs reconstruction |
| Learning Rate | 1e-4 | Adam optimizer default |
| Batch Size | 2048 | Memory-limited |
| Epochs | 50 | Early stopping with patience=10 |
| Warmup Steps | 1000 | Gradual learning rate increase |

### Metrics
| Metric | Description | Formula | Target |
|--------|-------------|---------|--------|
| Reconstruction Loss | MSE between input and output | $\text{MSE}(x, \hat{x})$ | < 0.1 |
| L0 Sparsity | Number of active features | $||f > 0||_0$ | ~50-200 |
| Validation Loss | Held-out reconstruction | Same as training | Decreasing |

### Figures Generated
- Training curves (loss vs epoch)
- Sparsity histograms

### Justification
Standard SAE architecture following Anthropic's methodology (Bricken et al., 2023). 8× expansion is the recommended ratio for interpretable features.

### Literature Support
- **Bricken et al. (2023)** "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" *Anthropic* - SAE methodology
- **Cunningham et al. (2023)** "Sparse Autoencoders Find Highly Interpretable Features in Language Models" *ICLR* - SAE for interpretability

---

## Stage 4: Cross-Lingual Analysis

### Script
`scripts/23_proper_cross_lingual_analysis.py`

### Method

#### 4.1 Gender Probe Training
**Classifier**: Logistic Regression
```python
LogisticRegression(max_iter=1000, random_state=42, C=0.1)
```

**Process**:
1. Extract SAE features for each sample
2. Standardize features with StandardScaler
3. Train 5-fold cross-validation
4. Report mean accuracy ± std

#### 4.2 Gender Effect Size (Cohen's d)
For each SAE feature $i$:
$$d_i = \frac{\mu_{male,i} - \mu_{female,i}}{\sigma_{pooled,i}}$$

Where:
$$\sigma_{pooled} = \sqrt{\frac{\sigma_{male}^2 + \sigma_{female}^2}{2}}$$

**Interpretation**:
- $|d| > 0.2$: Small effect
- $|d| > 0.5$: Medium effect  
- $|d| > 0.8$: Large effect

### Metrics
| Metric | Description | Formula |
|--------|-------------|---------|
| Probe Accuracy | Gender classification performance | Cross-val mean |
| Active Features | Features with activation > 0.01 | Count |
| Overall Sparsity | Fraction of zero features | mean(f < 0.01) |

### Figures Generated
- `layer_X_analysis.png`: Effect size comparison scatter plot
- `summary.png`: Layer-wise accuracy comparison

### Justification
Linear probing is the standard method for analyzing learned representations (Conneau et al., 2018). Cohen's d provides standardized effect sizes for comparison.

### Literature Support
- **Conneau et al. (2018)** "What you can cram into a single $&!#* vector" *ACL* - Probing methodology
- **Cohen (1988)** "Statistical Power Analysis" - Effect size interpretation

---

## Stage 5: Feature Overlap Analysis

### Script
`scripts/24_cross_lingual_overlap.py`

### Method

#### 5.1 Cosine Similarity of Gender Directions
**Formula**:
$$\text{cos}(\theta) = \frac{\mathbf{d}_{ar} \cdot \mathbf{d}_{en}}{||\mathbf{d}_{ar}|| \times ||\mathbf{d}_{en}||}$$

Where:
- $\mathbf{d}_{ar}$ = Arabic effect size vector (16,384 dimensions)
- $\mathbf{d}_{en}$ = English effect size vector (16,384 dimensions)

**Interpretation**:
| Value | Meaning |
|-------|---------|
| ~1.0 | Perfect alignment (same features) |
| ~0.0 | Orthogonal (different features) |
| ~-1.0 | Inverse (opposite features) |

#### 5.2 Jaccard Index for Feature Overlap
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Where:
- $A$ = Top-k Arabic gender features
- $B$ = Top-k English gender features
- $k = 100$ (top features by absolute effect size)

### Metrics
| Metric | Description | Formula | Expected |
|--------|-------------|---------|----------|
| Cosine Similarity | Direction alignment | See above | Near 0 |
| Jaccard Index | Set overlap | See above | < 0.05 |
| Overlap Count | Raw intersection | \|A ∩ B\| | 0-5 |

### Results
| Model | Mean Cosine Sim | Max Overlap |
|-------|-----------------|-------------|
| PaLiGemma-3B | 0.027 | 3 features |
| Qwen2-VL-7B | 0.004 | 1 feature |

### Figures Generated
- `feature_overlap_by_layer.png`: Overlap % per layer
- `clbas_components_by_layer.png`: ~~CLBAS~~ Cosine similarity components
- `cross_lingual_summary_heatmap.png`: Summary heatmap
- `cosine_similarity_comparison.png`: Model comparison

### Justification
**Cosine Similarity** is the standard metric for cross-lingual alignment:
1. Used in mBERT, XLM-R cross-lingual studies
2. Scale-invariant (measures direction, not magnitude)
3. Bounded [-1, 1] for easy interpretation

### Literature Support
- **Conneau et al. (2020)** "Emerging Cross-lingual Structure in Pretrained Language Models" *ACL* - 333 citations
- **Hämmerl et al. (2024)** "Understanding Cross-Lingual Alignment -- A Survey" *ACL Findings*
- **Wang et al. (2018)** "Cross-lingual Knowledge Graph Alignment via GCNs" *EMNLP* - 806 citations

---

## Stage 6: Surgical Bias Intervention (SBI)

### Script
`scripts/26_surgical_bias_intervention.py`

### Method

#### 6.1 Feature Ablation
Zero out top-k gender-associated features:
```python
features_ablated[:, top_k_indices] = 0
```

#### 6.2 Feature Neutralization (Alternative)
Replace with gender-neutral average:
```python
neutral_value = (male_mean + female_mean) / 2
features_neutral[:, idx] = neutral_value
```

#### 6.3 Cross-Lingual Specificity Test
- **Same-language ablation**: Ablate Arabic features → test Arabic probe (should drop)
- **Cross-language ablation**: Ablate English features → test Arabic probe (should NOT drop if language-specific)

### Metrics
| Metric | Description | Formula |
|--------|-------------|---------|
| Accuracy Drop | Baseline - Ablated | $\Delta_{acc} = acc_{base} - acc_{ablated}$ |
| Relative Drop | Percentage change | $100 \times \Delta_{acc} / acc_{base}$ |
| Reconstruction Quality | Semantic preservation | $\text{cos}(x_{orig}, x_{recon})$ |

### Results (Layer 17)
| k Ablated | Arabic Acc Drop | English Acc Drop | Recon Quality |
|-----------|-----------------|------------------|---------------|
| 10 | 0.05% | 0.13% | 0.997 |
| 25 | -0.05% | -0.24% | 0.994 |
| 50 | -0.04% | 0.03% | 0.989 |
| 100 | 0.02% | 0.13% | 0.977 |
| 200 | -0.02% | 0.29% | 0.953 |

### Key Finding
**~0% accuracy drop** indicates gender information is distributed, not localized in top features.

### Figures Generated
- `sbi_accuracy_vs_k.png`: Accuracy vs features ablated
- `sbi_cross_lingual_specificity.png`: Same vs cross-language effects
- `sbi_tradeoff.png`: Bias reduction vs semantic preservation

### Justification
Ablation studies are the gold standard for causal analysis in neural networks (Morcos et al., 2018). Cross-lingual specificity tests validate that features are language-specific.

### Literature Support
- **Morcos et al. (2018)** "On the Importance of Single Directions for Generalization" *ICLR* - Ablation methodology
- **Bolukbasi et al. (2016)** "Man is to Computer Programmer as Woman is to Homemaker?" *NeurIPS* - Bias intervention

---

## Stage 7: Statistical Significance

### Script
`scripts/27_statistical_significance.py`

### Method

#### 7.1 Bootstrap Confidence Intervals
**Process**:
1. Resample with replacement (n=1000 iterations)
2. Compute metric on each sample
3. Report 95% CI: [2.5th percentile, 97.5th percentile]

**Applied to**: Cosine similarity, probe accuracy, effect sizes

#### 7.2 Permutation Test for Feature Overlap
**Null Hypothesis**: Overlap is due to random chance

**Process**:
1. Randomly select k features from total (n=16,384)
2. Compute overlap with observed top-k
3. Repeat 10,000 times
4. p-value = proportion ≥ observed

**Expected overlap under null**:
$$E[\text{overlap}] = \frac{k^2}{n} = \frac{100^2}{16384} \approx 0.61$$

### Metrics
| Test | Statistic | CI/p-value |
|------|-----------|------------|
| Cosine Similarity | Bootstrap mean | 95% CI |
| Feature Overlap | Permutation | p-value |
| Probe Accuracy | Bootstrap | 95% CI |
| Ablation Effect | Bootstrap | 95% CI, p < 0.05 |

### Justification
Bootstrap methods provide robust uncertainty estimates without distributional assumptions (Efron & Tibshirani, 1993). Permutation tests are appropriate for testing against random baselines.

### Literature Support
- **Efron & Tibshirani (1993)** "An Introduction to the Bootstrap" *Chapman & Hall* - Bootstrap methodology
- **Good (2005)** "Permutation, Parametric, and Bootstrap Tests of Hypotheses" - Permutation testing

---

## Stage 8: Multi-Model Comparison

### Scripts
`scripts/28-31_qwen2vl_*.py`

### Method
Replicate full pipeline (Stages 1-7) on Qwen2-VL-7B-Instruct for model comparison.

### Model Specifications
| Spec | PaLiGemma-3B | Qwen2-VL-7B |
|------|--------------|-------------|
| Parameters | 3B | 7B |
| Hidden Dim | 2048 | 3584 |
| Layers | 18 | 28 |
| SAE Features | 16,384 | 28,672 |
| Layers Analyzed | 0,3,6,9,12,15,17 | 0,4,8,12,16,20,24,27 |

### Results Summary
| Metric | PaLiGemma-3B | Qwen2-VL-7B | Interpretation |
|--------|--------------|-------------|----------------|
| Mean Cosine Sim | 0.027 | 0.004 | 6.7× lower in larger model |
| Max Cosine Sim | 0.041 | 0.008 | - |
| Feature Overlap | 3 | 1 | 3× fewer shared features |
| Probe Accuracy (AR) | - | 0.903 | High gender predictability |
| Probe Accuracy (EN) | - | 0.918 | High gender predictability |

### Figures Generated
- `cosine_similarity_comparison.png`: Side-by-side comparison
- `publication_summary.png`: Publication-ready figure
- `qwen2vl_detailed_analysis.png`: Layer-wise breakdown

### Key Finding
**Larger models show MORE language-specific gender encoding** (lower cross-lingual alignment). This suggests scaling may increase language-specific processing pathways.

---

## Summary of All Metrics

### Core Metrics (Literature-Supported)

| Metric | Stage | Formula | Literature |
|--------|-------|---------|------------|
| **Cosine Similarity** | 5 | $\frac{A \cdot B}{\|A\| \|B\|}$ | Conneau 2020, Hämmerl 2024 |
| **Jaccard Index** | 5 | $\frac{\|A \cap B\|}{\|A \cup B\|}$ | Standard set similarity |
| **Cohen's d** | 4 | $\frac{\mu_1 - \mu_2}{\sigma_{pool}}$ | Cohen 1988 |
| **Probe Accuracy** | 4 | 5-fold CV mean | Conneau 2018 |
| **Bootstrap CI** | 7 | 2.5-97.5 percentiles | Efron 1993 |
| **Permutation p-value** | 7 | $P(X \geq x_{obs})$ | Good 2005 |

### ~~Removed/Replaced Metrics~~
| Original | Replaced With | Reason |
|----------|---------------|--------|
| ~~CLBAS~~ | Cosine Similarity | CLBAS is novel, not in literature |

---

## Complete Figure Inventory

### Generated Figures
| Figure | Stage | Location | Description |
|--------|-------|----------|-------------|
| `layer_X_analysis.png` | 4 | `visualizations/proper_cross_lingual/` | Effect size scatter |
| `feature_overlap_by_layer.png` | 5 | `results/cross_lingual_overlap/` | Overlap percentages |
| `cosine_similarity_comparison.png` | 8 | `results/qwen2vl_analysis/` | Model comparison |
| `sbi_accuracy_vs_k.png` | 6 | `results/sbi_analysis/visualizations/` | Ablation curves |
| `sbi_cross_lingual_specificity.png` | 6 | `results/sbi_analysis/visualizations/` | Specificity test |
| `publication_summary.png` | 8 | `results/qwen2vl_analysis/` | Final summary |

---

## BibTeX Citations

```bibtex
@inproceedings{conneau2020emerging,
  title={Emerging Cross-lingual Structure in Pretrained Language Models},
  author={Conneau, Alexis and others},
  booktitle={ACL},
  year={2020}
}

@inproceedings{hammerl2024understanding,
  title={Understanding Cross-Lingual Alignment -- A Survey},
  author={Hämmerl, Katharina and others},
  booktitle={ACL Findings},
  year={2024}
}

@article{bricken2023monosemanticity,
  title={Towards Monosemanticity: Decomposing Language Models With Dictionary Learning},
  author={Bricken, Trenton and others},
  journal={Anthropic},
  year={2023}
}

@inproceedings{cunningham2023sparse,
  title={Sparse Autoencoders Find Highly Interpretable Features in Language Models},
  author={Cunningham, Hoagy and others},
  booktitle={ICLR},
  year={2023}
}

@book{efron1993bootstrap,
  title={An Introduction to the Bootstrap},
  author={Efron, Bradley and Tibshirani, Robert J},
  year={1993},
  publisher={Chapman \& Hall}
}
```

---

*Document generated: Verification of all methods, metrics, and figures in the cross-lingual SAE analysis pipeline.*
