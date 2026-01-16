# Mechanistic Interpretability Analysis Summary

## Project Status

✅ **Completed Steps:**

### 1. Data Preparation
- **Dataset**: 2,000 image-caption pairs (English + Arabic)
- **Source**: Gender-labeled images with cross-lingual captions
- **Split**: 500 samples per language for analysis (memory-efficient sampling)

### 2. Activation Extraction
- **Model**: `google/gemma-3-4b-it` (Vision-Language Model)
- **Precision**: float32 (for numerical stability)
- **Layers Extracted**: [2, 6, 10, 14, 18, 22, 26, 30] (8 layers across the model)
- **Activation Shape**: [batch_size, 278 tokens, 2560 hidden_dim]
- **Total Data**: ~87GB raw activations (40 chunks per language)
- **Sampled Data**: ~21GB (500 samples per language for analysis)

### 3. Sparse Autoencoder (SAE) Training
- **Layers Trained**: [10, 14, 18, 22] (middle layers for primary analysis)
- **Architecture**:
  - Input dimension: 2560
  - Expansion factor: 8x
  - Hidden dimension: 20,480 features
  - Activation: ReLU
  - Top-k sparsity: 32 active features
- **Training**:
  - L1 coefficient: 0.0005
  - Normalized decoder weights
  - Float32 precision
- **Checkpoint Size**: ~401MB per layer
- **Sparsity Achieved**: ~32 active features per token (L0)

### 4. Feature Analysis & Visualization (In Progress)
Currently running comprehensive analysis including:

#### A. SAE Feature Statistics
- Feature activation patterns (frequency, mean, max)
- Dead feature analysis
- Sparsity metrics (L0 per sample)
- Gender-specific activation patterns

#### B. ViT-Prisma Mechanistic Analysis
- **Factored Matrix Analysis**:
  - Effective rank computation
  - Information content quantification
  - Cross-lingual comparison

- **Feature Alignment**:
  - Cross-lingual feature similarity
  - Shared vs language-specific features
  - Alignment ratio metrics

- **Gender Bias Analysis**:
  - Cohen's d effect sizes
  - Male-biased vs female-biased features
  - Cross-lingual gender bias correlation

#### C. Visualizations Being Generated
1. **Feature Statistics Plots**: Distribution of activations, frequencies, sparsity
2. **Gender Bias Plots**: Top male/female-biased features per language
3. **Embedding Visualizations**: t-SNE and PCA projections of feature space
4. **Cross-Lingual Comparisons**: Correlation plots between English and Arabic

## Key Research Questions

1. **Universal vs Language-Specific Encoding**:
   - Do English and Arabic share the same gender representation features?
   - Measured by: Feature alignment ratio, correlation metrics

2. **Gender Bias Magnitude**:
   - How strongly are features biased toward male/female concepts?
   - Measured by: Cohen's d effect sizes, differential activation

3. **Layer-wise Evolution**:
   - How does gender representation evolve across model depth?
   - Measured by: Layer-by-layer comparison of alignment and bias

4. **Morphological vs Semantic Gender**:
   - Does Arabic grammatical gender correlate with semantic gender bias?
   - Measured by: Morphological analysis + feature activation patterns

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│  VLM: Gemma-3-4B-IT (Vision-Language Model)            │
│  ├─ Vision Encoder (processes images)                   │
│  └─ Language Model (34 layers, 2560 hidden dim)         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Activation Extraction (8 layers)                       │
│  └─ Hook-based extraction at transformer layers         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Sparse Autoencoders (4 SAEs trained)                   │
│  ├─ Input: 2560-dim activations                         │
│  ├─ Encoder: Linear → ReLU → Top-K(32)                  │
│  ├─ Hidden: 20,480 features (8x expansion)              │
│  └─ Decoder: Linear → Reconstruction                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Mechanistic Analysis                                    │
│  ├─ Feature Statistics (activation patterns)            │
│  ├─ ViT-Prisma (factored matrices, interactions)        │
│  ├─ Cross-Lingual Alignment (feature matching)          │
│  └─ Gender Bias Metrics (effect sizes, correlations)    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Visualizations & Results                                │
│  ├─ Statistical plots (distributions, correlations)     │
│  ├─ Gender bias heatmaps                                 │
│  ├─ t-SNE/PCA embeddings                                 │
│  └─ JSON results for further analysis                   │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
sae_captioning_project/
├── checkpoints/
│   ├── activations_english_sample.pt (11GB, 500 samples)
│   ├── activations_arabic_sample.pt (11GB, 500 samples)
│   ├── sae_layer_10.pt (401MB)
│   ├── sae_layer_14.pt (401MB)
│   ├── sae_layer_18.pt (401MB)
│   └── sae_layer_22.pt (401MB)
├── results/
│   └── comprehensive_analysis_results.json (pending)
├── visualizations/
│   └── (plots being generated)
└── logs/
    └── comprehensive_analysis.log (in progress)
```

## Next Steps

Once the current analysis completes:

1. **Review Results**:
   - Examine JSON analysis results
   - Study generated visualizations
   - Identify key patterns and insights

2. **Generate Report**:
   - Compile findings across all layers
   - Create summary figures
   - Write mechanistic interpretability report

3. **Further Analysis** (optional):
   - Causal interventions (activation patching)
   - Feature steering experiments
   - Logit lens analysis
   - Attention pattern visualization

## Configuration

Model: `google/gemma-3-4b-it`
- Total layers: 34
- Hidden dimension: 2560
- dtype: float32

Data:
- Languages: English, Arabic
- Samples: 500 per language (analysis subset)
- Total dataset: 2000 per language

SAE:
- Expansion: 8x (2560 → 20,480)
- Sparsity: Top-32
- L1: 0.0005

Analysis Layers: [10, 14, 18, 22]

---

*Analysis initiated: 2025-12-16*
*Expected completion: ~10-15 minutes per layer*
