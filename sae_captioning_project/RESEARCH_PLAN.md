# Cross-Lingual SAE Analysis for Vision-Language Model Gender Bias
## Comprehensive Research Plan

**Project**: Mechanistic Interpretability for Cross-Lingual Gender Bias Analysis  
**Model**: Gemma-3-4B-IT  
**Languages**: English & Arabic  
**Last Updated**: January 9, 2026

---

## 🎯 Research Title
**Cross-Lingual SAE Analysis for Vision-Language Model Gender Bias**

---

## 📚 Core Research Questions

| # | Research Question | Method |
|---|-------------------|--------|
| **RQ1** | **Where do gender representations diverge between Arabic and English?** | Layer-wise SAE analysis + LogitLens |
| **RQ2** | **Are there language-specific gender features?** | Cross-Lingual Feature Alignment (CLFA) |
| **RQ3** | **Can we steer the model to reduce bias?** | Surgical Bias Intervention (SBI) |
| **RQ4** | **Grammatical vs semantic gender:** How does Arabic morphological gender (ة endings) differ from semantic associations? | Morphological & Semantic Gender Analyzers |

---

## 🏗️ CLMB Framework (Novel Contribution)

### 1. HBL: Hierarchical Bias Localization
Identify which model component (Vision → Projection → Language) contributes most to gender bias using Bias Attribution Scores.

**Metric - Bias Attribution Score (BAS):**
$$BAS_{component} = \sum_i |a_i^{male} - a_i^{female}| \times w_i$$

Where:
- $a_i^{male/female}$ = mean activation for gender-specific samples
- $w_i$ = feature importance from SAE analysis

### 2. CLFA: Cross-Lingual Feature Alignment
Use optimal transport (Wasserstein distance) to find which SAE features encode the same concepts in Arabic vs English.

**Alignment via Optimal Transport:**
$$\gamma^* = \arg\min_\gamma \sum_{i,j} C_{ij} \gamma_{ij}$$

### 3. SBI: Surgical Bias Intervention
- **Ablation**: Zero out bias-encoding features
- **Neutralization**: Average male/female feature values  
- **Amplification**: Boost fairness-promoting features

### 4. CLBAS: Cross-Lingual Bias Alignment Score
Novel metric measuring bias consistency across languages:

$$CLBAS = \frac{\sum_{(f_{ar}, f_{en}) \in \text{aligned}} |bias(f_{ar}) - bias(f_{en})| \times sim(f_{ar}, f_{en})}{\sum_{(f_{ar}, f_{en}) \in \text{aligned}} sim(f_{ar}, f_{en})}$$

**Interpretation:**
- Low CLBAS (→ 0): Same stereotypes in both languages
- High CLBAS (→ 1): Language-specific stereotypes

---

## 🔬 Methodology Pipeline

### Stage 1: Data Preparation ✅
- 40,455 image captions from bilingual Arabic-English dataset
- Gender labels (male/female/unknown)
- Paired English + Arabic prompts for same images

### Stage 2: Activation Extraction ✅
- Model: **Gemma-3-4B-IT** (4B parameter multilingual LLM)
- Layers extracted: 0, 3, 6, 9, 12, 15, 17 (spanning early→late)
- ~22GB per layer checkpoint (merged activations)

### Stage 3: SAE Training 🔄
**Sparse Autoencoders** trained separately for each language/layer:
- Architecture: 2048 → 16,384 hidden dimensions (8× expansion)
- L1 regularization for sparsity (l1_coef = 5e-4)
- 50 epochs, batch size 256

**SAE Models Status:**
| Language | Layers Trained |
|----------|---------------|
| English | 0, 3, 6, 9, 12, 15, 17 ✅ |
| Arabic | 0 ✅, 3 🔄, 6 ✅ |

### Stage 4: Feature Analysis
- Identify gender-encoding features via linear probing
- Compute feature importance scores
- Find shared vs language-specific features

### Stage 5: Steering Experiments
- Ablate gender features → measure caption quality change
- Test cross-lingual steering (Arabic features → English outputs)
- Quantify bias reduction vs semantic degradation

### Stage 6: Visualization
- Layer-wise gender encoding heatmaps
- Feature activation scatter plots
- Cross-lingual alignment diagrams

### Stage 7: Integrated Mechanistic Analysis
Using **ViT-Prisma** tools:
- **FactoredMatrix**: SVD, rank, information entropy analysis
- **LogitLens**: Track where gender prediction emerges
- **InteractionPatternAnalyzer**: Find feature interactions
- **TransformerProbeAnalyzer**: Linear probes per layer
- **MorphologicalGenderAnalyzer**: Arabic suffix detection (ة, ها)
- **SemanticGenderAnalyzer**: Meaning-based gender associations

---

## 🛠️ Technical Implementation

### Model Architecture
```
Gemma-3-4B-IT
├── Embedding Layer
├── Transformer Blocks (0-41)
│   ├── Layer 0  → SAE (English ✅, Arabic ✅)
│   ├── Layer 3  → SAE (English ✅, Arabic 🔄)
│   ├── Layer 6  → SAE (English ✅, Arabic ✅)
│   ├── Layer 9  → SAE (English ✅)
│   ├── Layer 12 → SAE (English ✅)
│   ├── Layer 15 → SAE (English ✅)
│   └── Layer 17 → SAE (English ✅)
└── Output Layer
```

### SAE Architecture
```
Input: d_model = 2048
Hidden: d_hidden = 16,384 (8× expansion)
Output: d_model = 2048

Loss = Reconstruction Loss + λ × L1(hidden activations)
     = MSE(input, output) + 5e-4 × ||h||_1
```

### Integrated Mechanistic Tools

#### ViT-Prisma Tools (6 classes)
1. **HookPoint**: Activation capture via forward hooks
2. **ActivationCache**: Multi-layer caching with cleanup
3. **FactoredMatrix**: SVD, rank, information content (Shannon entropy), PCA
4. **LogitLens**: Layer-wise prediction tracking
5. **InteractionPatternAnalyzer**: Feature interaction discovery
6. **TransformerProbeAnalyzer**: Linear probe training/evaluation

#### Multilingual Feature Tools (5 classes)
1. **CrossLingualFeatureAligner**: Cosine similarity-based feature matching
2. **MorphologicalGenderAnalyzer**: Arabic feminine suffix detection (ة, ها, تها, نها)
3. **SemanticGenderAnalyzer**: Semantic gender associations
4. **ContrastiveLanguageAnalyzer**: Language-specific effect measurement
5. **LanguageSpecificFeatureIdentifier**: Unique feature detection

---

## 📊 Expected Outputs

### Quantitative Findings
- Layer-wise bias attribution scores
- Feature alignment scores (English ↔ Arabic)
- CLBAS metric for the model
- Steering effectiveness metrics

### Qualitative Insights
- Which layers encode grammatical vs semantic gender
- Whether Arabic morphology (تاء مربوطة) has dedicated features
- Cross-lingual steering effectiveness

### Artifacts
- Trained SAE models per layer/language (~268MB each)
- Feature importance rankings
- Visualizations and plots
- W&B experiment tracking logs

---

## 📁 Project Structure

```
sae_captioning_project/
├── configs/
│   └── config.yaml              # Main configuration
├── scripts/
│   ├── 01_prepare_data.py       # Dataset preparation
│   ├── 02_extract_activations.py # Activation extraction
│   ├── 03_train_sae.py          # SAE training
│   ├── 04_analyze_features.py   # Feature analysis
│   ├── 05_steering_experiments.py # Intervention experiments
│   ├── 06_generate_visualizations.py # Create plots
│   ├── 07_integrated_mechanistic_analysis.py # Mechanistic analysis
│   └── train_sae_ncc.py         # NCC-format SAE training
├── src/
│   ├── models/sae.py            # SAE architecture
│   ├── mechanistic/
│   │   ├── prisma_integration.py    # ViT-Prisma tools
│   │   └── multilingual_features.py # Multilingual analyzers
│   └── analysis/                # Analysis utilities
├── checkpoints/
│   ├── saes/                    # Trained SAE models
│   └── full_layers_ncc/         # Extracted activations
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Prepared samples
├── results/                     # Analysis outputs
├── visualizations/              # Generated plots
└── logs/                        # Training logs
```

---

## 🔮 Research Significance

This work provides:

1. **First mechanistic analysis** of gender bias in multilingual VLMs
2. **Novel CLMB framework** for cross-lingual bias study
3. **Surgical intervention techniques** that could debias without retraining
4. **Insights into Arabic-specific** morphological gender encoding

---

## 📚 Key References

1. Elhage et al. "Toy Models of Superposition" - Feature interaction theory
2. Geva et al. "Transformer Interpretability Beyond Attention Visualization" - Logit lens
3. Hewitt & Liang "Designing and Interpreting Probes with Control Tasks" - Probing methodology
4. Cunningham et al. "Sparse Autoencoders Find Highly Interpretable Features" - SAE methodology

---

## 📅 Timeline & Milestones

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1 | Data preparation & activation extraction | ✅ Complete |
| Phase 2 | SAE training (all layers, both languages) | 🔄 In Progress |
| Phase 3 | Feature analysis & probing | ⏳ Pending |
| Phase 4 | Steering experiments | ⏳ Pending |
| Phase 5 | Mechanistic analysis & visualization | ⏳ Pending |
| Phase 6 | Paper writing & documentation | ⏳ Pending |

---

*Generated: January 9, 2026*
