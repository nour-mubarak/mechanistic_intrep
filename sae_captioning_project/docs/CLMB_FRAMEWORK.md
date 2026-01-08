# CLMB: Cross-Lingual Multimodal Bias Framework

## Novel Methodological Contribution

This framework introduces a novel approach to understanding and mitigating gender bias in multilingual Vision-Language Models (VLMs) through mechanistic interpretability.

## Key Innovation

**CLMB addresses three limitations of existing bias research:**

1. **Where does bias enter?** - Existing work treats VLMs as black boxes
2. **Is bias language-universal?** - Do Arabic and English share bias circuits?
3. **Can we surgically fix it?** - Without retraining the entire model

## Framework Components

### 1. HBL: Hierarchical Bias Localization

**Purpose**: Identify which model component (Vision, Projection, or Language) contributes most to gender bias.

**Method**:
```
Image → [Vision Encoder] → [Projection Layer] → [Language Model] → Caption
              ↓                    ↓                   ↓
         V-Bias Score        P-Bias Score        L-Bias Score
```

**Metric - Bias Attribution Score (BAS)**:
$$BAS_{component} = \sum_i |a_i^{male} - a_i^{female}| \times w_i$$

Where:
- $a_i^{male/female}$ = mean activation for gender-specific samples
- $w_i$ = feature importance from SAE analysis

### 2. CLFA: Cross-Lingual Feature Alignment

**Purpose**: Discover which SAE features encode the same concepts in Arabic vs English.

**Method**:
- Use Wasserstein distance to measure distribution similarity between features
- Apply optimal transport to find minimal-cost feature matching
- Identify language-specific vs shared features

**Alignment via Optimal Transport**:
$$\gamma^* = \arg\min_\gamma \sum_{i,j} C_{ij} \gamma_{ij}$$

Subject to: marginal constraints ensuring proper coupling

### 3. SBI: Surgical Bias Intervention

**Purpose**: Mitigate bias through targeted SAE feature modification.

**Intervention Types**:
1. **Ablation**: Set bias features to zero
   $$f'_i = f_i \times \mathbb{1}[i \notin \text{bias\_features}]$$

2. **Neutralization**: Average male/female feature values
   $$f'_i = \frac{f_i^{male} + f_i^{female}}{2}$$

3. **Amplification**: Boost fairness-promoting features
   $$f'_i = \alpha \times f_i$$ for $i \in \text{fair\_features}$

### 4. CLBAS: Cross-Lingual Bias Alignment Score

**Purpose**: Novel metric measuring how consistently bias manifests across languages.

**Formula**:
$$CLBAS = \frac{\sum_{(f_{ar}, f_{en}) \in \text{aligned}} |bias(f_{ar}) - bias(f_{en})| \times sim(f_{ar}, f_{en})}{\sum_{(f_{ar}, f_{en}) \in \text{aligned}} sim(f_{ar}, f_{en})}$$

**Interpretation**:
- **Low CLBAS** (→ 0): Consistent bias across languages (same stereotypes)
- **High CLBAS** (→ 1): Language-specific bias (different stereotypes)

## Research Questions Addressed

1. **RQ1**: Where in VLM architecture does gender bias predominantly emerge?
   - Answer: HBL analysis across Vision/Projection/Language

2. **RQ2**: Do bias representations align across Arabic and English?
   - Answer: CLFA alignment scores + CLBAS metric

3. **RQ3**: Can we surgically mitigate bias while preserving caption quality?
   - Answer: SBI intervention with semantic preservation tracking

## Multi-Model Comparative Study

### Supported Models (Memory-Efficient for NCC)

| Model | Category | Memory | Arabic Support |
|-------|----------|--------|----------------|
| PaLiGemma-3B | English | 12GB | ✗ |
| BLIP-2-OPT-2.7B | English | 10GB | ✗ |
| Qwen-VL-Chat | Multilingual | 18GB | ✓ |
| mBLIP-mT0-XL | Multilingual | 12GB | ✓ |
| Peacock-7B | Arabic | 16GB | ✓ |
| Arabic-BLIP | Arabic | 3GB | ✓ |

### Comparative Analysis Framework

```python
from clmb import CLMBFramework, MultiModelExtractor

# Analyze multiple models
models = ["google/paligemma-3b-pt-224", "Qwen/Qwen-VL-Chat", "UBC-NLP/Peacock"]

results = {}
for model_id in models:
    framework = CLMBFramework(model_name=model_id)
    result = framework.analyze(images, english_captions, arabic_captions, genders)
    results[model_id] = result
    
# Compare CLBAS across models
for model_id, result in results.items():
    print(f"{model_id}: CLBAS={result.clbas:.4f}, Dominant={result.dominant_bias_source}")
```

## Expected Contributions

1. **Novel Framework**: First mechanistic interpretability approach to VLM bias
2. **New Metric (CLBAS)**: Quantifies cross-lingual bias alignment
3. **Interpretable Interventions**: Surgical bias mitigation without retraining
4. **Arabic-English Study**: Understudied language pair in VLM research

## File Structure

```
src/clmb/
├── __init__.py           # Main CLMBFramework class
├── hbl.py                # Hierarchical Bias Localization
├── clfa.py               # Cross-Lingual Feature Alignment  
├── sbi.py                # Surgical Bias Intervention
├── extractors.py         # Multi-model activation extractors
└── models.py             # Model registry and configs

scripts/
├── 19_clmb_analysis.py   # Main analysis script
└── slurm_19_clmb_analysis.sh  # SLURM job script

configs/
└── clmb_config.yaml      # Configuration file
```

## Running the Analysis

```bash
# After SAE training completes
sbatch scripts/slurm_19_clmb_analysis.sh

# Or run directly
python scripts/19_clmb_analysis.py --config configs/clmb_config.yaml
```

## Citation

If you use CLMB in your research, please cite:
```bibtex
@inproceedings{clmb2025,
  title={CLMB: Cross-Lingual Multimodal Bias Analysis via Mechanistic Interpretability},
  author={...},
  booktitle={...},
  year={2025}
}
```

## Status

- [x] HBL implementation
- [x] CLFA implementation  
- [x] SBI implementation
- [x] Multi-model extractors
- [x] CLBAS metric
- [ ] Full evaluation (awaiting SAE training)
- [ ] Multi-model comparison
- [ ] Paper results
