# Cross-Lingual SAE Mechanistic Interpretability Pipeline
## Complete Documentation for Model-Agnostic Analysis

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Core Components](#core-components)
4. [Scripts Reference](#scripts-reference)
5. [Configuration Files](#configuration-files)
6. [Data Flow](#data-flow)
7. [Results & Findings](#results--findings)
8. [Adapting for New Models](#adapting-for-new-models)
9. [Key Metrics & Methods](#key-metrics--methods)

---

## 🎯 Project Overview

### Research Question
**Do multilingual vision-language models use the same or different internal features to encode gender across languages (Arabic vs English)?**

### Key Findings (PaLiGemma-3B)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Feature Overlap | 0.48% | Near-zero shared features |
| CLBAS Score | 0.025 | No cross-lingual alignment |
| Arabic Baseline Accuracy | 88-90% | Strong gender encoding |
| English Baseline Accuracy | 83-86% | Moderate gender encoding |
| Ablation Effect | <0.3% | Gender is distributed |

### Novel Contributions
1. **CLBAS Metric**: Cross-Lingual Bias Alignment Score
2. **Language-Specific Gender Circuits**: Different features per language
3. **Distributed Gender Encoding**: Not concentrated in top-k features
4. **SBI Framework**: Surgical Bias Intervention for causal analysis

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE STAGES                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STAGE 1: DATA PREPARATION                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 01_prepare_data.py                                           │   │
│  │ - Load image-caption pairs                                   │   │
│  │ - Extract gender labels from captions                        │   │
│  │ - Create train/val splits                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  STAGE 2: ACTIVATION EXTRACTION                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 02_extract_activations.py (per language)                     │   │
│  │ 18_extract_full_activations_ncc.py (Arabic)                  │   │
│  │ 22_extract_english_activations.py (English)                  │   │
│  │ - Load VLM model                                             │   │
│  │ - Hook into transformer layers                               │   │
│  │ - Save activations per layer (~22GB Arabic, ~79MB English)   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  STAGE 3: SAE TRAINING                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 03_train_sae.py (per language, per layer)                    │   │
│  │ - Train Sparse Autoencoder (2048 → 16384)                    │   │
│  │ - L1 regularization for sparsity                             │   │
│  │ - Save model weights and training history                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  STAGE 4: FEATURE ANALYSIS                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 04_analyze_features.py                                       │   │
│  │ - Compute effect sizes (Cohen's d)                           │   │
│  │ - Identify gender-associated features                        │   │
│  │ - Statistical tests                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  STAGE 5: CROSS-LINGUAL ANALYSIS                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 24_cross_lingual_overlap.py                                  │   │
│  │ - Compare Arabic vs English feature spaces                   │   │
│  │ - Compute CLBAS metric                                       │   │
│  │ - Feature overlap statistics                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  STAGE 6: SURGICAL BIAS INTERVENTION                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 26_surgical_bias_intervention.py                             │   │
│  │ - Ablate top-k features                                      │   │
│  │ - Measure probe accuracy drop                                │   │
│  │ - Cross-lingual ablation tests                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  STAGE 7: STATISTICAL VALIDATION                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 27_statistical_significance.py                               │   │
│  │ - Bootstrap confidence intervals                             │   │
│  │ - Permutation tests                                          │   │
│  │ - P-values for all findings                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Core Components

### Source Code (`src/`)

#### `src/models/sae.py` - Sparse Autoencoder
```python
class SAEConfig:
    d_model: int = 2048        # Input dimension (model hidden size)
    expansion_factor: int = 8   # Hidden = d_model * expansion_factor
    l1_coefficient: float = 5e-4
    
class SparseAutoencoder:
    def encode(x) -> features    # x → sparse features
    def decode(features) -> x    # features → reconstruction
```

#### `src/models/hooks.py` - Activation Hooks
```python
class ActivationHook:
    # Hooks into transformer layers to capture activations
    # Supports: model.model.layers[i].output
```

#### `src/clmb/` - Cross-Lingual Mechanistic Bias Framework
- `hbl.py`: Hierarchical Bias Localization
- `clfa.py`: Cross-Lingual Feature Alignment (Optimal Transport)
- `sbi.py`: Surgical Bias Intervention
- `extractors.py`: Model-agnostic activation extraction

#### `src/analysis/`
- `features.py`: Feature analysis utilities
- `metrics.py`: CLBAS, effect sizes, overlap metrics

---

## 📜 Scripts Reference

### Essential Scripts (Ordered Pipeline)

| # | Script | Purpose | Input | Output |
|---|--------|---------|-------|--------|
| 01 | `01_prepare_data.py` | Prepare dataset | Raw captions + images | `data/processed/samples.csv` |
| 02 | `02_extract_activations.py` | Extract activations (generic) | Model + images | `checkpoints/activations/` |
| 18 | `18_extract_full_activations_ncc.py` | Extract Arabic activations (NCC) | PaLiGemma + Arabic captions | `checkpoints/full_layers_ncc/` |
| 22 | `22_extract_english_activations.py` | Extract English activations | PaLiGemma + English captions | `checkpoints/full_layers_ncc/` |
| 03 | `03_train_sae.py` | Train SAE | Activations | `checkpoints/saes/` |
| 04 | `04_analyze_features.py` | Analyze features | SAE + activations | `results/feature_stats_*.csv` |
| 24 | `24_cross_lingual_overlap.py` | Cross-lingual comparison | Arabic + English SAEs | `results/cross_lingual_overlap/` |
| 25 | `25_cross_lingual_feature_interpretation.py` | Interpret features | SAEs + activations | `results/feature_interpretation/` |
| 26 | `26_surgical_bias_intervention.py` | Causal ablation tests | SAEs + activations | `results/sbi_analysis/` |
| 27 | `27_statistical_significance.py` | Statistical tests | All results | `results/statistical_tests/` |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `merge_checkpoints.py` | Merge chunked activation files |
| `consolidate_results.py` | Consolidate analysis results |
| `run_full_pipeline.py` | Run entire pipeline locally |

### SLURM Job Scripts

| Script | Purpose |
|--------|---------|
| `slurm_00_full_pipeline.sh` | Full automated pipeline |
| `slurm_02_parallel_extraction.sh` | Parallel activation extraction |
| `slurm_03_train_all_saes.sh` | Train all SAEs |
| `slurm_24_cross_lingual_overlap.sh` | Cross-lingual analysis |
| `slurm_26_sbi_array.sh` | SBI with array jobs |
| `slurm_27_statistical_tests.sh` | Statistical validation |

---

## ⚙️ Configuration Files

### `configs/config.yaml` - Main Configuration
```yaml
model:
  name: "google/paligemma-3b-pt-224"
  device: "cuda"
  dtype: "float32"

sae:
  d_model: 2048
  expansion_factor: 8
  l1_coefficient: 0.0005
  learning_rate: 0.0003
  batch_size: 256
  epochs: 50

layers:
  extraction: [0, 3, 6, 9, 12, 15, 17]
  analysis: [0, 3, 6, 9, 12, 15, 17]

languages:
  - arabic
  - english
```

### `configs/clmb_config.yaml` - CLMB Framework Config
```yaml
clmb:
  optimal_transport:
    reg: 0.1
    num_iterations: 100
  
  sbi:
    k_values: [10, 25, 50, 100, 200]
    intervention_types: [ablation, neutralization]
```

---

## 📊 Data Flow

```
data/raw/
├── captions.csv          # Columns: image_id, arabic_caption, english_caption
└── images/               # COCO-style images

data/processed/
├── samples.csv           # Processed with gender labels
└── data_summary.yaml     # Statistics

checkpoints/
├── full_layers_ncc/
│   └── layer_checkpoints/
│       ├── layer_0_arabic.pt    # ~22GB per layer
│       ├── layer_0_english.pt   # ~79MB per layer
│       └── ...
└── saes/
    ├── sae_arabic_layer_0.pt    # ~256MB each
    ├── sae_arabic_layer_0_history.json
    ├── sae_english_layer_0.pt
    └── ...

results/
├── cross_lingual_overlap/
│   └── cross_lingual_overlap_results.json
├── feature_interpretation/
│   └── feature_interpretation_results.json
├── sbi_analysis/
│   ├── sbi_results.json
│   └── visualizations/
├── statistical_tests/
│   └── statistical_significance_results.json
└── feature_stats_layer_*.csv
```

---

## 📈 Results & Findings

### Cross-Lingual Overlap Results
```json
{
  "mean_clbas": 0.025,
  "mean_overlap_pct": 0.48,
  "layers": {
    "0": {"clbas": 0.013, "overlap": 0.0},
    "3": {"clbas": 0.011, "overlap": 0.33},
    "6": {"clbas": 0.015, "overlap": 0.33},
    "9": {"clbas": 0.028, "overlap": 2.0},
    "12": {"clbas": 0.039, "overlap": 0.33},
    "15": {"clbas": 0.028, "overlap": 0.33},
    "17": {"clbas": 0.041, "overlap": 0.0}
  }
}
```

### SBI Results
- Same-language ablation: 0-0.3% accuracy drop
- Cross-language ablation: 0% accuracy drop (confirms language specificity)
- Reconstruction quality: >0.99 (semantic preservation)

### Probe Accuracies
| Language | Mean Accuracy | Std |
|----------|---------------|-----|
| Arabic | 88.5% | 0.5% |
| English | 85.1% | 1.5% |

---

## 🔄 Adapting for New Models

### Step 1: Model Configuration
Modify `configs/config.yaml`:
```yaml
model:
  name: "meta-llama/Llama-3.2-11B-Vision"  # Or any HF model
  device: "cuda"
  dtype: "bfloat16"  # Adjust based on model
  
sae:
  d_model: 4096  # Match model hidden size
  expansion_factor: 8
```

### Step 2: Activation Extraction Hook
Modify `src/models/hooks.py` for new model architecture:

```python
# For LLaMA-style models:
hook_name = f"model.layers.{layer_idx}"

# For CLIP-style models:
hook_name = f"vision_model.encoder.layers.{layer_idx}"

# For Gemma-style models:
hook_name = f"model.model.layers.{layer_idx}"
```

### Step 3: Create Model-Specific Extraction Script
Copy `18_extract_full_activations_ncc.py` and modify:

```python
# Change model loading
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(...)

# Change hook registration
def get_hook_name(layer_idx):
    return f"model.layers.{layer_idx}"  # Model-specific
```

### Step 4: Adjust SAE Architecture
If model has different hidden size:
```python
config = SAEConfig(
    d_model=4096,  # LLaMA hidden size
    expansion_factor=8,  # → 32768 features
    l1_coefficient=5e-4
)
```

### Step 5: Run Pipeline
```bash
# 1. Extract activations
python scripts/02_extract_activations.py --model meta-llama/Llama-3.2-11B-Vision

# 2. Train SAEs
python scripts/03_train_sae.py --layers 0,8,16,24,32

# 3. Analyze
python scripts/24_cross_lingual_overlap.py
python scripts/26_surgical_bias_intervention.py
python scripts/27_statistical_significance.py
```

---

## 📐 Key Metrics & Methods

### CLBAS (Cross-Lingual Bias Alignment Score)
```python
def compute_clbas(ar_effect_sizes, en_effect_sizes):
    cosine = cosine_similarity(ar_effect_sizes, en_effect_sizes)
    spearman = spearmanr(ar_effect_sizes, en_effect_sizes)
    pearson = pearsonr(ar_effect_sizes, en_effect_sizes)
    return (abs(cosine) + abs(spearman) + abs(pearson)) / 3
```
- Range: 0-1
- 0 = No alignment (languages use different features)
- 1 = Perfect alignment (same features)

### Cohen's d Effect Size
```python
def cohens_d(male_vals, female_vals):
    pooled_std = sqrt((std(male)**2 + std(female)**2) / 2)
    return (mean(male) - mean(female)) / pooled_std
```
- |d| < 0.2: Small effect
- |d| 0.2-0.5: Medium effect
- |d| > 0.8: Large effect

### Feature Overlap
```python
def compute_overlap(ar_top_k, en_top_k, k=100):
    overlap = len(set(ar_top_k[:k]) & set(en_top_k[:k]))
    jaccard = overlap / (2*k - overlap)
    return overlap, jaccard
```

### Surgical Ablation
```python
def ablate(features, indices):
    features[:, indices] = 0
    return features

# Same-language: Ablate Arabic features, test on Arabic
# Cross-language: Ablate English features, test on Arabic (should have no effect)
```

### Bootstrap Confidence Intervals
```python
def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    samples = [np.random.choice(data, len(data), replace=True).mean() 
               for _ in range(n_bootstrap)]
    return np.percentile(samples, [2.5, 97.5])
```

---

## 📁 File Checklist for New Model

### Must Have
- [ ] `configs/config.yaml` - Updated for new model
- [ ] `src/models/sae.py` - SAE with correct d_model
- [ ] `src/models/hooks.py` - Hooks for new architecture
- [ ] Extraction script adapted for model

### Should Have
- [ ] SLURM scripts adjusted for memory/time
- [ ] Test run on single layer before full pipeline

### Nice to Have
- [ ] W&B integration for tracking
- [ ] Model-specific visualization scripts

---

## 🚀 Quick Start for New Model

```bash
# 1. Clone project
git clone https://github.com/nour-mubarak/mechanistic_intrep.git
cd mechanistic_intrep/sae_captioning_project

# 2. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure for new model
# Edit configs/config.yaml

# 4. Run pipeline
bash scripts/slurm_00_full_pipeline.sh  # Or run_full_pipeline.py locally
```

---

## 📚 References

- Sparse Autoencoders: Cunningham et al. (2023)
- Mechanistic Interpretability: Elhage et al. (2022)
- ViT-Prisma: [GitHub](https://github.com/soniajoseph/ViT-Prisma)
- Cross-Lingual NLP: Pires et al. (2019)

---

*Documentation generated: January 16, 2026*
*Model: PaLiGemma-3B-pt-224*
*Framework: Cross-Lingual Mechanistic Bias (CLMB)*
