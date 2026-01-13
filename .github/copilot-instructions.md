# Cross-Lingual SAE Mechanistic Interpretability - Copilot Instructions

## Project Overview
Sparse Autoencoder (SAE) analysis pipeline for mechanistic interpretability of Gemma-3-4B vision-language model, investigating cross-lingual gender bias in Arabic-English image captioning.

## Architecture
```
sae_captioning_project/
├── src/
│   ├── models/          # SAE architecture (sae.py, hooks.py)
│   ├── clmb/            # Novel CLMB framework components
│   │   ├── hbl.py       # Hierarchical Bias Localization
│   │   ├── clfa.py      # Cross-Lingual Feature Alignment (optimal transport)
│   │   └── sbi.py       # Surgical Bias Intervention
│   ├── mechanistic/     # ViT-Prisma integration (LogitLens, probes)
│   └── analysis/        # Feature analysis and metrics
├── scripts/             # Pipeline stages 01-23 + SLURM jobs
├── configs/config.yaml  # Central configuration
└── checkpoints/         # Activations (~30GB) and trained SAEs
```

## Key Patterns

### SAE Configuration
Always use `SAEConfig` from `src/models/sae.py`:
```python
from src.models.sae import SAEConfig, SparseAutoencoder
config = SAEConfig(d_model=2048, expansion_factor=8, l1_coefficient=5e-4)
```

### Activation Loading
Use the pattern in `scripts/03_train_sae.py` - supports both merged files and chunked checkpoints:
```python
# Merged: checkpoints/full_layers_ncc/activations_{language}.pt
# Chunks: checkpoints/full_layers_ncc/layer_checkpoints/layer_{N}_{lang}_*.pt
```

### Layer Indexing
Gemma-3-4B has 42 layers (0-41). Primary analysis layers: `[0, 3, 6, 9, 12, 15, 17]`. Extract via hook names: `model.model.layers[{layer_idx}]`

### Cross-Lingual Analysis
When comparing Arabic/English, always process both languages in the same script to ensure paired samples. Use `CrossLingualFeatureAligner` from `src/clmb/clfa.py`.

## Critical Workflows

### Running on NCC Cluster
```bash
# Full automated pipeline with SLURM dependencies
bash scripts/slurm_00_full_pipeline.sh

# Individual steps (dependencies handled automatically)
sbatch scripts/slurm_02_parallel_extraction.sh
sbatch scripts/slurm_03_train_all_saes.sh
```

### Local Development
```bash
pip install -e . --break-system-packages
python scripts/run_full_pipeline.py --config configs/config.yaml
```

### Memory Management
- Use `torch.float32` for stability (float16 causes NaN issues with this model)
- Process batch_size=1 for extraction (Gemma-3 processor limitation)
- Clear GPU memory between layers: `gc.collect(); torch.cuda.empty_cache()`

## Novel CLMB Framework Components
1. **HBL** (`src/clmb/hbl.py`): Bias Attribution Score decomposing Vision→Projection→Language contributions
2. **CLFA** (`src/clmb/clfa.py`): Optimal transport alignment of SAE features across languages
3. **SBI** (`src/clmb/sbi.py`): Ablation, neutralization, and amplification interventions
4. **CLBAS**: Cross-Lingual Bias Alignment Score metric (see `RESEARCH_PLAN.md`)

## Configuration
All parameters in `configs/config.yaml`. Key sections:
- `model.name`: Currently `google/paligemma-3b-pt-224`
- `sae.expansion_factor`: 8× (2048→16384 hidden dims)
- `layers.extraction`: Which layers to analyze

## File Naming Conventions
- Scripts: `{NN}_{name}.py` (numbered pipeline order)
- SLURM: `slurm_{NN}_{name}.sh` 
- Checkpoints: `layer_{N}_{language}_chunk_{M}.pt`
- SAE models: `checkpoints/saes/layer_{N}/sae_final.pt`

## Dependencies
Requires: `torch>=2.1.0`, `transformers>=4.40.0`, `camel-tools` (Arabic NLP), `POT` (optimal transport)
