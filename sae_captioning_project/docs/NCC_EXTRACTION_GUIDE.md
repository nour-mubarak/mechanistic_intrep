# Full Layer Activation Extraction with NCC

**Created**: January 5, 2026
**Model**: google/gemma-3-4b-it (34 layers, 0-33)
**Dataset**: 2000 samples (full dataset)
**Methodology**: Neural Corpus Compilation (NCC)

---

## Overview

This guide documents the full-layer activation extraction process using Neural Corpus Compilation (NCC) methodology. The extraction covers **all 34 layers** of Gemma-3-4B on the **full 2000-sample dataset** for both English and Arabic prompts.

### What is NCC (Neural Corpus Compilation)?

Neural Corpus Compilation is a methodology for efficient large-scale activation extraction that emphasizes:

1. **Streaming Processing**: Activations are processed in a streaming fashion to minimize memory usage
2. **Layer-wise Storage**: Each layer's activations are saved separately for parallel processing
3. **Aggressive Memory Management**: GPU memory is cleared after each layer extraction
4. **Recovery from Failures**: Layer-wise checkpoints enable resuming from interruptions
5. **Efficient Tensor Storage**: Optimized storage formats for downstream processing

### Why Full Layer Extraction?

Previous analyses focused on layers 10, 14, 18, and 22. However, comprehensive mechanistic interpretability requires understanding how representations evolve across **all layers**:

- **Early Layers (0-10)**: Vision encoding and basic feature extraction
- **Middle Layers (11-20)**: Cross-modal alignment and concept formation
- **Deep Layers (21-33)**: High-level reasoning and output generation

Full layer extraction enables:
- Fine-grained analysis of representation evolution
- Identification of critical transition points
- Complete understanding of bias emergence and propagation

---

## Quick Start

### Basic Usage

Extract all 34 layers on the full dataset:

```bash
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml
```

### Alternative: Use the Shell Script

```bash
./scripts/run_full_extraction.sh
```

### Custom Layer Ranges

Extract specific layer ranges (e.g., early and late layers):

```bash
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 0-10 24-33
```

Extract individual layers:

```bash
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --layer-ranges 0 5 10 15 20 25 30 33
```

### Adjust Batch Size

For GPUs with different memory (default: 1 for 24GB):

```bash
# Larger GPU (40GB+)
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --batch-size 2

# Smaller GPU (16GB)
python scripts/18_extract_full_activations_ncc.py \
    --config configs/config.yaml \
    --batch-size 1 \
    --checkpoint-interval 25
```

---

## Technical Specifications

### Extraction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--layer-ranges` | 0-33 (all) | Layer ranges to extract |
| `--batch-size` | 1 | Samples per batch |
| `--checkpoint-interval` | 50 | Save every N samples |
| `--languages` | english, arabic | Languages to process |
| `--output-dir` | checkpoints/full_layers_ncc | Output directory |

### System Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (tested on 23.65GB)
- **RAM**: 32GB+ recommended
- **Disk Space**: ~15GB per language (34 layers × 2000 samples)
- **Processing Time**: ~45-60 minutes per language

### Memory Management

The NCC implementation uses several strategies to minimize memory usage:

1. **Immediate CPU Transfer**: Activations are moved to CPU immediately after extraction
2. **Detached Tensors**: Prevents gradient graph buildup
3. **Aggressive Cache Clearing**: `torch.cuda.empty_cache()` after each layer
4. **Batch Processing**: Small batch size (1) for stability
5. **Chunked Saves**: Checkpoint every 50 samples to prevent accumulation

---

## Output Structure

### Directory Layout

```
checkpoints/full_layers_ncc/
├── extraction_metadata.json          # Extraction configuration and stats
├── activations_english_all_layers.pt # Combined English activations
├── activations_arabic_all_layers.pt  # Combined Arabic activations
└── layer_checkpoints/                # Layer-wise storage (NCC)
    ├── layer_0_english.pt
    ├── layer_0_arabic.pt
    ├── layer_1_english.pt
    ├── layer_1_arabic.pt
    ...
    ├── layer_33_english.pt
    └── layer_33_arabic.pt
```

### File Formats

#### Combined File (e.g., `activations_english_all_layers.pt`)

```python
{
    'activations': {
        0: torch.Tensor,    # Shape: [2000, seq_len, 2560]
        1: torch.Tensor,
        ...
        33: torch.Tensor
    },
    'genders': List[str],           # ['male', 'female', ...]
    'image_ids': List[str],         # ['000000000139', ...]
    'layers': List[int],            # [0, 1, 2, ..., 33]
    'hidden_size': 2560,
    'timestamp': '2026-01-05T...',
    'model': 'google/gemma-3-4b-it',
    'num_layers_extracted': 34
}
```

#### Layer Checkpoint (e.g., `layer_10_english.pt`)

```python
{
    'layer': 10,
    'activations': torch.Tensor,    # Shape: [2000, seq_len, 2560]
    'genders': List[str],
    'image_ids': List[str],
    'hidden_size': 2560,
    'model': 'google/gemma-3-4b-it',
    'timestamp': '2026-01-05T...'
}
```

#### Metadata File (`extraction_metadata.json`)

```json
{
    "model": "google/gemma-3-4b-it",
    "hidden_size": 2560,
    "num_layers": 34,
    "layers_extracted": [0, 1, 2, ..., 33],
    "num_samples": 2000,
    "batch_size": 1,
    "checkpoint_interval": 50,
    "extraction_date": "2026-01-05T...",
    "methodology": "NCC (Neural Corpus Compilation)"
}
```

---

## Usage Examples

### Load All Layers (Combined File)

```python
import torch

# Load combined file
data = torch.load('checkpoints/full_layers_ncc/activations_english_all_layers.pt')

# Access specific layer
layer_10_activations = data['activations'][10]  # Shape: [2000, seq_len, 2560]

# Get metadata
genders = data['genders']
image_ids = data['image_ids']

print(f"Layer 10 shape: {layer_10_activations.shape}")
print(f"Total samples: {len(genders)}")
```

### Load Single Layer (Layer Checkpoint)

```python
import torch

# Load specific layer
layer_data = torch.load('checkpoints/full_layers_ncc/layer_checkpoints/layer_22_english.pt')

activations = layer_data['activations']
genders = layer_data['genders']

print(f"Layer 22 shape: {activations.shape}")
```

### Compare Across All Layers

```python
import torch
from pathlib import Path

checkpoint_dir = Path('checkpoints/full_layers_ncc/layer_checkpoints')

# Analyze specific sample across all layers
sample_idx = 0

layer_norms = {}
for layer in range(34):
    layer_file = checkpoint_dir / f'layer_{layer}_english.pt'
    if layer_file.exists():
        data = torch.load(layer_file)
        act = data['activations'][sample_idx]  # Shape: [seq_len, 2560]
        layer_norms[layer] = torch.norm(act, dim=-1).mean().item()

# Plot evolution
import matplotlib.pyplot as plt
plt.plot(list(layer_norms.keys()), list(layer_norms.values()))
plt.xlabel('Layer')
plt.ylabel('Average Activation Norm')
plt.title('Activation Magnitude Across Layers')
plt.savefig('layer_evolution.png')
```

---

## NCC Methodology Details

### 1. Streaming Architecture

```python
def extract_activations_batch_ncc(model, processor, images, prompts, layers, device):
    """
    NCC Key Principle: Stream activations directly to CPU to minimize GPU memory.
    """
    activations = {layer: [] for layer in layers}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Extract activation
            act = output[0] if isinstance(output, tuple) else output

            # NCC: Immediate detach + CPU transfer + float32 conversion
            act = act.detach().cpu().float()

            activations[layer_idx].append(act)

            # NCC: Clear GPU cache after each layer
            torch.cuda.empty_cache()

        return hook

    # Register hooks, forward pass, remove hooks
    # ...
```

### 2. Layer-wise Storage

Each layer is saved independently:

```python
def save_layer_checkpoint(layer_idx, activations, genders, image_ids, checkpoint_dir, language, metadata):
    """
    NCC: Save each layer separately for:
    - Parallel downstream processing
    - Recovery from failures
    - Selective layer loading
    """
    checkpoint_path = checkpoint_dir / f'layer_{layer_idx}_{language}.pt'

    torch.save({
        'layer': layer_idx,
        'activations': activations,
        'genders': genders,
        'image_ids': image_ids,
        **metadata
    }, checkpoint_path)
```

### 3. Checkpointing Strategy

Activations are accumulated in chunks and saved periodically:

```python
# Process samples in batches
for batch_idx in range(0, len(dataset), batch_size):
    # Extract batch
    batch_activations = extract_activations_batch_ncc(...)

    # Accumulate
    chunk_activations[layer].append(batch_activations[layer])
    samples_processed += batch_size

    # Save checkpoint every N samples
    if samples_processed >= checkpoint_interval:
        # Save current chunk
        # Clear chunk data
        # Free memory
```

### 4. Memory Optimization

The implementation includes several aggressive memory management techniques:

```python
# After each batch
del inputs
gc.collect()
torch.cuda.empty_cache()

# After each checkpoint
del stacked_chunk
gc.collect()
torch.cuda.empty_cache()

# Periodic cleanup
if batch_idx % (batch_size * 5) == 0:
    gc.collect()
    torch.cuda.empty_cache()
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: CUDA OOM during extraction

**Solutions**:
1. Reduce batch size to 1 (default)
2. Decrease checkpoint interval: `--checkpoint-interval 25`
3. Extract fewer layers at a time: `--layer-ranges 0-16`
4. Clear GPU memory: `nvidia-smi` and kill other processes

### NaN Values in Activations

**Symptoms**: "NaN detected in layer X activations!"

**Solutions**:
- Activations are automatically converted to zeros (warning logged)
- Checkpoint is skipped if NaN values persist
- Check model dtype (should be float32)

### Slow Extraction Speed

**Symptoms**: <0.5 samples/sec

**Expected Speed**: ~0.8-1.2 samples/sec for batch_size=1

**Solutions**:
1. Check GPU utilization: `nvidia-smi`
2. Ensure no other processes using GPU
3. Verify disk I/O speed (SSD recommended)

### Incomplete Extraction

**Symptoms**: Missing layer checkpoint files

**Recovery**:
- NCC enables resuming from last checkpoint
- Check `extraction_metadata.json` for expected layers
- Re-run with `--layer-ranges` for missing layers only

---

## Performance Benchmarks

### Tested Configuration
- **GPU**: NVIDIA GPU with 23.65 GB memory
- **Model**: google/gemma-3-4b-it
- **Dataset**: 2000 samples
- **Batch Size**: 1

### Expected Performance

| Metric | Value |
|--------|-------|
| Samples/sec | 0.8-1.2 |
| Time per language | 45-60 min |
| Total time (2 languages) | 90-120 min |
| GPU memory usage | 18-22 GB |
| Disk usage per language | ~15 GB |

### Layer-wise Extraction Speed

Early layers (0-10): Faster (~1.2 samples/sec)
Middle layers (11-20): Moderate (~1.0 samples/sec)
Deep layers (21-33): Slower (~0.8 samples/sec)

---

## Integration with Downstream Analysis

### SAE Training

Train SAEs on extracted activations:

```bash
# Train SAE for layer 10
python scripts/03_train_sae.py \
    --config configs/config.yaml \
    --layer 10 \
    --activations checkpoints/full_layers_ncc/layer_checkpoints/layer_10_english.pt

# Train SAEs for all layers (batch script)
for layer in {0..33}; do
    python scripts/03_train_sae.py \
        --config configs/config.yaml \
        --layer $layer \
        --activations checkpoints/full_layers_ncc/layer_checkpoints/layer_${layer}_english.pt
done
```

### Comprehensive Analysis

Run comprehensive analysis across all layers:

```bash
python scripts/09_comprehensive_analysis.py \
    --config configs/config.yaml \
    --layers $(seq 0 33)  # All 34 layers
```

### Fine-grained Cross-Layer Analysis

Compare adjacent layers to identify transition points:

```bash
python scripts/16_cross_layer_analysis.py \
    --config configs/config.yaml \
    --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 33
```

---

## Scientific Applications

### 1. Layer-by-Layer Bias Evolution

Analyze how gender bias emerges and evolves through all 34 layers:

```python
# Extract gender differentials for all layers
differentials = {}
for layer in range(34):
    # Load layer activations
    # Compute gender-specific feature importance
    # Store differential

# Plot evolution
plt.plot(range(34), list(differentials.values()))
plt.xlabel('Layer')
plt.ylabel('Gender Differential')
plt.title('Gender Bias Evolution Across All Layers')
```

### 2. Critical Transition Identification

Identify layers where representations undergo major changes:

```python
# Compare adjacent layer representations
transition_scores = []
for layer in range(33):
    act_l = load_layer(layer)
    act_l1 = load_layer(layer + 1)

    # Compute representational similarity
    similarity = compute_cka(act_l, act_l1)
    transition_scores.append(similarity)

# Low similarity = major transition
critical_layers = [i for i, s in enumerate(transition_scores) if s < threshold]
```

### 3. Multi-Resolution Analysis

Compare coarse-grained (8 layers) vs fine-grained (34 layers) analysis:

- Validate findings from layers 10, 14, 18, 22
- Identify missed patterns in intermediate layers
- Optimize layer selection for future studies

---

## Weights & Biases Integration

The extraction automatically logs to W&B if enabled:

```yaml
# configs/config.yaml
logging:
  use_wandb: true
  wandb_project: "sae-captioning-bias"
  wandb_entity: "nourmubarak"
```

### Logged Metrics

- `{language}/samples_processed`: Current sample count
- `{language}/samples_per_sec`: Extraction speed
- `{language}/progress`: Percentage complete
- `{language}/gpu_memory_allocated_gb`: GPU memory usage
- `{language}/layer_{X}_shape`: Shape of layer X activations
- `{language}/gender_{male/female}_count`: Gender distribution

---

## Future Extensions

### 1. Token-level NCC

Extract activations at token granularity for fine-grained analysis:

```python
# Current: [num_samples, seq_len, hidden_size]
# Future: Store token-level metadata
{
    'activations': torch.Tensor,
    'token_ids': List[List[int]],
    'token_positions': List[List[int]],
    'attention_masks': List[List[bool]]
}
```

### 2. Distributed Extraction

Parallelize across multiple GPUs:

```bash
# GPU 0: Layers 0-16
CUDA_VISIBLE_DEVICES=0 python scripts/18_extract_full_activations_ncc.py \
    --layer-ranges 0-16 &

# GPU 1: Layers 17-33
CUDA_VISIBLE_DEVICES=1 python scripts/18_extract_full_activations_ncc.py \
    --layer-ranges 17-33 &
```

### 3. Compression

Apply compression to reduce disk usage:

```python
# Quantization: float32 -> float16
activations = activations.half()

# Sparse storage: only save top-k activations per token
top_k_values, top_k_indices = torch.topk(activations, k=32, dim=-1)
```

---

## References

1. **Neural Corpus Compilation**: Efficient large-scale activation extraction methodology
2. **Cross-Layer Analysis**: Understanding representation evolution through network depth
3. **Mechanistic Interpretability**: Identifying causal mechanisms of model behavior
4. **Sparse Autoencoders**: Decomposing activations into interpretable features

---

## Contact

For questions or issues with NCC extraction:
- Check troubleshooting section above
- Review extraction logs in `logs/`
- Inspect metadata in `checkpoints/full_layers_ncc/extraction_metadata.json`

---

**Last Updated**: January 5, 2026
**Version**: 1.0
**Status**: Ready for production use
