# Full Layer Activation Extraction on Durham NCC

**Quick reference guide for extracting activations from all 34 Gemma-3-4B layers using Durham's NCC cluster.**

---

## TL;DR

```bash
# 1. SSH to NCC
ssh username@ncc.durham.ac.uk

# 2. Navigate to project
cd /path/to/mechanistic_intrep/sae_captioning_project

# 3. Update email in scripts
sed -i 's/your.email@durham.ac.uk/your_actual_email@durham.ac.uk/g' scripts/slurm_*.sh

# 4. Submit parallel extraction (FASTEST - 40-60 min)
bash scripts/slurm_parallel_extraction.sh

# 5. Monitor
squeue -u $USER
tail -f logs/extract_layers_*.out

# 6. Verify completion (should show 68 files: 34 layers × 2 languages)
ls checkpoints/full_layers_ncc/layer_checkpoints/ | wc -l
```

---

## Available Scripts

| Script | Purpose | Time | GPUs | Command |
|--------|---------|------|------|---------|
| **slurm_parallel_extraction.sh** | **Fastest: 4 parallel jobs** | 40-60 min | 4 | `bash scripts/slurm_parallel_extraction.sh` |
| slurm_extract_full_activations.sh | All layers sequentially | 2 hours | 1 | `sbatch scripts/slurm_extract_full_activations.sh` |
| slurm_extract_layer_ranges.sh | Custom layer ranges | Variable | 1 | `sbatch scripts/slurm_extract_layer_ranges.sh` |

**Recommended**: Use `slurm_parallel_extraction.sh` for fastest results.

---

## Extraction Details

- **Model**: google/gemma-3-4b-it (34 layers, 0-33)
- **Dataset**: 2000 samples (English + Arabic)
- **Output**: ~45 GB total
- **Location**: `checkpoints/full_layers_ncc/`

---

## Output Structure

```
checkpoints/full_layers_ncc/
├── extraction_metadata.json               # Extraction config
├── activations_english_all_layers.pt      # Combined file (all layers)
├── activations_arabic_all_layers.pt       # Combined file (all layers)
└── layer_checkpoints/                     # Individual layer files
    ├── layer_0_english.pt  (450 MB)
    ├── layer_0_arabic.pt   (450 MB)
    ├── layer_1_english.pt
    ...
    └── layer_33_arabic.pt
```

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/extract_layers_00_08_<job_id>.out

# Check all job outputs
watch -n 5 'squeue -u $USER'
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Job stays pending | Check `squeue` for queue position, may need to wait |
| OOM error | Already using minimum batch_size=1; increase memory: `--mem=128G` |
| Missing modules | Ensure `source venv/bin/activate` is in SLURM script |
| Quota exceeded | Check `quota -s`, clean up old files |

---

## Next Steps After Extraction

1. **Verify completion**:
   ```bash
   # Should show 68 files
   ls checkpoints/full_layers_ncc/layer_checkpoints/ | wc -l
   ```

2. **Train SAEs** on all layers:
   ```bash
   for layer in {0..33}; do
       sbatch scripts/slurm_train_sae.sh --layer $layer
   done
   ```

3. **Run comprehensive analysis**:
   ```bash
   sbatch scripts/slurm_comprehensive_analysis.sh --layers $(seq 0 33)
   ```

---

## Documentation

- **Detailed NCC Guide**: [docs/DURHAM_NCC_GUIDE.md](docs/DURHAM_NCC_GUIDE.md)
- **Extraction Methodology**: [docs/NCC_EXTRACTION_GUIDE.md](docs/NCC_EXTRACTION_GUIDE.md)
- **Durham NCC Portal**: https://nccadmin.webspace.durham.ac.uk/

---

## Quick Validation

After jobs complete, verify outputs:

```bash
python << EOF
import torch
import json

# Check metadata
with open('checkpoints/full_layers_ncc/extraction_metadata.json') as f:
    metadata = json.load(f)
print(f"Layers extracted: {len(metadata['layers_extracted'])}")
print(f"Total samples: {metadata['num_samples']}")

# Verify a layer file
data = torch.load('checkpoints/full_layers_ncc/layer_checkpoints/layer_10_english.pt')
print(f"\nLayer 10 English:")
print(f"  Shape: {data['activations'].shape}")
print(f"  Samples: {len(data['genders'])}")
print(f"  Male samples: {data['genders'].count('male')}")
print(f"  Female samples: {data['genders'].count('female')}")
EOF
```

Expected output:
```
Layers extracted: 34
Total samples: 2000

Layer 10 English:
  Shape: torch.Size([2000, 278, 2560])
  Samples: 2000
  Male samples: 1000
  Female samples: 1000
```

---

## Performance Summary

| Method | Time | GPUs | Recommendation |
|--------|------|------|----------------|
| Parallel (4 jobs) | **40-60 min** | 4 | ✅ **Recommended** |
| Sequential | 2 hours | 1 | Use if GPU queue is long |
| Custom ranges | Variable | 1-N | For specific layers only |

---

**Created**: January 5, 2026
**Status**: Ready for production use on Durham NCC
