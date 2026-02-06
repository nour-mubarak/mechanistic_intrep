#!/usr/bin/env python3
"""
Investigate Arabic activation file dimension mismatch for PaLiGemma.
Compares Arabic vs English file structure to identify the issue.
"""
import torch
import os
import sys

BASE = "/home2/jmsk62/mechanistic_intrep/sae_captioning_project"
ACT_DIR = os.path.join(BASE, "checkpoints/full_layers_ncc/layer_checkpoints")
SAE_DIR = os.path.join(BASE, "checkpoints/saes")

print("=" * 70)
print("INVESTIGATING ARABIC ACTIVATION FILE DIMENSION MISMATCH")
print("=" * 70)

# 1. Inspect English file (known working)
print("\n" + "=" * 70)
print("1. ENGLISH ACTIVATION FILES (WORKING)")
print("=" * 70)

for layer in [3, 6, 9, 12, 15, 17]:
    eng_path = os.path.join(ACT_DIR, f"layer_{layer}_english.pt")
    if os.path.exists(eng_path):
        size_mb = os.path.getsize(eng_path) / (1024**2)
        data = torch.load(eng_path, map_location='cpu')
        print(f"\nLayer {layer} English ({size_mb:.1f} MB):")
        if isinstance(data, torch.Tensor):
            print(f"  Type: Tensor, Shape: {data.shape}, Dtype: {data.dtype}")
        elif isinstance(data, dict):
            print(f"  Type: Dict, Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"    '{k}': Tensor shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"    '{k}': List, len={len(v)}")
                    if len(v) > 0 and isinstance(v[0], torch.Tensor):
                        print(f"      First element: Tensor shape={v[0].shape}")
                else:
                    print(f"    '{k}': {type(v).__name__} = {v}")
        elif isinstance(data, list):
            print(f"  Type: List, Length: {len(data)}")
            if len(data) > 0:
                print(f"  First element type: {type(data[0]).__name__}")
                if isinstance(data[0], torch.Tensor):
                    print(f"  First element shape: {data[0].shape}")
        else:
            print(f"  Type: {type(data).__name__}")
        del data
    else:
        print(f"  Layer {layer} English: NOT FOUND")

# 2. Inspect Arabic file (problematic) - use mmap to avoid loading full 21GB
print("\n" + "=" * 70)
print("2. ARABIC ACTIVATION FILES (PROBLEMATIC)")
print("=" * 70)

# Try with layer 3 first (smallest layer number available)
for layer in [3, 6, 9, 12, 15, 17]:
    ar_path = os.path.join(ACT_DIR, f"layer_{layer}_arabic.pt")
    if os.path.exists(ar_path):
        size_gb = os.path.getsize(ar_path) / (1024**3)
        print(f"\nLayer {layer} Arabic ({size_gb:.1f} GB):")
        
        # Try to load metadata only using weights_only and map_location
        try:
            # Load with mmap to avoid full memory consumption
            data = torch.load(ar_path, map_location='cpu', weights_only=False)
            
            if isinstance(data, torch.Tensor):
                print(f"  Type: Tensor, Shape: {data.shape}, Dtype: {data.dtype}")
                print(f"  Expected shape for 2048-dim: [N, 2048]")
                if len(data.shape) == 2:
                    print(f"  Samples: {data.shape[0]}, Features: {data.shape[1]}")
                elif len(data.shape) == 3:
                    print(f"  Dim0: {data.shape[0]}, Dim1: {data.shape[1]}, Dim2: {data.shape[2]}")
                    print(f"  Possible: [batch, seq_len, d_model] or [n_images, patches, d_model]")
            elif isinstance(data, dict):
                print(f"  Type: Dict, Keys: {list(data.keys())}")
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    '{k}': Tensor shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, list):
                        print(f"    '{k}': List, len={len(v)}")
                        if len(v) > 0 and isinstance(v[0], torch.Tensor):
                            print(f"      First element: Tensor shape={v[0].shape}")
                            if len(v) > 1:
                                print(f"      Second element: Tensor shape={v[1].shape}")
                            # Check if all have same shape
                            shapes = set()
                            for idx in range(min(10, len(v))):
                                if isinstance(v[idx], torch.Tensor):
                                    shapes.add(v[idx].shape)
                            print(f"      Unique shapes in first 10: {shapes}")
                    else:
                        val_str = str(v)[:100]
                        print(f"    '{k}': {type(v).__name__} = {val_str}")
            elif isinstance(data, list):
                print(f"  Type: List, Length: {len(data)}")
                if len(data) > 0:
                    el = data[0]
                    print(f"  First element type: {type(el).__name__}")
                    if isinstance(el, torch.Tensor):
                        print(f"  First element shape: {el.shape}")
                    if len(data) > 1:
                        el2 = data[1]
                        print(f"  Second element type: {type(el2).__name__}")
                        if isinstance(el2, torch.Tensor):
                            print(f"  Second element shape: {el2.shape}")
                    # Check shapes
                    shapes = set()
                    for idx in range(min(10, len(data))):
                        if isinstance(data[idx], torch.Tensor):
                            shapes.add(data[idx].shape)
                    print(f"  Unique shapes in first 10: {shapes}")
            else:
                print(f"  Type: {type(data).__name__}")
            
            del data
            
        except Exception as e:
            print(f"  ERROR loading: {type(e).__name__}: {e}")
        
        # Only examine one layer to avoid memory issues
        print("\n  (Examining only one Arabic layer to avoid memory issues)")
        break
    else:
        print(f"  Layer {layer} Arabic: NOT FOUND")

# 3. Inspect SAE checkpoint to compare expected dimensions
print("\n" + "=" * 70)
print("3. SAE CHECKPOINT DIMENSIONS")
print("=" * 70)

sae_files = [f for f in os.listdir(SAE_DIR) if f.endswith('.pt')] if os.path.exists(SAE_DIR) else []
# Look for PaLiGemma SAE files
pali_saes = [f for f in sae_files if 'paligemma' in f.lower() or '2048' in f]
if not pali_saes:
    # Try listing all to find relevant ones
    print(f"  All SAE files: {sae_files[:10]}")
    # Check first few
    for f in sae_files[:3]:
        ckpt = torch.load(os.path.join(SAE_DIR, f), map_location='cpu')
        d_model = ckpt.get('d_model', 'N/A')
        d_hidden = ckpt.get('d_hidden', 'N/A')
        layer = ckpt.get('layer', 'N/A')
        lang = ckpt.get('language', 'N/A')
        print(f"  {f}: d_model={d_model}, d_hidden={d_hidden}, layer={layer}, language={lang}")
        del ckpt
else:
    for f in pali_saes:
        ckpt = torch.load(os.path.join(SAE_DIR, f), map_location='cpu')
        d_model = ckpt.get('d_model', 'N/A')
        d_hidden = ckpt.get('d_hidden', 'N/A')
        print(f"  {f}: d_model={d_model}, d_hidden={d_hidden}")
        del ckpt

# 4. Check the compute script that failed
print("\n" + "=" * 70)
print("4. ERROR CONTEXT")
print("=" * 70)
print("  Previous error: 'output with shape [16384] doesn't match the broadcast shape [268, 16384]'")
print("  This suggests Arabic data has shape [268, ...] somewhere")
print("  268 could be: number of images, sequence positions, or batch chunks")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
