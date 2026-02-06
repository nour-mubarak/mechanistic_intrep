#!/usr/bin/env python3
"""
Memory-efficient investigation of Arabic activation files.
Uses partial unpickling to read metadata without loading full tensors.
"""
import torch
import pickle
import io
import os
import struct
import zipfile

BASE = "/home2/jmsk62/mechanistic_intrep/sae_captioning_project"
ACT_DIR = os.path.join(BASE, "checkpoints/full_layers_ncc/layer_checkpoints")

print("=" * 70)
print("MEMORY-EFFICIENT ARABIC FILE INVESTIGATION")
print("=" * 70)

# Strategy: PyTorch .pt files are ZIP archives containing pickled metadata
# We can read the pickle part to get tensor shapes without loading tensor data

for layer in [3, 6, 9, 12, 15, 17]:
    ar_path = os.path.join(ACT_DIR, f"layer_{layer}_arabic.pt")
    if not os.path.exists(ar_path):
        continue
    
    size_gb = os.path.getsize(ar_path) / (1024**3)
    print(f"\nLayer {layer} Arabic ({size_gb:.1f} GB):")
    
    try:
        # Try reading as a zip file to examine contents
        if zipfile.is_zipfile(ar_path):
            print("  Format: ZIP-based PyTorch archive")
            with zipfile.ZipFile(ar_path, 'r') as zf:
                names = zf.namelist()
                print(f"  ZIP entries ({len(names)}):")
                for name in names[:20]:
                    info = zf.getinfo(name)
                    print(f"    {name} ({info.compress_size} bytes)")
                if len(names) > 20:
                    print(f"    ... and {len(names) - 20} more")
                
                # Read the data.pkl to get structure
                pkl_entries = [n for n in names if n.endswith('data.pkl') or n.endswith('.pkl')]
                if pkl_entries:
                    print(f"\n  Reading pickle metadata from: {pkl_entries[0]}")
                    with zf.open(pkl_entries[0]) as pkl_file:
                        # Use a custom unpickler that records tensor shapes
                        class ShapeOnlyUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if name == '_rebuild_tensor_v2':
                                    return self._rebuild_tensor_v2
                                return super().find_class(module, name)
                            
                            @staticmethod
                            def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                                print(f"    Found tensor: shape={size}, stride={stride}")
                                return f"TENSOR(shape={size})"
                        
                        # This may partially work - at minimum we see structure
                        try:
                            unpickler = ShapeOnlyUnpickler(pkl_file)
                            result = unpickler.load()
                            if isinstance(result, dict):
                                print(f"\n  File structure (dict keys): {list(result.keys())}")
                                for k, v in result.items():
                                    if isinstance(v, str) and v.startswith("TENSOR"):
                                        print(f"    '{k}': {v}")
                                    elif isinstance(v, list):
                                        print(f"    '{k}': list, len={len(v)}")
                                    else:
                                        val_str = str(v)[:80]
                                        print(f"    '{k}': {type(v).__name__} = {val_str}")
                        except Exception as e:
                            print(f"    Pickle parse error: {e}")
        else:
            print("  Format: Legacy PyTorch format (not zip-based)")
            # For legacy format, try to read header
            with open(ar_path, 'rb') as f:
                magic = f.read(8)
                print(f"  Magic bytes: {magic[:8]}")
    
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Only check one layer
    break

# Also try the backup_layer6 smaller Arabic file (2.1GB)
print("\n" + "=" * 70)
print("CHECKING BACKUP_LAYER6 ARABIC FILE (smaller, 2.1GB)")
print("=" * 70)

backup_path = os.path.join(ACT_DIR, "backup_layer6/layer_6_arabic.pt")
if os.path.exists(backup_path):
    size_gb = os.path.getsize(backup_path) / (1024**3)
    print(f"  Size: {size_gb:.1f} GB")
    try:
        data = torch.load(backup_path, map_location='cpu')
        if isinstance(data, torch.Tensor):
            print(f"  Type: Tensor, Shape: {data.shape}, Dtype: {data.dtype}")
        elif isinstance(data, dict):
            print(f"  Type: Dict, Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"    '{k}': Tensor shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"    '{k}': list, len={len(v)}")
                    if len(v) > 0 and isinstance(v[0], torch.Tensor):
                        shapes = set()
                        for i in range(min(5, len(v))):
                            if isinstance(v[i], torch.Tensor):
                                shapes.add(tuple(v[i].shape))
                        print(f"      Element shapes: {shapes}")
                else:
                    print(f"    '{k}': {type(v).__name__} = {str(v)[:80]}")
        del data
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
else:
    print("  File not found")

# Check chunk files
print("\n" + "=" * 70)
print("CHECKING ARABIC CHUNK FILES (backup_layer6)")
print("=" * 70)

chunk_dir = os.path.join(ACT_DIR, "backup_layer6")
if os.path.exists(chunk_dir):
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if 'chunk' in f])
    print(f"  Found {len(chunk_files)} chunk files")
    if chunk_files:
        # Load first chunk
        first_chunk = os.path.join(chunk_dir, chunk_files[0])
        size_mb = os.path.getsize(first_chunk) / (1024**2)
        print(f"  First chunk: {chunk_files[0]} ({size_mb:.1f} MB)")
        try:
            data = torch.load(first_chunk, map_location='cpu')
            if isinstance(data, torch.Tensor):
                print(f"    Type: Tensor, Shape: {data.shape}, Dtype: {data.dtype}")
            elif isinstance(data, dict):
                print(f"    Type: Dict, Keys: {list(data.keys())}")
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        print(f"      '{k}': Tensor shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"      '{k}': {type(v).__name__} = {str(v)[:80]}")
            del data
        except Exception as e:
            print(f"    Error: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("QUICK SIZE ANALYSIS")
print("=" * 70)
# English: 10000 samples x 2048 dims x 4 bytes = ~78MB ✓
# Arabic: 21GB → 21*1024^3 / 4 / 2048 ≈ 2,752,512 samples (if [N, 2048])
#   OR: 21GB → could be [N_images, seq_len, 2048] 3D tensor
eng_expected = 10000 * 2048 * 4
ar_size = 21 * (1024**3)
n_2d = ar_size / (2048 * 4)
print(f"  English: 10,000 samples × 2,048 = {eng_expected / (1024**2):.1f} MB (matches)")
print(f"  Arabic 21GB as [N, 2048] → N = {n_2d:,.0f} samples")
print(f"  Arabic 21GB as [N, 268, 2048] → N = {n_2d / 268:,.0f} images × 268 tokens each")
print(f"  Arabic 21GB as [N, 256, 2048] → N = {n_2d / 256:,.0f} images × 256 tokens each")
print(f"")
print(f"  268 could be: image patches (16×16=256 + 12 special tokens?)")
print(f"  If PaLiGemma uses 256 image patches + text tokens = 268 seq positions")
