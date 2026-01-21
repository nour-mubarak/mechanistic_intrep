#!/usr/bin/env python3
"""Quick test of Qwen2-VL model loading and inference."""

import torch
import time
from pathlib import Path
from PIL import Image
import pandas as pd

print("Starting Qwen2-VL test...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
print("\n1. Loading model...")
start = time.time()

from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    trust_remote_code=True
)

print(f"   Model loaded in {time.time() - start:.1f}s")
print(f"   Model type: {type(model).__name__}")

# Check model structure
print("\n2. Checking model structure...")
print(f"   model.model type: {type(model.model).__name__}")
if hasattr(model.model, 'language_model'):
    print(f"   language_model.layers: {len(model.model.language_model.layers)}")
else:
    print("   No language_model found")

# Load a test image
print("\n3. Loading test image...")
df = pd.read_csv('data/processed/samples.csv')
images_dir = Path('data/raw/images')

# Find first image with valid gender
test_row = None
for idx, row in df.iterrows():
    if row.get('ground_truth_gender') in ['male', 'female']:
        img_path = images_dir / row['image']
        if img_path.exists():
            test_row = row
            break

if test_row is None:
    print("   ERROR: No valid test image found")
    exit(1)

img_path = images_dir / test_row['image']
print(f"   Image: {img_path}")
print(f"   Gender: {test_row['ground_truth_gender']}")
print(f"   Caption: {test_row['ar_caption'][:50]}...")

# Prepare input
print("\n4. Preparing input...")
start = time.time()

image = Image.open(img_path).convert("RGB")
print(f"   Image size: {image.size}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"   Text length: {len(text)}")

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True
)

print(f"   Input keys: {list(inputs.keys())}")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"   {k}: {v.shape}, dtype={v.dtype}")

print(f"   Input preparation took {time.time() - start:.1f}s")

# Move to device
print("\n5. Moving inputs to device...")
device = next(model.parameters()).device
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Forward pass
print("\n6. Running forward pass...")
start = time.time()

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print(f"   Forward pass took {time.time() - start:.1f}s")
print(f"   Output keys: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}")

if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
    print(f"   Hidden states: {len(outputs.hidden_states)} layers")
    print(f"   Layer 0 shape: {outputs.hidden_states[0].shape}")

print("\n✓ Test completed successfully!")
