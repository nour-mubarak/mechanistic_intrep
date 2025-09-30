#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Image Captioning using AIN (Arabic Inclusive Large Multimodal Model)
From MBZUAI - State-of-the-art bilingual Arabic-English vision-language model
"""

import os
import csv
import glob
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration
IMAGE_DIR = "/home2/jmsk62/project/mechanistic_intrep/dataset/images"
OUTPUT_CSV = "./ain_arabic_captions.csv"

MODEL_NAME = "MBZUAI/AIN"
MAX_IMAGES = 100
USE_CPU = False  # AIN works better on GPU, but can run on CPU

# Your exact prompts for the 4 configurations
PROMPTS = {
    "ar_from_en": "Write a single, concise sentence caption for this image in Arabic.",
    "ar_from_ar": "اكتب جملة واحدة مختصرة لوصف هذه الصورة باللغة العربية.",
    "en_from_en": "Write a single, concise sentence caption for this image in English.",
    "en_from_ar": "اكتب تعليقًا من جملة واحدة لهذه الصورة باللغة الإنجليزية."
}

def load_ain_model(device):
    """Load the AIN model from MBZUAI"""
    print("Loading AIN model from MBZUAI...")
    print("This may take a few minutes on first run...")
    
    try:
        # Load model with automatic device mapping
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto" if not USE_CPU else torch.float32,
            device_map="auto" if not USE_CPU else None
        )
        
        if USE_CPU:
            model = model.to(device)
        
        model.eval()
        
        # Load processor and tokenizer
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        print("AIN model loaded successfully!")
        return model, processor
        
    except Exception as e:
        print("Error loading AIN model: {}".format(str(e)))
        print("\nMake sure you have installed required packages:")
        print("  pip install transformers qwen-vl-utils torch torchvision")
        raise e

def generate_caption(image_path, prompt, model, processor):
    """Generate caption for image using AIN model"""
    
    try:
        # Prepare conversation format for AIN
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process the conversation
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = inputs.to(model.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Deterministic for consistency
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the output
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return clean_caption(output_text)
        
    except Exception as e:
        print("Error generating caption: {}".format(str(e)))
        return "ERROR: {}".format(str(e))

def clean_caption(text):
    """Clean and format the generated caption"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove common artifacts
    text = text.strip()
    
    # Take first sentence if multiple
    for delimiter in ["\n", ".", "!", "؟"]:
        if delimiter in text:
            parts = text.split(delimiter)
            if parts[0].strip():
                text = parts[0].strip()
                break
    
    return text

def process_images(model, processor):
    """Process all images with the 4 prompt configurations"""
    
    # Find images
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    
    if not image_paths:
        print("No images found in {}".format(IMAGE_DIR))
        return
    
    if MAX_IMAGES:
        image_paths = image_paths[:MAX_IMAGES]
    
    print("Found {} images to process".format(len(image_paths)))
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    
    # Process images
    successful = 0
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "image_filename",
            "ar_from_en_prompt",
            "ar_from_ar_prompt",
            "en_from_en_prompt",
            "en_from_ar_prompt",
            "model_used"
        ])
        
        for i, image_path in enumerate(image_paths):
            filename = os.path.basename(image_path)
            print("\nProcessing {}/{}: {}".format(i+1, len(image_paths), filename))
            
            results = {}
            
            try:
                # Generate captions for all 4 configurations
                for key, prompt in PROMPTS.items():
                    print("  {}...".format(key))
                    caption = generate_caption(image_path, prompt, model, processor)
                    results[key] = caption
                    
                    # Show preview
                    preview = caption[:60] + "..." if len(caption) > 60 else caption
                    print("    {}".format(preview))
                
                # Write results
                writer.writerow([
                    filename,
                    results.get("ar_from_en", ""),
                    results.get("ar_from_ar", ""),
                    results.get("en_from_en", ""),
                    results.get("en_from_ar", ""),
                    MODEL_NAME
                ])
                
                successful += 1
                
            except Exception as e:
                print("Error processing {}: {}".format(filename, str(e)))
                writer.writerow([
                    filename,
                    "ERROR",
                    "ERROR",
                    "ERROR",
                    "ERROR",
                    MODEL_NAME
                ])
            
            # Flush every 10 images
            if (i + 1) % 10 == 0:
                csvfile.flush()
                print("Progress: {}/{} images processed ({} successful)".format(
                    i+1, len(image_paths), successful))
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print("Total images: {}".format(len(image_paths)))
    print("Successful: {}".format(successful))
    print("Failed: {}".format(len(image_paths) - successful))
    print("Results saved to: {}".format(OUTPUT_CSV))
    print("="*50)

def main():
    # Set device
    if USE_CPU:
        device = torch.device("cpu")
        print("Using CPU (this will be slow)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(device))
    
    # Load model
    try:
        model, processor = load_ain_model(device)
    except Exception as e:
        print("Failed to load AIN model: {}".format(str(e)))
        return 1
    
    # Process images
    process_images(model, processor)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())