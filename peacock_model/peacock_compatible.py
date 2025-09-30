#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-Language Arabic Captioning: CLIP Vision Encoder + Arabic LLM
Uses CLIP to extract visual features and projects them into the LLM's text space
for direct Arabic caption generation.
"""

import os
import csv
import glob
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
IMAGE_DIR = "/home2/jmsk62/project/mechanistic_intrep/dataset/images"
OUTPUT_CSV = "./vision_arabic_llm_captions.csv"

CLIP_VISION_MODEL = "openai/clip-vit-base-patch32"
ARABIC_LLM = "tiiuae/falcon-7b-instruct"

MAX_IMAGES = 100
USE_CPU = True

# Prompts
PROMPTS = {
    "ar_from_en": "Describe this image in Arabic in one sentence:",
    "ar_from_ar": "صف هذه الصورة بالعربية في جملة واحدة:",
    "en_from_en": "Describe this image in English in one sentence:",
    "en_from_ar": "اكتب وصف لهذه الصورة بالإنجليزية في جملة واحدة:"
}

class VisionProjector(nn.Module):
    """Projects CLIP vision features to LLM text embedding space"""
    def __init__(self, vision_dim=768, text_dim=4544):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, text_dim),
            nn.LayerNorm(text_dim)
        )
    
    def forward(self, vision_features):
        return self.projection(vision_features)

def load_models(device):
    """Load vision encoder, projector, and Arabic LLM"""
    print("Loading CLIP vision encoder...")
    vision_processor = CLIPImageProcessor.from_pretrained(CLIP_VISION_MODEL)
    vision_model = CLIPVisionModel.from_pretrained(CLIP_VISION_MODEL)
    vision_model.to(device)
    vision_model.eval()
    
    print("Loading Arabic LLM...")
    llm_tokenizer = AutoTokenizer.from_pretrained(ARABIC_LLM)
    llm_model = AutoModelForCausalLM.from_pretrained(ARABIC_LLM)
    
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    llm_model.to(device)
    llm_model.eval()
    
    # Create projection layer
    print("Initializing vision-language projector...")
    vision_dim = vision_model.config.hidden_size
    text_dim = llm_model.config.hidden_size
    projector = VisionProjector(vision_dim, text_dim)
    projector.to(device)
    
    return vision_processor, vision_model, projector, llm_tokenizer, llm_model

def extract_visual_features(image, vision_processor, vision_model):
    """Extract features from image using CLIP vision encoder"""
    inputs = vision_processor(images=image, return_tensors="pt").to(vision_model.device)
    
    with torch.no_grad():
        outputs = vision_model(**inputs)
        # Use the pooled output (CLS token)
        visual_features = outputs.pooler_output
    
    return visual_features

def generate_caption_with_visual_context(visual_features, prompt, projector, tokenizer, model):
    """Generate caption by conditioning LLM on visual features"""
    
    # Project visual features to text space
    projected_features = projector(visual_features)
    
    # Encode the text prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_embeds = model.get_input_embeddings()(prompt_ids)
    
    # Concatenate visual features with prompt embeddings
    # Shape: [batch, seq_len + 1, hidden_dim]
    combined_embeds = torch.cat([
        projected_features.unsqueeze(1),  # Visual "token"
        prompt_embeds
    ], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long).to(model.device)
    
    # Generate from combined embeddings
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode - skip the input length
    generated_ids = outputs[0][combined_embeds.shape[1]:]
    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return clean_caption(caption)

def clean_caption(text):
    """Clean generated caption"""
    if not text:
        return ""
    
    # Remove newlines
    text = text.replace("\n", " ").replace("\r", " ").strip()
    
    # Take first sentence
    for stop in [".", "!", "؟", "…"]:
        if stop in text:
            text = text.split(stop)[0].strip()
            break
    
    # Limit length
    if len(text) > 100:
        text = text[:100].strip()
    
    return text

def process_image(image_path, vision_proc, vision_model, projector, llm_tok, llm_model):
    """Process single image through all configurations"""
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Extract visual features once
        visual_features = extract_visual_features(image, vision_proc, vision_model)
        
        results = {}
        
        # Generate captions for all 4 configurations
        for key, prompt in PROMPTS.items():
            print("  {}...".format(key))
            caption = generate_caption_with_visual_context(
                visual_features, prompt, projector, llm_tok, llm_model
            )
            results[key] = caption
            print("    {}".format(caption[:60]))
        
        return results
        
    except Exception as e:
        print("Error processing image: {}".format(str(e)))
        return {key: "ERROR" for key in PROMPTS.keys()}

def main():
    device = torch.device("cpu" if USE_CPU else "cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    
    # Load models
    try:
        vision_proc, vision_model, projector, llm_tok, llm_model = load_models(device)
    except Exception as e:
        print("Failed to load models: {}".format(str(e)))
        return 1
    
    # Find images
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_paths:
        print("No images found")
        return 1
    
    if MAX_IMAGES:
        image_paths = image_paths[:MAX_IMAGES]
    
    print("Found {} images".format(len(image_paths)))
    
    # Process images
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_filename",
            "ar_from_en_prompt",
            "ar_from_ar_prompt", 
            "en_from_en_prompt",
            "en_from_ar_prompt"
        ])
        
        for i, img_path in enumerate(image_paths):
            print("\nProcessing {}/{}: {}".format(
                i+1, len(image_paths), os.path.basename(img_path)
            ))
            
            results = process_image(
                img_path, vision_proc, vision_model, projector, llm_tok, llm_model
            )
            
            writer.writerow([
                os.path.basename(img_path),
                results.get("ar_from_en", ""),
                results.get("ar_from_ar", ""),
                results.get("en_from_en", ""),
                results.get("en_from_ar", "")
            ])
            
            if (i + 1) % 10 == 0:
                f.flush()
    
    print("\nComplete! Results saved to: {}".format(OUTPUT_CSV))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())