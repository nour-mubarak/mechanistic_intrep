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
OUTPUT_CSV = "./ain_arabic_captions_full_2.csv"

MODEL_NAME = "MBZUAI/AIN"
MAX_IMAGES = None  # Set to None to process all images
USE_CPU = False  # Keep False for GPU - much faster

# Your exact prompts for the 4 configurations
PROMPTS = {
    "ar_from_en": "Write a single, concise sentence caption for this image in Arabic.",
    "ar_from_ar": "اكتب جملة واحدة مختصرة لوصف هذه الصورة باللغة العربية.",
    "en_from_en": "Write a single, concise sentence caption for this image in English.",
    "en_from_ar": "اكتب تعليقًا من جملة واحدة لهذه الصورة باللغة الإنجليزية. يجب أن يكون الرد باللغة الإنجليزية فقط. Answer in English only:"  # Explicit English instruction
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

######### working on ain_arabic_captions_full.csv ##################
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Arabic Image Captioning using AIN (Arabic Inclusive Large Multimodal Model)
# From MBZUAI - State-of-the-art bilingual Arabic-English vision-language model
# """

# import os
# import csv
# import glob
# import torch
# from PIL import Image
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info

# # Configuration
# IMAGE_DIR = "/home2/jmsk62/project/mechanistic_intrep/dataset/images"
# OUTPUT_CSV = "./ain_arabic_captions_full.csv"

# MODEL_NAME = "MBZUAI/AIN"
# MAX_IMAGES = None  # Set to None to process all images
# USE_CPU = False  # Keep False for GPU - much faster

# # Your exact prompts for the 4 configurations
# PROMPTS = {
#     "ar_from_en": "Write a single, concise sentence caption for this image in Arabic.",
#     "ar_from_ar": "اكتب جملة واحدة مختصرة لوصف هذه الصورة باللغة العربية.",
#     "en_from_en": "Write a single, concise sentence caption for this image in English.",
#     "en_from_ar": "اكتب جملة واحدة مختصرة لوصف هذه الصورة باللغة الإنجليزية."
# }

# def load_ain_model(device):
#     """Load the AIN model from MBZUAI"""
#     print("Loading AIN model from MBZUAI...")
#     print("This may take a few minutes on first run...")
    
#     try:
#         # Load model with automatic device mapping
#         model = Qwen2VLForConditionalGeneration.from_pretrained(
#             MODEL_NAME,
#             torch_dtype="auto" if not USE_CPU else torch.float32,
#             device_map="auto" if not USE_CPU else None
#         )
        
#         if USE_CPU:
#             model = model.to(device)
        
#         model.eval()
        
#         # Load processor and tokenizer
#         processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
#         print("AIN model loaded successfully!")
#         return model, processor
        
#     except Exception as e:
#         print("Error loading AIN model: {}".format(str(e)))
#         print("\nMake sure you have installed required packages:")
#         print("  pip install transformers qwen-vl-utils torch torchvision")
#         raise e

# def generate_caption(image_path, prompt, model, processor):
#     """Generate caption for image using AIN model"""
    
#     try:
#         # Prepare conversation format for AIN
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image_path},
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
        
#         # Process the conversation
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
        
#         # Process vision information
#         image_inputs, video_inputs = process_vision_info(messages)
        
#         # Prepare inputs
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
        
#         # Move to device
#         inputs = inputs.to(model.device)
        
#         # Generate caption
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 do_sample=False,  # Deterministic for consistency
#             )
        
#         # Trim input tokens from output
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] 
#             for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
        
#         # Decode the output
#         output_text = processor.batch_decode(
#             generated_ids_trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )[0]
        
#         return clean_caption(output_text)
        
#     except Exception as e:
#         print("Error generating caption: {}".format(str(e)))
#         return "ERROR: {}".format(str(e))

# def clean_caption(text):
#     """Clean and format the generated caption"""
#     if not text:
#         return ""
    
#     # Remove extra whitespace
#     text = " ".join(text.split())
    
#     # Remove common artifacts
#     text = text.strip()
    
#     # Take first sentence if multiple
#     for delimiter in ["\n", ".", "!", "؟"]:
#         if delimiter in text:
#             parts = text.split(delimiter)
#             if parts[0].strip():
#                 text = parts[0].strip()
#                 break
    
#     return text

# def process_images(model, processor):
#     """Process all images with the 4 prompt configurations"""
    
#     # Find images
#     image_paths = []
#     for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
#         image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
#         image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    
#     if not image_paths:
#         print("No images found in {}".format(IMAGE_DIR))
#         return
    
#     if MAX_IMAGES:
#         image_paths = image_paths[:MAX_IMAGES]
    
#     print("Found {} images to process".format(len(image_paths)))
    
#     # Create output directory
#     os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    
#     # Process images
#     successful = 0
    
#     with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([
#             "image_filename",
#             "ar_from_en_prompt",
#             "ar_from_ar_prompt",
#             "en_from_en_prompt",
#             "en_from_ar_prompt",
#             "model_used"
#         ])
        
#         for i, image_path in enumerate(image_paths):
#             filename = os.path.basename(image_path)
#             print("\nProcessing {}/{}: {}".format(i+1, len(image_paths), filename))
            
#             results = {}
            
#             try:
#                 # Generate captions for all 4 configurations
#                 for key, prompt in PROMPTS.items():
#                     print("  {}...".format(key))
#                     caption = generate_caption(image_path, prompt, model, processor)
#                     results[key] = caption
                    
#                     # Show preview
#                     preview = caption[:60] + "..." if len(caption) > 60 else caption
#                     print("    {}".format(preview))
                
#                 # Write results
#                 writer.writerow([
#                     filename,
#                     results.get("ar_from_en", ""),
#                     results.get("ar_from_ar", ""),
#                     results.get("en_from_en", ""),
#                     results.get("en_from_ar", ""),
#                     MODEL_NAME
#                 ])
                
#                 successful += 1
                
#             except Exception as e:
#                 print("Error processing {}: {}".format(filename, str(e)))
#                 writer.writerow([
#                     filename,
#                     "ERROR",
#                     "ERROR",
#                     "ERROR",
#                     "ERROR",
#                     MODEL_NAME
#                 ])
            
#             # Flush every 10 images
#             if (i + 1) % 10 == 0:
#                 csvfile.flush()
#                 print("Progress: {}/{} images processed ({} successful)".format(
#                     i+1, len(image_paths), successful))
    
#     print("\n" + "="*50)
#     print("Processing Complete!")
#     print("Total images: {}".format(len(image_paths)))
#     print("Successful: {}".format(successful))
#     print("Failed: {}".format(len(image_paths) - successful))
#     print("Results saved to: {}".format(OUTPUT_CSV))
#     print("="*50)

# def main():
#     # Set device
#     if USE_CPU:
#         device = torch.device("cpu")
#         print("Using CPU (this will be slow)")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("Using device: {}".format(device))
    
#     # Load model
#     try:
#         model, processor = load_ain_model(device)
#     except Exception as e:
#         print("Failed to load AIN model: {}".format(str(e)))
#         return 1
    
#     # Process images
#     process_images(model, processor)
    
#     return 0

# if __name__ == "__main__":
#     import sys
#     sys.exit(main())


#################### working on 100 images claude ####################
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Arabic Image Captioning using AIN (Arabic Inclusive Large Multimodal Model)
# From MBZUAI - State-of-the-art bilingual Arabic-English vision-language model
# """

# import os
# import csv
# import glob
# import torch
# from PIL import Image
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info

# # Configuration
# IMAGE_DIR = "/home2/jmsk62/project/mechanistic_intrep/dataset/images"
# OUTPUT_CSV = "./ain_arabic_captions.csv"

# MODEL_NAME = "MBZUAI/AIN"
# MAX_IMAGES = 100
# USE_CPU = False  # AIN works better on GPU, but can run on CPU

# # Your exact prompts for the 4 configurations
# PROMPTS = {
#     "ar_from_en": "Write a single, concise sentence caption for this image in Arabic.",
#     "ar_from_ar": "اكتب جملة واحدة مختصرة لوصف هذه الصورة باللغة العربية.",
#     "en_from_en": "Write a single, concise sentence caption for this image in English.",
#     "en_from_ar": "اكتب جملة واحدة مختصرة لوصف هذه الصورة باللغة الإنجليزية."
# }

# def load_ain_model(device):
#     """Load the AIN model from MBZUAI"""
#     print("Loading AIN model from MBZUAI...")
#     print("This may take a few minutes on first run...")
    
#     try:
#         # Load model with automatic device mapping
#         model = Qwen2VLForConditionalGeneration.from_pretrained(
#             MODEL_NAME,
#             torch_dtype="auto" if not USE_CPU else torch.float32,
#             device_map="auto" if not USE_CPU else None
#         )
        
#         if USE_CPU:
#             model = model.to(device)
        
#         model.eval()
        
#         # Load processor and tokenizer
#         processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
#         print("AIN model loaded successfully!")
#         return model, processor
        
#     except Exception as e:
#         print("Error loading AIN model: {}".format(str(e)))
#         print("\nMake sure you have installed required packages:")
#         print("  pip install transformers qwen-vl-utils torch torchvision")
#         raise e

# def generate_caption(image_path, prompt, model, processor):
#     """Generate caption for image using AIN model"""
    
#     try:
#         # Prepare conversation format for AIN
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image_path},
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
        
#         # Process the conversation
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
        
#         # Process vision information
#         image_inputs, video_inputs = process_vision_info(messages)
        
#         # Prepare inputs
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
        
#         # Move to device
#         inputs = inputs.to(model.device)
        
#         # Generate caption
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 do_sample=False,  # Deterministic for consistency
#             )
        
#         # Trim input tokens from output
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] 
#             for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
        
#         # Decode the output
#         output_text = processor.batch_decode(
#             generated_ids_trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )[0]
        
#         return clean_caption(output_text)
        
#     except Exception as e:
#         print("Error generating caption: {}".format(str(e)))
#         return "ERROR: {}".format(str(e))

# def clean_caption(text):
#     """Clean and format the generated caption"""
#     if not text:
#         return ""
    
#     # Remove extra whitespace
#     text = " ".join(text.split())
    
#     # Remove common artifacts
#     text = text.strip()
    
#     # Take first sentence if multiple
#     for delimiter in ["\n", ".", "!", "؟"]:
#         if delimiter in text:
#             parts = text.split(delimiter)
#             if parts[0].strip():
#                 text = parts[0].strip()
#                 break
    
#     return text

# def process_images(model, processor):
#     """Process all images with the 4 prompt configurations"""
    
#     # Find images
#     image_paths = []
#     for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
#         image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
#         image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    
#     if not image_paths:
#         print("No images found in {}".format(IMAGE_DIR))
#         return
    
#     if MAX_IMAGES:
#         image_paths = image_paths[:MAX_IMAGES]
    
#     print("Found {} images to process".format(len(image_paths)))
    
#     # Create output directory
#     os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    
#     # Process images
#     successful = 0
    
#     with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([
#             "image_filename",
#             "ar_from_en_prompt",
#             "ar_from_ar_prompt",
#             "en_from_en_prompt",
#             "en_from_ar_prompt",
#             "model_used"
#         ])
        
#         for i, image_path in enumerate(image_paths):
#             filename = os.path.basename(image_path)
#             print("\nProcessing {}/{}: {}".format(i+1, len(image_paths), filename))
            
#             results = {}
            
#             try:
#                 # Generate captions for all 4 configurations
#                 for key, prompt in PROMPTS.items():
#                     print("  {}...".format(key))
#                     caption = generate_caption(image_path, prompt, model, processor)
#                     results[key] = caption
                    
#                     # Show preview
#                     preview = caption[:60] + "..." if len(caption) > 60 else caption
#                     print("    {}".format(preview))
                
#                 # Write results
#                 writer.writerow([
#                     filename,
#                     results.get("ar_from_en", ""),
#                     results.get("ar_from_ar", ""),
#                     results.get("en_from_en", ""),
#                     results.get("en_from_ar", ""),
#                     MODEL_NAME
#                 ])
                
#                 successful += 1
                
#             except Exception as e:
#                 print("Error processing {}: {}".format(filename, str(e)))
#                 writer.writerow([
#                     filename,
#                     "ERROR",
#                     "ERROR",
#                     "ERROR",
#                     "ERROR",
#                     MODEL_NAME
#                 ])
            
#             # Flush every 10 images
#             if (i + 1) % 10 == 0:
#                 csvfile.flush()
#                 print("Progress: {}/{} images processed ({} successful)".format(
#                     i+1, len(image_paths), successful))
    
#     print("\n" + "="*50)
#     print("Processing Complete!")
#     print("Total images: {}".format(len(image_paths)))
#     print("Successful: {}".format(successful))
#     print("Failed: {}".format(len(image_paths) - successful))
#     print("Results saved to: {}".format(OUTPUT_CSV))
#     print("="*50)

# def main():
#     # Set device
#     if USE_CPU:
#         device = torch.device("cpu")
#         print("Using CPU (this will be slow)")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("Using device: {}".format(device))
    
#     # Load model
#     try:
#         model, processor = load_ain_model(device)
#     except Exception as e:
#         print("Failed to load AIN model: {}".format(str(e)))
#         return 1
    
#     # Process images
#     process_images(model, processor)
    
#     return 0

# if __name__ == "__main__":
#     import sys
#     sys.exit(main())

#########################################
# #!/usr/bin/env python3
# """
# Arabic-English Image Captioning using AIN (Arabic Inclusive Large Multimodal Model)

# This script provides a complete implementation for bilingual image captioning
# using the AIN model from MBZUAI. It supports both Arabic and English caption generation.

# """

# import torch
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from PIL import Image
# import requests
# from typing import Optional, Union
# import argparse
# import os


# class ArabicImageCaptioner:
#     """
#     A class for generating Arabic and English image captions using the AIN model.
#     """
    
#     def __init__(self, model_name: str = "MBZUAI/AIN", device: str = "auto"):
#         """
#         Initialize the Arabic Image Captioner.
        
#         Args:
#             model_name (str): The Hugging Face model name
#             device (str): Device to run the model on ('auto', 'cuda', 'cpu')
#         """
#         self.model_name = model_name
#         self.device = device
#         self.model = None
#         self.processor = None
        
#         print(f"Initializing Arabic Image Captioner with model: {model_name}")
#         self._load_model()
    
#     def _load_model(self):
#         """Load the model and processor."""
#         try:
#             print("Loading model and processor...")
            
#             # Load the model
#             self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#                 self.model_name,
#                 torch_dtype="auto",
#                 device_map=self.device
#             )
            
#             # Load the processor
#             self.processor = AutoProcessor.from_pretrained(self.model_name)
            
#             print("Model and processor loaded successfully!")
            
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             print("Please ensure you have the required dependencies installed:")
#             print("pip install transformers torch qwen-vl-utils pillow")
#             raise
    
#     def generate_caption(
#         self, 
#         image_path: str, 
#         language: str = "arabic",
#         custom_prompt: Optional[str] = None,
#         max_new_tokens: int = 128
#     ) -> str:
#         """
#         Generate a caption for the given image.
        
#         Args:
#             image_path (str): Path to the image file or URL
#             language (str): Target language ('arabic' or 'english')
#             custom_prompt (str, optional): Custom prompt for caption generation
#             max_new_tokens (int): Maximum number of tokens to generate
            
#         Returns:
#             str: Generated caption
#         """
        
#         # Load image
#         if image_path.startswith(('http://', 'https://')):
#             image = Image.open(requests.get(image_path, stream=True).raw)
#         else:
#             image = Image.open(image_path)
        
#         # Prepare prompt based on language
#         if custom_prompt:
#             prompt = custom_prompt
#         elif language.lower() == "arabic":
#             prompt = "يرجى وصف هذه الصورة بالتفصيل."  # "Please describe this image in detail."
#         else:
#             prompt = "Please describe this image in detail."
        
#         # Prepare messages
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image"},
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
        
#         # Apply chat template
#         text_prompt = self.processor.apply_chat_template(
#             messages, add_generation_prompt=True
#         )
        
#         # Process inputs
#         inputs = self.processor(
#             text=[text_prompt], 
#             images=[image], 
#             padding=True, 
#             return_tensors="pt"
#         )
        
#         # Move inputs to device
#         if torch.cuda.is_available() and self.device != "cpu":
#             inputs = inputs.to("cuda")
        
#         # Generate caption
#         with torch.no_grad():
#             output_ids = self.model.generate(
#                 **inputs, 
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False
#             )
        
#         # Decode output
#         generated_ids = [
#             output_ids[len(input_ids):] 
#             for input_ids, output_ids in zip(inputs.input_ids, output_ids)
#         ]
        
#         output_text = self.processor.batch_decode(
#             generated_ids, 
#             skip_special_tokens=True, 
#             clean_up_tokenization_spaces=True
#         )
        
#         return output_text[0] if output_text else ""
    
#     def generate_bilingual_captions(
#         self, 
#         image_path: str, 
#         max_new_tokens: int = 128
#     ) -> dict:
#         """
#         Generate both Arabic and English captions for an image.
        
#         Args:
#             image_path (str): Path to the image file or URL
#             max_new_tokens (int): Maximum number of tokens to generate
            
#         Returns:
#             dict: Dictionary containing both Arabic and English captions
#         """
        
#         print(f"Generating bilingual captions for: {image_path}")
        
#         # Generate Arabic caption
#         print("Generating Arabic caption...")
#         arabic_caption = self.generate_caption(
#             image_path, 
#             language="arabic", 
#             max_new_tokens=max_new_tokens
#         )
        
#         # Generate English caption
#         print("Generating English caption...")
#         english_caption = self.generate_caption(
#             image_path, 
#             language="english", 
#             max_new_tokens=max_new_tokens
#         )
        
#         return {
#             "arabic": arabic_caption,
#             "english": english_caption
#         }


# def main():
#     """Main function to run the image captioning script."""
    
#     parser = argparse.ArgumentParser(description="Arabic-English Image Captioning")
#     parser.add_argument("--image", required=True, help="Path to image file or URL")
#     parser.add_argument("--language", choices=["arabic", "english", "both"], 
#                        default="both", help="Target language for caption")
#     parser.add_argument("--prompt", help="Custom prompt for caption generation")
#     parser.add_argument("--max_tokens", type=int, default=128, 
#                        help="Maximum number of tokens to generate")
#     parser.add_argument("--model", default="MBZUAI/AIN", 
#                        help="Hugging Face model name")
    
#     args = parser.parse_args()
    
#     # Check if image exists (for local files)
#     if not args.image.startswith(('http://', 'https://')) and not os.path.exists(args.image):
#         print(f"Error: Image file or directory '{args.image}' not found.")
#         return

#     # Initialize captioner
#     try:
#         captioner = ArabicImageCaptioner(model_name=args.model)
#     except Exception as e:
#         print(f"Failed to initialize captioner: {e}")
#         return

#     # Batch processing for directory
#     if not args.image.startswith(('http://', 'https://')) and os.path.isdir(args.image):
#         image_dir = args.image
#         image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
#         for fname in image_files:
#             fpath = os.path.join(image_dir, fname)
#             try:
#                 if args.language == "both":
#                     captions = captioner.generate_bilingual_captions(
#                         fpath,
#                         max_new_tokens=args.max_tokens
#                     )
#                     print("\n" + "="*50)
#                     print(f"BILINGUAL IMAGE CAPTIONS for {fname}")
#                     print("="*50)
#                     print(f"\nArabic Caption:")
#                     print(f"{captions['arabic']}")
#                     print(f"\nEnglish Caption:")
#                     print(f"{captions['english']}")
#                 else:
#                     caption = captioner.generate_caption(
#                         fpath,
#                         language=args.language,
#                         custom_prompt=args.prompt,
#                         max_new_tokens=args.max_tokens
#                     )
#                     print("\n" + "="*50)
#                     print(f"{args.language.upper()} IMAGE CAPTION for {fname}")
#                     print("="*50)
#                     print(f"\n{caption}")
#             except Exception as e:
#                 print(f"Error generating caption for {fname}: {e}")
#     else:
#         # Single image
#         try:
#             if args.language == "both":
#                 captions = captioner.generate_bilingual_captions(
#                     args.image, 
#                     max_new_tokens=args.max_tokens
#                 )
#                 print("\n" + "="*50)
#                 print("BILINGUAL IMAGE CAPTIONS")
#                 print("="*50)
#                 print(f"\nArabic Caption:")
#                 print(f"{captions['arabic']}")
#                 print(f"\nEnglish Caption:")
#                 print(f"{captions['english']}")
#             else:
#                 caption = captioner.generate_caption(
#                     args.image,
#                     language=args.language,
#                     custom_prompt=args.prompt,
#                     max_new_tokens=args.max_tokens
#                 )
#                 print("\n" + "="*50)
#                 print(f"{args.language.upper()} IMAGE CAPTION")
#                 print("="*50)
#                 print(f"\n{caption}")
#         except Exception as e:
#             print(f"Error generating caption: {e}")


# if __name__ == "__main__":
#     main()
