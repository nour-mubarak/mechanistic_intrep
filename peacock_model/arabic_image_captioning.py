#!/usr/bin/env python3
"""
Arabic-English Image Captioning using AIN (Arabic Inclusive Large Multimodal Model)

This script provides a complete implementation for bilingual image captioning
using the AIN model from MBZUAI. It supports both Arabic and English caption generation.

Author: Manus AI
Date: September 2025
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from typing import Optional, Union
import argparse
import os


class ArabicImageCaptioner:
    """
    A class for generating Arabic and English image captions using the AIN model.
    """
    
    def __init__(self, model_name: str = "MBZUAI/AIN", device: str = "auto"):
        """
        Initialize the Arabic Image Captioner.
        
        Args:
            model_name (str): The Hugging Face model name
            device (str): Device to run the model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
        print(f"Initializing Arabic Image Captioner with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the model and processor."""
        try:
            print("Loading model and processor...")
            
            # Load the model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device
            )
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            print("Model and processor loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have the required dependencies installed:")
            print("pip install transformers torch qwen-vl-utils pillow")
            raise
    
    def generate_caption(
        self, 
        image_path: str, 
        language: str = "arabic",
        custom_prompt: Optional[str] = None,
        max_new_tokens: int = 128
    ) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image_path (str): Path to the image file or URL
            language (str): Target language ('arabic' or 'english')
            custom_prompt (str, optional): Custom prompt for caption generation
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated caption
        """
        
        # Load image
        if image_path.startswith(('http://', 'https://')):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        
        # Prepare prompt based on language
        if custom_prompt:
            prompt = custom_prompt
        elif language.lower() == "arabic":
            prompt = "يرجى وصف هذه الصورة بالتفصيل."  # "Please describe this image in detail."
        else:
            prompt = "Please describe this image in detail."
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text_prompt], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        )
        
        # Move inputs to device
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = inputs.to("cuda")
        
        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        return output_text[0] if output_text else ""
    
    def generate_bilingual_captions(
        self, 
        image_path: str, 
        max_new_tokens: int = 128
    ) -> dict:
        """
        Generate both Arabic and English captions for an image.
        
        Args:
            image_path (str): Path to the image file or URL
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            dict: Dictionary containing both Arabic and English captions
        """
        
        print(f"Generating bilingual captions for: {image_path}")
        
        # Generate Arabic caption
        print("Generating Arabic caption...")
        arabic_caption = self.generate_caption(
            image_path, 
            language="arabic", 
            max_new_tokens=max_new_tokens
        )
        
        # Generate English caption
        print("Generating English caption...")
        english_caption = self.generate_caption(
            image_path, 
            language="english", 
            max_new_tokens=max_new_tokens
        )
        
        return {
            "arabic": arabic_caption,
            "english": english_caption
        }


def main():
    """Main function to run the image captioning script."""
    
    parser = argparse.ArgumentParser(description="Arabic-English Image Captioning")
    parser.add_argument("--image", required=True, help="Path to image file or URL")
    parser.add_argument("--language", choices=["arabic", "english", "both"], 
                       default="both", help="Target language for caption")
    parser.add_argument("--prompt", help="Custom prompt for caption generation")
    parser.add_argument("--max_tokens", type=int, default=128, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--model", default="MBZUAI/AIN", 
                       help="Hugging Face model name")
    
    args = parser.parse_args()
    
    # Check if image exists (for local files)
    if not args.image.startswith(('http://', 'https://')) and not os.path.exists(args.image):
        print(f"Error: Image file or directory '{args.image}' not found.")
        return

    # Initialize captioner
    try:
        captioner = ArabicImageCaptioner(model_name=args.model)
    except Exception as e:
        print(f"Failed to initialize captioner: {e}")
        return

    # Batch processing for directory
    if not args.image.startswith(('http://', 'https://')) and os.path.isdir(args.image):
        image_dir = args.image
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        for fname in image_files:
            fpath = os.path.join(image_dir, fname)
            try:
                if args.language == "both":
                    captions = captioner.generate_bilingual_captions(
                        fpath,
                        max_new_tokens=args.max_tokens
                    )
                    print("\n" + "="*50)
                    print(f"BILINGUAL IMAGE CAPTIONS for {fname}")
                    print("="*50)
                    print(f"\nArabic Caption:")
                    print(f"{captions['arabic']}")
                    print(f"\nEnglish Caption:")
                    print(f"{captions['english']}")
                else:
                    caption = captioner.generate_caption(
                        fpath,
                        language=args.language,
                        custom_prompt=args.prompt,
                        max_new_tokens=args.max_tokens
                    )
                    print("\n" + "="*50)
                    print(f"{args.language.upper()} IMAGE CAPTION for {fname}")
                    print("="*50)
                    print(f"\n{caption}")
            except Exception as e:
                print(f"Error generating caption for {fname}: {e}")
    else:
        # Single image
        try:
            if args.language == "both":
                captions = captioner.generate_bilingual_captions(
                    args.image, 
                    max_new_tokens=args.max_tokens
                )
                print("\n" + "="*50)
                print("BILINGUAL IMAGE CAPTIONS")
                print("="*50)
                print(f"\nArabic Caption:")
                print(f"{captions['arabic']}")
                print(f"\nEnglish Caption:")
                print(f"{captions['english']}")
            else:
                caption = captioner.generate_caption(
                    args.image,
                    language=args.language,
                    custom_prompt=args.prompt,
                    max_new_tokens=args.max_tokens
                )
                print("\n" + "="*50)
                print(f"{args.language.upper()} IMAGE CAPTION")
                print("="*50)
                print(f"\n{caption}")
        except Exception as e:
            print(f"Error generating caption: {e}")


if __name__ == "__main__":
    main()
