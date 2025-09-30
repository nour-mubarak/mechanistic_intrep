#!/usr/bin/env python3
"""
Batch Arabic-English Image Captioning with Custom Prompts

This script processes a dataset of images using the AIN model with custom prompts
for generating both Arabic and English captions in different prompt configurations.

Author: Manus AI
Date: September 2025
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import pandas as pd
import os
import argparse
from tqdm import tqdm
import time
from typing import List, Dict, Optional


class BatchArabicImageCaptioner:
    """
    A class for batch processing images with custom Arabic and English prompts.
    """
    
    def __init__(self, model_name: str = "MBZUAI/AIN", device: str = "auto"):
        """
        Initialize the Batch Arabic Image Captioner.
        
        Args:
            model_name (str): The Hugging Face model name
            device (str): Device to run the model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
        # Define the custom prompts
        self.prompts = {
            'ENGLISH_PROMPT_FOR_ENGLISH': "Write a single, concise sentence caption for this image in English.",
            'ARABIC_PROMPT_FOR_ENGLISH': "اكتب تعليقًا من جملة واحدة لهذه الصورة باللغة الإنجليزية.",
            'ENGLISH_PROMPT_FOR_ARABIC': "Write a single, concise sentence caption for this image in Arabic.",
            'ARABIC_PROMPT_FOR_ARABIC': "اكتب جملة واحدة مختصرة لوصف هذه الصورة باللغة العربية."
        }
        
        print(f"Initializing Batch Arabic Image Captioner with model: {model_name}")
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
            raise
    
    def _generate_single_caption(
        self, 
        image_path: str, 
        prompt: str,
        max_new_tokens: int = 128
    ) -> str:
        """
        Generate a single caption for an image with a specific prompt.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): The prompt to use for caption generation
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated caption
        """
        try:
            # Load image
            image = Image.open(image_path)
            
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
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return f"Error: {str(e)}"
    
    def process_image_with_all_prompts(
        self, 
        image_path: str,
        max_new_tokens: int = 128
    ) -> Dict[str, str]:
        """
        Process a single image with all four prompt configurations.
        
        Args:
            image_path (str): Path to the image file
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            dict: Dictionary containing all four caption variations
        """
        results = {}
        
        # Generate captions with each prompt
        for prompt_name, prompt_text in self.prompts.items():
            print(f"  Processing with {prompt_name}...")
            caption = self._generate_single_caption(
                image_path, 
                prompt_text, 
                max_new_tokens
            )
            results[prompt_name] = caption
            
            # Small delay to prevent overwhelming the GPU
            time.sleep(0.1)
        
        return results
    
    def process_dataset(
        self, 
        image_folder: str,
        image_list: Optional[List[str]] = None,
        output_file: str = "batch_captions_results.csv",
        max_new_tokens: int = 128
    ) -> pd.DataFrame:
        """
        Process a dataset of images and generate captions with all prompt configurations.
        
        Args:
            image_folder (str): Path to the folder containing images
            image_list (List[str], optional): List of specific image filenames to process
            output_file (str): Output CSV file name
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            pd.DataFrame: DataFrame containing all results
        """
        
        # Get list of images to process
        if image_list is None:
            # Process all images in the folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_list = [
                f for f in os.listdir(image_folder) 
                if os.path.splitext(f.lower())[1] in image_extensions
            ]
        
        print(f"Processing {len(image_list)} images...")
        
        # Initialize results list
        results = []
        
        # Process each image
        for i, image_filename in enumerate(tqdm(image_list, desc="Processing images")):
            image_path = os.path.join(image_folder, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping...")
                continue
            
            print(f"\nProcessing image {i+1}/{len(image_list)}: {image_filename}")
            
            # Generate captions with all prompts
            captions = self.process_image_with_all_prompts(image_path, max_new_tokens)
            
            # Create result row
            result_row = {
                'image_filename': image_filename,
                'ar_from_en_prompt': captions.get('ENGLISH_PROMPT_FOR_ARABIC', ''),
                'ar_from_ar_prompt': captions.get('ARABIC_PROMPT_FOR_ARABIC', ''),
                'en_from_en_prompt': captions.get('ENGLISH_PROMPT_FOR_ENGLISH', ''),
                'en_from_ar_prompt': captions.get('ARABIC_PROMPT_FOR_ENGLISH', ''),
                'model_used': self.model_name
            }
            
            results.append(result_row)
            
            # Save intermediate results every 10 images
            if (i + 1) % 10 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(f"temp_{output_file}", index=False)
                print(f"Saved intermediate results after {i+1} images")
        
        # Create final DataFrame
        df_results = pd.DataFrame(results)
        
        # Save to CSV
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df_results


def main():
    """Main function to run the batch processing script."""
    
    parser = argparse.ArgumentParser(description="Batch Arabic-English Image Captioning")
    parser.add_argument("--image_folder", required=True, 
                       help="Path to folder containing images")
    parser.add_argument("--image_list", 
                       help="Path to text file containing list of image filenames (one per line)")
    parser.add_argument("--output", default="batch_captions_results.csv",
                       help="Output CSV file name")
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--model", default="MBZUAI/AIN",
                       help="Hugging Face model name")
    
    args = parser.parse_args()
    
    # Check if image folder exists
    if not os.path.exists(args.image_folder):
        print(f"Error: Image folder '{args.image_folder}' not found.")
        return
    
    # Load image list if provided
    image_list = None
    if args.image_list:
        if os.path.exists(args.image_list):
            with open(args.image_list, 'r') as f:
                image_list = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(image_list)} image filenames from {args.image_list}")
        else:
            print(f"Warning: Image list file '{args.image_list}' not found. Processing all images in folder.")
    
    # Initialize batch captioner
    try:
        captioner = BatchArabicImageCaptioner(model_name=args.model)
    except Exception as e:
        print(f"Failed to initialize captioner: {e}")
        return
    
    # Process the dataset
    try:
        results_df = captioner.process_dataset(
            image_folder=args.image_folder,
            image_list=image_list,
            output_file=args.output,
            max_new_tokens=args.max_tokens
        )
        
        print(f"\nBatch processing completed!")
        print(f"Processed {len(results_df)} images")
        print(f"Results saved to: {args.output}")
        
        # Display sample results
        print("\nSample results:")
        print(results_df.head())
        
    except Exception as e:
        print(f"Error during batch processing: {e}")


if __name__ == "__main__":
    main()
