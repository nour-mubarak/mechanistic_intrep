#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arabic Image Captioning using CLIP + Arabic Language Model
Combines CLIP for image understanding with Arabic LLMs for caption generation
"""

import os
import sys
import glob
import csv
import time
import argparse
from typing import Optional, List, Dict
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from transformers import (
        CLIPProcessor,
        CLIPModel,
        AutoTokenizer,
        AutoModelForCausalLM,
        BlipProcessor,
        BlipForConditionalGeneration,
    )
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install: pip install torch transformers sentence-transformers pillow")
    sys.exit(1)

# ---- Configuration ----
DEFAULT_IMAGE_DIR = "/home2/jmsk62/project/mechanistic_intrep/dataset/images"
DEFAULT_OUTPUT_CSV = "/home2/jmsk62/project/mechanistic_intrep/arabic_clip_results.csv"

# Model configurations - multiple approaches
CLIP_MODELS = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]

ARABIC_LLM_MODELS = [
    "tiiuae/falcon-7b-instruct",  # Has some Arabic capability
    "microsoft/DialoGPT-medium",  # Fallback
    "aubmindlab/bert-base-arabertv2",  # Arabic-specific (encoder only)
]

# Arabic embedding models for similarity matching
ARABIC_EMBEDDING_MODELS = [
    "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

# Predefined templates for both languages
ARABIC_TEMPLATES = {
    "person": [
        "شخص يقف في المكان",
        "رجل يرتدي ملابس",
        "امرأة تقوم بنشاط",
        "أشخاص يتفاعلون معاً",
    ],
    "animal": [
        "حيوان في البيئة الطبيعية",
        "قط يلعب",
        "كلب يجري",
        "طائر يطير في السماء",
    ],
    "outdoor": ["منظر طبيعي جميل", "مشهد خارجي", "حديقة خضراء", "جبال وسماء"],
    "indoor": ["غرفة مرتبة", "مطبخ نظيف", "صالة مريحة", "مكتب للعمل"],
    "vehicle": ["سيارة على الطريق", "وسيلة نقل", "مركبة متحركة", "نقل عام"],
    "food": ["طعام شهي", "وجبة لذيذة", "أكل صحي", "مأكولات متنوعة"],
}

ENGLISH_TEMPLATES = {
    "person": [
        "A person standing in a location",
        "A man wearing clothes",
        "A woman doing an activity",
        "People interacting together",
    ],
    "animal": [
        "An animal in its natural environment",
        "A cat playing",
        "A dog running",
        "A bird flying in the sky",
    ],
    "outdoor": [
        "A beautiful natural landscape",
        "An outdoor scene",
        "A green garden",
        "Mountains and sky",
    ],
    "indoor": [
        "A tidy room",
        "A clean kitchen",
        "A comfortable living room",
        "A workspace",
    ],
    "vehicle": [
        "A car on the road",
        "A means of transportation",
        "A moving vehicle",
        "Public transportation",
    ],
    "food": ["Delicious food", "A tasty meal", "Healthy food", "Various dishes"],
}


class ArabicImageCaptioner:
    def caption_image(self, image_path: str, languages=["arabic", "english"]) -> dict:
        """Generate captions for the given image in Arabic and/or English."""
        # Extract image features
        image_features = self.get_image_features(image_path)
        if image_features is None:
            return {"error": "Failed to extract image features"}

        # Classify image category
        category = self.classify_image_category(image_features)
        result = {"category": category}

        # Generate captions for requested languages
        for lang in languages:
            if lang.lower() == "arabic":
                config = ("arabic", "arabic", "", "arabic_caption")
            elif lang.lower() == "english":
                config = ("arabic", "english", "", "english_caption")
            else:
                continue
            caption = self.generate_caption_from_config(
                image_features, category, config
            )
            result[config[3]] = caption
        return result

    def __init__(self, force_cpu: bool = False):
        self.device = (
            "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        self.clip_model = None
        self.clip_processor = None
        self.arabic_embedder = None
        self.llm_model = None
        self.llm_tokenizer = None

        self._load_models()

    def _load_models(self):
        """Load CLIP and Arabic models with fallbacks"""

        # Load CLIP model
        print("Loading CLIP model...")
        for clip_model_name in CLIP_MODELS:
            try:
                self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(
                    self.device
                )
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
                print(f"✓ Loaded CLIP model: {clip_model_name}")
                break
            except Exception as e:
                print(f"Failed to load {clip_model_name}: {e}")

        if self.clip_model is None:
            print("Failed to load any CLIP model. Trying BLIP as fallback...")
            try:
                self.clip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self.device)
                self.clip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                print("✓ Loaded BLIP as fallback")
            except Exception as e:
                print(f"Failed to load BLIP fallback: {e}")
                raise Exception("Could not load any vision model")

        # Load Arabic sentence embedder
        print("Loading Arabic embedding model...")
        for embed_model_name in ARABIC_EMBEDDING_MODELS:
            try:
                self.arabic_embedder = SentenceTransformer(embed_model_name)
                print(f"✓ Loaded embedding model: {embed_model_name}")
                break
            except Exception as e:
                print(f"Failed to load {embed_model_name}: {e}")

        # Optional: Load Arabic LLM (if available)
        print("Loading Arabic LLM (optional)...")
        for llm_model_name in ARABIC_LLM_MODELS:
            try:
                if "bert" in llm_model_name.lower():
                    continue  # Skip encoder-only models for generation

                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name
                ).to(self.device)

                # Add pad token if missing
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                print(f"✓ Loaded LLM: {llm_model_name}")
                break
            except Exception as e:
                print(f"Failed to load LLM {llm_model_name}: {e}")

        if self.llm_model is None:
            print("Warning: No Arabic LLM loaded. Will use template matching only.")

    def get_image_features(self, image_path: str):
        """Extract features from image using CLIP"""
        try:
            image = Image.open(image_path).convert("RGB")

            if hasattr(self.clip_model, "get_image_features"):
                # CLIP model
                inputs = self.clip_processor(images=image, return_tensors="pt").to(
                    self.device
                )
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                return image_features
            else:
                # BLIP model - generate basic caption first
                inputs = self.clip_processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.clip_model.generate(**inputs, max_new_tokens=20)
                caption = self.clip_processor.decode(
                    output[0], skip_special_tokens=True
                )
                return caption

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def classify_image_category(self, image_features) -> str:
        """Classify image into predefined categories using CLIP"""
        if isinstance(image_features, str):
            # BLIP caption - simple keyword matching
            caption_lower = image_features.lower()
            if any(
                word in caption_lower for word in ["person", "man", "woman", "people"]
            ):
                return "person"
            elif any(
                word in caption_lower for word in ["cat", "dog", "bird", "animal"]
            ):
                return "animal"
            elif any(
                word in caption_lower for word in ["car", "truck", "vehicle", "bike"]
            ):
                return "vehicle"
            elif any(word in caption_lower for word in ["food", "eat", "meal"]):
                return "food"
            elif any(
                word in caption_lower for word in ["room", "kitchen", "house", "indoor"]
            ):
                return "indoor"
            else:
                return "outdoor"

        # CLIP features - use text classification
        try:
            category_texts = [
                "a photo of a person",
                "a photo of an animal",
                "an outdoor scene",
                "an indoor scene",
                "a photo of a vehicle",
                "a photo of food",
            ]

            text_inputs = self.clip_processor(
                text=category_texts, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)

            # Calculate similarity
            similarities = F.cosine_similarity(image_features, text_features)
            best_category_idx = similarities.argmax().item()

            categories = ["person", "animal", "outdoor", "indoor", "vehicle", "food"]
            return categories[best_category_idx]

        except Exception as e:
            print(f"Error in category classification: {e}")
            return "outdoor"  # Default fallback

    def generate_caption_from_config(
        self, image_features, category: str, config: tuple
    ) -> str:
        """Generate caption using specific prompt configuration"""
        prompt_lang, output_lang, prompt_text, column_name = config

        if self.llm_model is None or self.llm_tokenizer is None:
            # Fallback to template if no LLM
            templates = ARABIC_TEMPLATES if output_lang == "AR" else ENGLISH_TEMPLATES
            category_templates = templates.get(category, templates["outdoor"])
            return category_templates[0]

        try:
            # Use the exact prompt from configuration

            # Use tokenizer() to get attention_mask
            enc = self.llm_tokenizer(prompt_text, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=40,  # Slightly longer for more descriptive captions
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )

            generated_text = self.llm_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # Extract only the generated part (after prompt)
            if prompt_text in generated_text:
                generated_caption = generated_text.replace(prompt_text, "").strip()
            else:
                generated_caption = generated_text.strip()

            return (
                generated_caption
                if generated_caption
                else self._get_fallback_caption(category, output_lang)
            )

        except Exception as e:
            print(f"Error in config-based generation: {e}")
            return self._get_fallback_caption(category, output_lang)

    def _get_fallback_caption(self, category: str, output_lang: str) -> str:
        """Get fallback caption from templates"""
        templates = ARABIC_TEMPLATES if output_lang == "AR" else ENGLISH_TEMPLATES
        category_templates = templates.get(category, templates["outdoor"])
        return category_templates[0]

    def generate_caption_llm(
        self, image_features, category: str, language: str = "arabic"
    ) -> str:
        """Generate caption using LLM for specified language"""
        if self.llm_model is None or self.llm_tokenizer is None:
            return None

        try:
            # Create appropriate prompt based on language
            if language == "arabic":
                prompt = f"صف هذه الصورة باللغة العربية في جملة واحدة. الصورة تحتوي على: {category}"
            else:  # English
                prompt = f"Describe this image in English in one sentence. The image contains: {category}"

            # Use tokenizer() to get attention_mask
            enc = self.llm_tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )

            generated_text = self.llm_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # Extract only the generated part (after prompt)
            if prompt in generated_text:
                generated_caption = generated_text.replace(prompt, "").strip()
            else:
                generated_caption = generated_text.strip()

            return generated_caption if generated_caption else None

        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return None

    def select_best_caption(
        self, candidates: List[str], language: str = "arabic"
    ) -> str:
        """Select best caption from candidates using appropriate embeddings"""
        if not candidates:
            return "صورة جميلة" if language == "arabic" else "A beautiful image"

        if len(candidates) == 1:
            return candidates[0]

        if self.arabic_embedder is None:
            return candidates[0]  # Use first if no embedder

        try:
            # Use embeddings to select most natural-sounding caption
            embeddings = self.arabic_embedder.encode(candidates)

            # Simple heuristic: select the one that's most different from others
            # (indicating uniqueness/specificity)
            similarities = []
            for i, emb1 in enumerate(embeddings):
                avg_sim = sum(
                    F.cosine_similarity(
                        torch.tensor(emb1).unsqueeze(0), torch.tensor(emb2).unsqueeze(0)
                    ).item()
                    for j, emb2 in enumerate(embeddings)
                    if i != j
                ) / (len(embeddings) - 1)
                similarities.append(avg_sim)

            # Select the one with moderate uniqueness (not too generic, not too weird)
            best_idx = similarities.index(sorted(similarities)[len(similarities) // 2])
            return candidates[best_idx]

        except Exception as e:
            print(f"Error in caption selection: {e}")
            return candidates[0]

    def caption_image_with_configs(self, image_path: str) -> Dict[str, str]:
        """Generate captions using all 4 test configurations"""
        try:
            # Extract image features
            image_features = self.get_image_features(image_path)
            if image_features is None:
                return {"error": "Failed to extract image features"}

            # Classify image category
            category = self.classify_image_category(image_features)

            results = {"category": category}

            # Generate captions for all 4 configurations
            for config in TEST_CONFIGS:
                prompt_lang, output_lang, prompt_text, column_name = config

                print(f"  {prompt_lang} -> {output_lang}...")
                caption = self.generate_caption_from_config(
                    image_features, category, config
                )
                results[column_name] = caption

                # Show preview
                preview = caption[:60] + "..." if len(caption) > 60 else caption
                print(f"    {preview}")

            results["method"] = "4-config-llm+template"
            return results

        except Exception as e:
            return {"error": f"Captioning failed: {str(e)}"}


# Define the 4 test configurations: (prompt_lang, output_lang, prompt_text, column_name)
TEST_CONFIGS = [
    (
        "english",
        "arabic",
        "صف هذه الصورة باللغة العربية في جملة واحدة.",
        "ar_from_en_prompt",
    ),
    ("arabic", "arabic", "صف هذه الصورة في جملة واحدة.", "ar_from_ar_prompt"),
    (
        "english",
        "english",
        "Describe this image in English in one sentence.",
        "en_from_en_prompt",
    ),
    (
        "arabic",
        "english",
        "Describe this image in English in one sentence.",
        "en_from_ar_prompt",
    ),
]


def process_images_with_configs(
    captioner: ArabicImageCaptioner,
    image_dir: str,
    output_csv: str,
    max_images: Optional[int] = None,
):
    """Process all images using 4-configuration approach"""

    # Find images
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    if max_images:
        image_files = image_files[:max_images]

    print(f"Processing {len(image_files)} images with 4-configuration approach...")

    # Create output directory
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    start_time = time.time()
    results = []

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "image_filename",
                "ar_from_en_prompt",  # Arabic from English prompt
                "ar_from_ar_prompt",  # Arabic from Arabic prompt
                "en_from_en_prompt",  # English from English prompt
                "en_from_ar_prompt",  # English from Arabic prompt
                "category",
                "method",
                "processing_time_sec",
                "status",
            ]
        )

        for i, image_path in enumerate(image_files):
            img_start = time.time()
            filename = os.path.basename(image_path)

            print(f"\n[{i+1}/{len(image_files)}] Processing: {filename}")

            # Generate captions using all 4 configs
            result = captioner.caption_image_with_configs(image_path)
            processing_time = time.time() - img_start

            if "error" in result:
                print(f"  Error: {result['error']}")
                writer.writerow(
                    [
                        filename,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        f"{processing_time:.2f}",
                        "error",
                    ]
                )
            else:
                print(f"  Category: {result['category']}")
                writer.writerow(
                    [
                        filename,
                        result.get("ar_from_en_prompt", ""),
                        result.get("ar_from_ar_prompt", ""),
                        result.get("en_from_en_prompt", ""),
                        result.get("en_from_ar_prompt", ""),
                        result["category"],
                        result["method"],
                        f"{processing_time:.2f}",
                        "success",
                    ]
                )

            results.append(result)

            # Flush every 5 images
            if (i + 1) % 5 == 0:
                csvfile.flush()
                print(f"  Progress: {i+1}/{len(image_files)} images processed")

    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(image_files):.2f} seconds")
    print(f"Results saved to: {output_csv}")

    # Show sample results
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        print(f"\nSuccessful captions: {len(successful_results)}/{len(results)}")
        print("\nSample results:")
        for i, result in enumerate(successful_results[:2]):
            print(f"\nImage {i+1} (Category: {result['category']}):")
            print(f"  AR from EN: {result.get('ar_from_en_prompt', '')}")
            print(f"  AR from AR: {result.get('ar_from_ar_prompt', '')}")
            print(f"  EN from EN: {result.get('en_from_en_prompt', '')}")
            print(f"  EN from AR: {result.get('en_from_ar_prompt', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="Bilingual Image Captioning with CLIP + Arabic LLM"
    )
    parser.add_argument(
        "--images_dir", default=DEFAULT_IMAGE_DIR, help="Directory containing images"
    )
    parser.add_argument(
        "--output_csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV file"
    )
    parser.add_argument(
        "--max_images", type=int, help="Maximum number of images to process"
    )
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--use_configs",
        action="store_true",
        help="Use 4-configuration approach (EN→AR, AR→AR, EN→EN, AR→EN)",
    )

    args = parser.parse_args()

    print("=== Bilingual Image Captioning with CLIP + Arabic LLM ===")
    print(f"Images directory: {args.images_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Max images: {args.max_images or 'All'}")
    print(f"Mode: {'4-Configuration' if args.use_configs else 'Simple bilingual'}")
    print(f"Force CPU: {args.force_cpu}")

    if args.use_configs:
        print("\n4-Configuration approach will test:")
        for config in TEST_CONFIGS:
            prompt_lang, output_lang, prompt_text, column_name = config
            print(f"  {prompt_lang}→{output_lang}: {prompt_text[:50]}...")

    # Initialize captioner
    try:
        captioner = ArabicImageCaptioner(force_cpu=args.force_cpu)
    except Exception as e:
        print(f"Failed to initialize captioner: {e}")
        return 1

    # Test with one image first
    test_images = glob.glob(os.path.join(args.images_dir, "*.jpg"))[:1]
    if test_images:
        print(f"\n=== Testing with: {os.path.basename(test_images[0])} ===")

        if args.use_configs:
            test_result = captioner.caption_image_with_configs(test_images[0])
            if "error" in test_result:
                print(f"Test failed: {test_result['error']}")
                return 1
            else:
                print(f"Test successful!")
                print(f"Category: {test_result['category']}")
                for config in TEST_CONFIGS:
                    _, _, _, column_name = config
                    caption = test_result.get(column_name, "")
                    print(f"{column_name}: {caption}")
        else:
            # Simple test with both languages
            test_result = captioner.caption_image(test_images[0], ["arabic", "english"])
            if "error" in test_result:
                print(f"Test failed: {test_result['error']}")
                return 1
            else:
                print(f"Test successful!")
                print(f"Arabic: {test_result.get('arabic_caption', '')}")
                print(f"English: {test_result.get('english_caption', '')}")
                print(f"Category: {test_result['category']}")

    # Process all images
    if args.use_configs:
        process_images_with_configs(
            captioner, args.images_dir, args.output_csv, args.max_images
        )
    else:
        # Fallback to simple bilingual processing using process_images_with_configs
        process_images_with_configs(
            captioner,
            args.images_dir,
            args.output_csv,
            args.max_images,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
