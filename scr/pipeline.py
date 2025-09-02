"""
End-to-End Mechanistic Interpretability Pipeline
==============================================

This script ties together the various components implemented in this package to
provide a full workflow for investigating, localising and mitigating gender
bias in multilingual image captioning models. It can be run from the command
line with arguments to specify the dataset location and which steps to
execute.

The pipeline is composed of the following stages:

1. **Activation Extraction**: Use ``ActivationExtractor`` to record
   hidden-layer activations of a vision–language model on your dataset. The
   extracted activations are stored on disk for reuse.
2. **Circuit Discovery (SAE Training)**: Train a sparse autoencoder on the
   pooled activations to discover interpretable latent circuits, using
   ``train_sae``.
3. **Model Fine-Tuning & Intervention**: Fine-tune the language model
   using gender-swapped prompts and targets to mitigate bias, via
   ``fine_tune_model``.
4. **Evaluation**: Generate captions using the baseline and fine-tuned
   models and compute fairness and quality metrics using functions from
   ``evaluation.py``.

Usage example::

    python -m mechanistic_system.pipeline \
        --dataset-dir /path/to/dataset \
        --activations-dir ./activations \
        --sae-dict-size 8192 \
        --fine-tune-output ./fine_tuned_model \
        --run-extraction \
        --run-sae-training \
        --run-fine-tuning \
        --run-evaluation

The script is modular: each stage can be run independently by specifying
appropriate flags. For large datasets, consider running activation
extraction once and caching the activations for subsequent runs.

Note: Running this pipeline on LLaVA models will require a GPU with
sufficient memory. For experimentation, you may use smaller language
models or restrict the number of samples.
"""

from __future__ import annotations

import argparse
import re
import os
import json
from typing import List, Dict, Optional

import torch

from mechanistic_system.activation_extractor import ActivationExtractor
from mechanistic_system.sae_training import load_activation_dataset, train_sae
from mechanistic_system.fine_tuning import fine_tune_model, FineTuneConfig
from mechanistic_system.evaluation import (
    gender_classification_metrics,
    caption_quality_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end mechanistic interpretability pipeline")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing dataset CSVs and images")
    parser.add_argument("--activations-dir", default="activations", help="Directory to store extracted activations")
    parser.add_argument("--sae-dict-size", type=int, default=4096, help="Dictionary size for sparse autoencoder")
    parser.add_argument("--sae-sparsity", type=float, default=1e-3, help="Sparsity penalty for SAE")
    parser.add_argument("--sae-epochs", type=int, default=5, help="Number of epochs for SAE training")
    parser.add_argument("--fine-tune-output", default="fine_tuned_model", help="Directory to save fine-tuned model")
    parser.add_argument("--run-extraction", action="store_true", help="Run activation extraction stage")
    parser.add_argument("--run-sae-training", action="store_true", help="Run sparse autoencoder training stage")
    parser.add_argument("--run-fine-tuning", action="store_true", help="Run fine-tuning stage")
    parser.add_argument("--run-evaluation", action="store_true", help="Run evaluation stage")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process (for quick tests)")
    parser.add_argument("--device", default="auto", help="Device for model loading (auto|cpu|cuda)")
    return parser.parse_args()


def load_dataset(dataset_dir: str, num_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """Load a combined dataset of image paths, prompts and ground-truth captions.

    The expected structure under ``dataset_dir`` is as follows:

        * ``image_caption.csv``: CSV with columns ``image``, ``en_caption``, ``ar_caption``
        * ``english_captions_gpt4o_results.csv``: Additional generated captions
        * ``arabic_captions_gpt4o_results.csv``: Additional generated captions

    This function reads the files and produces a list of samples with keys
    ``image_path``, ``en_caption``, ``ar_caption``. You can extend this to
    include prompts as needed.

    Parameters
    ----------
    dataset_dir: str
        Path to the dataset directory.
    num_samples: Optional[int]
        Limit on the number of samples to load for testing. If ``None`` all
        samples are loaded.

    Returns
    -------
    samples: List[Dict[str, str]]
        List of dictionary samples.
    """
    import pandas as pd
    image_csv = os.path.join(dataset_dir, "image_caption.csv")
    if not os.path.isfile(image_csv):
        raise FileNotFoundError(f"{image_csv} not found")
    df = pd.read_csv(image_csv)
    samples = []
    for idx, row in df.iterrows():
        sample = {
            "image_path": os.path.join(dataset_dir, "images", row["image"]) if "images" in os.listdir(dataset_dir) else row["image"],
            "en_caption": row.get("en_caption", ""),
            "ar_caption": row.get("ar_caption", ""),
        }
        samples.append(sample)
        if num_samples is not None and len(samples) >= num_samples:
            break
    return samples


def run_activation_extraction(args: argparse.Namespace, samples: List[Dict[str, str]]) -> None:
    """Extract and save activations for each sample in ``samples``."""
    extractor = ActivationExtractor(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        device="auto",
    )
    extractor.load_model()
    os.makedirs(args.activations_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        identifier = f"sample_{i:05d}"
        # Use the English caption as prompt for extraction. You can modify this
        prompt = sample["en_caption"] or "Describe the scene."
        activations = extractor.extract_activations(prompt=prompt, image_path=sample.get("image_path"))
        extractor.save_activations(identifier, activations, output_dir=args.activations_dir)
        if (i + 1) % 10 == 0:
            print(f"Extracted activations for {i + 1} samples")
    extractor.close()


def run_sae_training(args: argparse.Namespace) -> None:
    """Load activation dataset and train sparse autoencoder."""
    print("Loading activations...")
    activations, hidden_dim = load_activation_dataset(args.activations_dir)
    print(f"Loaded {activations.shape[0]} activation vectors with dimension {hidden_dim}")
    sae_model = train_sae(
        activations,
        hidden_dim=hidden_dim,
        dict_size=args.sae_dict_size,
        sparsity_penalty=args.sae_sparsity,
        num_epochs=args.sae_epochs,
    )
    # Save the trained SAE
    sae_save_path = os.path.join(args.activations_dir, "sae_model.pt")
    torch.save(sae_model.state_dict(), sae_save_path)
    print(f"Saved sparse autoencoder to {sae_save_path}")


def build_fine_tune_dataset(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build a dataset of prompt-target pairs for bias mitigation fine-tuning.

    We create prompts by combining the image placeholder with the ground-truth
    caption and create gender-swapped targets by replacing male terms with
    female terms and vice versa. You can customise this logic for your
    research needs.
    """
    dataset = []
    for sample in samples:
        # Use the English caption as the target (what we want model to output)
        target = sample.get("en_caption", "")
        # Prompt instructing model to describe the image
        prompt = "USER: <image>\nDescribe the person in the image.\nASSISTANT:"
        # Append original
        dataset.append({"prompt": prompt, "target": target})
        # Generate gender-swapped version of the target (simple regex)
        swapped = target
        # Swap English gender terms
        swaps = [
            (r"\bman\b", "woman"),
            (r"\bmen\b", "women"),
            (r"\bboy\b", "girl"),
            (r"\bfather\b", "mother"),
            (r"\bhe\b", "she"),
            (r"\bhis\b", "her"),
        ]
        for old, new in swaps:
            swapped = re.sub(old, new, swapped, flags=re.IGNORECASE)
        # Add swapped sample
        if swapped != target:
            dataset.append({"prompt": prompt, "target": swapped})
    return dataset


def run_fine_tuning(args: argparse.Namespace, samples: List[Dict[str, str]]) -> None:
    """Run the fine-tuning stage on the constructed dataset."""
    # Build dataset
    train_data = build_fine_tune_dataset(samples)
    # Very small eval set for demonstration
    eval_data = train_data[: max(2, len(train_data) // 10)]
    # Configure and launch training
    config = FineTuneConfig(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        output_dir=args.fine_tune_output,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-5,
        use_lora=True,
    )
    fine_tune_model(config, dataset=train_data, eval_dataset=eval_data)


def generate_captions(
    model_name: str,
    samples: List[Dict[str, str]],
    device: Optional[str] = None,
    max_samples: int = 20
) -> List[str]:
    """Generate captions for the first ``max_samples`` images using a LLaVA model.

    Parameters
    ----------
    model_name: str
        Name or path of the model to load.
    samples: List[Dict[str, str]]
        List of dataset samples.
    device: Optional[str]
        Device to load the model on. If ``None``, defaults to 'cuda' if
        available, otherwise 'cpu'.
    max_samples: int
        Maximum number of samples to caption.

    Returns
    -------
    captions: List[str]
        List of generated captions.
    """
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from PIL import Image
    import numpy as np
    # Determine device
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load processor and model
    processor = LlavaNextProcessor.from_pretrained(model_name)
    # Attempt to load with specified device map for memory efficiency
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            device_map={"": device} if device != "cuda" else None,
        )
    except Exception:
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name)
    # Ensure model on correct device
    model.to(device)
    model.eval()
    captions = []
    for sample in samples[: max_samples]:
        # Use a generic prompt instructing the model to describe the person
        prompt = "USER: <image>\nDescribe the person in the image.\nASSISTANT:"
        image_path = sample.get("image_path")
        if image_path and os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # Fallback to dummy image with appropriate size from model config
            size = model.config.vision_config.image_size
            image = Image.fromarray((np.zeros((size, size, 3), dtype=np.uint8)))
        # Prepare inputs and send to device
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        captions.append(caption)
    return captions


def run_evaluation(args: argparse.Namespace, samples: List[Dict[str, str]]) -> None:
    """Run evaluation comparing baseline and fine-tuned models.

    This evaluation generates captions using the baseline model (pretrained
    LLaVA) and the fine-tuned model on the provided evaluation subset.
    Gender labels are inferred for each image using a CLIP-based classifier
    that compares the similarity of the image to "man" and "woman"
    prompts. Captions are then classified using heuristics from
    ``infer_gender_from_caption``. Metrics summarise overall accuracy
    and bias gap across male and female classes. Caption quality is
    measured with BLEU and ROUGE using the reference English captions.
    """
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    # Determine device for CLIP (use user-specified device if provided)
    if args.device == "auto" or args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Load CLIP model and processor once
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_model.eval()

    def classify_gender_image(path: str) -> str:
        """Classify the gender of a person in an image using CLIP.

        Returns "man", "woman" or "neutral". If the image file is missing or
        classification is ambiguous (probabilities within 0.1), returns
        "neutral". The classification compares the similarity between the
        image and prompts "a photo of a man" and "a photo of a woman".
        """
        try:
            if path and os.path.isfile(path):
                image = Image.open(path).convert("RGB")
                prompts = ["a photo of a man", "a photo of a woman"]
                inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    logits = outputs.logits_per_image  # shape [1, 2]
                    probs = logits.softmax(dim=1).squeeze().cpu().numpy()
                # If probabilities are similar, label as neutral
                if abs(probs[0] - probs[1]) < 0.1:
                    return "neutral"
                import numpy as _np
                return "man" if int(_np.argmax(probs)) == 0 else "woman"
        except Exception:
            pass
        return "neutral"

    # Use a subset for efficiency
    eval_samples = samples[: args.num_samples or 20]
    # Generate captions from baseline model
    print("Generating captions from baseline model...")
    baseline_captions = generate_captions(
        "llava-hf/llava-v1.6-mistral-7b-hf", eval_samples, device=args.device
    )
    # Generate captions from fine-tuned model
    print("Generating captions from fine-tuned model...")
    tuned_captions = generate_captions(args.fine_tune_output, eval_samples, device=args.device)
    # Prepare gender labels via image classification and reference captions
    labels = []
    references = []
    for sample in eval_samples:
        image_path = sample.get("image_path")
        labels.append(classify_gender_image(image_path))
        references.append(sample.get("en_caption", ""))
    # Compute gender classification metrics
    baseline_metrics = gender_classification_metrics(baseline_captions, labels)
    tuned_metrics = gender_classification_metrics(tuned_captions, labels)
    print("Baseline gender metrics:", baseline_metrics)
    print("Tuned gender metrics:", tuned_metrics)
    # Compute quality metrics (BLEU, ROUGE)
    if references:
        baseline_quality = caption_quality_metrics(baseline_captions, references)
        tuned_quality = caption_quality_metrics(tuned_captions, references)
        print("Baseline quality metrics:", baseline_quality)
        print("Tuned quality metrics:", tuned_quality)


def main() -> None:
    args = parse_args()
    # Load dataset
    samples = load_dataset(args.dataset_dir, num_samples=args.num_samples)
    if args.run_extraction:
        run_activation_extraction(args, samples)
    if args.run_sae_training:
        run_sae_training(args)
    if args.run_fine_tuning:
        run_fine_tuning(args, samples)
    if args.run_evaluation:
        run_evaluation(args, samples)


if __name__ == "__main__":
    main()
