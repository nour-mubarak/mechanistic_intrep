"""
Dataset Utilities for Cross-Lingual Captioning Analysis
========================================================

Data loading, preprocessing, and batch generation for
Arabic-English image captioning bias analysis.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from dataclasses import dataclass
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CaptionSample:
    """Single sample for captioning analysis."""
    image_id: str
    image_path: Path
    english_prompt: str
    arabic_prompt: str
    ground_truth_gender: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CrossLingualCaptionDataset(Dataset):
    """
    Dataset for cross-lingual image captioning analysis.
    
    Supports:
    - Paired Arabic/English prompts for same images
    - Ground truth gender labels
    - Flexible prompt templates
    - Image preprocessing
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        csv_path: Optional[Union[str, Path]] = None,
        english_prompt_template: str = "Describe the person in this image.",
        arabic_prompt_template: str = "صف الشخص في هذه الصورة.",
        image_size: int = 448,
        max_samples: Optional[int] = None,
        transform: Optional[Any] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            csv_path: Path to CSV with metadata (image_id, ground_truth_gender)
            english_prompt_template: Template for English prompts
            arabic_prompt_template: Template for Arabic prompts
            image_size: Target image size
            max_samples: Maximum samples to load
            transform: Optional image transform
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.english_template = english_prompt_template
        self.arabic_template = arabic_prompt_template
        self.transform = transform
        
        self.samples: List[CaptionSample] = []
        self._load_data(csv_path, max_samples)
    
    def _load_data(
        self,
        csv_path: Optional[Union[str, Path]],
        max_samples: Optional[int]
    ) -> None:
        """Load dataset from directory and optional CSV."""
        
        if csv_path and Path(csv_path).exists():
            # Load from CSV
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                if max_samples and idx >= max_samples:
                    break
                
                image_id = str(row.get("image_id", row.get("image", row.get("id", idx))))
                
                # Find image file
                image_path = self._find_image(image_id)
                if image_path is None:
                    continue
                
                # Get prompts (use templates or custom from CSV)
                english_prompt = row.get("english_prompt", self.english_template)
                arabic_prompt = row.get("arabic_prompt", self.arabic_template)
                
                # Get gender label
                gender = row.get("ground_truth_gender", row.get("gender", None))
                
                self.samples.append(CaptionSample(
                    image_id=image_id,
                    image_path=image_path,
                    english_prompt=english_prompt,
                    arabic_prompt=arabic_prompt,
                    ground_truth_gender=gender,
                    metadata=row.to_dict() if hasattr(row, 'to_dict') else None
                ))
        else:
            # Load from directory structure
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
            image_dir = self.data_dir / "images"
            
            if not image_dir.exists():
                image_dir = self.data_dir
            
            image_files = [
                f for f in image_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            
            for idx, image_path in enumerate(image_files):
                if max_samples and idx >= max_samples:
                    break
                
                self.samples.append(CaptionSample(
                    image_id=image_path.stem,
                    image_path=image_path,
                    english_prompt=self.english_template,
                    arabic_prompt=self.arabic_template,
                    ground_truth_gender=None
                ))
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def _find_image(self, image_id: str) -> Optional[Path]:
        """Find image file by ID."""
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '']  # Empty string for when image_id already has extension
        search_dirs = [
            self.data_dir / "images",
            self.data_dir,
            self.data_dir.parent / "raw" / "images",  # Also check raw images directory
            Path("data/raw/images"),  # Absolute fallback
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for ext in extensions:
                path = search_dir / f"{image_id}{ext}"
                if path.exists():
                    return path

        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample.image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image_id": sample.image_id,
            "image": image,
            "english_prompt": sample.english_prompt,
            "arabic_prompt": sample.arabic_prompt,
            "ground_truth_gender": sample.ground_truth_gender,
        }
    
    def get_paired_dataloader(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get paired dataloaders for English and Arabic prompts.
        
        Returns two dataloaders that iterate through the same images
        but with different prompts.
        """
        english_loader = DataLoader(
            _PromptWrapper(self, "english"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
        
        arabic_loader = DataLoader(
            _PromptWrapper(self, "arabic"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # Keep same order as English
            collate_fn=self._collate_fn
        )
        
        return english_loader, arabic_loader
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function."""
        return {
            "image_ids": [b["image_id"] for b in batch],
            "images": [b["image"] for b in batch],
            "prompts": [b["prompt"] for b in batch],
            "genders": [b["ground_truth_gender"] for b in batch],
        }


class _PromptWrapper(Dataset):
    """Wrapper to select specific prompt language."""
    
    def __init__(self, dataset: CrossLingualCaptionDataset, language: str):
        self.dataset = dataset
        self.language = language
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt_key = f"{self.language}_prompt"
        return {
            "image_id": item["image_id"],
            "image": item["image"],
            "prompt": item[prompt_key],
            "ground_truth_gender": item["ground_truth_gender"],
        }


class ActivationDataset(Dataset):
    """
    Dataset for loading pre-extracted activations.
    
    Used for SAE training after activation extraction.
    """
    
    def __init__(
        self,
        activation_path: Union[str, Path],
        layer: int,
        flatten_sequence: bool = True,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize activation dataset.
        
        Args:
            activation_path: Path to saved activations
            layer: Layer index to load
            flatten_sequence: Whether to flatten sequence dimension
            max_tokens: Maximum tokens to use
        """
        self.layer = layer
        self.flatten_sequence = flatten_sequence
        
        # Load activations
        data = torch.load(activation_path)
        
        if isinstance(data, dict):
            self.activations = data.get(f"layer_{layer}", data.get(layer))
        else:
            self.activations = data
        
        if self.activations is None:
            raise ValueError(f"No activations found for layer {layer}")
        
        # Flatten if requested
        if flatten_sequence and len(self.activations.shape) == 3:
            # (batch, seq, hidden) -> (batch * seq, hidden)
            self.activations = self.activations.view(-1, self.activations.shape[-1])
        
        # Limit tokens if specified
        if max_tokens and len(self.activations) > max_tokens:
            indices = torch.randperm(len(self.activations))[:max_tokens]
            self.activations = self.activations[indices]
        
        logger.info(f"Loaded {len(self.activations)} activation vectors")
    
    def __len__(self) -> int:
        return len(self.activations)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


def create_dataloaders(
    config: Dict[str, Any],
    processor: Any = None
) -> Dict[str, DataLoader]:
    """
    Create all dataloaders from config.
    
    Args:
        config: Configuration dictionary
        processor: Model processor for preprocessing
        
    Returns:
        Dictionary of dataloaders
    """
    data_dir = Path(config["paths"]["data_dir"])
    data_config = config["data"]
    
    # Check for existing processed data
    processed_dir = Path(config["paths"]["processed_data"])
    csv_path = processed_dir / "samples.csv"
    
    if not csv_path.exists():
        csv_path = data_dir / "captions.csv"
    
    # Create dataset
    dataset = CrossLingualCaptionDataset(
        data_dir=data_dir,
        csv_path=csv_path if csv_path.exists() else None,
        english_prompt_template=data_config.get("english_prompt", "Describe the person in this image."),
        arabic_prompt_template=data_config.get("arabic_prompt", "صف الشخص في هذه الصورة."),
        image_size=data_config.get("image_size", 448),
        max_samples=data_config.get("num_samples", None)
    )
    
    # Split into train/val
    val_size = int(len(dataset) * data_config.get("val_split", 0.1))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    batch_size = data_config.get("batch_size", 8)
    num_workers = data_config.get("num_workers", 4)
    
    loaders = {
        "train_english": DataLoader(
            _PromptWrapper(train_dataset.dataset, "english"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=CrossLingualCaptionDataset._collate_fn,
            sampler=torch.utils.data.SubsetRandomSampler(train_dataset.indices)
        ),
        "train_arabic": DataLoader(
            _PromptWrapper(train_dataset.dataset, "arabic"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=CrossLingualCaptionDataset._collate_fn,
            sampler=torch.utils.data.SubsetRandomSampler(train_dataset.indices)
        ),
        "val_english": DataLoader(
            _PromptWrapper(val_dataset.dataset, "english"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=CrossLingualCaptionDataset._collate_fn,
            sampler=torch.utils.data.SubsetRandomSampler(val_dataset.indices)
        ),
        "val_arabic": DataLoader(
            _PromptWrapper(val_dataset.dataset, "arabic"),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=CrossLingualCaptionDataset._collate_fn,
            sampler=torch.utils.data.SubsetRandomSampler(val_dataset.indices)
        ),
    }
    
    return loaders


def prepare_sample_data(
    output_dir: Union[str, Path],
    num_samples: int = 100
) -> None:
    """
    Create sample dataset for testing.
    
    Generates synthetic data for pipeline testing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample CSV
    data = {
        "image_id": [f"sample_{i:04d}" for i in range(num_samples)],
        "english_prompt": ["Describe the person in this image."] * num_samples,
        "arabic_prompt": ["صف الشخص في هذه الصورة."] * num_samples,
        "ground_truth_gender": np.random.choice(
            ["male", "female"], size=num_samples
        ).tolist()
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "samples.csv", index=False)
    
    # Create sample images (solid color placeholders)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        img = Image.new("RGB", (448, 448), color=(128, 128, 128))
        img.save(image_dir / f"sample_{i:04d}.jpg")
    
    logger.info(f"Created sample dataset with {num_samples} samples at {output_dir}")
