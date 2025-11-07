#!/usr/bin/env python3
"""
Script 01: Data Preparation
===========================

Prepares and validates the dataset for SAE analysis.
Creates processed data directory with standardized format.

Usage:
    python scripts/01_prepare_data.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import prepare_sample_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_images(image_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Validate that all images exist and are readable."""
    valid_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        image_id = str(row.get('image_id', row.get('id', idx)))
        
        # Try to find image
        found = False
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            image_path = image_dir / f"{image_id}{ext}"
            if image_path.exists():
                try:
                    img = Image.open(image_path)
                    img.verify()
                    valid_rows.append(row)
                    found = True
                    break
                except Exception as e:
                    logger.warning(f"Invalid image {image_path}: {e}")
        
        if not found:
            logger.warning(f"Image not found for ID: {image_id}")
    
    return pd.DataFrame(valid_rows)


def prepare_prompts(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Ensure prompts are properly set."""
    data_config = config.get('data', {})
    
    # Set default prompts if not present
    if 'english_prompt' not in df.columns:
        df['english_prompt'] = data_config.get(
            'english_prompt',
            "Describe the person in this image in detail."
        )
    
    if 'arabic_prompt' not in df.columns:
        df['arabic_prompt'] = data_config.get(
            'arabic_prompt',
            "صف الشخص في هذه الصورة بالتفصيل."
        )
    
    return df


def validate_gender_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize gender labels."""
    if 'ground_truth_gender' not in df.columns:
        if 'gender' in df.columns:
            df['ground_truth_gender'] = df['gender']
        else:
            logger.warning("No gender labels found. Setting to 'unknown'")
            df['ground_truth_gender'] = 'unknown'
    
    # Standardize labels
    label_map = {
        'm': 'male', 'M': 'male', 'Male': 'male', 'MALE': 'male',
        'f': 'female', 'F': 'female', 'Female': 'female', 'FEMALE': 'female',
        'man': 'male', 'woman': 'female', 'boy': 'male', 'girl': 'female',
    }
    
    df['ground_truth_gender'] = df['ground_truth_gender'].map(
        lambda x: label_map.get(x, x) if pd.notna(x) else 'unknown'
    )
    
    # Report distribution
    gender_dist = df['ground_truth_gender'].value_counts()
    logger.info(f"Gender distribution:\n{gender_dist}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare data for SAE analysis")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample data for testing')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup paths
    raw_dir = Path(config['paths']['raw_data'])
    processed_dir = Path(config['paths']['processed_data'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample:
        logger.info("Creating sample data for testing...")
        prepare_sample_data(raw_dir, num_samples=100)
        logger.info(f"Sample data created at {raw_dir}")
    
    # Check for existing data
    csv_path = raw_dir / 'captions.csv'
    if not csv_path.exists():
        csv_path = raw_dir / 'samples.csv'
    
    if not csv_path.exists():
        logger.error(f"No CSV file found at {raw_dir}")
        logger.info("Use --create-sample to generate test data")
        return 1
    
    # Load data
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Find image directory
    image_dir = raw_dir / 'images'
    if not image_dir.exists():
        image_dir = raw_dir
    
    # Validate images
    logger.info("Validating images...")
    df = validate_images(image_dir, df)
    logger.info(f"{len(df)} valid samples after image validation")
    
    # Prepare prompts
    logger.info("Preparing prompts...")
    df = prepare_prompts(df, config)
    
    # Validate gender labels
    logger.info("Validating gender labels...")
    df = validate_gender_labels(df)
    
    # Limit samples if configured
    max_samples = config['data'].get('num_samples')
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=config.get('seed', 42))
        logger.info(f"Sampled {max_samples} samples")
    
    # Save processed data
    output_csv = processed_dir / 'samples.csv'
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved processed data to {output_csv}")
    
    # Create symlink to images
    processed_images = processed_dir / 'images'
    if not processed_images.exists():
        processed_images.symlink_to(image_dir.absolute())
        logger.info(f"Created symlink to images: {processed_images}")
    
    # Save summary statistics
    summary = {
        'total_samples': len(df),
        'gender_distribution': df['ground_truth_gender'].value_counts().to_dict(),
        'has_custom_prompts': 'english_prompt' in df.columns,
    }
    
    summary_path = processed_dir / 'data_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f)
    logger.info(f"Saved data summary to {summary_path}")
    
    logger.info("Data preparation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
