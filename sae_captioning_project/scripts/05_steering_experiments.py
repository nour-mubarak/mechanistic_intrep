#!/usr/bin/env python3
"""
Script 05: Steering Experiments
===============================

Tests the effect of steering SAE features on caption generation.

Usage:
    python scripts/05_steering_experiments.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import json
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder
from src.data import CrossLingualCaptionDataset
from src.analysis.metrics import (
    compute_caption_gender_bias,
    compute_steering_effectiveness
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sae(checkpoint_path: Path, device: str = "cuda") -> SparseAutoencoder:
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    sae = SparseAutoencoder(config)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    return sae


def load_analysis_results(results_dir: Path) -> dict:
    """Load feature analysis results."""
    path = results_dir / 'feature_analysis.json'
    if not path.exists():
        raise FileNotFoundError(f"Analysis results not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


class SteeringHook:
    """Hook for steering SAE features during generation."""
    
    def __init__(
        self,
        sae: SparseAutoencoder,
        feature_indices: List[int],
        steering_strength: float,
        device: str = "cuda"
    ):
        self.sae = sae
        self.feature_indices = feature_indices
        self.steering_strength = steering_strength
        self.device = device
    
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()
        
        # Get original shape
        original_shape = hidden_states.shape
        batch, seq, hidden = original_shape
        
        # Flatten for SAE
        flat = hidden_states.view(-1, hidden)
        
        # Encode
        with torch.no_grad():
            features = self.sae.encode(flat)
            
            # Apply steering to selected features
            for idx in self.feature_indices:
                features[:, idx] *= self.steering_strength
            
            # Decode
            modified = self.sae.decode(features)
        
        # Reshape
        modified = modified.view(original_shape)
        
        if rest:
            return (modified,) + rest
        return modified


def generate_with_steering(
    model,
    processor,
    images: List,
    prompts: List[str],
    sae: SparseAutoencoder,
    feature_indices: List[int],
    steering_strength: float,
    layer_idx: int,
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> List[str]:
    """Generate captions with feature steering."""
    
    # Setup steering hook
    hook = SteeringHook(sae, feature_indices, steering_strength, device)
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    
    try:
        # Prepare inputs
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # Decode
        captions = processor.batch_decode(outputs, skip_special_tokens=True)
        
    finally:
        handle.remove()
    
    return captions


def run_steering_experiment(
    model,
    processor,
    dataset: CrossLingualCaptionDataset,
    sae: SparseAutoencoder,
    feature_indices: List[int],
    layer_idx: int,
    steering_strengths: List[float],
    language: str,
    max_samples: int = 100,
    batch_size: int = 4,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Run steering experiment with multiple strengths."""
    
    # Get samples
    indices = list(range(min(max_samples, len(dataset))))
    samples = [dataset[i] for i in indices]
    
    images = [s['image'] for s in samples]
    prompts = [s[f'{language}_prompt'] for s in samples]
    genders = [s['ground_truth_gender'] for s in samples]
    
    results = {
        'strengths': steering_strengths,
        'accuracy_by_strength': [],
        'bias_by_strength': [],
        'neutral_rate_by_strength': [],
        'captions_by_strength': {},
    }
    
    # Generate baseline (strength = 1.0, no change)
    logger.info("Generating baseline captions...")
    baseline_captions = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        
        inputs = processor(
            images=batch_images,
            text=batch_prompts,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        captions = processor.batch_decode(outputs, skip_special_tokens=True)
        baseline_captions.extend(captions)
    
    # Compute baseline metrics
    baseline_bias = compute_caption_gender_bias(baseline_captions, genders, language)
    results['baseline'] = {
        'accuracy': baseline_bias['accuracy'],
        'bias_score': baseline_bias['bias_score'],
        'neutral_rate': baseline_bias['neutral_rate'],
    }
    logger.info(f"Baseline accuracy: {baseline_bias['accuracy']:.2%}")
    
    # Test each steering strength
    for strength in tqdm(steering_strengths, desc="Testing strengths"):
        logger.info(f"\nTesting steering strength: {strength}")
        
        steered_captions = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            
            batch_captions = generate_with_steering(
                model=model,
                processor=processor,
                images=batch_images,
                prompts=batch_prompts,
                sae=sae,
                feature_indices=feature_indices,
                steering_strength=strength,
                layer_idx=layer_idx,
                device=device
            )
            steered_captions.extend(batch_captions)
        
        # Compute metrics
        steered_bias = compute_caption_gender_bias(steered_captions, genders, language)
        
        results['accuracy_by_strength'].append(steered_bias['accuracy'])
        results['bias_by_strength'].append(steered_bias['bias_score'])
        results['neutral_rate_by_strength'].append(steered_bias['neutral_rate'])
        results['captions_by_strength'][str(strength)] = steered_captions[:10]  # Save sample
        
        logger.info(f"  Accuracy: {steered_bias['accuracy']:.2%}")
        logger.info(f"  Bias score: {steered_bias['bias_score']:.3f}")
    
    # Find best steering
    best_idx = max(range(len(steering_strengths)),
                   key=lambda i: results['accuracy_by_strength'][i])
    results['best_steering'] = {
        'strength': steering_strengths[best_idx],
        'original_accuracy': baseline_bias['accuracy'],
        'steered_accuracy': results['accuracy_by_strength'][best_idx],
        'improvement': results['accuracy_by_strength'][best_idx] - baseline_bias['accuracy'],
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run steering experiments")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--layer', type=int, default=None,
                       help='Layer to steer (default: most divergent)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup paths
    checkpoint_dir = Path(config['paths']['checkpoints'])
    results_dir = Path(config['paths']['results'])
    processed_dir = Path(config['paths']['processed_data'])
    
    # Load analysis results
    logger.info("Loading analysis results...")
    analysis_results = load_analysis_results(results_dir)
    
    # Determine layer to steer
    if args.layer is not None:
        layer_idx = args.layer
    else:
        # Use most divergent layer
        layer_divergence = analysis_results.get('layer_divergence', {})
        if layer_divergence:
            # Convert keys to int
            layer_divergence = {int(k): v for k, v in layer_divergence.items()}
            layer_idx = min(layer_divergence, key=layer_divergence.get)
        else:
            # Default to middle layer
            layer_idx = 14
    
    logger.info(f"Using layer {layer_idx} for steering")
    
    # Load model
    logger.info("Loading model...")
    model_name = config['model']['name']
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Load SAE
    sae_path = checkpoint_dir / f'sae_layer_{layer_idx}.pt'
    if not sae_path.exists():
        logger.error(f"SAE not found: {sae_path}")
        return 1
    
    sae = load_sae(sae_path, args.device)
    
    # Get features to steer
    layer_results = analysis_results['layers'].get(str(layer_idx), {})
    en_gender_features = layer_results.get('english_gender_features', {})
    
    # Combine male and female associated features
    feature_indices = (
        en_gender_features.get('male_associated', [])[:10] +
        en_gender_features.get('female_associated', [])[:10]
    )
    
    if not feature_indices:
        logger.warning("No gender features found, using top 20 features by effect size")
        en_stats = layer_results.get('english_stats', [])
        sorted_stats = sorted(
            en_stats,
            key=lambda x: abs(x.get('gender_effect_size', 0) or 0),
            reverse=True
        )
        feature_indices = [s['feature_idx'] for s in sorted_stats[:20]]
    
    logger.info(f"Steering {len(feature_indices)} features: {feature_indices[:5]}...")
    
    # Load dataset
    csv_path = processed_dir / 'samples.csv'
    dataset = CrossLingualCaptionDataset(
        data_dir=processed_dir,
        csv_path=csv_path
    )
    
    # Get steering strengths
    steering_config = config.get('steering', {})
    strengths = steering_config.get('strengths', [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    max_samples = steering_config.get('num_samples_per_condition', 100)
    
    # Run experiments for both languages
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'layer': layer_idx,
        'feature_indices': feature_indices,
        'strengths': strengths,
        'languages': {}
    }
    
    for language in ['english', 'arabic']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running steering experiment for {language.upper()}")
        logger.info(f"{'='*50}")
        
        results = run_steering_experiment(
            model=model,
            processor=processor,
            dataset=dataset,
            sae=sae,
            feature_indices=feature_indices,
            layer_idx=layer_idx,
            steering_strengths=strengths,
            language=language,
            max_samples=max_samples,
            device=args.device
        )
        
        all_results['languages'][language] = results
        
        # Log summary
        logger.info(f"\n{language.upper()} Results:")
        logger.info(f"  Baseline accuracy: {results['baseline']['accuracy']:.2%}")
        logger.info(f"  Best steering strength: {results['best_steering']['strength']}")
        logger.info(f"  Best accuracy: {results['best_steering']['steered_accuracy']:.2%}")
        logger.info(f"  Improvement: {results['best_steering']['improvement']:+.2%}")
    
    # Save results
    output_path = results_dir / 'steering_experiments.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved steering results to {output_path}")
    
    logger.info("\nSteering experiments complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
