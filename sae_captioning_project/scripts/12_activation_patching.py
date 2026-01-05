#!/usr/bin/env python3
"""
Activation Patching Experiments
=================================

Use activation patching to causally verify which features drive gender predictions.

Interventions:
1. Feature ablation: Zero out specific features and measure impact
2. Feature amplification: Increase feature activations
3. Feature swapping: Replace male->female feature patterns and vice versa

Usage:
    python scripts/12_activation_patching.py --config configs/config.yaml --layer 10
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import argparse
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActivationPatcher:
    """Perform activation patching experiments to test causal importance of features."""

    def __init__(self, config: Dict, layer: int, device: str = 'cuda'):
        """
        Initialize activation patcher.

        Args:
            config: Configuration dictionary
            layer: Layer to patch
            device: Device to use
        """
        self.config = config
        self.layer = layer
        self.device = device

        # Load paths
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.data_file = Path(config['paths']['processed_data']) / 'samples.csv'
        self.image_dir = Path(config['paths']['processed_data']) / 'images'
        self.output_dir = Path(config['paths']['visualizations']) / f'activation_patching_layer_{layer}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results to get top biased features
        results_file = Path(config['paths']['results']) / 'comprehensive_analysis_results.json'
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        # Load SAE
        self.sae = self._load_sae()

        # Load data
        self.data_df = pd.read_csv(self.data_file)
        logger.info(f"Loaded {len(self.data_df)} samples from {self.data_file}")

        # Load VLM model for generation
        logger.info("Loading VLM model for generation...")
        self.model_name = config['model']['name']
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()

    def _load_sae(self) -> SparseAutoencoder:
        """Load trained SAE for the layer."""
        sae_path = self.checkpoint_dir / f'sae_layer_{self.layer}.pt'
        logger.info(f"Loading SAE from {sae_path}")

        checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
        sae_config = checkpoint['config']

        sae = SparseAutoencoder(sae_config)
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.to(self.device)
        sae.eval()

        return sae

    def ablate_features(
        self,
        activations: torch.Tensor,
        feature_indices: List[int],
        ablation_type: str = 'zero'
    ) -> torch.Tensor:
        """
        Ablate specific SAE features.

        Args:
            activations: Original activations [batch, seq, hidden]
            feature_indices: List of feature indices to ablate
            ablation_type: 'zero' or 'mean'

        Returns:
            Modified activations with features ablated
        """
        # Encode through SAE
        orig_shape = activations.shape
        activations_flat = activations.reshape(-1, activations.shape[-1]).to(self.device)

        with torch.no_grad():
            reconstruction, features, _ = self.sae(activations_flat)

            # Ablate features
            if ablation_type == 'zero':
                features[:, feature_indices] = 0
            elif ablation_type == 'mean':
                for idx in feature_indices:
                    features[:, idx] = features[:, idx].mean()

            # Decode to get modified activations
            modified_activations = self.sae.decode(features)

        return modified_activations.reshape(orig_shape).cpu()

    def amplify_features(
        self,
        activations: torch.Tensor,
        feature_indices: List[int],
        amplification_factor: float = 2.0
    ) -> torch.Tensor:
        """
        Amplify specific SAE features.

        Args:
            activations: Original activations
            feature_indices: Features to amplify
            amplification_factor: Multiplication factor

        Returns:
            Modified activations
        """
        orig_shape = activations.shape
        activations_flat = activations.reshape(-1, activations.shape[-1]).to(self.device)

        with torch.no_grad():
            reconstruction, features, _ = self.sae(activations_flat)

            # Amplify features
            features[:, feature_indices] *= amplification_factor

            # Decode
            modified_activations = self.sae.decode(features)

        return modified_activations.reshape(orig_shape).cpu()

    def generate_caption_with_patching(
        self,
        image_path: Path,
        prompt: str,
        feature_indices: Optional[List[int]] = None,
        intervention_type: str = 'none',
        intervention_strength: float = 1.0
    ) -> Dict:
        """
        Generate caption with optional activation patching.

        Args:
            image_path: Path to image
            prompt: Text prompt
            feature_indices: Features to intervene on
            intervention_type: 'none', 'ablate', 'amplify'
            intervention_strength: Strength of intervention

        Returns:
            Dict with generated caption and metadata
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Hook to intervene on activations
        intervention_applied = False

        def intervention_hook(module, input, output):
            nonlocal intervention_applied

            if not intervention_applied and feature_indices is not None:
                # Get activations (output is a tuple, first element is hidden states)
                hidden_states = output[0] if isinstance(output, tuple) else output

                # Apply intervention
                if intervention_type == 'ablate':
                    hidden_states = self.ablate_features(
                        hidden_states.cpu(),
                        feature_indices,
                        ablation_type='zero'
                    ).to(self.device)
                elif intervention_type == 'amplify':
                    hidden_states = self.amplify_features(
                        hidden_states.cpu(),
                        feature_indices,
                        amplification_factor=intervention_strength
                    ).to(self.device)

                intervention_applied = True

                # Return modified output
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states

            return output

        # Register hook on the target layer
        # For Gemma/Llama-style models, layers are in model.language_model.model.layers
        target_module = None
        if hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'model'):
                target_module = self.model.language_model.model.layers[self.layer]

        hook_handle = None
        if target_module is not None and intervention_type != 'none':
            hook_handle = target_module.register_forward_hook(intervention_hook)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config['model']['max_new_tokens'],
                do_sample=False
            )

        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()

        # Decode
        caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return {
            'caption': caption,
            'intervention_type': intervention_type,
            'feature_indices': feature_indices,
            'intervention_strength': intervention_strength,
            'intervention_applied': intervention_applied
        }

    def run_ablation_experiment(
        self,
        num_samples: int = 20,
        num_features: int = 5
    ):
        """
        Run feature ablation experiment on sample images.

        Args:
            num_samples: Number of images to test
            num_features: Number of top features to ablate
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Ablation Experiment - Layer {self.layer}")
        logger.info(f"{'='*60}\n")

        # Get top biased features
        layer_results = self.results['layers'][str(self.layer)]
        male_biased = layer_results['gender_bias']['english']['top_male_biased'][:num_features]
        female_biased = layer_results['gender_bias']['english']['top_female_biased'][:num_features]

        logger.info(f"Male-biased features to ablate: {male_biased}")
        logger.info(f"Female-biased features to ablate: {female_biased}")

        # Sample images (CSV uses 'ground_truth_gender' column)
        male_samples = self.data_df[self.data_df['ground_truth_gender'] == 'male'].sample(
            min(num_samples // 2, len(self.data_df[self.data_df['ground_truth_gender'] == 'male']))
        )
        female_samples = self.data_df[self.data_df['ground_truth_gender'] == 'female'].sample(
            min(num_samples // 2, len(self.data_df[self.data_df['ground_truth_gender'] == 'female']))
        )

        samples = pd.concat([male_samples, female_samples])
        logger.info(f"Selected {len(male_samples)} male and {len(female_samples)} female samples")

        results = []

        for _, sample in tqdm(samples.iterrows(), total=len(samples), desc="Running ablations"):
            # CSV uses 'image' column, and it already includes .jpg extension
            image_path = self.image_dir / sample['image']
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            prompt = self.config['data']['english_prompt']

            # 1. Baseline (no intervention)
            baseline = self.generate_caption_with_patching(
                image_path, prompt, intervention_type='none'
            )

            # 2. Ablate male-biased features
            ablate_male = self.generate_caption_with_patching(
                image_path, prompt, male_biased, intervention_type='ablate'
            )

            # 3. Ablate female-biased features
            ablate_female = self.generate_caption_with_patching(
                image_path, prompt, female_biased, intervention_type='ablate'
            )

            results.append({
                'image_id': sample['image'],
                'gender': sample['ground_truth_gender'],
                'baseline_caption': baseline['caption'],
                'ablate_male_caption': ablate_male['caption'],
                'ablate_female_caption': ablate_female['caption'],
                'original_caption': sample.get('en_caption', 'N/A')
            })

        # Save results
        results_df = pd.DataFrame(results)
        output_file = self.output_dir / 'ablation_results.csv'
        results_df.to_csv(output_file, index=False)

        logger.info(f"\nSaved ablation results to: {output_file}")

        # Create summary visualization
        self.visualize_ablation_results(results_df)

        return results_df

    def visualize_ablation_results(self, results_df: pd.DataFrame):
        """Create visualization summarizing ablation results."""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Layer {self.layer} - Ablation Experiment Results',
                     fontsize=18, fontweight='bold')

        # Show example results
        for idx, ax in enumerate(axes.flat[:4]):
            if idx >= len(results_df):
                ax.axis('off')
                continue

            row = results_df.iloc[idx]
            # image_id already includes .jpg extension
            image_path = self.image_dir / row['image_id']

            if image_path.exists():
                img = Image.open(image_path)
                ax.imshow(img)

            text = f"Gender: {row['gender']}\n\n"
            text += f"Baseline:\n{row['baseline_caption'][:80]}...\n\n"
            text += f"Ablate Male Features:\n{row['ablate_male_caption'][:80]}...\n\n"
            text += f"Ablate Female Features:\n{row['ablate_female_caption'][:80]}..."

            ax.set_title(text, fontsize=9, loc='left')
            ax.axis('off')

        plt.tight_layout()
        output_path = self.output_dir / 'ablation_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ablation summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Activation Patching Experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--layer', type=int, required=True, help='Layer to patch')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to test (default: 10)')
    parser.add_argument('--num-features', type=int, default=5,
                       help='Number of features to ablate (default: 5)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create patcher
    patcher = ActivationPatcher(config, args.layer)

    # Run ablation experiment
    patcher.run_ablation_experiment(
        num_samples=args.num_samples,
        num_features=args.num_features
    )

    logger.info("\n" + "="*60)
    logger.info("Activation patching complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
