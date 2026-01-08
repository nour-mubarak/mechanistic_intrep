"""
Hierarchical Bias Localization (HBL)
=====================================

Novel methodology component that systematically decomposes where bias enters the model:
- Vision encoder bias
- Cross-modal projection bias  
- Language decoder bias

Computes Bias Attribution Score (BAS) for each component.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


@dataclass
class BiasAttributionResult:
    """Result of bias attribution analysis."""
    vision_bias_score: float
    projection_bias_score: float
    language_bias_score: float
    total_bias: float
    layer_scores: Dict[str, List[float]]
    feature_importance: Dict[str, np.ndarray]
    dominant_component: str  # 'vision', 'projection', or 'language'


class HierarchicalBiasLocalizer:
    """
    Hierarchical Bias Localization (HBL) - Novel Methodology Component
    
    Decomposes bias across model components:
    Image → [Vision Encoder] → [Projection] → [Language Model] → Caption
                  ↓                  ↓                ↓
             V-Features         Bridge          L-Features
                  ↓                  ↓                ↓
            Vision Bias      Transfer Bias    Linguistic Bias
    """
    
    def __init__(self, model: Any, model_config: Any, device: str = "cuda"):
        self.model = model
        self.config = model_config
        self.device = device
        
        # Storage for activations
        self.vision_activations: Dict[int, List[torch.Tensor]] = {}
        self.language_activations: Dict[int, List[torch.Tensor]] = {}
        self.projection_activations: List[torch.Tensor] = []
        
        # Hooks
        self.hooks: List[Any] = []
        
        # Results cache
        self.gender_activations: Dict[str, Dict] = {'male': {}, 'female': {}}
        
    def _register_hooks(self, layers: Dict[str, List]):
        """Register forward hooks to capture activations."""
        
        def make_vision_hook(layer_idx: int):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                act = act.detach().cpu().float()
                if layer_idx not in self.vision_activations:
                    self.vision_activations[layer_idx] = []
                self.vision_activations[layer_idx].append(act)
            return hook
        
        def make_language_hook(layer_idx: int):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                act = act.detach().cpu().float()
                if layer_idx not in self.language_activations:
                    self.language_activations[layer_idx] = []
                self.language_activations[layer_idx].append(act)
            return hook
        
        def projection_hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            self.projection_activations.append(act.detach().cpu().float())
        
        # Register vision hooks
        for idx, layer in enumerate(layers.get('vision', [])):
            h = layer.register_forward_hook(make_vision_hook(idx))
            self.hooks.append(h)
        
        # Register language hooks
        for idx, layer in enumerate(layers.get('language', [])):
            h = layer.register_forward_hook(make_language_hook(idx))
            self.hooks.append(h)
        
        # Register projection hook
        if layers.get('projection') is not None:
            h = layers['projection'].register_forward_hook(projection_hook)
            self.hooks.append(h)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    def _clear_activations(self):
        """Clear stored activations."""
        self.vision_activations.clear()
        self.language_activations.clear()
        self.projection_activations.clear()
    
    def extract_activations_by_gender(
        self,
        dataloader: Any,
        processor: Any,
        layers: Dict[str, List],
        language: str = 'english'
    ) -> Dict[str, Dict]:
        """
        Extract activations grouped by gender.
        
        Returns:
            Dict with 'male' and 'female' keys, each containing
            vision, projection, and language activations.
        """
        self._register_hooks(layers)
        
        gender_data = {
            'male': {'vision': {}, 'language': {}, 'projection': []},
            'female': {'vision': {}, 'language': {}, 'projection': []}
        }
        
        prompt_key = f'{language}_prompt'
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting {language} activations"):
                images = batch['image']
                prompts = batch[prompt_key]
                genders = batch['ground_truth_gender']
                
                # Process batch
                for img, prompt, gender in zip(images, prompts, genders):
                    if gender not in ['male', 'female']:
                        continue
                    
                    self._clear_activations()
                    
                    # Forward pass
                    inputs = processor(images=img, text=prompt, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    _ = self.model(**inputs)
                    
                    # Store activations by gender
                    for layer_idx, acts in self.vision_activations.items():
                        if layer_idx not in gender_data[gender]['vision']:
                            gender_data[gender]['vision'][layer_idx] = []
                        gender_data[gender]['vision'][layer_idx].extend(acts)
                    
                    for layer_idx, acts in self.language_activations.items():
                        if layer_idx not in gender_data[gender]['language']:
                            gender_data[gender]['language'][layer_idx] = []
                        gender_data[gender]['language'][layer_idx].extend(acts)
                    
                    gender_data[gender]['projection'].extend(self.projection_activations)
                    
                    # Clear GPU memory
                    torch.cuda.empty_cache()
        
        self._remove_hooks()
        return gender_data
    
    def compute_bias_attribution_score(
        self,
        male_activations: List[torch.Tensor],
        female_activations: List[torch.Tensor],
        feature_importance: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Bias Attribution Score (BAS) for a component.
        
        BAS = Σ |activation_male - activation_female| × feature_importance
        
        Higher BAS indicates more gender-biased representations.
        """
        if not male_activations or not female_activations:
            return 0.0
        
        # Stack activations
        male_stack = torch.cat(male_activations, dim=0).mean(dim=0)  # Average over samples
        female_stack = torch.cat(female_activations, dim=0).mean(dim=0)
        
        # Compute absolute difference
        diff = torch.abs(male_stack - female_stack)
        
        # Apply feature importance weighting if provided
        if feature_importance is not None:
            importance = torch.from_numpy(feature_importance).float()
            if importance.shape != diff.shape:
                importance = importance.mean()  # Fallback to scalar
            diff = diff * importance
        
        # Aggregate to single score
        bas = diff.mean().item()
        
        return bas
    
    def localize_bias(
        self,
        gender_data: Dict[str, Dict],
        sae_feature_importance: Optional[Dict[str, np.ndarray]] = None
    ) -> BiasAttributionResult:
        """
        Main HBL method: Localize bias across model components.
        
        Returns:
            BiasAttributionResult with scores for each component.
        """
        layer_scores = {'vision': [], 'language': []}
        feature_importance = {}
        
        # Compute vision bias scores per layer
        vision_scores = []
        for layer_idx in sorted(gender_data['male']['vision'].keys()):
            male_acts = gender_data['male']['vision'].get(layer_idx, [])
            female_acts = gender_data['female']['vision'].get(layer_idx, [])
            
            importance = None
            if sae_feature_importance and f'vision_{layer_idx}' in sae_feature_importance:
                importance = sae_feature_importance[f'vision_{layer_idx}']
            
            score = self.compute_bias_attribution_score(male_acts, female_acts, importance)
            vision_scores.append(score)
            layer_scores['vision'].append(score)
        
        vision_bias = np.mean(vision_scores) if vision_scores else 0.0
        
        # Compute language bias scores per layer
        language_scores = []
        for layer_idx in sorted(gender_data['male']['language'].keys()):
            male_acts = gender_data['male']['language'].get(layer_idx, [])
            female_acts = gender_data['female']['language'].get(layer_idx, [])
            
            importance = None
            if sae_feature_importance and f'language_{layer_idx}' in sae_feature_importance:
                importance = sae_feature_importance[f'language_{layer_idx}']
            
            score = self.compute_bias_attribution_score(male_acts, female_acts, importance)
            language_scores.append(score)
            layer_scores['language'].append(score)
        
        language_bias = np.mean(language_scores) if language_scores else 0.0
        
        # Compute projection bias
        male_proj = gender_data['male']['projection']
        female_proj = gender_data['female']['projection']
        projection_bias = self.compute_bias_attribution_score(male_proj, female_proj)
        
        # Total bias (weighted sum - weights can be learned or preset)
        alpha, beta, gamma = 0.3, 0.2, 0.5  # Default weights
        total_bias = alpha * vision_bias + beta * projection_bias + gamma * language_bias
        
        # Determine dominant component
        bias_components = {
            'vision': vision_bias,
            'projection': projection_bias,
            'language': language_bias
        }
        dominant = max(bias_components, key=bias_components.get)
        
        return BiasAttributionResult(
            vision_bias_score=vision_bias,
            projection_bias_score=projection_bias,
            language_bias_score=language_bias,
            total_bias=total_bias,
            layer_scores=layer_scores,
            feature_importance=feature_importance,
            dominant_component=dominant
        )
    
    def analyze_model(
        self,
        dataloader: Any,
        processor: Any,
        layers: Dict[str, List],
        languages: List[str] = ['english', 'arabic']
    ) -> Dict[str, BiasAttributionResult]:
        """
        Full HBL analysis for multiple languages.
        
        Returns:
            Dict mapping language to BiasAttributionResult.
        """
        results = {}
        
        for language in languages:
            logger.info(f"Analyzing bias for {language}...")
            
            # Extract gender-separated activations
            gender_data = self.extract_activations_by_gender(
                dataloader, processor, layers, language
            )
            
            # Localize bias
            result = self.localize_bias(gender_data)
            results[language] = result
            
            logger.info(f"{language} results:")
            logger.info(f"  Vision bias:     {result.vision_bias_score:.4f}")
            logger.info(f"  Projection bias: {result.projection_bias_score:.4f}")
            logger.info(f"  Language bias:   {result.language_bias_score:.4f}")
            logger.info(f"  Dominant:        {result.dominant_component}")
        
        return results
    
    def save_results(self, results: Dict[str, BiasAttributionResult], output_path: Path):
        """Save HBL results to JSON."""
        output = {}
        for lang, result in results.items():
            output[lang] = {
                'vision_bias_score': result.vision_bias_score,
                'projection_bias_score': result.projection_bias_score,
                'language_bias_score': result.language_bias_score,
                'total_bias': result.total_bias,
                'dominant_component': result.dominant_component,
                'layer_scores': result.layer_scores,
            }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved HBL results to {output_path}")
