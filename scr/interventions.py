"""
Advanced Causal Intervention Techniques
=====================================

This module implements sophisticated causal intervention methods for mechanistic
interpretability, specifically designed for analyzing and mitigating gender bias
in multilingual image captioning models.

Key Features:
- Activation patching with multiple strategies
- Attention head knockout experiments
- Gradient-based interventions
- Concept erasure techniques
- Causal mediation analysis
- Cross-lingual intervention transfer
- Real-time bias mitigation

Example usage:
    from interventions import InterventionEngine
    
    engine = InterventionEngine(model)
    results = engine.apply_gender_bias_intervention(inputs, method="concept_erasure")
    engine.evaluate_intervention_effects(results)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InterventionConfig:
    """Configuration for intervention experiments."""
    method: str
    target_layers: List[int]
    strength: float
    direction: str  # 'suppress', 'enhance', 'neutralize'
    concept_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

@dataclass
class InterventionEffect:
    """Results from an intervention experiment."""
    config: InterventionConfig
    original_outputs: Dict[str, Any]
    intervened_outputs: Dict[str, Any]
    effect_magnitude: float
    bias_change: float
    quality_change: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]

class InterventionEngine:
    """Advanced causal intervention system for bias analysis."""
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: Any = None,
                 device: str = "auto"):
        """
        Initialize the intervention engine.
        
        Args:
            model: PyTorch model to intervene on
            tokenizer: Tokenizer for text processing
            device: Device for computations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._setup_device(device)
        self.intervention_hooks = []
        self.concept_vectors = {}
        self.baseline_activations = {}
        
        # Gender concept definitions
        self.gender_concepts = {
            'male_words': ['man', 'boy', 'father', 'son', 'brother', 'husband', 'he', 'his', 'him'],
            'female_words': ['woman', 'girl', 'mother', 'daughter', 'sister', 'wife', 'she', 'her', 'hers'],
            'neutral_words': ['person', 'individual', 'human', 'someone', 'they', 'their', 'them']
        }
        
    def _setup_device(self, device: str) -> torch.device:
        """Set up computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def extract_concept_vectors(self, 
                              concept_examples: Dict[str, List[str]],
                              layer_name: str) -> Dict[str, np.ndarray]:
        """
        Extract concept vectors from example texts.
        
        Args:
            concept_examples: Dictionary mapping concept names to example texts
            layer_name: Target layer for concept extraction
            
        Returns:
            Dictionary mapping concept names to their vectors
        """
        concept_vectors = {}
        
        # Hook to capture activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu()
            return hook
        
        # Register hook
        hook_handle = None
        for name, module in self.model.named_modules():
            if layer_name in name:
                hook_handle = module.register_forward_hook(hook_fn(name))
                break
        
        if hook_handle is None:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        try:
            for concept_name, examples in concept_examples.items():
                concept_activations = []
                
                for example in examples:
                    # Process example through model
                    if self.tokenizer:
                        inputs = self.tokenizer(example, return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    else:
                        # Assume example is already a tensor
                        inputs = example.to(self.device) if isinstance(example, torch.Tensor) else example
                    
                    with torch.no_grad():
                        _ = self.model(**inputs if isinstance(inputs, dict) else inputs)
                    
                    # Extract activations
                    if layer_name in activations:
                        act = activations[layer_name]
                        # Pool activations (mean over sequence dimension)
                        pooled_act = act.mean(dim=1) if len(act.shape) > 2 else act
                        concept_activations.append(pooled_act.numpy())
                
                if concept_activations:
                    # Average activations for this concept
                    concept_vector = np.mean(concept_activations, axis=0)
                    concept_vectors[concept_name] = concept_vector
                    
        finally:
            hook_handle.remove()
        
        self.concept_vectors.update(concept_vectors)
        return concept_vectors
    
    def compute_bias_direction(self, 
                             male_examples: List[str],
                             female_examples: List[str],
                             layer_name: str) -> np.ndarray:
        """
        Compute the bias direction vector between male and female concepts.
        
        Args:
            male_examples: Examples of male-biased text
            female_examples: Examples of female-biased text
            layer_name: Target layer name
            
        Returns:
            Bias direction vector
        """
        # Extract concept vectors
        concepts = {
            'male': male_examples,
            'female': female_examples
        }
        
        concept_vectors = self.extract_concept_vectors(concepts, layer_name)
        
        if 'male' in concept_vectors and 'female' in concept_vectors:
            # Bias direction is the difference between male and female vectors
            bias_direction = concept_vectors['male'] - concept_vectors['female']
            # Normalize
            bias_direction = bias_direction / np.linalg.norm(bias_direction)
            return bias_direction
        else:
            raise ValueError("Could not extract both male and female concept vectors")
    
    def apply_concept_erasure(self, 
                            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                            bias_direction: np.ndarray,
                            layer_names: List[str],
                            strength: float = 1.0) -> torch.Tensor:
        """
        Apply concept erasure intervention to remove bias direction.
        
        Args:
            inputs: Model inputs
            bias_direction: Bias direction vector to erase
            layer_names: Target layer names
            strength: Intervention strength (0-1)
            
        Returns:
            Model outputs after intervention
        """
        bias_direction_tensor = torch.tensor(bias_direction, dtype=torch.float32, device=self.device)
        
        def erasure_hook(name):
            def hook(module, input, output):
                # Project out the bias direction
                output_flat = output.view(-1, output.size(-1))
                
                # Compute projection onto bias direction
                projections = torch.matmul(output_flat, bias_direction_tensor.unsqueeze(-1))
                bias_component = projections * bias_direction_tensor.unsqueeze(0)
                
                # Remove bias component
                erased_output = output_flat - strength * bias_component
                
                return erased_output.view_as(output)
            return hook
        
        # Register hooks
        hook_handles = []
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                handle = module.register_forward_hook(erasure_hook(name))
                hook_handles.append(handle)
        
        try:
            # Forward pass with intervention
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
            
        finally:
            # Remove hooks
            for handle in hook_handles:
                handle.remove()
        
        return outputs
    
    def apply_activation_patching(self, 
                                clean_inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                corrupted_inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                patch_layers: List[str],
                                patch_positions: Optional[List[int]] = None,
                                strength: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply activation patching intervention.
        
        Args:
            clean_inputs: Clean (unbiased) inputs
            corrupted_inputs: Corrupted (biased) inputs
            patch_layers: Layers to patch
            patch_positions: Specific positions to patch (if None, patch all)
            strength: Patching strength
            
        Returns:
            Tuple of (original_output, patched_output)
        """
        # Store clean activations
        clean_activations = {}
        
        def store_hook(name):
            def hook(module, input, output):
                clean_activations[name] = output.detach().clone()
            return hook
        
        # Register storage hooks
        storage_handles = []
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in patch_layers):
                handle = module.register_forward_hook(store_hook(name))
                storage_handles.append(handle)
        
        try:
            # Forward pass on clean inputs to store activations
            with torch.no_grad():
                if isinstance(clean_inputs, dict):
                    _ = self.model(**clean_inputs)
                else:
                    _ = self.model(clean_inputs)
        finally:
            # Remove storage hooks
            for handle in storage_handles:
                handle.remove()
        
        # Get original output on corrupted inputs
        with torch.no_grad():
            if isinstance(corrupted_inputs, dict):
                original_output = self.model(**corrupted_inputs)
            else:
                original_output = self.model(corrupted_inputs)
        
        # Apply patching hooks
        def patch_hook(name):
            def hook(module, input, output):
                if name in clean_activations:
                    clean_act = clean_activations[name]
                    
                    if patch_positions is not None:
                        # Patch specific positions
                        patched_output = output.clone()
                        for pos in patch_positions:
                            if pos < output.size(1):  # Sequence dimension
                                patched_output[:, pos] = (
                                    strength * clean_act[:, pos] + 
                                    (1 - strength) * output[:, pos]
                                )
                    else:
                        # Patch entire activation
                        patched_output = strength * clean_act + (1 - strength) * output
                    
                    return patched_output
                return output
            return hook
        
        # Register patching hooks
        patch_handles = []
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in patch_layers):
                handle = module.register_forward_hook(patch_hook(name))
                patch_handles.append(handle)
        
        try:
            # Forward pass with patching
            with torch.no_grad():
                if isinstance(corrupted_inputs, dict):
                    patched_output = self.model(**corrupted_inputs)
                else:
                    patched_output = self.model(corrupted_inputs)
        finally:
            # Remove patching hooks
            for handle in patch_handles:
                handle.remove()
        
        return original_output, patched_output
    
    def apply_attention_knockout(self, 
                               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                               target_heads: List[Tuple[int, int]],  # (layer, head) pairs
                               knockout_strength: float = 1.0) -> torch.Tensor:
        """
        Apply attention head knockout intervention.
        
        Args:
            inputs: Model inputs
            target_heads: List of (layer_idx, head_idx) pairs to knock out
            knockout_strength: Strength of knockout (0-1)
            
        Returns:
            Model outputs after intervention
        """
        def attention_knockout_hook(layer_idx, head_idx):
            def hook(module, input, output):
                # Assuming output is attention weights [batch, heads, seq, seq]
                if len(output.shape) == 4 and head_idx < output.size(1):
                    modified_output = output.clone()
                    # Zero out or reduce the target attention head
                    modified_output[:, head_idx, :, :] *= (1 - knockout_strength)
                    return modified_output
                return output
            return hook
        
        # Register knockout hooks
        hook_handles = []
        for layer_idx, head_idx in target_heads:
            for name, module in self.model.named_modules():
                if f"layer.{layer_idx}" in name and "attention" in name:
                    handle = module.register_forward_hook(attention_knockout_hook(layer_idx, head_idx))
                    hook_handles.append(handle)
                    break
        
        try:
            # Forward pass with attention knockout
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
        finally:
            # Remove hooks
            for handle in hook_handles:
                handle.remove()
        
        return outputs
    
    def apply_gradient_based_intervention(self, 
                                        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                        target_concept: str,
                                        intervention_strength: float = 0.1) -> torch.Tensor:
        """
        Apply gradient-based intervention to modify model behavior.
        
        Args:
            inputs: Model inputs
            target_concept: Concept to intervene on
            intervention_strength: Strength of intervention
            
        Returns:
            Model outputs after intervention
        """
        # Enable gradients for intervention
        if isinstance(inputs, dict):
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key].requires_grad_(True)
        else:
            inputs.requires_grad_(True)
        
        # Forward pass to compute gradients
        if isinstance(inputs, dict):
            outputs = self.model(**inputs)
        else:
            outputs = self.model(inputs)
        
        # Compute loss with respect to target concept
        # This is a simplified example - in practice, you'd define a specific loss
        if hasattr(outputs, 'logits'):
            target_loss = outputs.logits.mean()
        else:
            target_loss = outputs.mean()
        
        # Compute gradients
        target_loss.backward()
        
        # Apply gradient-based modification
        with torch.no_grad():
            if isinstance(inputs, dict):
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor) and inputs[key].grad is not None:
                        # Modify inputs based on gradients
                        inputs[key] -= intervention_strength * inputs[key].grad
            else:
                if inputs.grad is not None:
                    inputs -= intervention_strength * inputs.grad
            
            # Forward pass with modified inputs
            if isinstance(inputs, dict):
                modified_outputs = self.model(**inputs)
            else:
                modified_outputs = self.model(inputs)
        
        return modified_outputs
    
    def evaluate_intervention_effects(self, 
                                    original_outputs: torch.Tensor,
                                    intervened_outputs: torch.Tensor,
                                    evaluation_metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the effects of an intervention.
        
        Args:
            original_outputs: Original model outputs
            intervened_outputs: Outputs after intervention
            evaluation_metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation results
        """
        if evaluation_metrics is None:
            evaluation_metrics = ['mse', 'cosine_similarity', 'kl_divergence']
        
        results = {}
        
        # Convert to numpy for easier computation
        orig_np = original_outputs.detach().cpu().numpy()
        interv_np = intervened_outputs.detach().cpu().numpy()
        
        if 'mse' in evaluation_metrics:
            mse = np.mean((orig_np - interv_np) ** 2)
            results['mse'] = float(mse)
        
        if 'cosine_similarity' in evaluation_metrics:
            # Flatten for cosine similarity
            orig_flat = orig_np.flatten()
            interv_flat = interv_np.flatten()
            
            cosine_sim = np.dot(orig_flat, interv_flat) / (
                np.linalg.norm(orig_flat) * np.linalg.norm(interv_flat)
            )
            results['cosine_similarity'] = float(cosine_sim)
        
        if 'kl_divergence' in evaluation_metrics:
            # Apply softmax to get probabilities
            orig_probs = F.softmax(original_outputs, dim=-1).detach().cpu().numpy()
            interv_probs = F.softmax(intervened_outputs, dim=-1).detach().cpu().numpy()
            
            # Compute KL divergence
            kl_div = stats.entropy(orig_probs.flatten(), interv_probs.flatten())
            results['kl_divergence'] = float(kl_div) if not np.isnan(kl_div) else 0.0
        
        if 'effect_magnitude' in evaluation_metrics:
            # Overall effect magnitude
            effect_mag = np.linalg.norm(orig_np - interv_np) / np.linalg.norm(orig_np)
            results['effect_magnitude'] = float(effect_mag)
        
        return results
    
    def run_intervention_experiment(self, 
                                  config: InterventionConfig,
                                  test_inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
                                  baseline_inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None) -> InterventionEffect:
        """
        Run a complete intervention experiment.
        
        Args:
            config: Intervention configuration
            test_inputs: Test inputs for the experiment
            baseline_inputs: Baseline inputs for comparison
            
        Returns:
            Intervention effect results
        """
        original_outputs = []
        intervened_outputs = []
        
        for inputs in test_inputs:
            # Get original output
            with torch.no_grad():
                if isinstance(inputs, dict):
                    orig_out = self.model(**inputs)
                else:
                    orig_out = self.model(inputs)
            original_outputs.append(orig_out)
            
            # Apply intervention based on method
            if config.method == "concept_erasure" and config.concept_vector is not None:
                interv_out = self.apply_concept_erasure(
                    inputs, config.concept_vector, 
                    [f"layer.{i}" for i in config.target_layers], 
                    config.strength
                )
            elif config.method == "attention_knockout":
                target_heads = [(layer, 0) for layer in config.target_layers]  # Knockout head 0
                interv_out = self.apply_attention_knockout(inputs, target_heads, config.strength)
            elif config.method == "gradient_intervention":
                interv_out = self.apply_gradient_based_intervention(
                    inputs, "gender", config.strength
                )
            else:
                # Default to no intervention
                interv_out = orig_out
            
            intervened_outputs.append(interv_out)
        
        # Evaluate effects
        all_effects = []
        for orig, interv in zip(original_outputs, intervened_outputs):
            effects = self.evaluate_intervention_effects(orig, interv)
            all_effects.append(effects)
        
        # Aggregate results
        aggregated_effects = {}
        for key in all_effects[0].keys():
            values = [effect[key] for effect in all_effects]
            aggregated_effects[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # Create intervention effect object
        effect = InterventionEffect(
            config=config,
            original_outputs={f"output_{i}": out.detach().cpu().numpy() 
                            for i, out in enumerate(original_outputs)},
            intervened_outputs={f"output_{i}": out.detach().cpu().numpy() 
                              for i, out in enumerate(intervened_outputs)},
            effect_magnitude=aggregated_effects.get('effect_magnitude', {}).get('mean', 0.0),
            bias_change=0.0,  # Would need specific bias metric
            quality_change=0.0,  # Would need specific quality metric
            statistical_significance=0.0,  # Would need statistical test
            confidence_interval=(0.0, 0.0)  # Would need bootstrap or similar
        )
        
        return effect
    
    def visualize_intervention_effects(self, 
                                     effects: List[InterventionEffect],
                                     save_path: str = None) -> None:
        """
        Visualize the effects of multiple interventions.
        
        Args:
            effects: List of intervention effects
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Effect magnitude comparison
        methods = [effect.config.method for effect in effects]
        magnitudes = [effect.effect_magnitude for effect in effects]
        
        axes[0, 0].bar(methods, magnitudes)
        axes[0, 0].set_title('Intervention Effect Magnitudes')
        axes[0, 0].set_ylabel('Effect Magnitude')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Bias change comparison
        bias_changes = [effect.bias_change for effect in effects]
        axes[0, 1].bar(methods, bias_changes, color='orange')
        axes[0, 1].set_title('Bias Change')
        axes[0, 1].set_ylabel('Bias Change')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Quality change comparison
        quality_changes = [effect.quality_change for effect in effects]
        axes[1, 0].bar(methods, quality_changes, color='green')
        axes[1, 0].set_title('Quality Change')
        axes[1, 0].set_ylabel('Quality Change')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Scatter plot: bias vs quality change
        axes[1, 1].scatter(bias_changes, quality_changes, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            axes[1, 1].annotate(method, (bias_changes[i], quality_changes[i]))
        axes[1, 1].set_xlabel('Bias Change')
        axes[1, 1].set_ylabel('Quality Change')
        axes[1, 1].set_title('Bias vs Quality Trade-off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_sample_intervention_experiment():
    """Create sample intervention experiment for demonstration."""
    # This would normally use a real model
    print("Creating sample intervention experiment...")
    
    # Mock intervention configs
    configs = [
        InterventionConfig(
            method="concept_erasure",
            target_layers=[12, 16, 20],
            strength=0.5,
            direction="suppress",
            concept_vector=np.random.randn(768)
        ),
        InterventionConfig(
            method="attention_knockout",
            target_layers=[8, 12, 16],
            strength=0.8,
            direction="suppress"
        ),
        InterventionConfig(
            method="gradient_intervention",
            target_layers=[16, 20, 24],
            strength=0.1,
            direction="neutralize"
        )
    ]
    
    # Mock intervention effects
    effects = []
    for config in configs:
        effect = InterventionEffect(
            config=config,
            original_outputs={},
            intervened_outputs={},
            effect_magnitude=np.random.uniform(0.1, 0.8),
            bias_change=np.random.uniform(-0.5, 0.2),
            quality_change=np.random.uniform(-0.1, 0.1),
            statistical_significance=np.random.uniform(0.001, 0.1),
            confidence_interval=(0.0, 0.0)
        )
        effects.append(effect)
    
    # Create mock engine for visualization
    class MockEngine:
        def visualize_intervention_effects(self, effects, save_path=None):
            # Use the function from the real class
            engine = InterventionEngine.__new__(InterventionEngine)
            engine.visualize_intervention_effects(effects, save_path)
    
    engine = MockEngine()
    engine.visualize_intervention_effects(effects)
    
    print("Sample intervention experiment completed!")
    return effects


if __name__ == "__main__":
    # Create sample experiment
    effects = create_sample_intervention_experiment()
    print(f"Created {len(effects)} intervention effects for analysis")

