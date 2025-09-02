"""
Advanced Circuit Discovery and Intervention Methods
=================================================

This module implements state-of-the-art mechanistic interpretability techniques
for discovering and intervening on neural circuits responsible for gender bias
in multilingual image captioning models.

Key Features:
- Activation patching for causal circuit discovery
- Gradient-based attribution methods
- Sparse probing for concept localization
- Causal intervention techniques
- Circuit ablation and knockout experiments
- Cross-lingual circuit comparison
- Gender-specific circuit analysis

Example usage:
    from circuit_discovery import CircuitDiscoverer
    
    discoverer = CircuitDiscoverer(model)
    circuits = discoverer.discover_gender_circuits(activations, labels)
    effects = discoverer.apply_interventions(circuits, test_data)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CircuitInfo:
    """Information about a discovered neural circuit."""
    name: str
    layer_indices: List[int]
    neuron_indices: List[int]
    activation_pattern: np.ndarray
    correlation_strength: float
    causal_effect: float
    confidence: float
    metadata: Dict[str, any] = None

@dataclass
class InterventionResult:
    """Results from a causal intervention experiment."""
    intervention_type: str
    target_circuit: str
    original_output: any
    intervened_output: any
    effect_size: float
    statistical_significance: float
    metadata: Dict[str, any] = None

class CircuitDiscoverer:
    """Advanced circuit discovery and intervention system."""
    
    def __init__(self, 
                 model: nn.Module = None,
                 device: str = "auto"):
        """
        Initialize the circuit discoverer.
        
        Args:
            model: PyTorch model to analyze
            device: Device for computations
        """
        self.model = model
        self.device = self._setup_device(device)
        self.activation_cache = {}
        self.gradient_cache = {}
        self.discovered_circuits = {}
        
        # Gender-related concept vectors
        self.gender_concepts = {
            'male_concepts': ['man', 'boy', 'father', 'son', 'brother', 'husband', 'he', 'his'],
            'female_concepts': ['woman', 'girl', 'mother', 'daughter', 'sister', 'wife', 'she', 'her'],
            'neutral_concepts': ['person', 'individual', 'human', 'someone', 'they', 'their']
        }
        
    def _setup_device(self, device: str) -> torch.device:
        """Set up computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def register_hooks(self, layer_names: List[str] = None) -> None:
        """
        Register forward and backward hooks for activation and gradient capture.
        
        Args:
            layer_names: Specific layers to hook (if None, hook all)
        """
        def forward_hook(name):
            def hook(module, input, output):
                self.activation_cache[name] = output.detach().clone()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradient_cache[name] = grad_output[0].detach().clone()
            return hook
        
        # Register hooks on specified layers
        if layer_names is None:
            # Hook all named modules
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    module.register_forward_hook(forward_hook(name))
                    module.register_backward_hook(backward_hook(name))
        else:
            # Hook specific layers
            for name, module in self.model.named_modules():
                if name in layer_names:
                    module.register_forward_hook(forward_hook(name))
                    module.register_backward_hook(backward_hook(name))
    
    def discover_circuits_by_correlation(self, 
                                       activations: Dict[str, np.ndarray],
                                       labels: List[str],
                                       threshold: float = 0.7) -> Dict[str, CircuitInfo]:
        """
        Discover circuits using correlation analysis.
        
        Args:
            activations: Dictionary mapping layer names to activation arrays
            labels: Gender labels for each sample
            threshold: Correlation threshold for circuit discovery
            
        Returns:
            Dictionary of discovered circuits
        """
        circuits = {}
        
        # Convert labels to numeric
        label_map = {'male': 1, 'female': -1, 'neutral': 0}
        numeric_labels = np.array([label_map.get(label, 0) for label in labels])
        
        for layer_name, layer_activations in activations.items():
            # Calculate correlations between each neuron and gender labels
            correlations = []
            
            for neuron_idx in range(layer_activations.shape[-1]):
                neuron_activations = layer_activations[..., neuron_idx].flatten()
                if len(neuron_activations) == len(numeric_labels):
                    correlation, p_value = stats.pearsonr(neuron_activations, numeric_labels)
                    correlations.append((neuron_idx, correlation, p_value))
            
            # Find highly correlated neurons
            significant_neurons = [
                (idx, corr, p_val) for idx, corr, p_val in correlations
                if abs(corr) > threshold and p_val < 0.05
            ]
            
            if significant_neurons:
                # Create circuit info
                neuron_indices = [idx for idx, _, _ in significant_neurons]
                correlations_vals = [corr for _, corr, _ in significant_neurons]
                
                circuit_name = f"{layer_name}_gender_circuit"
                
                # Calculate activation pattern for this circuit
                circuit_activations = layer_activations[..., neuron_indices]
                activation_pattern = np.mean(circuit_activations, axis=0)
                
                circuit_info = CircuitInfo(
                    name=circuit_name,
                    layer_indices=[int(layer_name.split('_')[-1])] if '_' in layer_name else [0],
                    neuron_indices=neuron_indices,
                    activation_pattern=activation_pattern,
                    correlation_strength=float(np.mean(np.abs(correlations_vals))),
                    causal_effect=0.0,  # To be determined by intervention
                    confidence=float(1.0 - np.mean([p_val for _, _, p_val in significant_neurons])),
                    metadata={
                        'discovery_method': 'correlation',
                        'threshold': threshold,
                        'num_neurons': len(neuron_indices),
                        'correlations': correlations_vals
                    }
                )
                
                circuits[circuit_name] = circuit_info
        
        self.discovered_circuits.update(circuits)
        return circuits
    
    def discover_circuits_by_probing(self, 
                                   activations: Dict[str, np.ndarray],
                                   labels: List[str],
                                   probe_type: str = "linear") -> Dict[str, CircuitInfo]:
        """
        Discover circuits using sparse probing techniques.
        
        Args:
            activations: Dictionary mapping layer names to activation arrays
            labels: Gender labels for each sample
            probe_type: Type of probe to use (linear, sparse)
            
        Returns:
            Dictionary of discovered circuits
        """
        circuits = {}
        
        # Prepare labels
        label_encoder = {'male': 0, 'female': 1, 'neutral': 2}
        encoded_labels = [label_encoder.get(label, 2) for label in labels]
        
        for layer_name, layer_activations in activations.items():
            # Flatten activations for probing
            flattened_activations = layer_activations.reshape(len(labels), -1)
            
            # Train probe
            if probe_type == "linear":
                probe = LogisticRegression(max_iter=1000, random_state=42)
            else:  # sparse probe
                probe = LogisticRegression(penalty='l1', solver='liblinear', 
                                         C=0.1, max_iter=1000, random_state=42)
            
            probe.fit(flattened_activations, encoded_labels)
            
            # Get feature importance (weights)
            if hasattr(probe, 'coef_'):
                weights = np.abs(probe.coef_).mean(axis=0)
                
                # Find most important features
                importance_threshold = np.percentile(weights, 95)  # Top 5%
                important_indices = np.where(weights > importance_threshold)[0]
                
                if len(important_indices) > 0:
                    # Convert flat indices back to neuron indices
                    if len(layer_activations.shape) > 2:
                        # For multi-dimensional activations
                        neuron_indices = important_indices % layer_activations.shape[-1]
                    else:
                        neuron_indices = important_indices
                    
                    circuit_name = f"{layer_name}_probe_circuit"
                    
                    # Calculate activation pattern
                    circuit_activations = flattened_activations[:, important_indices]
                    activation_pattern = np.mean(circuit_activations, axis=0)
                    
                    # Evaluate probe performance
                    predictions = probe.predict(flattened_activations)
                    accuracy = accuracy_score(encoded_labels, predictions)
                    
                    circuit_info = CircuitInfo(
                        name=circuit_name,
                        layer_indices=[int(layer_name.split('_')[-1])] if '_' in layer_name else [0],
                        neuron_indices=neuron_indices.tolist(),
                        activation_pattern=activation_pattern,
                        correlation_strength=float(accuracy),  # Use accuracy as strength measure
                        causal_effect=0.0,
                        confidence=float(accuracy),
                        metadata={
                            'discovery_method': 'probing',
                            'probe_type': probe_type,
                            'probe_accuracy': accuracy,
                            'num_features': len(important_indices),
                            'feature_weights': weights[important_indices].tolist()
                        }
                    )
                    
                    circuits[circuit_name] = circuit_info
        
        self.discovered_circuits.update(circuits)
        return circuits
    
    def discover_circuits_by_gradients(self, 
                                     model_outputs: Dict[str, torch.Tensor],
                                     target_concept: str = "gender") -> Dict[str, CircuitInfo]:
        """
        Discover circuits using gradient-based attribution.
        
        Args:
            model_outputs: Dictionary containing model outputs and gradients
            target_concept: Target concept for attribution
            
        Returns:
            Dictionary of discovered circuits
        """
        circuits = {}
        
        if not self.gradient_cache:
            print("Warning: No gradients found. Make sure to call register_hooks() first.")
            return circuits
        
        for layer_name, gradients in self.gradient_cache.items():
            # Calculate gradient magnitudes
            grad_magnitudes = torch.abs(gradients).mean(dim=0)
            
            # Find neurons with high gradient magnitudes
            threshold = torch.quantile(grad_magnitudes.flatten(), 0.95)  # Top 5%
            important_neurons = torch.where(grad_magnitudes > threshold)
            
            if len(important_neurons[0]) > 0:
                # Convert to numpy indices
                if len(important_neurons) == 1:
                    neuron_indices = important_neurons[0].cpu().numpy().tolist()
                else:
                    # For multi-dimensional gradients, take the last dimension
                    neuron_indices = important_neurons[-1].cpu().numpy().tolist()
                
                circuit_name = f"{layer_name}_gradient_circuit"
                
                # Get corresponding activations
                if layer_name in self.activation_cache:
                    activations = self.activation_cache[layer_name]
                    circuit_activations = activations[..., neuron_indices]
                    activation_pattern = circuit_activations.mean(dim=0).cpu().numpy()
                else:
                    activation_pattern = np.zeros(len(neuron_indices))
                
                circuit_info = CircuitInfo(
                    name=circuit_name,
                    layer_indices=[int(layer_name.split('_')[-1])] if '_' in layer_name else [0],
                    neuron_indices=neuron_indices,
                    activation_pattern=activation_pattern,
                    correlation_strength=float(grad_magnitudes[important_neurons].mean()),
                    causal_effect=0.0,
                    confidence=0.8,  # Default confidence for gradient-based discovery
                    metadata={
                        'discovery_method': 'gradients',
                        'target_concept': target_concept,
                        'gradient_threshold': float(threshold),
                        'num_neurons': len(neuron_indices)
                    }
                )
                
                circuits[circuit_name] = circuit_info
        
        self.discovered_circuits.update(circuits)
        return circuits
    
    def apply_activation_patching(self, 
                                clean_input: torch.Tensor,
                                corrupted_input: torch.Tensor,
                                circuit: CircuitInfo,
                                patch_strength: float = 1.0) -> InterventionResult:
        """
        Apply activation patching intervention.
        
        Args:
            clean_input: Clean input tensor
            corrupted_input: Corrupted input tensor
            circuit: Circuit to patch
            patch_strength: Strength of the intervention (0-1)
            
        Returns:
            Intervention result
        """
        if self.model is None:
            raise ValueError("Model not provided for intervention")
        
        # Get original outputs
        with torch.no_grad():
            original_output = self.model(corrupted_input)
            clean_output = self.model(clean_input)
        
        # Define patching hook
        def patching_hook(module, input, output):
            # Get clean activations for the circuit neurons
            clean_activations = self.activation_cache.get(circuit.name.split('_')[0], output)
            
            # Apply patch
            patched_output = output.clone()
            for neuron_idx in circuit.neuron_indices:
                if neuron_idx < output.shape[-1]:
                    patched_output[..., neuron_idx] = (
                        patch_strength * clean_activations[..., neuron_idx] +
                        (1 - patch_strength) * output[..., neuron_idx]
                    )
            
            return patched_output
        
        # Register patching hook
        target_layer = None
        for name, module in self.model.named_modules():
            if circuit.name.split('_')[0] in name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Target layer not found for circuit: {circuit.name}")
        
        hook_handle = target_layer.register_forward_hook(patching_hook)
        
        try:
            # Get patched output
            with torch.no_grad():
                patched_output = self.model(corrupted_input)
            
            # Calculate effect size
            original_loss = F.mse_loss(original_output, clean_output)
            patched_loss = F.mse_loss(patched_output, clean_output)
            effect_size = float((original_loss - patched_loss) / original_loss)
            
        finally:
            hook_handle.remove()
        
        # Update circuit causal effect
        circuit.causal_effect = effect_size
        
        result = InterventionResult(
            intervention_type="activation_patching",
            target_circuit=circuit.name,
            original_output=original_output.cpu().numpy(),
            intervened_output=patched_output.cpu().numpy(),
            effect_size=effect_size,
            statistical_significance=0.0,  # Would need multiple runs for significance
            metadata={
                'patch_strength': patch_strength,
                'original_loss': float(original_loss),
                'patched_loss': float(patched_loss)
            }
        )
        
        return result
    
    def apply_neuron_ablation(self, 
                            input_tensor: torch.Tensor,
                            circuit: CircuitInfo,
                            ablation_value: float = 0.0) -> InterventionResult:
        """
        Apply neuron ablation intervention.
        
        Args:
            input_tensor: Input tensor
            circuit: Circuit to ablate
            ablation_value: Value to set ablated neurons to
            
        Returns:
            Intervention result
        """
        if self.model is None:
            raise ValueError("Model not provided for intervention")
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(input_tensor)
        
        # Define ablation hook
        def ablation_hook(module, input, output):
            ablated_output = output.clone()
            for neuron_idx in circuit.neuron_indices:
                if neuron_idx < output.shape[-1]:
                    ablated_output[..., neuron_idx] = ablation_value
            return ablated_output
        
        # Register ablation hook
        target_layer = None
        for name, module in self.model.named_modules():
            if circuit.name.split('_')[0] in name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Target layer not found for circuit: {circuit.name}")
        
        hook_handle = target_layer.register_forward_hook(ablation_hook)
        
        try:
            # Get ablated output
            with torch.no_grad():
                ablated_output = self.model(input_tensor)
            
            # Calculate effect size
            effect_size = float(F.mse_loss(original_output, ablated_output))
            
        finally:
            hook_handle.remove()
        
        result = InterventionResult(
            intervention_type="neuron_ablation",
            target_circuit=circuit.name,
            original_output=original_output.cpu().numpy(),
            intervened_output=ablated_output.cpu().numpy(),
            effect_size=effect_size,
            statistical_significance=0.0,
            metadata={
                'ablation_value': ablation_value,
                'num_ablated_neurons': len(circuit.neuron_indices)
            }
        )
        
        return result
    
    def compare_cross_lingual_circuits(self, 
                                     english_circuits: Dict[str, CircuitInfo],
                                     arabic_circuits: Dict[str, CircuitInfo]) -> Dict[str, float]:
        """
        Compare circuits discovered in English vs Arabic processing.
        
        Args:
            english_circuits: Circuits discovered for English
            arabic_circuits: Circuits discovered for Arabic
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {
            'english_circuit_count': len(english_circuits),
            'arabic_circuit_count': len(arabic_circuits),
            'shared_layers': 0,
            'correlation_similarity': 0.0,
            'activation_similarity': 0.0
        }
        
        # Find shared layers
        en_layers = set()
        ar_layers = set()
        
        for circuit in english_circuits.values():
            en_layers.update(circuit.layer_indices)
        
        for circuit in arabic_circuits.values():
            ar_layers.update(circuit.layer_indices)
        
        shared_layers = en_layers.intersection(ar_layers)
        comparison['shared_layers'] = len(shared_layers)
        
        # Calculate similarity metrics
        if english_circuits and arabic_circuits:
            # Correlation strength similarity
            en_correlations = [c.correlation_strength for c in english_circuits.values()]
            ar_correlations = [c.correlation_strength for c in arabic_circuits.values()]
            
            if len(en_correlations) == len(ar_correlations):
                correlation_sim, _ = stats.pearsonr(en_correlations, ar_correlations)
                comparison['correlation_similarity'] = float(correlation_sim) if not np.isnan(correlation_sim) else 0.0
            
            # Activation pattern similarity (for circuits in shared layers)
            activation_similarities = []
            for en_name, en_circuit in english_circuits.items():
                for ar_name, ar_circuit in arabic_circuits.items():
                    if set(en_circuit.layer_indices).intersection(set(ar_circuit.layer_indices)):
                        # Calculate cosine similarity of activation patterns
                        if len(en_circuit.activation_pattern) == len(ar_circuit.activation_pattern):
                            similarity = np.dot(en_circuit.activation_pattern, ar_circuit.activation_pattern) / (
                                np.linalg.norm(en_circuit.activation_pattern) * 
                                np.linalg.norm(ar_circuit.activation_pattern)
                            )
                            activation_similarities.append(similarity)
            
            if activation_similarities:
                comparison['activation_similarity'] = float(np.mean(activation_similarities))
        
        return comparison
    
    def evaluate_circuit_robustness(self, 
                                   circuit: CircuitInfo,
                                   test_inputs: List[torch.Tensor],
                                   noise_levels: List[float] = [0.1, 0.2, 0.3]) -> Dict[str, float]:
        """
        Evaluate the robustness of a discovered circuit.
        
        Args:
            circuit: Circuit to evaluate
            test_inputs: List of test input tensors
            noise_levels: Levels of noise to test robustness
            
        Returns:
            Dictionary containing robustness metrics
        """
        if self.model is None:
            raise ValueError("Model not provided for evaluation")
        
        robustness_metrics = {
            'baseline_consistency': 0.0,
            'noise_robustness': {},
            'activation_stability': 0.0
        }
        
        # Baseline consistency across inputs
        baseline_activations = []
        
        for input_tensor in test_inputs:
            with torch.no_grad():
                _ = self.model(input_tensor)
                if circuit.name.split('_')[0] in self.activation_cache:
                    activations = self.activation_cache[circuit.name.split('_')[0]]
                    circuit_activations = activations[..., circuit.neuron_indices]
                    baseline_activations.append(circuit_activations.cpu().numpy())
        
        if baseline_activations:
            # Calculate consistency (inverse of variance)
            all_activations = np.concatenate(baseline_activations, axis=0)
            consistency = 1.0 / (1.0 + np.var(all_activations))
            robustness_metrics['baseline_consistency'] = float(consistency)
        
        # Noise robustness
        for noise_level in noise_levels:
            noise_activations = []
            
            for input_tensor in test_inputs:
                # Add noise to input
                noise = torch.randn_like(input_tensor) * noise_level
                noisy_input = input_tensor + noise
                
                with torch.no_grad():
                    _ = self.model(noisy_input)
                    if circuit.name.split('_')[0] in self.activation_cache:
                        activations = self.activation_cache[circuit.name.split('_')[0]]
                        circuit_activations = activations[..., circuit.neuron_indices]
                        noise_activations.append(circuit_activations.cpu().numpy())
            
            if noise_activations and baseline_activations:
                # Calculate similarity between noisy and baseline activations
                similarities = []
                for baseline, noisy in zip(baseline_activations, noise_activations):
                    similarity = np.corrcoef(baseline.flatten(), noisy.flatten())[0, 1]
                    if not np.isnan(similarity):
                        similarities.append(similarity)
                
                if similarities:
                    robustness_metrics['noise_robustness'][noise_level] = float(np.mean(similarities))
        
        return robustness_metrics
    
    def save_circuits(self, filepath: str) -> None:
        """Save discovered circuits to file."""
        circuits_data = {}
        
        for name, circuit in self.discovered_circuits.items():
            circuits_data[name] = {
                'name': circuit.name,
                'layer_indices': circuit.layer_indices,
                'neuron_indices': circuit.neuron_indices,
                'activation_pattern': circuit.activation_pattern.tolist(),
                'correlation_strength': circuit.correlation_strength,
                'causal_effect': circuit.causal_effect,
                'confidence': circuit.confidence,
                'metadata': circuit.metadata
            }
        
        with open(filepath, 'w') as f:
            json.dump(circuits_data, f, indent=2)
        
        print(f"Circuits saved to {filepath}")
    
    def load_circuits(self, filepath: str) -> Dict[str, CircuitInfo]:
        """Load circuits from file."""
        with open(filepath, 'r') as f:
            circuits_data = json.load(f)
        
        circuits = {}
        for name, data in circuits_data.items():
            circuit = CircuitInfo(
                name=data['name'],
                layer_indices=data['layer_indices'],
                neuron_indices=data['neuron_indices'],
                activation_pattern=np.array(data['activation_pattern']),
                correlation_strength=data['correlation_strength'],
                causal_effect=data['causal_effect'],
                confidence=data['confidence'],
                metadata=data.get('metadata', {})
            )
            circuits[name] = circuit
        
        self.discovered_circuits.update(circuits)
        return circuits


def create_sample_circuit_analysis():
    """Create sample circuit discovery analysis for demonstration."""
    # Mock data for demonstration
    np.random.seed(42)
    
    # Create mock activations
    mock_activations = {
        'layer_12': np.random.randn(100, 768),  # 100 samples, 768 neurons
        'layer_16': np.random.randn(100, 768),
        'layer_20': np.random.randn(100, 768)
    }
    
    # Create mock labels
    mock_labels = ['male'] * 30 + ['female'] * 40 + ['neutral'] * 30
    
    # Initialize circuit discoverer
    discoverer = CircuitDiscoverer()
    
    # Discover circuits using different methods
    print("Discovering circuits using correlation analysis...")
    correlation_circuits = discoverer.discover_circuits_by_correlation(
        mock_activations, mock_labels, threshold=0.3
    )
    
    print("Discovering circuits using probing...")
    probe_circuits = discoverer.discover_circuits_by_probing(
        mock_activations, mock_labels, probe_type="sparse"
    )
    
    # Print results
    print(f"\nDiscovered {len(correlation_circuits)} circuits via correlation")
    print(f"Discovered {len(probe_circuits)} circuits via probing")
    
    for name, circuit in correlation_circuits.items():
        print(f"\nCircuit: {name}")
        print(f"  Layers: {circuit.layer_indices}")
        print(f"  Neurons: {len(circuit.neuron_indices)}")
        print(f"  Correlation strength: {circuit.correlation_strength:.3f}")
        print(f"  Confidence: {circuit.confidence:.3f}")
    
    return discoverer


if __name__ == "__main__":
    # Create sample analysis
    discoverer = create_sample_circuit_analysis()
    print("Circuit discovery analysis completed!")

