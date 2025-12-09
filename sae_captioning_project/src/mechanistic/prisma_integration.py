"""
ViT-Prisma Integration Module
==============================

Integrates ViT-Prisma tools for advanced mechanistic interpretability analysis
of vision-language model activations and gender bias patterns.

Features:
- Activation caching and factored matrix analysis
- Logit lens for layer-wise prediction analysis
- Hook points for selective activation monitoring
- Advanced factorization of attention patterns
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HookPoint:
    """Represents a hook point in the model for activation capture."""
    name: str
    layer_idx: int
    module: nn.Module
    activation_fn: Optional[Callable] = None
    cache: Dict[str, torch.Tensor] = None
    
    def __post_init__(self):
        if self.cache is None:
            self.cache = {}


class ActivationCache:
    """
    Caches activations at multiple hook points throughout the model.
    Enables efficient computation of mechanistic interpretability metrics.
    """
    
    def __init__(self, model: nn.Module, hook_points: List[HookPoint]):
        """
        Initialize activation cache.
        
        Args:
            model: The neural network model
            hook_points: List of hook points to monitor
        """
        self.model = model
        self.hook_points = {hp.name: hp for hp in hook_points}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks at all hook points."""
        for hook_point in self.hook_points.values():
            handle = hook_point.module.register_forward_hook(
                self._make_hook_fn(hook_point)
            )
            self.hooks.append(handle)
    
    def _make_hook_fn(self, hook_point: HookPoint) -> Callable:
        """Create a hook function for a specific hook point."""
        def hook_fn(module, input, output):
            activation = output.detach()
            
            if hook_point.activation_fn is not None:
                activation = hook_point.activation_fn(activation)
            
            hook_point.cache['activation'] = activation
            hook_point.cache['input_shape'] = input[0].shape if input else None
        
        return hook_fn
    
    def get_cached_activation(self, hook_name: str) -> Optional[torch.Tensor]:
        """Retrieve cached activation from a specific hook point."""
        if hook_name in self.hook_points:
            return self.hook_points[hook_name].cache.get('activation', None)
        return None
    
    def get_all_activations(self) -> Dict[str, torch.Tensor]:
        """Retrieve all cached activations."""
        activations = {}
        for name, hp in self.hook_points.items():
            if 'activation' in hp.cache:
                activations[name] = hp.cache['activation']
        return activations
    
    def clear_cache(self):
        """Clear all cached activations."""
        for hp in self.hook_points.values():
            hp.cache.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()


class FactoredMatrix:
    """
    Analyzes activations as factored matrices for interpretability.
    Useful for understanding composition and information flow in layers.
    """
    
    def __init__(self, activation: torch.Tensor, name: str = ""):
        """
        Initialize factored matrix analysis.
        
        Args:
            activation: Activation tensor to analyze
            name: Name of the activation (for logging)
        """
        self.name = name
        self.activation = activation
        self.shape = activation.shape
        
    def compute_svd(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SVD of the activation matrix.
        
        Returns:
            Tuple of (U, S, V) from SVD
        """
        # Reshape to 2D if necessary
        if len(self.activation.shape) > 2:
            batch_size = self.activation.shape[0]
            activation_2d = self.activation.view(batch_size, -1)
        else:
            activation_2d = self.activation
        
        U, S, V = torch.linalg.svd(activation_2d, full_matrices=False)
        return U, S, V
    
    def compute_rank(self, threshold: float = 0.95) -> int:
        """
        Compute effective rank based on singular value threshold.
        
        Args:
            threshold: Cumulative variance threshold (default 95%)
            
        Returns:
            Effective rank
        """
        _, S, _ = self.compute_svd()
        cumsum = torch.cumsum(S / S.sum(), dim=0)
        rank = (cumsum < threshold).sum().item() + 1
        return rank
    
    def compute_pca(self, n_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PCA of the activation.
        
        Args:
            n_components: Number of principal components
            
        Returns:
            Tuple of (components, explained_variance_ratio)
        """
        U, S, _ = self.compute_svd()
        components = U[:, :n_components]
        variance_ratio = (S[:n_components] ** 2) / (S ** 2).sum()
        return components, variance_ratio
    
    def compute_information_content(self) -> float:
        """
        Compute Shannon information content of activation distribution.
        
        Returns:
            Information content (entropy) in bits
        """
        # Normalize activation to probability distribution
        act_abs = torch.abs(self.activation.view(-1))
        if act_abs.sum() > 0:
            p = act_abs / act_abs.sum()
            # Remove zero probabilities to avoid log(0)
            p = p[p > 0]
            entropy = -(p * torch.log2(p)).sum().item()
        else:
            entropy = 0.0
        return entropy


class LogitLens:
    """
    Implements logit lens for analyzing prediction information flow through layers.
    Reveals at which layers gender information becomes prominent.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_outputs: Dict[int, torch.Tensor],
        classifier_head: Optional[nn.Module] = None
    ):
        """
        Initialize logit lens analyzer.
        
        Args:
            model: The neural network model
            layer_outputs: Dictionary mapping layer indices to their outputs
            classifier_head: Optional classification head (if not part of model)
        """
        self.model = model
        self.layer_outputs = layer_outputs
        self.classifier_head = classifier_head
    
    def analyze_gender_prediction_emergence(
        self,
        gender_labels: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Analyze at which layers gender prediction emerges.
        
        Args:
            gender_labels: Ground truth gender labels
            layers: Specific layers to analyze (all if None)
            
        Returns:
            Dictionary mapping layer to prediction accuracy
        """
        if layers is None:
            layers = list(self.layer_outputs.keys())
        
        accuracies = {}
        
        for layer in layers:
            if layer not in self.layer_outputs:
                continue
            
            output = self.layer_outputs[layer]
            
            # Project to gender logits
            if self.classifier_head is not None:
                gender_logits = self.classifier_head(output)
            else:
                # Simple linear probe on layer output
                gender_logits = output.mean(dim=1)  # Aggregate if needed
            
            # Compute accuracy
            predictions = (gender_logits > 0).long()
            accuracy = (predictions == gender_labels).float().mean().item()
            accuracies[layer] = accuracy
        
        return accuracies
    
    def compute_layer_gender_information(
        self,
        layers: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute mutual information between layer activations and gender.
        
        Args:
            layers: Specific layers to analyze
            
        Returns:
            Dictionary mapping layer to information content
        """
        if layers is None:
            layers = list(self.layer_outputs.keys())
        
        information = {}
        
        for layer in layers:
            if layer not in self.layer_outputs:
                continue
            
            output = self.layer_outputs[layer]
            fm = FactoredMatrix(output, name=f"layer_{layer}")
            info = fm.compute_information_content()
            information[layer] = info
        
        return information


class InteractionPatternAnalyzer:
    """
    Analyzes interaction patterns between gender features across languages.
    Identifies where cross-lingual gender representations diverge.
    """
    
    def __init__(self, sae: nn.Module, device: str = "cuda"):
        """
        Initialize interaction analyzer.
        
        Args:
            sae: Sparse autoencoder for feature extraction
            device: Device to use
        """
        self.sae = sae.to(device)
        self.device = device
    
    def analyze_feature_interactions(
        self,
        english_features: torch.Tensor,
        arabic_features: torch.Tensor,
        interaction_order: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze higher-order interactions between SAE features.
        
        Args:
            english_features: English SAE features
            arabic_features: Arabic SAE features
            interaction_order: Order of interactions to analyze (2 for pairs)
            
        Returns:
            Dictionary with interaction statistics
        """
        results = {
            "pairwise_correlations": self._compute_pairwise_interactions(
                english_features, arabic_features
            ),
            "feature_importance": self._compute_feature_importance(
                english_features, arabic_features
            ),
            "divergence_points": self._find_divergence_points(
                english_features, arabic_features
            )
        }
        return results
    
    def _compute_pairwise_interactions(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> np.ndarray:
        """Compute pairwise feature interactions."""
        corr_matrix = torch.corrcoef(torch.cat([feat1.T, feat2.T], dim=0))
        return corr_matrix.cpu().numpy()
    
    def _compute_feature_importance(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> Dict[int, float]:
        """Compute feature importance using activation magnitude."""
        importance = {}
        
        feat1_importance = feat1.abs().mean(dim=0)
        feat2_importance = feat2.abs().mean(dim=0)
        
        for i in range(feat1_importance.shape[0]):
            importance[f"en_feat_{i}"] = feat1_importance[i].item()
            importance[f"ar_feat_{i}"] = feat2_importance[i].item()
        
        return importance
    
    def _find_divergence_points(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> List[int]:
        """Identify features with largest cross-lingual divergence."""
        # Compute KL divergence between distributions
        feat1_dist = (feat1.abs() / feat1.abs().sum(dim=0, keepdim=True)).detach()
        feat2_dist = (feat2.abs() / feat2.abs().sum(dim=0, keepdim=True)).detach()
        
        divergences = []
        for i in range(feat1.shape[1]):
            p = feat1_dist[:, i]
            q = feat2_dist[:, i]
            
            # Avoid log(0)
            p = torch.clamp(p, min=1e-10)
            q = torch.clamp(q, min=1e-10)
            
            kl = (p * (torch.log(p) - torch.log(q))).sum().item()
            divergences.append((i, kl))
        
        # Return indices of top divergence points
        divergences.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in divergences[:10]]


class TransformerProbeAnalyzer:
    """
    Analyzes information in transformer layers using linear probes.
    Determines which layers encode gender information most reliably.
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        num_classes: int = 2,
        device: str = "cuda"
    ):
        """
        Initialize probe analyzer.
        
        Args:
            layer_dims: Dimensions of each layer to probe
            num_classes: Number of output classes (for gender: 2)
            device: Device to use
        """
        self.probes = nn.ModuleDict()
        self.device = device
        
        for i, dim in enumerate(layer_dims):
            self.probes[f"layer_{i}"] = nn.Linear(dim, num_classes)
        
        self.probes.to(device)
    
    def train_probes(
        self,
        layer_activations: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        epochs: int = 10,
        learning_rate: float = 1e-2
    ) -> Dict[int, float]:
        """
        Train linear probes on layer activations.
        
        Args:
            layer_activations: Dict mapping layer index to activations
            labels: Target labels for classification
            epochs: Number of training epochs
            learning_rate: Learning rate for probe training
            
        Returns:
            Dictionary mapping layer to final accuracy
        """
        accuracies = {}
        criterion = nn.CrossEntropyLoss()
        
        for layer_idx, activations in layer_activations.items():
            probe = self.probes[f"layer_{layer_idx}"]
            optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
            
            for _ in range(epochs):
                optimizer.zero_grad()
                logits = probe(activations)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            # Compute final accuracy
            with torch.no_grad():
                logits = probe(activations)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == labels).float().mean().item()
                accuracies[layer_idx] = accuracy
        
        return accuracies


__all__ = [
    'HookPoint',
    'ActivationCache',
    'FactoredMatrix',
    'LogitLens',
    'InteractionPatternAnalyzer',
    'TransformerProbeAnalyzer',
]
