"""
Cross-Lingual Feature Alignment (CLFA)
=======================================

Novel methodology component that discovers which SAE features
encode the same concepts in Arabic vs English.

Uses Wasserstein distance and optimal transport for alignment.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import json

try:
    import ot  # POT: Python Optimal Transport
    HAS_OT = True
except ImportError:
    HAS_OT = False
    print("Warning: POT library not found. Install with: pip install POT")

logger = logging.getLogger(__name__)


@dataclass
class FeatureAlignment:
    """Result of feature alignment between languages."""
    arabic_feature_idx: int
    english_feature_idx: int
    similarity_score: float
    wasserstein_distance: float
    semantic_category: Optional[str] = None


@dataclass
class CLFAResult:
    """Full CLFA analysis result."""
    alignments: List[FeatureAlignment]
    alignment_matrix: np.ndarray
    transport_plan: Optional[np.ndarray]
    mean_alignment_score: float
    language_specific_arabic: List[int]  # Features only active for Arabic
    language_specific_english: List[int]  # Features only active for English
    shared_features: List[Tuple[int, int]]  # (arabic_idx, english_idx) pairs


class CrossLingualFeatureAligner:
    """
    Cross-Lingual Feature Alignment (CLFA) - Novel Methodology Component
    
    Discovers semantic correspondences between Arabic and English 
    feature representations using optimal transport.
    
    Key Innovation: Uses SAE features (sparse, interpretable) rather than
    dense activations, enabling principled alignment.
    """
    
    def __init__(
        self,
        arabic_sae: Any,
        english_sae: Any,
        similarity_threshold: float = 0.7,
        device: str = "cuda"
    ):
        self.arabic_sae = arabic_sae
        self.english_sae = english_sae
        self.threshold = similarity_threshold
        self.device = device
        
        # Feature statistics
        self.arabic_feature_means: Optional[np.ndarray] = None
        self.english_feature_means: Optional[np.ndarray] = None
        self.arabic_feature_vars: Optional[np.ndarray] = None
        self.english_feature_vars: Optional[np.ndarray] = None
    
    def compute_feature_statistics(
        self,
        arabic_activations: Dict[int, torch.Tensor],
        english_activations: Dict[int, torch.Tensor]
    ):
        """
        Compute statistics of SAE feature activations.
        
        Args:
            arabic_activations: {layer_idx: tensor of shape (N, num_features)}
            english_activations: Same format
        """
        # Encode through SAE to get sparse features
        arabic_features = self._get_sae_features(arabic_activations, self.arabic_sae)
        english_features = self._get_sae_features(english_activations, self.english_sae)
        
        # Compute mean and variance per feature
        self.arabic_feature_means = arabic_features.mean(axis=0)
        self.arabic_feature_vars = arabic_features.var(axis=0)
        self.english_feature_means = english_features.mean(axis=0)
        self.english_feature_vars = english_features.var(axis=0)
        
        logger.info(f"Arabic features: {arabic_features.shape}")
        logger.info(f"English features: {english_features.shape}")
    
    def _get_sae_features(
        self,
        activations: Dict[int, torch.Tensor],
        sae: Any
    ) -> np.ndarray:
        """
        Extract sparse SAE features from activations.
        
        Returns:
            np.ndarray of shape (total_samples, num_sae_features)
        """
        all_features = []
        
        for layer_idx, acts in activations.items():
            if hasattr(sae, 'encode'):
                # Standard SAE interface
                with torch.no_grad():
                    if isinstance(acts, np.ndarray):
                        acts = torch.from_numpy(acts).float()
                    acts = acts.to(self.device)
                    features = sae.encode(acts)
                    all_features.append(features.cpu().numpy())
            else:
                # Fallback: use activations directly
                if isinstance(acts, torch.Tensor):
                    all_features.append(acts.cpu().numpy())
                else:
                    all_features.append(acts)
        
        # Concatenate across layers
        return np.concatenate(all_features, axis=0)
    
    def compute_wasserstein_matrix(
        self,
        arabic_features: np.ndarray,
        english_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise Wasserstein distances between features.
        
        Returns:
            np.ndarray of shape (num_arabic_features, num_english_features)
        """
        n_arabic = arabic_features.shape[1]
        n_english = english_features.shape[1]
        
        wasserstein_matrix = np.zeros((n_arabic, n_english))
        
        for i in range(n_arabic):
            for j in range(n_english):
                # 1D Wasserstein distance between feature distributions
                w_dist = wasserstein_distance(
                    arabic_features[:, i],
                    english_features[:, j]
                )
                wasserstein_matrix[i, j] = w_dist
        
        return wasserstein_matrix
    
    def compute_cosine_similarity_matrix(
        self,
        arabic_features: np.ndarray,
        english_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between feature activation patterns.
        
        Uses the mean activation pattern per feature.
        """
        # Normalize feature means
        arabic_norms = np.linalg.norm(self.arabic_feature_means)
        english_norms = np.linalg.norm(self.english_feature_means)
        
        if arabic_norms > 0 and english_norms > 0:
            arabic_normalized = self.arabic_feature_means / arabic_norms
            english_normalized = self.english_feature_means / english_norms
            
            # Compute cosine similarity
            similarity = 1 - cdist(
                arabic_normalized.reshape(-1, 1).T,
                english_normalized.reshape(-1, 1).T,
                metric='cosine'
            )
            return similarity.squeeze()
        
        return np.zeros((len(self.arabic_feature_means), len(self.english_feature_means)))
    
    def optimal_transport_alignment(
        self,
        arabic_features: np.ndarray,
        english_features: np.ndarray,
        reg: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use optimal transport to find feature correspondences.
        
        Returns:
            Tuple of (transport_plan, alignment_matrix)
        """
        if not HAS_OT:
            logger.warning("POT library not available, using greedy alignment")
            return self._greedy_alignment(arabic_features, english_features)
        
        # Feature distributions (as marginals)
        n_arabic = arabic_features.shape[1]
        n_english = english_features.shape[1]
        
        # Uniform distributions
        a = np.ones(n_arabic) / n_arabic
        b = np.ones(n_english) / n_english
        
        # Cost matrix (Wasserstein distances)
        C = self.compute_wasserstein_matrix(arabic_features, english_features)
        
        # Normalize cost matrix
        C = C / C.max() if C.max() > 0 else C
        
        # Solve entropic regularized OT
        transport_plan = ot.sinkhorn(a, b, C, reg)
        
        # Convert transport plan to alignment matrix (thresholded)
        alignment_matrix = (transport_plan > (1 / (n_arabic * n_english))).astype(float)
        
        return transport_plan, alignment_matrix
    
    def _greedy_alignment(
        self,
        arabic_features: np.ndarray,
        english_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedy alignment fallback when POT is not available.
        Uses Hungarian algorithm.
        """
        # Cost matrix
        C = self.compute_wasserstein_matrix(arabic_features, english_features)
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(C)
        
        # Create alignment matrix
        alignment_matrix = np.zeros(C.shape)
        alignment_matrix[row_ind, col_ind] = 1.0
        
        return alignment_matrix, alignment_matrix
    
    def identify_language_specific_features(
        self,
        arabic_features: np.ndarray,
        english_features: np.ndarray,
        activity_threshold: float = 0.01
    ) -> Tuple[List[int], List[int]]:
        """
        Identify features that are language-specific.
        
        Features active for one language but not the other.
        """
        # Compute activity levels (mean activation)
        arabic_activity = arabic_features.mean(axis=0)
        english_activity = english_features.mean(axis=0)
        
        # Active features
        arabic_active = np.where(arabic_activity > activity_threshold)[0]
        english_active = np.where(english_activity > activity_threshold)[0]
        
        # Find overlap
        shared = set(arabic_active) & set(english_active)
        
        arabic_specific = [i for i in arabic_active if i not in shared]
        english_specific = [i for i in english_active if i not in shared]
        
        return arabic_specific, english_specific
    
    def align_features(
        self,
        arabic_activations: Dict[int, torch.Tensor],
        english_activations: Dict[int, torch.Tensor]
    ) -> CLFAResult:
        """
        Main CLFA method: Align features across languages.
        
        Returns:
            CLFAResult with alignments and analysis.
        """
        # Get SAE features
        arabic_features = self._get_sae_features(arabic_activations, self.arabic_sae)
        english_features = self._get_sae_features(english_activations, self.english_sae)
        
        # Compute statistics
        self.compute_feature_statistics(arabic_activations, english_activations)
        
        # Optimal transport alignment
        transport_plan, alignment_matrix = self.optimal_transport_alignment(
            arabic_features, english_features
        )
        
        # Extract alignments
        alignments = []
        wasserstein_matrix = self.compute_wasserstein_matrix(arabic_features, english_features)
        
        for i in range(alignment_matrix.shape[0]):
            for j in range(alignment_matrix.shape[1]):
                if alignment_matrix[i, j] > 0:
                    sim_score = 1.0 - (wasserstein_matrix[i, j] / wasserstein_matrix.max())
                    
                    if sim_score >= self.threshold:
                        alignments.append(FeatureAlignment(
                            arabic_feature_idx=i,
                            english_feature_idx=j,
                            similarity_score=sim_score,
                            wasserstein_distance=wasserstein_matrix[i, j]
                        ))
        
        # Identify language-specific features
        arabic_specific, english_specific = self.identify_language_specific_features(
            arabic_features, english_features
        )
        
        # Shared features from alignments
        shared = [(a.arabic_feature_idx, a.english_feature_idx) for a in alignments]
        
        # Mean alignment score
        mean_score = np.mean([a.similarity_score for a in alignments]) if alignments else 0.0
        
        return CLFAResult(
            alignments=alignments,
            alignment_matrix=alignment_matrix,
            transport_plan=transport_plan,
            mean_alignment_score=mean_score,
            language_specific_arabic=arabic_specific,
            language_specific_english=english_specific,
            shared_features=shared
        )
    
    def compute_clbas_metric(
        self,
        clfa_result: CLFAResult,
        arabic_bias_scores: np.ndarray,
        english_bias_scores: np.ndarray
    ) -> float:
        """
        Compute Cross-Lingual Bias Alignment Score (CLBAS).
        
        CLBAS = Σ |bias(f_ar) - bias(f_en)| × alignment_score(f_ar, f_en)
                / Σ alignment_score(f_ar, f_en)
        
        Lower CLBAS = More aligned bias across languages (consistent treatment)
        Higher CLBAS = Different bias in different languages (language-specific bias)
        """
        if not clfa_result.alignments:
            return 0.0
        
        numerator = 0.0
        denominator = 0.0
        
        for alignment in clfa_result.alignments:
            ar_idx = alignment.arabic_feature_idx
            en_idx = alignment.english_feature_idx
            
            if ar_idx < len(arabic_bias_scores) and en_idx < len(english_bias_scores):
                bias_diff = abs(arabic_bias_scores[ar_idx] - english_bias_scores[en_idx])
                weight = alignment.similarity_score
                
                numerator += bias_diff * weight
                denominator += weight
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def save_results(self, result: CLFAResult, output_path: Path):
        """Save CLFA results to JSON."""
        output = {
            'alignments': [
                {
                    'arabic_idx': a.arabic_feature_idx,
                    'english_idx': a.english_feature_idx,
                    'similarity': a.similarity_score,
                    'wasserstein_distance': a.wasserstein_distance,
                    'semantic_category': a.semantic_category
                }
                for a in result.alignments
            ],
            'mean_alignment_score': result.mean_alignment_score,
            'arabic_specific_features': result.language_specific_arabic,
            'english_specific_features': result.language_specific_english,
            'shared_features': result.shared_features,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved CLFA results to {output_path}")
        logger.info(f"Found {len(result.alignments)} aligned feature pairs")
        logger.info(f"Arabic-specific: {len(result.language_specific_arabic)}")
        logger.info(f"English-specific: {len(result.language_specific_english)}")
