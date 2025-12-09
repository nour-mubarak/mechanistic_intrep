"""
Multilingual LLM Features Integration
======================================

Integrates multilingual-llm-features tools for cross-lingual mechanistic analysis.
Provides tools for analyzing language-specific feature representations and
morphological vs. semantic gender encoding in Arabic and English.

Features:
- Cross-lingual feature alignment
- Morphological gender analysis for Arabic
- Semantic gender analysis
- Language-specific feature identification
- Contrastive analysis between languages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class LanguageFeatureProfile:
    """Profile of gender features for a specific language."""
    language: str
    male_features: List[int]
    female_features: List[int]
    neutral_features: List[int]
    morphological_features: Optional[List[int]] = None  # For Arabic
    semantic_features: Optional[List[int]] = None
    feature_strength: Dict[int, float] = None  # Feature importance scores


class CrossLingualFeatureAligner:
    """
    Aligns gender features across languages to find shared vs. language-specific
    representations and analyze their divergence patterns.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize cross-lingual aligner.
        
        Args:
            device: Device to use
        """
        self.device = device
        self.alignments = {}
    
    def align_features(
        self,
        english_features: torch.Tensor,
        arabic_features: torch.Tensor,
        similarity_threshold: float = 0.7
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Align features across languages using cosine similarity.
        
        Args:
            english_features: English feature space (batch, num_features)
            arabic_features: Arabic feature space (batch, num_features)
            similarity_threshold: Minimum similarity for alignment
            
        Returns:
            Dictionary with aligned features and similarity scores
        """
        # Compute feature means across samples
        en_mean = english_features.mean(dim=0)
        ar_mean = arabic_features.mean(dim=0)
        
        # Normalize
        en_norm = F.normalize(en_mean, p=2, dim=-1)
        ar_norm = F.normalize(ar_mean, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.mm(en_norm.unsqueeze(0), ar_norm.unsqueeze(1)).squeeze()
        
        # Find aligned pairs
        aligned_pairs = []
        for en_idx in range(len(en_mean)):
            for ar_idx in range(len(ar_mean)):
                sim = similarity[en_idx, ar_idx].item() if len(similarity.shape) > 0 else similarity.item()
                if sim >= similarity_threshold:
                    aligned_pairs.append((en_idx, ar_idx, sim))
        
        return {
            "aligned_pairs": aligned_pairs,
            "similarity_matrix": similarity.cpu().numpy(),
            "threshold": similarity_threshold
        }
    
    def compute_alignment_statistics(
        self,
        alignment_result: Dict
    ) -> Dict[str, float]:
        """
        Compute statistics about feature alignment.
        
        Args:
            alignment_result: Result from align_features
            
        Returns:
            Dictionary with alignment statistics
        """
        pairs = alignment_result["aligned_pairs"]
        similarities = [sim for _, _, sim in pairs]
        
        if not similarities:
            return {
                "num_aligned": 0,
                "mean_similarity": 0,
                "alignment_ratio": 0
            }
        
        return {
            "num_aligned": len(pairs),
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "alignment_ratio": len(pairs) / (len(pairs) + 1e-8)
        }


class MorphologicalGenderAnalyzer:
    """
    Analyzes morphological gender in Arabic, which differs significantly
    from semantic gender encoding.
    """
    
    def __init__(self):
        """Initialize morphological analyzer."""
        self.arabic_fem_suffixes = ['ة', 'ها', 'تها', 'نها']
        self.arabic_fem_prefixes = []
    
    def extract_morphological_gender(
        self,
        arabic_captions: List[str]
    ) -> Dict[str, List[str]]:
        """
        Extract morphologically gendered words from Arabic captions.
        
        Args:
            arabic_captions: List of Arabic caption texts
            
        Returns:
            Dictionary with feminine and masculine words
        """
        feminine_words = set()
        masculine_words = set()
        
        for caption in arabic_captions:
            words = caption.split()
            for word in words:
                # Check for feminine suffixes (simplified)
                if any(word.endswith(suffix) for suffix in self.arabic_fem_suffixes):
                    feminine_words.add(word)
                else:
                    masculine_words.add(word)
        
        return {
            "feminine_words": list(feminine_words),
            "masculine_words": list(masculine_words),
            "total_feminine": len(feminine_words),
            "total_masculine": len(masculine_words)
        }
    
    def analyze_morphological_features(
        self,
        sae_features: torch.Tensor,
        morphological_labels: Dict[str, List[str]],
        sample_words: List[str]
    ) -> Dict[str, float]:
        """
        Analyze which SAE features respond to morphological gender.
        
        Args:
            sae_features: SAE feature activations
            morphological_labels: Morphological gender labels
            sample_words: Words in samples
            
        Returns:
            Dictionary mapping features to morphological responsivity
        """
        results = {}
        
        fem_mask = np.array([
            any(w in morphological_labels["feminine_words"] for w in sample_words)
            for _ in range(sae_features.shape[0])
        ])
        
        for feat_idx in range(sae_features.shape[1]):
            feat_acts = sae_features[:, feat_idx].cpu().numpy()
            
            if fem_mask.sum() > 0 and (~fem_mask).sum() > 0:
                t_stat, p_value = stats.ttest_ind(
                    feat_acts[fem_mask],
                    feat_acts[~fem_mask]
                )
                results[feat_idx] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "effect_size": abs(t_stat) / np.sqrt(len(feat_acts))
                }
        
        return results


class SemanticGenderAnalyzer:
    """
    Analyzes semantic gender associations (e.g., "doctor" vs "nurse" associations)
    independent of morphological markers.
    """
    
    def __init__(self):
        """Initialize semantic analyzer."""
        # Common semantic gender associations
        self.male_semantic_words = {
            'man', 'boy', 'king', 'prince', 'father', 'son',
            'رجل', 'ولد', 'ملك', 'أمير', 'الأب', 'الابن'
        }
        self.female_semantic_words = {
            'woman', 'girl', 'queen', 'princess', 'mother', 'daughter',
            'امرأة', 'بنت', 'ملكة', 'أميرة', 'الأم', 'الابنة'
        }
    
    def extract_semantic_gender(
        self,
        captions: List[str]
    ) -> Dict[str, float]:
        """
        Extract semantic gender from captions.
        
        Args:
            captions: List of captions
            
        Returns:
            Dictionary with semantic gender scores
        """
        male_count = 0
        female_count = 0
        
        for caption in captions:
            words_lower = caption.lower().split()
            male_count += sum(1 for w in words_lower if w in self.male_semantic_words)
            female_count += sum(1 for w in words_lower if w in self.female_semantic_words)
        
        return {
            "male_semantic_count": male_count,
            "female_semantic_count": female_count,
            "semantic_gender_ratio": (male_count + 1e-8) / (female_count + 1e-8)
        }
    
    def analyze_semantic_features(
        self,
        sae_features: torch.Tensor,
        semantic_labels: Dict[str, float]
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze which SAE features respond to semantic gender.
        
        Args:
            sae_features: SAE feature activations
            semantic_labels: Semantic gender labels
            
        Returns:
            Dictionary mapping features to semantic responsivity
        """
        results = {}
        
        semantic_scores = np.array([
            semantic_labels.get("semantic_gender_ratio", 1.0)
            for _ in range(sae_features.shape[0])
        ])
        
        for feat_idx in range(sae_features.shape[1]):
            feat_acts = sae_features[:, feat_idx].cpu().numpy()
            correlation = np.corrcoef(feat_acts, semantic_scores)[0, 1]
            
            results[feat_idx] = {
                "semantic_correlation": float(correlation) if not np.isnan(correlation) else 0,
                "is_semantic_responsive": abs(correlation) > 0.3
            }
        
        return results


class ContrastiveLanguageAnalyzer:
    """
    Performs contrastive analysis to understand language-specific effects
    in gender encoding.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize contrastive analyzer.
        
        Args:
            device: Device to use
        """
        self.device = device
    
    def compare_language_feature_spaces(
        self,
        english_features: torch.Tensor,
        arabic_features: torch.Tensor,
        english_labels: List[str],
        arabic_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Compare gender feature spaces between English and Arabic.
        
        Args:
            english_features: English SAE features
            arabic_features: Arabic SAE features
            english_labels: English gender labels
            arabic_labels: Arabic gender labels
            
        Returns:
            Dictionary with comparative analysis
        """
        results = {}
        
        # Mask for male samples
        en_male_mask = np.array([label == "male" for label in english_labels])
        ar_male_mask = np.array([label == "male" for label in arabic_labels])
        
        # Compute gender separation in each language
        results["english_gender_separation"] = self._compute_gender_separation(
            english_features, en_male_mask
        )
        results["arabic_gender_separation"] = self._compute_gender_separation(
            arabic_features, ar_male_mask
        )
        
        # Analyze shared vs. language-specific encoding
        results["shared_encoding"] = self._analyze_shared_encoding(
            english_features, arabic_features, en_male_mask, ar_male_mask
        )
        
        return results
    
    def _compute_gender_separation(
        self,
        features: torch.Tensor,
        male_mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute gender separation in feature space."""
        male_feats = features[male_mask].mean(dim=0)
        female_feats = features[~male_mask].mean(dim=0)
        
        # Compute separation distance
        separation = torch.norm(male_feats - female_feats).item()
        
        # Compute angle between gender directions
        male_norm = F.normalize(male_feats, p=2, dim=0)
        female_norm = F.normalize(female_feats, p=2, dim=0)
        dot_product = torch.clamp((male_norm * female_norm).sum(), min=-1.0, max=1.0)
        angle = torch.acos(dot_product).item()
        
        return {
            "separation_distance": separation,
            "gender_angle_rad": angle,
            "male_mean": male_feats.mean().item(),
            "female_mean": female_feats.mean().item()
        }
    
    def _analyze_shared_encoding(
        self,
        en_features: torch.Tensor,
        ar_features: torch.Tensor,
        en_male_mask: np.ndarray,
        ar_male_mask: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze shared vs. language-specific gender encoding."""
        # Compute gender directions
        en_gender_dir = en_features[en_male_mask].mean(dim=0) - en_features[~en_male_mask].mean(dim=0)
        ar_gender_dir = ar_features[ar_male_mask].mean(dim=0) - ar_features[~ar_male_mask].mean(dim=0)
        
        # Normalize
        en_gender_norm = F.normalize(en_gender_dir, p=2, dim=0)
        ar_gender_norm = F.normalize(ar_gender_dir, p=2, dim=0)
        
        # Compute angle between gender directions
        dot_product = torch.clamp((en_gender_norm * ar_gender_norm).sum(), min=-1.0, max=1.0)
        shared_angle = torch.acos(dot_product).item()
        
        # Features that contribute most to each direction
        en_top_features = torch.topk(en_gender_dir.abs(), k=10)[1].tolist()
        ar_top_features = torch.topk(ar_gender_dir.abs(), k=10)[1].tolist()
        
        shared = set(en_top_features) & set(ar_top_features)
        
        return {
            "gender_direction_angle_rad": shared_angle,
            "shared_top_features": list(shared),
            "num_shared_top_features": len(shared),
            "english_top_features": en_top_features,
            "arabic_top_features": ar_top_features
        }


class LanguageSpecificFeatureIdentifier:
    """
    Identifies and characterizes language-specific features that don't
    have clear counterparts in the other language.
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize identifier.
        
        Args:
            similarity_threshold: Threshold for considering features similar
        """
        self.similarity_threshold = similarity_threshold
    
    def identify_language_specific_features(
        self,
        english_features: torch.Tensor,
        arabic_features: torch.Tensor,
        english_gender_features: List[int],
        arabic_gender_features: List[int]
    ) -> Dict[str, List[int]]:
        """
        Identify features unique to each language.
        
        Args:
            english_features: English feature activations
            arabic_features: Arabic feature activations
            english_gender_features: Gender-related feature indices for English
            arabic_gender_features: Gender-related feature indices for Arabic
            
        Returns:
            Dictionary with language-specific and shared features
        """
        # Compute feature correlations between languages
        en_mean = english_features.mean(dim=0)
        ar_mean = arabic_features.mean(dim=0)
        
        en_norm = F.normalize(en_mean, p=2)
        ar_norm = F.normalize(ar_mean, p=2)
        
        # Find best matching features
        english_specific = []
        arabic_specific = []
        shared = []
        
        for en_feat in english_gender_features:
            if en_feat >= len(en_norm):
                continue
            
            # Find best match in Arabic
            best_similarity = 0
            best_ar_feat = -1
            
            for ar_feat in arabic_gender_features:
                if ar_feat >= len(ar_norm):
                    continue
                
                similarity = (en_norm[en_feat] * ar_norm[ar_feat]).item()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_ar_feat = ar_feat
            
            if best_similarity >= self.similarity_threshold and best_ar_feat >= 0:
                shared.append((en_feat, best_ar_feat))
            else:
                english_specific.append(en_feat)
        
        # Find Arabic features without English matches
        for ar_feat in arabic_gender_features:
            found = any(pair[1] == ar_feat for pair in shared)
            if not found:
                arabic_specific.append(ar_feat)
        
        return {
            "english_specific": english_specific,
            "arabic_specific": arabic_specific,
            "shared_features": shared,
            "num_english_specific": len(english_specific),
            "num_arabic_specific": len(arabic_specific),
            "num_shared": len(shared)
        }


__all__ = [
    'LanguageFeatureProfile',
    'CrossLingualFeatureAligner',
    'MorphologicalGenderAnalyzer',
    'SemanticGenderAnalyzer',
    'ContrastiveLanguageAnalyzer',
    'LanguageSpecificFeatureIdentifier',
]
