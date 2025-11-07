"""
Feature Analysis for Cross-Lingual Gender Bias
===============================================

Tools for analyzing SAE features to understand gender encoding
differences between Arabic and English in vision-language models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for a single SAE feature."""
    feature_idx: int
    mean_activation: float
    std_activation: float
    sparsity: float  # Fraction of samples where feature fires
    max_activation: float
    
    # Gender-specific stats
    male_mean: Optional[float] = None
    female_mean: Optional[float] = None
    gender_diff: Optional[float] = None  # male_mean - female_mean
    gender_p_value: Optional[float] = None
    gender_effect_size: Optional[float] = None  # Cohen's d


@dataclass
class CrossLingualComparison:
    """Comparison of features between languages."""
    shared_gender_features: List[int]
    english_specific_features: List[int]
    arabic_specific_features: List[int]
    
    overlap_ratio: float
    correlation: float
    
    # Per-feature comparison
    feature_correlations: Dict[int, float] = field(default_factory=dict)


class GenderFeatureAnalyzer:
    """
    Analyzes SAE features for gender-related patterns.
    
    Identifies which features correlate with gender in captions
    and compares these patterns across languages.
    """
    
    def __init__(
        self,
        sae: torch.nn.Module,
        device: str = "cuda",
        significance_level: float = 0.05,
        effect_size_threshold: float = 0.3
    ):
        """
        Initialize analyzer.
        
        Args:
            sae: Trained SAE model
            device: Device to use
            significance_level: P-value threshold for significance
            effect_size_threshold: Minimum Cohen's d for relevance
        """
        self.sae = sae.to(device)
        self.sae.eval()
        self.device = device
        self.significance_level = significance_level
        self.effect_size_threshold = effect_size_threshold
    
    @torch.no_grad()
    def compute_feature_activations(
        self,
        activations: torch.Tensor,
        aggregate: str = "mean"
    ) -> torch.Tensor:
        """
        Compute SAE feature activations for input activations.
        
        Args:
            activations: Input activations (batch, seq, hidden) or (batch, hidden)
            aggregate: How to aggregate over sequence ("mean", "max", "last")
            
        Returns:
            Feature activations (batch, n_features)
        """
        activations = activations.to(self.device)
        
        # Handle sequence dimension
        if len(activations.shape) == 3:
            batch, seq, hidden = activations.shape
            
            # Reshape for SAE
            flat = activations.view(-1, hidden)
            features = self.sae.encode(flat)
            features = features.view(batch, seq, -1)
            
            # Aggregate over sequence
            if aggregate == "mean":
                features = features.mean(dim=1)
            elif aggregate == "max":
                features = features.max(dim=1)[0]
            elif aggregate == "last":
                features = features[:, -1, :]
            else:
                raise ValueError(f"Unknown aggregate: {aggregate}")
        else:
            features = self.sae.encode(activations)
        
        return features.cpu()
    
    def compute_feature_stats(
        self,
        feature_activations: torch.Tensor,
        gender_labels: List[str]
    ) -> List[FeatureStats]:
        """
        Compute statistics for each feature.
        
        Args:
            feature_activations: (n_samples, n_features)
            gender_labels: Gender label per sample
            
        Returns:
            List of FeatureStats for each feature
        """
        n_features = feature_activations.shape[1]
        stats_list = []
        
        # Convert labels to masks
        male_mask = torch.tensor([g == "male" for g in gender_labels])
        female_mask = torch.tensor([g == "female" for g in gender_labels])
        
        for feat_idx in tqdm(range(n_features), desc="Computing feature stats"):
            feat_acts = feature_activations[:, feat_idx]
            
            # Basic stats
            basic = FeatureStats(
                feature_idx=feat_idx,
                mean_activation=feat_acts.mean().item(),
                std_activation=feat_acts.std().item(),
                sparsity=(feat_acts > 0).float().mean().item(),
                max_activation=feat_acts.max().item()
            )
            
            # Gender-specific stats (if labels available)
            if male_mask.any() and female_mask.any():
                male_acts = feat_acts[male_mask].numpy()
                female_acts = feat_acts[female_mask].numpy()
                
                basic.male_mean = float(np.mean(male_acts))
                basic.female_mean = float(np.mean(female_acts))
                basic.gender_diff = basic.male_mean - basic.female_mean
                
                # Statistical test
                if len(male_acts) > 1 and len(female_acts) > 1:
                    try:
                        _, p_value = stats.mannwhitneyu(
                            male_acts, female_acts, alternative='two-sided'
                        )
                        basic.gender_p_value = float(p_value)
                        
                        # Cohen's d
                        pooled_std = np.sqrt(
                            (np.var(male_acts) + np.var(female_acts)) / 2
                        )
                        if pooled_std > 0:
                            basic.gender_effect_size = float(
                                basic.gender_diff / pooled_std
                            )
                    except Exception as e:
                        logger.warning(f"Stats error for feature {feat_idx}: {e}")
            
            stats_list.append(basic)
        
        return stats_list
    
    def identify_gender_features(
        self,
        feature_stats: List[FeatureStats],
        top_k: int = 100
    ) -> Dict[str, List[int]]:
        """
        Identify features associated with gender.
        
        Args:
            feature_stats: List of feature statistics
            top_k: Number of top features to return
            
        Returns:
            Dictionary with male_associated, female_associated, and neutral features
        """
        significant_features = []
        
        for fs in feature_stats:
            if (fs.gender_p_value is not None and 
                fs.gender_p_value < self.significance_level and
                fs.gender_effect_size is not None and
                abs(fs.gender_effect_size) > self.effect_size_threshold):
                significant_features.append(fs)
        
        # Sort by effect size
        male_features = sorted(
            [f for f in significant_features if f.gender_diff and f.gender_diff > 0],
            key=lambda x: x.gender_effect_size or 0,
            reverse=True
        )[:top_k]
        
        female_features = sorted(
            [f for f in significant_features if f.gender_diff and f.gender_diff < 0],
            key=lambda x: abs(x.gender_effect_size or 0),
            reverse=True
        )[:top_k]
        
        return {
            "male_associated": [f.feature_idx for f in male_features],
            "female_associated": [f.feature_idx for f in female_features],
            "male_stats": [f for f in male_features],
            "female_stats": [f for f in female_features],
        }
    
    def compare_languages(
        self,
        english_features: torch.Tensor,
        arabic_features: torch.Tensor,
        english_labels: List[str],
        arabic_labels: List[str]
    ) -> CrossLingualComparison:
        """
        Compare gender-related features between English and Arabic.
        
        Args:
            english_features: Feature activations for English prompts
            arabic_features: Feature activations for Arabic prompts
            english_labels: Gender labels for English samples
            arabic_labels: Gender labels for Arabic samples
            
        Returns:
            CrossLingualComparison with detailed comparison
        """
        # Compute stats for each language
        en_stats = self.compute_feature_stats(english_features, english_labels)
        ar_stats = self.compute_feature_stats(arabic_features, arabic_labels)
        
        # Identify gender features for each
        en_gender = self.identify_gender_features(en_stats)
        ar_gender = self.identify_gender_features(ar_stats)
        
        # Combine male and female features for comparison
        en_all = set(en_gender["male_associated"] + en_gender["female_associated"])
        ar_all = set(ar_gender["male_associated"] + ar_gender["female_associated"])
        
        # Compute overlap
        shared = en_all & ar_all
        en_specific = en_all - ar_all
        ar_specific = ar_all - en_all
        
        overlap_ratio = len(shared) / len(en_all | ar_all) if (en_all | ar_all) else 0
        
        # Compute correlation of feature activations
        en_mean = english_features.mean(dim=0)
        ar_mean = arabic_features.mean(dim=0)
        correlation = float(F.cosine_similarity(
            en_mean.unsqueeze(0), ar_mean.unsqueeze(0)
        ).item())
        
        # Per-feature correlations for shared features
        feature_corrs = {}
        for feat_idx in shared:
            en_act = english_features[:, feat_idx].numpy()
            ar_act = arabic_features[:, feat_idx].numpy()
            corr, _ = stats.spearmanr(en_act, ar_act)
            feature_corrs[feat_idx] = float(corr) if not np.isnan(corr) else 0.0
        
        return CrossLingualComparison(
            shared_gender_features=list(shared),
            english_specific_features=list(en_specific),
            arabic_specific_features=list(ar_specific),
            overlap_ratio=overlap_ratio,
            correlation=correlation,
            feature_correlations=feature_corrs
        )
    
    def compute_layer_divergence(
        self,
        english_activations: Dict[int, torch.Tensor],
        arabic_activations: Dict[int, torch.Tensor],
        english_labels: List[str],
        arabic_labels: List[str]
    ) -> Dict[int, float]:
        """
        Compute how gender feature overlap changes across layers.
        
        Args:
            english_activations: Dict mapping layer -> activations
            arabic_activations: Dict mapping layer -> activations
            english_labels: Gender labels
            arabic_labels: Gender labels
            
        Returns:
            Dict mapping layer -> overlap ratio
        """
        layer_overlaps = {}
        
        for layer_idx in tqdm(english_activations.keys(), desc="Analyzing layers"):
            en_acts = english_activations[layer_idx]
            ar_acts = arabic_activations[layer_idx]
            
            en_features = self.compute_feature_activations(en_acts)
            ar_features = self.compute_feature_activations(ar_acts)
            
            comparison = self.compare_languages(
                en_features, ar_features,
                english_labels, arabic_labels
            )
            
            layer_overlaps[layer_idx] = comparison.overlap_ratio
        
        return layer_overlaps


class FeatureClustering:
    """
    Cluster SAE features to identify semantic groups.
    """
    
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.pca = None
    
    def fit(self, feature_directions: torch.Tensor) -> np.ndarray:
        """
        Cluster feature directions.
        
        Args:
            feature_directions: (n_features, d_model) feature direction vectors
            
        Returns:
            Cluster labels for each feature
        """
        directions = feature_directions.numpy()
        
        # Reduce dimensionality for clustering
        self.pca = PCA(n_components=min(50, directions.shape[1]))
        reduced = self.pca.fit_transform(directions)
        
        # Cluster
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = self.kmeans.fit_predict(reduced)
        
        return labels
    
    def get_cluster_summary(
        self,
        labels: np.ndarray,
        feature_stats: List[FeatureStats]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Summarize each cluster.
        
        Args:
            labels: Cluster labels
            feature_stats: Statistics for each feature
            
        Returns:
            Summary for each cluster
        """
        summaries = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_features = [
                fs for fs, in_cluster in zip(feature_stats, cluster_mask)
                if in_cluster
            ]
            
            if not cluster_features:
                continue
            
            # Compute cluster statistics
            effect_sizes = [
                fs.gender_effect_size for fs in cluster_features
                if fs.gender_effect_size is not None
            ]
            
            summaries[cluster_id] = {
                "n_features": len(cluster_features),
                "mean_sparsity": np.mean([fs.sparsity for fs in cluster_features]),
                "mean_effect_size": np.mean(effect_sizes) if effect_sizes else 0,
                "gender_bias_direction": (
                    "male" if np.mean(effect_sizes) > 0 else "female"
                ) if effect_sizes else "neutral",
                "feature_indices": [fs.feature_idx for fs in cluster_features],
            }
        
        return summaries


def compute_embedding_space_analysis(
    english_features: torch.Tensor,
    arabic_features: torch.Tensor,
    labels: List[str],
    method: str = "tsne"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D embeddings for visualization.
    
    Args:
        english_features: English feature activations
        arabic_features: Arabic feature activations
        labels: Gender labels
        method: "tsne" or "pca"
        
    Returns:
        Tuple of (combined_embeddings, language_labels)
    """
    # Combine features
    combined = torch.cat([english_features, arabic_features], dim=0).numpy()
    
    # Create language labels
    n_en = len(english_features)
    lang_labels = ["english"] * n_en + ["arabic"] * len(arabic_features)
    
    # Compute embeddings
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    
    embeddings = reducer.fit_transform(combined)
    
    return embeddings, lang_labels


def serialize_analysis_results(
    feature_stats: List[FeatureStats],
    comparison: CrossLingualComparison,
    layer_divergence: Dict[int, float]
) -> Dict[str, Any]:
    """
    Serialize analysis results to JSON-compatible format.
    """
    return {
        "feature_stats": [
            {
                "feature_idx": fs.feature_idx,
                "mean_activation": fs.mean_activation,
                "std_activation": fs.std_activation,
                "sparsity": fs.sparsity,
                "max_activation": fs.max_activation,
                "male_mean": fs.male_mean,
                "female_mean": fs.female_mean,
                "gender_diff": fs.gender_diff,
                "gender_p_value": fs.gender_p_value,
                "gender_effect_size": fs.gender_effect_size,
            }
            for fs in feature_stats
        ],
        "cross_lingual_comparison": {
            "shared_gender_features": comparison.shared_gender_features,
            "english_specific_features": comparison.english_specific_features,
            "arabic_specific_features": comparison.arabic_specific_features,
            "overlap_ratio": comparison.overlap_ratio,
            "correlation": comparison.correlation,
            "feature_correlations": comparison.feature_correlations,
        },
        "layer_divergence": layer_divergence,
    }
