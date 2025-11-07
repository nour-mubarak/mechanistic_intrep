"""
Analysis module for SAE feature and bias analysis.
"""

from .features import (
    FeatureStats,
    CrossLingualComparison,
    GenderFeatureAnalyzer,
    FeatureClustering,
    compute_embedding_space_analysis,
    serialize_analysis_results,
)

from .metrics import (
    compute_reconstruction_metrics,
    compute_sparsity_metrics,
    compute_feature_coverage,
    compute_caption_gender_bias,
    compute_cross_lingual_bias_comparison,
    compute_cohens_d,
    compute_bootstrap_ci,
    compute_steering_effectiveness,
)

__all__ = [
    "FeatureStats",
    "CrossLingualComparison",
    "GenderFeatureAnalyzer",
    "FeatureClustering",
    "compute_embedding_space_analysis",
    "serialize_analysis_results",
    "compute_reconstruction_metrics",
    "compute_sparsity_metrics",
    "compute_feature_coverage",
    "compute_caption_gender_bias",
    "compute_cross_lingual_bias_comparison",
    "compute_cohens_d",
    "compute_bootstrap_ci",
    "compute_steering_effectiveness",
]
