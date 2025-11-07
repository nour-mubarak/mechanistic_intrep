"""
Evaluation Metrics for SAE and Bias Analysis
=============================================

Metrics for evaluating SAE quality and measuring bias in captions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.metrics import mutual_info_score
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SAE Quality Metrics
# ============================================================================

def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        original: Original activations
        reconstructed: SAE reconstructed activations
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # MSE
        mse = torch.nn.functional.mse_loss(reconstructed, original).item()
        
        # Normalized MSE (relative to variance)
        var = original.var().item()
        nmse = mse / var if var > 0 else float('inf')
        
        # Cosine similarity
        flat_orig = original.view(-1)
        flat_recon = reconstructed.view(-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_orig.unsqueeze(0), flat_recon.unsqueeze(0)
        ).item()
        
        # Explained variance
        explained_var = 1 - nmse
        
        return {
            "mse": mse,
            "nmse": nmse,
            "cosine_similarity": cos_sim,
            "explained_variance": max(0, explained_var),
        }


def compute_sparsity_metrics(features: torch.Tensor) -> Dict[str, float]:
    """
    Compute sparsity metrics for SAE features.
    
    Args:
        features: SAE feature activations (batch, n_features)
        
    Returns:
        Dictionary of sparsity metrics
    """
    with torch.no_grad():
        # L0 sparsity (average number of active features)
        l0 = (features > 0).float().sum(dim=-1).mean().item()
        
        # L0 ratio (fraction of active features)
        l0_ratio = l0 / features.shape[-1]
        
        # Gini coefficient (measure of activation inequality)
        sorted_acts = torch.sort(features.abs().mean(dim=0))[0]
        n = len(sorted_acts)
        cumsum = torch.cumsum(sorted_acts, dim=0)
        gini = (2 * torch.sum((torch.arange(1, n+1).float() * sorted_acts)) / 
                (n * sorted_acts.sum()) - (n + 1) / n).item()
        
        # Dead features (never activate)
        dead_ratio = (features.max(dim=0)[0] == 0).float().mean().item()
        
        return {
            "l0_sparsity": l0,
            "l0_ratio": l0_ratio,
            "gini_coefficient": gini,
            "dead_feature_ratio": dead_ratio,
        }


def compute_feature_coverage(
    features: torch.Tensor,
    percentile: float = 90
) -> Dict[str, float]:
    """
    Analyze how features cover the activation space.
    
    Args:
        features: SAE feature activations
        percentile: Percentile for coverage computation
        
    Returns:
        Coverage metrics
    """
    with torch.no_grad():
        # Per-feature activation frequency
        activation_freq = (features > 0).float().mean(dim=0)
        
        # Features needed for X% of total activation
        total_activation = features.sum()
        sorted_contrib = torch.sort(features.sum(dim=0), descending=True)[0]
        cumsum = torch.cumsum(sorted_contrib, dim=0)
        threshold = total_activation * (percentile / 100)
        features_for_coverage = (cumsum < threshold).sum().item() + 1
        
        return {
            "mean_activation_frequency": activation_freq.mean().item(),
            "median_activation_frequency": activation_freq.median().item(),
            f"features_for_{percentile}pct_coverage": features_for_coverage,
            "coverage_ratio": features_for_coverage / features.shape[-1],
        }


# ============================================================================
# Gender Bias Metrics
# ============================================================================

# Gender word lists for Arabic and English
ENGLISH_MALE_WORDS = {
    'he', 'him', 'his', 'man', 'men', 'boy', 'boys', 'male', 'father',
    'son', 'brother', 'husband', 'grandfather', 'uncle', 'nephew',
    'gentleman', 'gentlemen', 'guy', 'guys', 'mr', 'sir'
}

ENGLISH_FEMALE_WORDS = {
    'she', 'her', 'hers', 'woman', 'women', 'girl', 'girls', 'female',
    'mother', 'daughter', 'sister', 'wife', 'grandmother', 'aunt', 'niece',
    'lady', 'ladies', 'miss', 'mrs', 'ms', 'madam'
}

ARABIC_MALE_WORDS = {
    'هو', 'رجل', 'رجال', 'ولد', 'أولاد', 'ذكر', 'أب', 'ابن', 'أخ',
    'زوج', 'جد', 'عم', 'خال', 'ابن أخ', 'سيد', 'شاب', 'فتى'
}

ARABIC_FEMALE_WORDS = {
    'هي', 'امرأة', 'نساء', 'بنت', 'بنات', 'أنثى', 'أم', 'ابنة', 'أخت',
    'زوجة', 'جدة', 'عمة', 'خالة', 'ابنة أخ', 'سيدة', 'فتاة', 'آنسة'
}


def compute_caption_gender_bias(
    captions: List[str],
    ground_truth_genders: List[str],
    language: str = "english"
) -> Dict[str, Any]:
    """
    Compute gender bias metrics for captions.
    
    Args:
        captions: List of generated captions
        ground_truth_genders: Ground truth gender labels
        language: "english" or "arabic"
        
    Returns:
        Dictionary of bias metrics
    """
    if language == "english":
        male_words = ENGLISH_MALE_WORDS
        female_words = ENGLISH_FEMALE_WORDS
    else:
        male_words = ARABIC_MALE_WORDS
        female_words = ARABIC_FEMALE_WORDS
    
    results = {
        "correct_gender": 0,
        "incorrect_gender": 0,
        "neutral": 0,
        "male_for_female": 0,  # Misgendering female as male
        "female_for_male": 0,  # Misgendering male as female
    }
    
    predicted_genders = []
    
    for caption, gt_gender in zip(captions, ground_truth_genders):
        # Tokenize caption
        if language == "english":
            words = set(caption.lower().split())
        else:
            words = set(caption.split())
        
        # Count gender words
        male_count = len(words & male_words)
        female_count = len(words & female_words)
        
        # Determine predicted gender
        if male_count > female_count:
            pred_gender = "male"
        elif female_count > male_count:
            pred_gender = "female"
        else:
            pred_gender = "neutral"
        
        predicted_genders.append(pred_gender)
        
        # Update counts
        if pred_gender == "neutral":
            results["neutral"] += 1
        elif pred_gender == gt_gender:
            results["correct_gender"] += 1
        else:
            results["incorrect_gender"] += 1
            if pred_gender == "male" and gt_gender == "female":
                results["male_for_female"] += 1
            elif pred_gender == "female" and gt_gender == "male":
                results["female_for_male"] += 1
    
    # Compute rates
    total = len(captions)
    non_neutral = total - results["neutral"]
    
    results["accuracy"] = results["correct_gender"] / non_neutral if non_neutral > 0 else 0
    results["neutral_rate"] = results["neutral"] / total
    results["male_bias_rate"] = results["male_for_female"] / total
    results["female_bias_rate"] = results["female_for_male"] / total
    
    # Compute bias score (-1 to 1, negative = female bias, positive = male bias)
    if results["male_for_female"] + results["female_for_male"] > 0:
        results["bias_score"] = (
            (results["male_for_female"] - results["female_for_male"]) /
            (results["male_for_female"] + results["female_for_male"])
        )
    else:
        results["bias_score"] = 0.0
    
    results["predicted_genders"] = predicted_genders
    
    return results


def compute_cross_lingual_bias_comparison(
    english_captions: List[str],
    arabic_captions: List[str],
    ground_truth_genders: List[str]
) -> Dict[str, Any]:
    """
    Compare gender bias between English and Arabic captions.
    
    Args:
        english_captions: Captions from English prompts
        arabic_captions: Captions from Arabic prompts
        ground_truth_genders: Ground truth labels
        
    Returns:
        Comparison metrics
    """
    en_bias = compute_caption_gender_bias(
        english_captions, ground_truth_genders, "english"
    )
    ar_bias = compute_caption_gender_bias(
        arabic_captions, ground_truth_genders, "arabic"
    )
    
    # Compute agreement between languages
    agreement = sum(
        1 for en, ar in zip(en_bias["predicted_genders"], ar_bias["predicted_genders"])
        if en == ar
    ) / len(english_captions)
    
    return {
        "english_bias": en_bias,
        "arabic_bias": ar_bias,
        "language_agreement": agreement,
        "bias_difference": en_bias["bias_score"] - ar_bias["bias_score"],
        "accuracy_difference": en_bias["accuracy"] - ar_bias["accuracy"],
    }


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Input data
        statistic: Statistic function to compute
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci)
    """
    point_estimate = statistic(data)
    
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(statistic(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_samples, alpha * 100)
    upper = np.percentile(bootstrap_samples, (1 - alpha) * 100)
    
    return point_estimate, lower, upper


# ============================================================================
# Steering Effectiveness Metrics
# ============================================================================

def compute_steering_effectiveness(
    original_captions: List[str],
    steered_captions: List[str],
    ground_truth_genders: List[str],
    language: str = "english"
) -> Dict[str, Any]:
    """
    Measure effectiveness of feature steering on bias.
    
    Args:
        original_captions: Captions without steering
        steered_captions: Captions with steering applied
        ground_truth_genders: Ground truth labels
        language: Language of captions
        
    Returns:
        Steering effectiveness metrics
    """
    orig_bias = compute_caption_gender_bias(
        original_captions, ground_truth_genders, language
    )
    steered_bias = compute_caption_gender_bias(
        steered_captions, ground_truth_genders, language
    )
    
    return {
        "original_accuracy": orig_bias["accuracy"],
        "steered_accuracy": steered_bias["accuracy"],
        "accuracy_change": steered_bias["accuracy"] - orig_bias["accuracy"],
        "original_bias_score": orig_bias["bias_score"],
        "steered_bias_score": steered_bias["bias_score"],
        "bias_reduction": abs(orig_bias["bias_score"]) - abs(steered_bias["bias_score"]),
        "neutral_rate_change": steered_bias["neutral_rate"] - orig_bias["neutral_rate"],
    }
