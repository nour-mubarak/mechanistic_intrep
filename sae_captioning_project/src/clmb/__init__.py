"""
CLMB: Cross-Lingual Multimodal Bias Localization & Intervention Framework
==========================================================================

A novel methodology for:
1. Localizing bias across modalities (vision ↔ language)
2. Comparing bias circuits across languages
3. Surgical intervention with quality preservation

Components:
- HBL: Hierarchical Bias Localization
- CLFA: Cross-Lingual Feature Alignment
- SBI: Surgical Bias Intervention
- CLBAS: Cross-Lingual Bias Alignment Score (new metric)
"""

from .models import MODEL_REGISTRY, get_model_config
from .hbl import HierarchicalBiasLocalizer, BiasAttributionResult
from .clfa import CrossLingualFeatureAligner, CLFAResult, FeatureAlignment
from .sbi import SurgicalBiasIntervention, SBIResult, InterventionResult
from .extractors import get_extractor, MultiModelExtractor, MODEL_ARCHITECTURES

__version__ = "0.1.0"
__author__ = "SAE Captioning Bias Research"

__all__ = [
    'MODEL_REGISTRY',
    'get_model_config',
    'HierarchicalBiasLocalizer',
    'BiasAttributionResult',
    'CrossLingualFeatureAligner',
    'CLFAResult',
    'FeatureAlignment',
    'SurgicalBiasIntervention',
    'SBIResult',
    'InterventionResult',
    'get_extractor',
    'MultiModelExtractor',
    'MODEL_ARCHITECTURES',
]
