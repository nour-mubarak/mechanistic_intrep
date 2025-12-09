"""
Mechanistic Interpretability Module
====================================

Comprehensive mechanistic interpretability toolkit combining:
- ViT-Prisma tools for activation analysis
- Multilingual LLM features for cross-lingual analysis
- Core transformer interpretability techniques
- Gender bias detection and steering
"""

from .prisma_integration import (
    HookPoint,
    ActivationCache,
    FactoredMatrix,
    LogitLens,
    InteractionPatternAnalyzer,
    TransformerProbeAnalyzer,
)

from .multilingual_features import (
    LanguageFeatureProfile,
    CrossLingualFeatureAligner,
    MorphologicalGenderAnalyzer,
    SemanticGenderAnalyzer,
    ContrastiveLanguageAnalyzer,
    LanguageSpecificFeatureIdentifier,
)

__all__ = [
    # ViT-Prisma tools
    'HookPoint',
    'ActivationCache',
    'FactoredMatrix',
    'LogitLens',
    'InteractionPatternAnalyzer',
    'TransformerProbeAnalyzer',
    # Multilingual tools
    'LanguageFeatureProfile',
    'CrossLingualFeatureAligner',
    'MorphologicalGenderAnalyzer',
    'SemanticGenderAnalyzer',
    'ContrastiveLanguageAnalyzer',
    'LanguageSpecificFeatureIdentifier',
]
