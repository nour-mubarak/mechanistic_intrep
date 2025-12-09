#!/usr/bin/env python3
"""
Integration Validation Script
==============================

Validates that all mechanistic interpretability tools are properly integrated
and can be imported and used without errors.

Usage:
    python validate_integration.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Verify all modules can be imported."""
    print("="*60)
    print("CHECKING IMPORTS")
    print("="*60)
    
    checks = []
    
    try:
        from src.mechanistic import (
            HookPoint,
            ActivationCache,
            FactoredMatrix,
            LogitLens,
            InteractionPatternAnalyzer,
            TransformerProbeAnalyzer,
        )
        print("✓ ViT-Prisma tools imported successfully")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Failed to import ViT-Prisma tools: {e}")
        checks.append(False)
    
    try:
        from src.mechanistic import (
            LanguageFeatureProfile,
            CrossLingualFeatureAligner,
            MorphologicalGenderAnalyzer,
            SemanticGenderAnalyzer,
            ContrastiveLanguageAnalyzer,
            LanguageSpecificFeatureIdentifier,
        )
        print("✓ Multilingual features imported successfully")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Failed to import multilingual features: {e}")
        checks.append(False)
    
    return all(checks)


def test_factored_matrix():
    """Test FactoredMatrix functionality."""
    print("\n" + "="*60)
    print("TESTING FACTORED MATRIX")
    print("="*60)
    
    try:
        from src.mechanistic import FactoredMatrix
        
        # Create test tensor
        test_activation = torch.randn(100, 256)
        fm = FactoredMatrix(test_activation, name="test")
        
        # Test SVD
        U, S, V = fm.compute_svd()
        print(f"✓ SVD computation: U={U.shape}, S={S.shape}, V={V.shape}")
        
        # Test rank
        rank = fm.compute_rank()
        print(f"✓ Effective rank (threshold=0.95): {rank}")
        
        # Test information content
        entropy = fm.compute_information_content()
        print(f"✓ Information content: {entropy:.3f} bits")
        
        # Test PCA
        components, variance = fm.compute_pca(n_components=10)
        print(f"✓ PCA: components shape={components.shape}, variance sum={variance.sum():.3f}")
        
        return True
    except Exception as e:
        print(f"✗ FactoredMatrix test failed: {e}")
        return False


def test_activation_cache():
    """Test ActivationCache functionality."""
    print("\n" + "="*60)
    print("TESTING ACTIVATION CACHE")
    print("="*60)
    
    try:
        from src.mechanistic import ActivationCache, HookPoint
        import torch.nn as nn
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        
        # Create hook points
        hook_points = [
            HookPoint("layer_0", 0, model[0]),
            HookPoint("layer_1", 1, model[1]),
        ]
        
        # Create cache
        cache = ActivationCache(model, hook_points)
        
        # Run forward pass
        x = torch.randn(5, 10)
        _ = model(x)
        
        # Check cached activations
        activations = cache.get_all_activations()
        print(f"✓ Cache captured {len(activations)} activations")
        
        # Verify activation shapes
        for name, act in activations.items():
            print(f"  - {name}: {act.shape}")
        
        # Clean up
        cache.remove_hooks()
        print("✓ Hooks removed successfully")
        
        return True
    except Exception as e:
        print(f"✗ ActivationCache test failed: {e}")
        return False


def test_feature_alignment():
    """Test CrossLingualFeatureAligner."""
    print("\n" + "="*60)
    print("TESTING FEATURE ALIGNMENT")
    print("="*60)
    
    try:
        from src.mechanistic import CrossLingualFeatureAligner
        
        # Create test features
        en_features = torch.randn(100, 200)
        ar_features = torch.randn(100, 200)
        
        aligner = CrossLingualFeatureAligner()
        alignment = aligner.align_features(en_features, ar_features, similarity_threshold=0.7)
        
        stats = aligner.compute_alignment_statistics(alignment)
        
        print(f"✓ Alignment completed")
        print(f"  - Aligned pairs: {stats['num_aligned']}")
        print(f"  - Mean similarity: {stats['mean_similarity']:.3f}")
        print(f"  - Alignment ratio: {stats['alignment_ratio']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Feature alignment test failed: {e}")
        return False


def test_morphological_analyzer():
    """Test MorphologicalGenderAnalyzer."""
    print("\n" + "="*60)
    print("TESTING MORPHOLOGICAL ANALYZER")
    print("="*60)
    
    try:
        from src.mechanistic import MorphologicalGenderAnalyzer
        
        analyzer = MorphologicalGenderAnalyzer()
        
        # Test captions
        captions = [
            "امرأة تحت الشمس",  # woman with feminine suffix
            "رجل في المنزل",     # man (no feminine suffix)
        ]
        
        extraction = analyzer.extract_morphological_gender(captions)
        
        print(f"✓ Morphological extraction completed")
        print(f"  - Feminine words: {extraction['total_feminine']}")
        print(f"  - Masculine words: {extraction['total_masculine']}")
        
        return True
    except Exception as e:
        print(f"✗ Morphological analyzer test failed: {e}")
        return False


def test_contrastive_analyzer():
    """Test ContrastiveLanguageAnalyzer."""
    print("\n" + "="*60)
    print("TESTING CONTRASTIVE ANALYZER")
    print("="*60)
    
    try:
        from src.mechanistic import ContrastiveLanguageAnalyzer
        
        # Create test data
        en_features = torch.randn(100, 256)
        ar_features = torch.randn(100, 256)
        en_labels = ["male" if i % 2 == 0 else "female" for i in range(100)]
        ar_labels = ["male" if i % 2 == 0 else "female" for i in range(100)]
        
        analyzer = ContrastiveLanguageAnalyzer()
        results = analyzer.compare_language_feature_spaces(
            en_features, ar_features,
            en_labels, ar_labels
        )
        
        print(f"✓ Contrastive analysis completed")
        print(f"  - English gender separation: {results['english_gender_separation']['separation_distance']:.3f}")
        print(f"  - Arabic gender separation: {results['arabic_gender_separation']['separation_distance']:.3f}")
        print(f"  - Gender direction angle: {results['shared_encoding']['gender_direction_angle_rad']:.3f} rad")
        
        return True
    except Exception as e:
        print(f"✗ Contrastive analyzer test failed: {e}")
        return False


def check_pipeline_integration():
    """Verify pipeline integration."""
    print("\n" + "="*60)
    print("CHECKING PIPELINE INTEGRATION")
    print("="*60)
    
    checks = []
    
    # Check if stage 7 is in pipeline
    pipeline_file = Path("scripts/run_full_pipeline.py")
    if pipeline_file.exists():
        content = pipeline_file.read_text()
        if "07_integrated_mechanistic_analysis.py" in content:
            print("✓ Stage 7 integrated into pipeline")
            checks.append(True)
        else:
            print("✗ Stage 7 not found in pipeline")
            checks.append(False)
    else:
        print("✗ Pipeline file not found")
        checks.append(False)
    
    # Check if analysis script exists
    analysis_script = Path("scripts/07_integrated_mechanistic_analysis.py")
    if analysis_script.exists():
        print("✓ Integrated analysis script exists")
        checks.append(True)
    else:
        print("✗ Integrated analysis script not found")
        checks.append(False)
    
    # Check documentation
    docs = [
        "MECHANISTIC_INTERPRETABILITY_GUIDE.md",
        "INTEGRATION_SUMMARY.md"
    ]
    
    for doc in docs:
        if Path(doc).exists():
            print(f"✓ {doc} exists")
            checks.append(True)
        else:
            print(f"✗ {doc} not found")
            checks.append(False)
    
    return all(checks)


def main():
    """Run all validation checks."""
    print("\n")
    print("#" * 60)
    print("# MECHANISTIC INTERPRETABILITY INTEGRATION VALIDATION")
    print("#" * 60)
    
    results = {
        "Imports": check_imports(),
        "Factored Matrix": test_factored_matrix(),
        "Activation Cache": test_activation_cache(),
        "Feature Alignment": test_feature_alignment(),
        "Morphological Analyzer": test_morphological_analyzer(),
        "Contrastive Analyzer": test_contrastive_analyzer(),
        "Pipeline Integration": check_pipeline_integration(),
    }
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED!")
        print("="*60)
        print("\nIntegration Status: COMPLETE ✓")
        print("\nYou can now use:")
        print("  - python scripts/07_integrated_mechanistic_analysis.py")
        print("  - python scripts/run_full_pipeline.py (includes stage 7)")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
