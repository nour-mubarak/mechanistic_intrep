#!/usr/bin/env python3
"""
Integrated Mechanistic Analysis Pipeline
==========================================

Combines ViT-Prisma tools, multilingual-llm-features, and core mechanistic
interpretability techniques for comprehensive gender bias analysis.

Usage:
    python scripts/07_integrated_mechanistic_analysis.py --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import json
from datetime import datetime
import wandb
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sae import SparseAutoencoder, SAEConfig
from src.mechanistic import (
    ActivationCache,
    HookPoint,
    LogitLens,
    FactoredMatrix,
    InteractionPatternAnalyzer,
    TransformerProbeAnalyzer,
    CrossLingualFeatureAligner,
    MorphologicalGenderAnalyzer,
    SemanticGenderAnalyzer,
    ContrastiveLanguageAnalyzer,
    LanguageSpecificFeatureIdentifier,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_activations_and_features(checkpoint_dir: Path) -> Dict:
    """Load precomputed activations and SAE features."""
    data = {}
    
    for lang in ['english', 'arabic']:
        try:
            path = checkpoint_dir / f'activations_{lang}.pt'
            data[lang] = torch.load(path, weights_only=False)
        except FileNotFoundError:
            logger.warning(f"Activations not found for {lang}")
    
    return data


def run_prisma_analysis(
    english_features: torch.Tensor,
    arabic_features: torch.Tensor,
    sae: SparseAutoencoder,
    device: str = "cuda"
) -> Dict:
    """
    Run ViT-Prisma-based mechanistic analysis.
    
    Includes:
    - Activation caching and factored matrix analysis
    - Logit lens for layer prediction emergence
    - Interaction pattern analysis
    """
    logger.info("Running ViT-Prisma Mechanistic Analysis...")
    results = {}
    
    # 1. Factored Matrix Analysis
    logger.info("  1. Analyzing factored matrices...")
    en_fm = FactoredMatrix(english_features, name="english")
    ar_fm = FactoredMatrix(arabic_features, name="arabic")
    
    en_rank = en_fm.compute_rank(threshold=0.95)
    ar_rank = ar_fm.compute_rank(threshold=0.95)
    
    results["factored_matrix"] = {
        "english_rank": en_rank,
        "arabic_rank": ar_rank,
        "english_information_content": en_fm.compute_information_content(),
        "arabic_information_content": ar_fm.compute_information_content(),
    }
    
    logger.info(f"    English effective rank: {en_rank}")
    logger.info(f"    Arabic effective rank: {ar_rank}")
    
    # 2. Interaction Pattern Analysis
    logger.info("  2. Analyzing interaction patterns...")
    interaction_analyzer = InteractionPatternAnalyzer(sae, device)
    interaction_results = interaction_analyzer.analyze_feature_interactions(
        english_features, arabic_features
    )
    
    results["interaction_patterns"] = {
        "num_pairwise_interactions": len(interaction_results["pairwise_correlations"]),
        "feature_importance_keys": list(interaction_results["feature_importance"].keys())[:20],
        "divergence_points": interaction_results["divergence_points"],
    }
    
    logger.info(f"    Found {len(interaction_results['divergence_points'])} divergence points")
    
    return results


def run_multilingual_analysis(
    english_features: torch.Tensor,
    arabic_features: torch.Tensor,
    english_labels: List[str],
    arabic_labels: List[str],
    english_captions: List[str],
    arabic_captions: List[str],
    config: dict
) -> Dict:
    """
    Run multilingual-llm-features analysis.
    
    Includes:
    - Cross-lingual feature alignment
    - Morphological gender analysis for Arabic
    - Semantic gender analysis
    - Contrastive language analysis
    - Language-specific feature identification
    """
    logger.info("Running Multilingual Features Analysis...")
    results = {}
    
    # 1. Cross-Lingual Feature Alignment
    logger.info("  1. Aligning features across languages...")
    aligner = CrossLingualFeatureAligner()
    alignment = aligner.align_features(
        english_features, arabic_features,
        similarity_threshold=0.7
    )
    alignment_stats = aligner.compute_alignment_statistics(alignment)
    
    results["feature_alignment"] = alignment_stats
    logger.info(f"    Aligned {alignment_stats['num_aligned']} feature pairs")
    logger.info(f"    Mean similarity: {alignment_stats['mean_similarity']:.3f}")
    
    # 2. Morphological Gender Analysis (Arabic)
    logger.info("  2. Analyzing Arabic morphological gender...")
    morph_analyzer = MorphologicalGenderAnalyzer()
    morph_extraction = morph_analyzer.extract_morphological_gender(arabic_captions)
    
    results["morphological_analysis"] = {
        "feminine_words_count": morph_extraction["total_feminine"],
        "masculine_words_count": morph_extraction["total_masculine"],
    }
    
    logger.info(f"    Feminine words: {morph_extraction['total_feminine']}")
    logger.info(f"    Masculine words: {morph_extraction['total_masculine']}")
    
    # 3. Semantic Gender Analysis
    logger.info("  3. Analyzing semantic gender associations...")
    semantic_analyzer = SemanticGenderAnalyzer()
    
    en_semantic = semantic_analyzer.extract_semantic_gender(english_captions)
    ar_semantic = semantic_analyzer.extract_semantic_gender(arabic_captions)
    
    results["semantic_analysis"] = {
        "english_semantic_ratio": en_semantic["semantic_gender_ratio"],
        "arabic_semantic_ratio": ar_semantic["semantic_gender_ratio"],
    }
    
    # 4. Contrastive Language Analysis
    logger.info("  4. Performing contrastive analysis...")
    contrastive_analyzer = ContrastiveLanguageAnalyzer()
    contrastive_results = contrastive_analyzer.compare_language_feature_spaces(
        english_features, arabic_features,
        english_labels, arabic_labels
    )
    
    results["contrastive_analysis"] = {
        "english_separation": contrastive_results["english_gender_separation"]["separation_distance"],
        "arabic_separation": contrastive_results["arabic_gender_separation"]["separation_distance"],
        "shared_features_count": len(contrastive_results["shared_encoding"]["shared_top_features"]),
        "gender_direction_angle": contrastive_results["shared_encoding"]["gender_direction_angle_rad"],
    }
    
    logger.info(f"    English gender separation: {results['contrastive_analysis']['english_separation']:.3f}")
    logger.info(f"    Arabic gender separation: {results['contrastive_analysis']['arabic_separation']:.3f}")
    
    return results


def generate_integrated_report(
    prisma_results: Dict,
    multilingual_results: Dict,
    output_path: Path
) -> None:
    """
    Generate comprehensive integrated mechanistic analysis report.
    
    Args:
        prisma_results: Results from ViT-Prisma analysis
        multilingual_results: Results from multilingual analysis
        output_path: Path to save report
    """
    logger.info("Generating integrated mechanistic analysis report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "prisma_analysis": prisma_results,
        "multilingual_analysis": multilingual_results,
        "summary": {
            "analyses_completed": [
                "Factored Matrix Analysis",
                "Interaction Pattern Analysis",
                "Feature Alignment",
                "Morphological Gender Analysis",
                "Semantic Gender Analysis",
                "Contrastive Language Analysis"
            ],
            "key_findings": [
                f"Gender direction angle between languages: {multilingual_results['contrastive_analysis']['gender_direction_angle']:.3f} rad",
                f"Shared top gender features: {multilingual_results['contrastive_analysis']['shared_features_count']}",
                f"Cross-lingual alignment: {multilingual_results['feature_alignment']['num_aligned']} aligned pairs",
            ]
        }
    }
    
    # Save JSON report
    with open(output_path / "integrated_mechanistic_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {output_path / 'integrated_mechanistic_analysis.json'}")


def main():
    parser = argparse.ArgumentParser(description="Integrated Mechanistic Analysis")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize wandb
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        wandb.init(
            project=config['logging'].get('wandb_project', 'sae-captioning-bias'),
            entity=config['logging'].get('wandb_entity', None),
            name=f"mechanistic-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config,
            tags=['mechanistic-analysis', 'prisma', 'multilingual']
        )

    # Setup paths
    checkpoint_dir = Path(config['paths']['checkpoints'])
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load activations
    logger.info("Loading precomputed activations...")
    activations = load_activations_and_features(checkpoint_dir)
    
    if not activations:
        logger.error("No activations found. Run 02_extract_activations.py first.")
        return 1

    # Load SAE for interaction analysis
    sae_path = checkpoint_dir / 'sae_layer_14.pt'
    if sae_path.exists():
        checkpoint = torch.load(sae_path, map_location=args.device, weights_only=False)
        sae = SparseAutoencoder(checkpoint['config'])
        sae.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("SAE not found for interaction analysis")
        sae = None

    # Extract features (assuming you have computed them)
    english_features = activations.get('english', {}).get('features')
    arabic_features = activations.get('arabic', {}).get('features')
    
    if english_features is None or arabic_features is None:
        logger.error("Feature activations not found. Run 04_analyze_features.py first.")
        return 1

    # Run Analyses
    logger.info("\n" + "="*60)
    logger.info("INTEGRATED MECHANISTIC INTERPRETABILITY ANALYSIS")
    logger.info("="*60)

    # ViT-Prisma Analysis
    prisma_results = run_prisma_analysis(
        english_features, arabic_features, sae, args.device
    )

    # Multilingual Analysis
    english_labels = activations.get('english', {}).get('genders', [])
    arabic_labels = activations.get('arabic', {}).get('genders', [])
    english_captions = activations.get('english', {}).get('captions', [])
    arabic_captions = activations.get('arabic', {}).get('captions', [])
    
    multilingual_results = run_multilingual_analysis(
        english_features, arabic_features,
        english_labels, arabic_labels,
        english_captions, arabic_captions,
        config
    )

    # Generate Report
    generate_integrated_report(prisma_results, multilingual_results, results_dir)

    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            "prisma_analysis": prisma_results,
            "multilingual_analysis": multilingual_results,
        })
        wandb.finish()

    logger.info("\n" + "="*60)
    logger.info("Integrated mechanistic analysis complete!")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
