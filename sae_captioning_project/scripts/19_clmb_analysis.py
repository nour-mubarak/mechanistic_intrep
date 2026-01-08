#!/usr/bin/env python3
"""
CLMB Analysis Pipeline
=======================

Main script to run the full Cross-Lingual Multimodal Bias (CLMB) analysis.

Components:
1. HBL (Hierarchical Bias Localization) - Where bias enters
2. CLFA (Cross-Lingual Feature Alignment) - Language correspondences
3. SBI (Surgical Bias Intervention) - Bias mitigation
4. CLBAS (Cross-Lingual Bias Alignment Score) - Quantitative metric

Usage:
    python scripts/19_clmb_analysis.py --config configs/clmb_config.yaml
"""

import argparse
import logging
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from clmb import CLMBFramework
from clmb.hbl import HierarchicalBiasLocalizer, BiasAttributionResult
from clmb.clfa import CrossLingualFeatureAligner, CLFAResult
from clmb.sbi import SurgicalBiasIntervention, SBIResult
from clmb.extractors import get_extractor, MultiModelExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_wandb(config: Dict, run_name: str):
    """Initialize Weights & Biases tracking."""
    if HAS_WANDB and config.get('use_wandb', True):
        wandb.init(
            project=config.get('wandb_project', 'clmb-analysis'),
            name=run_name,
            config=config,
            tags=['clmb', 'bias-analysis', 'multimodal']
        )
        return True
    return False


def load_dataset(config: Dict) -> Dict[str, List]:
    """
    Load the Arabic-English image captioning dataset.
    
    Returns:
        Dict with 'images', 'english_captions', 'arabic_captions', 'genders'
    """
    data_dir = Path(config['data_dir'])
    
    # Load from processed data
    english_captions = []
    arabic_captions = []
    images = []
    genders = []
    
    # Check for parquet files
    english_file = data_dir / 'train_df_en.parquet'
    arabic_file = data_dir / 'train_df_ar.parquet'
    
    if english_file.exists() and arabic_file.exists():
        import pandas as pd
        
        en_df = pd.read_parquet(english_file)
        ar_df = pd.read_parquet(arabic_file)
        
        # Merge on image_id
        merged = en_df.merge(ar_df, on='image_id', suffixes=('_en', '_ar'))
        
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Loading data"):
            english_captions.append(row['caption_en'])
            arabic_captions.append(row['caption_ar'])
            
            # Load image
            img_path = data_dir / 'images' / row['image_id']
            if img_path.exists():
                images.append(Image.open(img_path).convert('RGB'))
            else:
                images.append(None)
            
            # Gender from metadata
            genders.append(row.get('ground_truth_gender', 'unknown'))
    else:
        # Fallback: load from CSV
        csv_file = data_dir / 'aligned_captions.csv'
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
                english_captions.append(row['english_caption'])
                arabic_captions.append(row['arabic_caption'])
                genders.append(row.get('gender', 'unknown'))
                
                img_path = data_dir / 'images' / row['image_path']
                if img_path.exists():
                    images.append(Image.open(img_path).convert('RGB'))
                else:
                    images.append(None)
    
    # Filter out None images
    valid_indices = [i for i, img in enumerate(images) if img is not None]
    
    return {
        'images': [images[i] for i in valid_indices],
        'english_captions': [english_captions[i] for i in valid_indices],
        'arabic_captions': [arabic_captions[i] for i in valid_indices],
        'genders': [genders[i] for i in valid_indices]
    }


def run_hbl_analysis(
    model_id: str,
    dataset: Dict,
    config: Dict,
    output_dir: Path
) -> Dict[str, BiasAttributionResult]:
    """
    Run Hierarchical Bias Localization analysis.
    """
    logger.info("="*60)
    logger.info("Running HBL (Hierarchical Bias Localization)")
    logger.info("="*60)
    
    # Load model
    extractor = get_extractor(model_id, config.get('device', 'cuda'))
    extractor.load_model()
    
    # Initialize HBL
    hbl = HierarchicalBiasLocalizer(
        extractor.model,
        extractor.architecture,
        device=config.get('device', 'cuda')
    )
    
    # Get layer modules
    layers = {
        'vision': [extractor.get_layer_module('vision', i) 
                   for i in range(0, extractor.architecture.num_vision_layers, 4)],
        'language': [extractor.get_layer_module('language', i)
                     for i in range(extractor.architecture.num_language_layers)],
        'projection': extractor.get_layer_module('projection', 0)
    }
    
    # Create dataloader-like structure
    samples = []
    for i in range(min(len(dataset['images']), config.get('max_samples', 1000))):
        samples.append({
            'image': dataset['images'][i],
            'english_prompt': f"Describe this image: {dataset['english_captions'][i]}",
            'arabic_prompt': f"صف هذه الصورة: {dataset['arabic_captions'][i]}",
            'ground_truth_gender': dataset['genders'][i]
        })
    
    # Run analysis
    results = hbl.analyze_model(
        samples,
        extractor.processor,
        layers,
        languages=['english', 'arabic']
    )
    
    # Save results
    hbl_output = output_dir / 'hbl_results.json'
    hbl.save_results(results, hbl_output)
    
    # Log to wandb
    if HAS_WANDB and wandb.run:
        for lang, result in results.items():
            wandb.log({
                f'hbl/{lang}/vision_bias': result.vision_bias_score,
                f'hbl/{lang}/projection_bias': result.projection_bias_score,
                f'hbl/{lang}/language_bias': result.language_bias_score,
                f'hbl/{lang}/total_bias': result.total_bias,
            })
    
    extractor.cleanup()
    return results


def run_clfa_analysis(
    arabic_activations: Dict[int, torch.Tensor],
    english_activations: Dict[int, torch.Tensor],
    arabic_sae: Any,
    english_sae: Any,
    config: Dict,
    output_dir: Path
) -> CLFAResult:
    """
    Run Cross-Lingual Feature Alignment analysis.
    """
    logger.info("="*60)
    logger.info("Running CLFA (Cross-Lingual Feature Alignment)")
    logger.info("="*60)
    
    # Initialize CLFA
    clfa = CrossLingualFeatureAligner(
        arabic_sae=arabic_sae,
        english_sae=english_sae,
        similarity_threshold=config.get('alignment_threshold', 0.7),
        device=config.get('device', 'cuda')
    )
    
    # Run alignment
    result = clfa.align_features(arabic_activations, english_activations)
    
    # Save results
    clfa_output = output_dir / 'clfa_results.json'
    clfa.save_results(result, clfa_output)
    
    # Log to wandb
    if HAS_WANDB and wandb.run:
        wandb.log({
            'clfa/num_alignments': len(result.alignments),
            'clfa/mean_alignment_score': result.mean_alignment_score,
            'clfa/arabic_specific_features': len(result.language_specific_arabic),
            'clfa/english_specific_features': len(result.language_specific_english),
            'clfa/shared_features': len(result.shared_features),
        })
    
    return result


def run_sbi_analysis(
    model_id: str,
    dataset: Dict,
    activations: Dict[int, torch.Tensor],
    sae: Any,
    config: Dict,
    output_dir: Path
) -> SBIResult:
    """
    Run Surgical Bias Intervention analysis.
    """
    logger.info("="*60)
    logger.info("Running SBI (Surgical Bias Intervention)")
    logger.info("="*60)
    
    # Load model
    extractor = get_extractor(model_id, config.get('device', 'cuda'))
    extractor.load_model()
    
    # Initialize SBI
    sbi = SurgicalBiasIntervention(
        model=extractor.model,
        sae=sae,
        processor=extractor.processor,
        device=config.get('device', 'cuda')
    )
    
    # Stack activations
    all_activations = torch.cat([v for v in activations.values()], dim=0)
    genders = dataset['genders'][:len(all_activations)]
    
    # Find optimal intervention
    result = sbi.find_optimal_intervention(
        activations=all_activations,
        genders=genders,
        test_data=dataset,
        max_features=config.get('max_intervention_features', 20),
        min_bias_reduction=config.get('min_bias_reduction', 0.1),
        max_semantic_drift=config.get('max_semantic_drift', 0.2)
    )
    
    # Save results
    sbi_output = output_dir / 'sbi_results.json'
    sbi.save_results(result, sbi_output)
    
    # Log to wandb
    if HAS_WANDB and wandb.run:
        wandb.log({
            'sbi/num_causal_features': len(result.bias_causal_features),
            'sbi/total_bias_reduction': result.total_bias_reduction,
            'sbi/semantic_preservation': result.semantic_preservation,
            'sbi/optimal_features': len(result.optimal_intervention.feature_indices) if result.optimal_intervention else 0,
        })
    
    extractor.cleanup()
    return result


def compute_clbas(
    clfa_result: CLFAResult,
    arabic_bias_scores: np.ndarray,
    english_bias_scores: np.ndarray
) -> float:
    """
    Compute Cross-Lingual Bias Alignment Score (CLBAS).
    
    Novel metric that measures how similarly bias manifests
    across aligned feature pairs in different languages.
    """
    if not clfa_result.alignments:
        return 0.0
    
    weighted_diffs = []
    weights = []
    
    for alignment in clfa_result.alignments:
        ar_idx = alignment.arabic_feature_idx
        en_idx = alignment.english_feature_idx
        
        if ar_idx < len(arabic_bias_scores) and en_idx < len(english_bias_scores):
            bias_diff = abs(arabic_bias_scores[ar_idx] - english_bias_scores[en_idx])
            weight = alignment.similarity_score
            
            weighted_diffs.append(bias_diff * weight)
            weights.append(weight)
    
    if sum(weights) == 0:
        return 0.0
    
    clbas = sum(weighted_diffs) / sum(weights)
    
    return clbas


def run_full_analysis(config: Dict):
    """
    Run the complete CLMB analysis pipeline.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"clmb_{timestamp}"
    
    # Setup output directory
    output_dir = Path(config['output_dir']) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    use_wandb = setup_wandb(config, run_name)
    
    logger.info("="*60)
    logger.info("CLMB (Cross-Lingual Multimodal Bias) Analysis")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = load_dataset(config)
    logger.info(f"Loaded {len(dataset['images'])} samples")
    
    # Results storage
    all_results = {}
    
    # Run for each model
    for model_id in config['models']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing model: {model_id}")
        logger.info(f"{'='*60}")
        
        model_output = output_dir / model_id.replace('/', '_')
        model_output.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. HBL Analysis
            hbl_results = run_hbl_analysis(model_id, dataset, config, model_output)
            all_results[f'{model_id}/hbl'] = hbl_results
            
            # Load SAE models if available
            sae_dir = Path(config.get('sae_dir', 'checkpoints/saes'))
            arabic_sae_path = sae_dir / 'arabic_sae.pt'
            english_sae_path = sae_dir / 'english_sae.pt'
            
            if arabic_sae_path.exists() and english_sae_path.exists():
                # Load SAEs
                arabic_sae = torch.load(arabic_sae_path)
                english_sae = torch.load(english_sae_path)
                
                # Load activations
                act_dir = Path(config.get('activations_dir', 'data/processed/activations'))
                arabic_activations = {}
                english_activations = {}
                
                for f in act_dir.glob('*arabic*.pt'):
                    layer_idx = int(f.stem.split('_')[-1])
                    arabic_activations[layer_idx] = torch.load(f)
                
                for f in act_dir.glob('*english*.pt'):
                    layer_idx = int(f.stem.split('_')[-1])
                    english_activations[layer_idx] = torch.load(f)
                
                # 2. CLFA Analysis
                if arabic_activations and english_activations:
                    clfa_result = run_clfa_analysis(
                        arabic_activations,
                        english_activations,
                        arabic_sae,
                        english_sae,
                        config,
                        model_output
                    )
                    all_results[f'{model_id}/clfa'] = clfa_result
                    
                    # 3. Compute CLBAS
                    # Get bias scores from HBL
                    ar_bias = np.array([r.language_bias_score for r in hbl_results.get('arabic', [])])
                    en_bias = np.array([r.language_bias_score for r in hbl_results.get('english', [])])
                    
                    if len(ar_bias) > 0 and len(en_bias) > 0:
                        clbas = compute_clbas(clfa_result, ar_bias, en_bias)
                        all_results[f'{model_id}/clbas'] = clbas
                        
                        logger.info(f"CLBAS Score: {clbas:.4f}")
                        
                        if use_wandb:
                            wandb.log({f'{model_id}/clbas': clbas})
                
                # 4. SBI Analysis
                sbi_result = run_sbi_analysis(
                    model_id,
                    dataset,
                    english_activations,
                    english_sae,
                    config,
                    model_output
                )
                all_results[f'{model_id}/sbi'] = sbi_result
            
            else:
                logger.warning("SAE models not found, skipping CLFA and SBI")
        
        except Exception as e:
            logger.error(f"Error analyzing {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate final report
    report = generate_report(all_results, config, output_dir)
    
    # Log final summary
    if use_wandb:
        wandb.log({"final_report": wandb.Table(data=[[report]], columns=["report"])})
        wandb.finish()
    
    logger.info("\n" + "="*60)
    logger.info("CLMB Analysis Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)
    
    return all_results


def generate_report(results: Dict, config: Dict, output_dir: Path) -> str:
    """Generate a summary report of the CLMB analysis."""
    
    report_lines = [
        "# CLMB Analysis Report",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        f"- Models: {config['models']}",
        f"- Max samples: {config.get('max_samples', 1000)}",
        "",
        "## Results Summary",
        ""
    ]
    
    for model_id in config['models']:
        model_key = model_id.replace('/', '_')
        report_lines.append(f"### {model_id}")
        report_lines.append("")
        
        # HBL Results
        hbl_key = f'{model_id}/hbl'
        if hbl_key in results:
            hbl = results[hbl_key]
            report_lines.append("**Hierarchical Bias Localization (HBL)**")
            for lang, res in hbl.items():
                report_lines.append(f"- {lang}: Vision={res.vision_bias_score:.4f}, "
                                   f"Projection={res.projection_bias_score:.4f}, "
                                   f"Language={res.language_bias_score:.4f}")
            report_lines.append("")
        
        # CLFA Results
        clfa_key = f'{model_id}/clfa'
        if clfa_key in results:
            clfa = results[clfa_key]
            report_lines.append("**Cross-Lingual Feature Alignment (CLFA)**")
            report_lines.append(f"- Aligned features: {len(clfa.alignments)}")
            report_lines.append(f"- Mean alignment score: {clfa.mean_alignment_score:.4f}")
            report_lines.append(f"- Arabic-specific: {len(clfa.language_specific_arabic)}")
            report_lines.append(f"- English-specific: {len(clfa.language_specific_english)}")
            report_lines.append("")
        
        # CLBAS
        clbas_key = f'{model_id}/clbas'
        if clbas_key in results:
            report_lines.append(f"**CLBAS Score: {results[clbas_key]:.4f}**")
            report_lines.append("")
        
        # SBI Results
        sbi_key = f'{model_id}/sbi'
        if sbi_key in results:
            sbi = results[sbi_key]
            report_lines.append("**Surgical Bias Intervention (SBI)**")
            report_lines.append(f"- Causal features identified: {len(sbi.bias_causal_features)}")
            report_lines.append(f"- Bias reduction: {sbi.total_bias_reduction:.4f}")
            report_lines.append(f"- Semantic preservation: {sbi.semantic_preservation:.4f}")
            report_lines.append("")
    
    report = "\n".join(report_lines)
    
    # Save report
    report_path = output_dir / 'CLMB_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Run CLMB Analysis')
    parser.add_argument('--config', type=str, default='configs/clmb_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model ID')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        # Default config
        config = {
            'models': ['google/paligemma-3b-pt-224'],
            'data_dir': 'data/processed',
            'output_dir': 'results/clmb',
            'activations_dir': 'data/processed/activations',
            'sae_dir': 'checkpoints/saes',
            'device': 'cuda',
            'max_samples': 1000,
            'alignment_threshold': 0.7,
            'max_intervention_features': 20,
            'min_bias_reduction': 0.1,
            'max_semantic_drift': 0.2,
            'use_wandb': True,
            'wandb_project': 'clmb-bias-analysis',
        }
    
    # Override with command line args
    if args.model:
        config['models'] = [args.model]
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.no_wandb:
        config['use_wandb'] = False
    
    # Run analysis
    run_full_analysis(config)


if __name__ == '__main__':
    main()
