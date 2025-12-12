#!/usr/bin/env python3
"""
Verify Arabic Checkpoints and Train SAE
Checks for NaN values in Arabic activation checkpoints and trains SAE on combined data.
"""

import torch
import sys
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import yaml
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_checkpoint_nan(checkpoint_path: Path) -> dict:
    """Verify a single checkpoint for NaN values."""
    try:
        data = torch.load(checkpoint_path, map_location='cpu')
        
        results = {
            'file': checkpoint_path.name,
            'has_nan': False,
            'nan_counts': {},
            'shapes': {}
        }
        
        if 'activations' in data:
            for layer_idx, acts in data['activations'].items():
                results['shapes'][layer_idx] = acts.shape
                nan_count = torch.isnan(acts).sum().item()
                if nan_count > 0:
                    results['has_nan'] = True
                    results['nan_counts'][layer_idx] = nan_count
        
        return results
    except Exception as e:
        logger.error(f"Error loading {checkpoint_path}: {e}")
        return {'file': checkpoint_path.name, 'error': str(e)}


def verify_all_arabic_checkpoints(checkpoint_dir: Path) -> dict:
    """Verify all Arabic checkpoint files for NaN values."""
    logger.info("=" * 60)
    logger.info("VERIFYING ARABIC CHECKPOINTS FOR NaN VALUES")
    logger.info("=" * 60)
    
    arabic_files = sorted(checkpoint_dir.glob("activations_arabic_chunk_*.pt"))
    
    if not arabic_files:
        logger.error("No Arabic checkpoint files found!")
        return {'status': 'error', 'message': 'No files found'}
    
    logger.info(f"Found {len(arabic_files)} Arabic checkpoint files")
    
    all_results = []
    total_nan_count = 0
    
    for checkpoint_file in tqdm(arabic_files, desc="Verifying checkpoints"):
        result = verify_checkpoint_nan(checkpoint_file)
        all_results.append(result)
        
        if result.get('has_nan', False):
            total_nan_count += sum(result['nan_counts'].values())
            logger.warning(f"⚠️  {result['file']}: Found NaN values!")
            for layer, count in result['nan_counts'].items():
                logger.warning(f"    Layer {layer}: {count} NaN values")
    
    # Summary
    files_with_nan = sum(1 for r in all_results if r.get('has_nan', False))
    
    logger.info("=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files checked: {len(all_results)}")
    logger.info(f"Files with NaN: {files_with_nan}")
    logger.info(f"Total NaN count: {total_nan_count}")
    
    if files_with_nan == 0:
        logger.info("✅ All Arabic checkpoints are clean (no NaN values)")
    else:
        logger.warning(f"⚠️  {files_with_nan} files contain NaN values")
    
    return {
        'status': 'success',
        'total_files': len(all_results),
        'files_with_nan': files_with_nan,
        'total_nan_count': total_nan_count,
        'results': all_results
    }


def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Verify and train SAE")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--verify-only', action='store_true', help='Only verify, do not train')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    project_root = config_path.parent.parent
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config: {config_path}")
    
    config = load_config(config_path)
    checkpoint_dir = project_root / "checkpoints"
    
    # Step 1: Verify Arabic checkpoints
    verification_results = verify_all_arabic_checkpoints(checkpoint_dir)
    
    if verification_results['status'] != 'success':
        logger.error("Verification failed!")
        return 1
    
    # Initialize wandb for logging
    wandb.init(
        project=config.get('wandb', {}).get('project', 'sae-captioning-bias'),
        name=f"checkpoint-verification-{wandb.util.generate_id()}",
        config={
            'verification_results': verification_results,
            'checkpoint_dir': str(checkpoint_dir)
        }
    )
    
    # Log verification results to wandb
    wandb.log({
        'verification/total_files': verification_results['total_files'],
        'verification/files_with_nan': verification_results['files_with_nan'],
        'verification/total_nan_count': verification_results['total_nan_count'],
        'verification/clean': verification_results['files_with_nan'] == 0
    })
    
    if args.verify_only:
        logger.info("Verification complete. Exiting (--verify-only flag set)")
        wandb.finish()
        return 0
    
    # Step 2: Train SAE on combined data
    if verification_results['files_with_nan'] > 0:
        logger.warning("⚠️  NaN values detected. Consider cleaning data before training.")
        response = input("Continue with training anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training cancelled by user")
            wandb.finish()
            return 0
    
    logger.info("=" * 60)
    logger.info("STARTING SAE TRAINING")
    logger.info("=" * 60)
    
    # Import and run the training script
    sys.path.insert(0, str(project_root / "scripts"))
    from train_sae_03 import main as train_main
    
    # Update sys.argv for the training script
    original_argv = sys.argv.copy()
    sys.argv = ['train_sae_03.py', '--config', str(config_path)]
    
    try:
        train_result = train_main()
        logger.info(f"Training completed with result: {train_result}")
    finally:
        sys.argv = original_argv
        wandb.finish()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
