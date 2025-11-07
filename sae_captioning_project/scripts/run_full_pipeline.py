#!/usr/bin/env python3
"""
Run Full Pipeline
=================

Executes the complete SAE cross-lingual analysis pipeline:
1. Data preparation
2. Activation extraction
3. SAE training
4. Feature analysis
5. Steering experiments
6. Visualization generation

Usage:
    python scripts/run_full_pipeline.py --config configs/config.yaml

For specific stages:
    python scripts/run_full_pipeline.py --config configs/config.yaml --stages 1,2,3
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
import yaml
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


STAGES = {
    1: ("Data Preparation", "01_prepare_data.py"),
    2: ("Activation Extraction", "02_extract_activations.py"),
    3: ("SAE Training", "03_train_sae.py"),
    4: ("Feature Analysis", "04_analyze_features.py"),
    5: ("Steering Experiments", "05_steering_experiments.py"),
    6: ("Visualization Generation", "06_generate_visualizations.py"),
}


def run_stage(
    stage_num: int,
    stage_name: str,
    script_name: str,
    config_path: str,
    scripts_dir: Path,
    extra_args: list = None
) -> bool:
    """Run a single pipeline stage."""
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STAGE {stage_num}: {stage_name}")
    logger.info("=" * 60)
    
    script_path = scripts_dir / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path), "--config", config_path]
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Stage {stage_num} completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage {stage_num} failed with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Stage {stage_num} failed with error: {e}")
        return False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_environment() -> bool:
    """Validate that required packages are available."""
    required_packages = [
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'matplotlib',
        'yaml',
        'tqdm',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing required packages: {missing}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_gpu_availability() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            logger.warning("No GPU detected. Pipeline will run on CPU (much slower)")
            return False
    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete SAE analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1 - Data Preparation: Validate and prepare dataset
  2 - Activation Extraction: Extract model activations
  3 - SAE Training: Train Sparse Autoencoders
  4 - Feature Analysis: Analyze gender features
  5 - Steering Experiments: Test feature interventions
  6 - Visualization: Generate plots and reports

Examples:
  # Run full pipeline
  python scripts/run_full_pipeline.py --config configs/config.yaml

  # Run only stages 1-3
  python scripts/run_full_pipeline.py --stages 1,2,3

  # Skip data preparation
  python scripts/run_full_pipeline.py --stages 2,3,4,5,6

  # Just create visualizations
  python scripts/run_full_pipeline.py --stages 6
        """
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--stages', type=str, default=None,
                       help='Comma-separated list of stages to run (default: all)')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--skip-gpu-check', action='store_true',
                       help='Skip GPU availability check')
    args = parser.parse_args()
    
    # Determine project root and scripts directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║  SAE Cross-Lingual Captioning Analysis Pipeline              ║
║  Mechanistic Interpretability for Vision-Language Models     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config: {args.config}")
    
    # Validate environment
    logger.info("\nValidating environment...")
    if not validate_environment():
        return 1
    
    # Check GPU
    if not args.skip_gpu_check:
        check_gpu_availability()
    
    # Load config
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    
    # Determine stages to run
    if args.stages:
        stages_to_run = [int(s.strip()) for s in args.stages.split(',')]
    else:
        stages_to_run = list(STAGES.keys())
    
    logger.info(f"Stages to run: {stages_to_run}")
    
    # Setup output directories
    for path_key in ['checkpoints', 'results', 'visualizations', 'logs']:
        path = project_root / config['paths'].get(path_key, path_key)
        path.mkdir(parents=True, exist_ok=True)
    
    # Track timing
    pipeline_start = time.time()
    stage_times = {}
    failed_stages = []
    
    # Run stages
    for stage_num in stages_to_run:
        if stage_num not in STAGES:
            logger.warning(f"Unknown stage {stage_num}, skipping")
            continue
        
        stage_name, script_name = STAGES[stage_num]
        
        # Handle special arguments for data preparation
        extra_args = None
        if stage_num == 1 and args.create_sample_data:
            extra_args = ['--create-sample']
        
        stage_start = time.time()
        success = run_stage(
            stage_num=stage_num,
            stage_name=stage_name,
            script_name=script_name,
            config_path=str(config_path),
            scripts_dir=script_dir,
            extra_args=extra_args
        )
        stage_times[stage_num] = time.time() - stage_start
        
        if not success:
            failed_stages.append(stage_num)
            logger.error(f"\nStage {stage_num} failed. Stopping pipeline.")
            break
    
    # Print summary
    total_time = time.time() - pipeline_start
    
    print("\n")
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print("\nStage timing:")
    for stage_num, elapsed in stage_times.items():
        status = "✓" if stage_num not in failed_stages else "✗"
        stage_name = STAGES[stage_num][0]
        print(f"  {status} Stage {stage_num} ({stage_name}): {elapsed:.1f}s")
    
    if failed_stages:
        print(f"\n❌ Pipeline failed at stage(s): {failed_stages}")
        return 1
    else:
        print("\n✅ Pipeline completed successfully!")
        
        # Print output locations
        print("\nOutput locations:")
        print(f"  Checkpoints: {project_root / config['paths']['checkpoints']}")
        print(f"  Results: {project_root / config['paths']['results']}")
        print(f"  Visualizations: {project_root / config['paths']['visualizations']}")
        
        return 0


if __name__ == '__main__':
    sys.exit(main())
