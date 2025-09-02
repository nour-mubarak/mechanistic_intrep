"""
Configuration Manager for Mechanistic Interpretability Experiments
================================================================

This module provides comprehensive configuration management using Hydra and OmegaConf
for organizing and tracking experiments in gender bias analysis.

Key Features:
- Hierarchical configuration management
- Experiment versioning and tracking
- Hyperparameter validation
- Configuration inheritance and overrides
- Integration with wandb tracking
- Reproducibility settings

Example usage:
    from config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config = config_manager.load_config("experiment_config.yaml")
    config_manager.setup_experiment(config)
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize_config_dir
from datetime import datetime
import torch
import numpy as np
import random

class ConfigManager:
    """Comprehensive configuration manager for experiments."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.current_config = None
        self.experiment_id = None
        
    def load_config(self, 
                   config_name: str = "experiment_config.yaml",
                   overrides: List[str] = None) -> DictConfig:
        """
        Load configuration from file with optional overrides.
        
        Args:
            config_name: Name of the configuration file
            overrides: List of configuration overrides
            
        Returns:
            Loaded configuration object
        """
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load base configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = OmegaConf.create(config_dict)
        
        # Apply overrides if provided
        if overrides:
            for override in overrides:
                if '=' in override:
                    key, value = override.split('=', 1)
                    # Try to convert value to appropriate type
                    try:
                        value = eval(value)
                    except:
                        pass  # Keep as string if eval fails
                    OmegaConf.set(config, key, value)
        
        self.current_config = config
        return config
    
    def validate_config(self, config: DictConfig) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        required_sections = ['experiment', 'model', 'dataset', 'training']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")
        
        # Validate model configuration
        if 'name' not in config.model:
            raise ValueError("Model name is required")
        
        # Validate dataset configuration
        if 'data_dir' not in config.dataset:
            raise ValueError("Dataset data_dir is required")
        
        # Validate training configuration
        if config.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate output directories
        if 'base_dir' in config.output:
            output_dir = Path(config.output.base_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def setup_experiment(self, config: DictConfig) -> str:
        """
        Set up experiment environment and directories.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        # Validate configuration
        self.validate_config(config)
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{config.experiment.name}_{timestamp}"
        
        # Create experiment directories
        base_dir = Path(config.output.base_dir) / self.experiment_id
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        if 'subdirs' in config.output:
            for subdir_name in config.output.subdirs.values():
                (base_dir / subdir_name).mkdir(exist_ok=True)
        
        # Update config with experiment-specific paths
        config.experiment.id = self.experiment_id
        config.experiment.output_dir = str(base_dir)
        
        # Set up reproducibility
        if 'reproducibility' in config:
            self.setup_reproducibility(config.reproducibility)
        
        # Save configuration to experiment directory
        config_save_path = base_dir / "config.yaml"
        with open(config_save_path, 'w') as f:
            OmegaConf.save(config, f)
        
        print(f"Experiment setup complete: {self.experiment_id}")
        print(f"Output directory: {base_dir}")
        
        return self.experiment_id
    
    def setup_reproducibility(self, repro_config: DictConfig) -> None:
        """
        Set up reproducibility settings.
        
        Args:
            repro_config: Reproducibility configuration
        """
        seed = repro_config.get('seed', 42)
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Set deterministic behavior
        if repro_config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = repro_config.get('benchmark', True)
        
        print(f"Reproducibility setup complete with seed: {seed}")
    
    def create_sweep_config(self, 
                          base_config: DictConfig,
                          parameter_space: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create wandb sweep configuration for hyperparameter optimization.
        
        Args:
            base_config: Base configuration
            parameter_space: Dictionary defining parameter search space
            
        Returns:
            Sweep configuration
        """
        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "bias_score",
                "goal": "minimize"
            },
            "parameters": parameter_space,
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 3,
                "eta": 2
            }
        }
        
        # Add project information
        if 'wandb' in base_config:
            sweep_config["project"] = base_config.wandb.get('project', 'mechanistic-bias')
            if 'entity' in base_config.wandb and base_config.wandb.entity:
                sweep_config["entity"] = base_config.wandb.entity
        
        return sweep_config
    
    def get_device_config(self, config: DictConfig) -> str:
        """
        Determine the appropriate device configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Device string (cuda, cpu, etc.)
        """
        device_setting = config.model.get('device', 'auto')
        
        if device_setting == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = device_setting
        
        # Validate device availability
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def update_config_from_wandb(self, config: DictConfig) -> DictConfig:
        """
        Update configuration with wandb sweep parameters.
        
        Args:
            config: Base configuration
            
        Returns:
            Updated configuration
        """
        try:
            import wandb
            
            # Get sweep parameters if running in a sweep
            if wandb.run and wandb.config:
                for key, value in wandb.config.items():
                    if '.' in key:
                        # Handle nested keys
                        keys = key.split('.')
                        current = config
                        for k in keys[:-1]:
                            if k not in current:
                                current[k] = {}
                            current = current[k]
                        current[keys[-1]] = value
                    else:
                        config[key] = value
        except ImportError:
            pass  # wandb not available
        
        return config
    
    def save_experiment_metadata(self, 
                               config: DictConfig,
                               additional_info: Dict[str, Any] = None) -> None:
        """
        Save experiment metadata for tracking and reproducibility.
        
        Args:
            config: Experiment configuration
            additional_info: Additional information to save
        """
        if not self.experiment_id:
            raise ValueError("Experiment not set up. Call setup_experiment first.")
        
        metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": OmegaConf.to_container(config, resolve=True),
            "system_info": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        if torch.cuda.is_available():
            metadata["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
        
        if additional_info:
            metadata.update(additional_info)
        
        # Save metadata
        output_dir = Path(config.experiment.output_dir)
        metadata_path = output_dir / "experiment_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Experiment metadata saved to: {metadata_path}")
    
    def load_experiment_config(self, experiment_id: str) -> DictConfig:
        """
        Load configuration from a previous experiment.
        
        Args:
            experiment_id: ID of the experiment to load
            
        Returns:
            Loaded configuration
        """
        # Search for experiment directory
        base_output_dir = Path("./results")
        experiment_dir = None
        
        for dir_path in base_output_dir.iterdir():
            if dir_path.is_dir() and experiment_id in dir_path.name:
                experiment_dir = dir_path
                break
        
        if not experiment_dir:
            raise FileNotFoundError(f"Experiment directory not found for ID: {experiment_id}")
        
        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        return OmegaConf.load(config_path)
    
    def create_config_variants(self, 
                             base_config: DictConfig,
                             variants: Dict[str, Dict[str, Any]]) -> Dict[str, DictConfig]:
        """
        Create configuration variants for different experiments.
        
        Args:
            base_config: Base configuration
            variants: Dictionary of variant specifications
            
        Returns:
            Dictionary of configuration variants
        """
        config_variants = {}
        
        for variant_name, changes in variants.items():
            # Create a copy of the base config
            variant_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
            
            # Apply changes
            for key, value in changes.items():
                OmegaConf.set(variant_config, key, value)
            
            # Update experiment name
            variant_config.experiment.name = f"{base_config.experiment.name}_{variant_name}"
            
            config_variants[variant_name] = variant_config
        
        return config_variants


def create_sample_configs():
    """Create sample configuration files for different experiment types."""
    config_manager = ConfigManager()
    
    # Create sweep parameter space
    parameter_space = {
        "training.learning_rate": {
            "distribution": "log_uniform",
            "min": 1e-6,
            "max": 1e-3
        },
        "training.batch_size": {
            "values": [2, 4, 8, 16]
        },
        "sae.dict_size": {
            "values": [4096, 8192, 16384]
        },
        "sae.sparsity_penalty": {
            "distribution": "log_uniform",
            "min": 1e-4,
            "max": 1e-2
        },
        "circuit_discovery.correlation_threshold": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 0.9
        }
    }
    
    # Load base config
    base_config = config_manager.load_config()
    
    # Create sweep config
    sweep_config = config_manager.create_sweep_config(base_config, parameter_space)
    
    # Save sweep config
    sweep_path = config_manager.config_dir / "sweep_config.yaml"
    with open(sweep_path, 'w') as f:
        yaml.dump(sweep_config, f, indent=2)
    
    print("Sample configurations created successfully!")
    return config_manager


if __name__ == "__main__":
    # Create sample configurations
    config_manager = create_sample_configs()
    print("Configuration manager setup complete!")

