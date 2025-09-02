"""
Weights & Biases Integration Module
=================================

This module provides comprehensive integration with Weights & Biases (wandb) for
experiment tracking, metric logging, artifact management, and custom visualizations
specifically designed for gender bias analysis in multilingual image captioning.

Key Features:
- Experiment initialization and configuration
- Real-time metric logging during training
- Artifact logging for models, datasets, and results
- Custom charts and visualizations
- Hyperparameter tracking and sweeps
- Cross-lingual experiment comparison
- Gender bias specific metrics and dashboards

Example usage:
    from wandb_integration import WandbTracker
    
    tracker = WandbTracker(project_name="gender-bias-analysis")
    tracker.init_experiment(config)
    tracker.log_training_metrics(metrics)
    tracker.log_bias_analysis(bias_results)
"""

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Union
import os
import json
from pathlib import Path
import torch
from datetime import datetime

class WandbTracker:
    """Comprehensive wandb integration for gender bias analysis experiments."""
    
    def __init__(self, 
                 project_name: str = "mechanistic-gender-bias",
                 entity: Optional[str] = None,
                 tags: List[str] = None):
        """
        Initialize the wandb tracker.
        
        Args:
            project_name: Name of the wandb project
            entity: Wandb entity (username or team)
            tags: List of tags for the experiment
        """
        self.project_name = project_name
        self.entity = entity
        self.tags = tags or ["gender-bias", "mechanistic-interpretability", "multilingual"]
        self.run = None
        self.experiment_config = {}
        
    def init_experiment(self, 
                       config: Dict[str, Any],
                       experiment_name: str = None,
                       notes: str = None) -> None:
        """
        Initialize a new wandb experiment.
        
        Args:
            config: Experiment configuration dictionary
            experiment_name: Name for this specific experiment
            notes: Additional notes about the experiment
        """
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"gender_bias_exp_{timestamp}"
        
        # Initialize wandb run
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=experiment_name,
            config=config,
            tags=self.tags,
            notes=notes,
            reinit=True
        )
        
        self.experiment_config = config
        
        # Log system information
        self.log_system_info()
        
        print(f"Initialized wandb experiment: {experiment_name}")
        print(f"Dashboard URL: {self.run.url}")
        
    def log_system_info(self) -> None:
        """Log system and environment information."""
        system_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "torch_version": torch.__version__ if torch else "Not available",
            "cuda_available": torch.cuda.is_available() if torch else False,
            "gpu_count": torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
        }
        
        if torch and torch.cuda.is_available():
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
            system_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        
        wandb.log({"system_info": system_info})
        
    def log_training_metrics(self, 
                           metrics: Dict[str, float],
                           step: Optional[int] = None,
                           epoch: Optional[int] = None) -> None:
        """
        Log training metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step number
            epoch: Training epoch number
        """
        log_dict = metrics.copy()
        
        if step is not None:
            log_dict["step"] = step
        if epoch is not None:
            log_dict["epoch"] = epoch
            
        wandb.log(log_dict)
        
    def log_bias_analysis(self, 
                         bias_results: Dict[str, Any],
                         prefix: str = "bias") -> None:
        """
        Log gender bias analysis results.
        
        Args:
            bias_results: Dictionary containing bias analysis results
            prefix: Prefix for metric names
        """
        # Flatten nested dictionaries
        flattened_results = {}
        
        def flatten_dict(d, parent_key='', sep='_'):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    flattened_results.update(flatten_dict(v, new_key, sep=sep))
                elif isinstance(v, (int, float, bool)):
                    flattened_results[f"{prefix}_{new_key}"] = v
                elif isinstance(v, str):
                    flattened_results[f"{prefix}_{new_key}"] = v
        
        flatten_dict(bias_results)
        wandb.log(flattened_results)
        
        # Create custom bias visualization
        if "gender_metrics" in bias_results:
            self.create_bias_chart(bias_results["gender_metrics"])
            
    def create_bias_chart(self, gender_metrics: Dict[str, float]) -> None:
        """
        Create custom bias visualization chart.
        
        Args:
            gender_metrics: Dictionary containing gender-specific metrics
        """
        # Create bar chart for gender metrics
        metrics = list(gender_metrics.keys())
        values = list(gender_metrics.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        ax.set_title('Gender Bias Metrics')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({"bias_metrics_chart": wandb.Image(fig)})
        plt.close(fig)
        
    def log_attention_analysis(self, 
                             attention_data: Dict[str, np.ndarray],
                             tokens: List[str] = None) -> None:
        """
        Log attention analysis results.
        
        Args:
            attention_data: Dictionary containing attention weights
            tokens: List of tokens for labeling
        """
        for layer_name, attention_weights in attention_data.items():
            # Create attention heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            
            sns.heatmap(attention_weights, 
                       ax=ax,
                       cmap='Blues',
                       xticklabels=tokens[:attention_weights.shape[1]] if tokens else False,
                       yticklabels=tokens[:attention_weights.shape[0]] if tokens else False)
            
            ax.set_title(f'Attention Patterns - {layer_name}')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({f"attention_{layer_name}": wandb.Image(fig)})
            plt.close(fig)
            
            # Log attention statistics
            attention_stats = {
                f"attention_{layer_name}_mean": float(np.mean(attention_weights)),
                f"attention_{layer_name}_std": float(np.std(attention_weights)),
                f"attention_{layer_name}_max": float(np.max(attention_weights)),
                f"attention_{layer_name}_sparsity": float(np.sum(attention_weights < 0.01) / attention_weights.size)
            }
            wandb.log(attention_stats)
            
    def log_circuit_discovery(self, 
                            circuits: Dict[str, Dict],
                            activations: Dict[str, np.ndarray] = None) -> None:
        """
        Log circuit discovery results.
        
        Args:
            circuits: Dictionary containing discovered circuits
            activations: Optional activation data for visualization
        """
        # Log circuit statistics
        circuit_stats = {}
        for circuit_name, circuit_data in circuits.items():
            circuit_stats[f"circuit_{circuit_name}_size"] = circuit_data.get("circuit_size", 0)
            circuit_stats[f"circuit_{circuit_name}_mean_correlation"] = circuit_data.get("mean_correlation", 0)
        
        wandb.log(circuit_stats)
        
        # Create circuit visualization
        if circuits:
            fig, axes = plt.subplots(1, len(circuits), figsize=(5*len(circuits), 6))
            if len(circuits) == 1:
                axes = [axes]
            
            for idx, (circuit_name, circuit_data) in enumerate(circuits.items()):
                neuron_indices = circuit_data.get("neuron_indices", [])
                correlations = circuit_data.get("correlations", [])
                
                if neuron_indices and correlations:
                    colors = ['red' if corr > 0 else 'blue' for corr in correlations]
                    axes[idx].bar(range(len(neuron_indices)), correlations, color=colors)
                    axes[idx].set_title(f'Circuit: {circuit_name}')
                    axes[idx].set_xlabel('Neuron Index')
                    axes[idx].set_ylabel('Gender Correlation')
                    axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            wandb.log({"circuit_discovery": wandb.Image(fig)})
            plt.close(fig)
            
    def log_cross_lingual_comparison(self, 
                                   english_results: Dict[str, float],
                                   arabic_results: Dict[str, float]) -> None:
        """
        Log cross-lingual comparison results.
        
        Args:
            english_results: Results for English captions
            arabic_results: Results for Arabic captions
        """
        # Log individual language results
        for key, value in english_results.items():
            wandb.log({f"english_{key}": value})
            
        for key, value in arabic_results.items():
            wandb.log({f"arabic_{key}": value})
            
        # Create comparison visualization
        common_metrics = set(english_results.keys()) & set(arabic_results.keys())
        
        if common_metrics:
            metrics = list(common_metrics)
            en_values = [english_results[m] for m in metrics]
            ar_values = [arabic_results[m] for m in metrics]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, en_values, width, label='English', color='#d62728')
            bars2 = ax.bar(x + width/2, ar_values, width, label='Arabic', color='#9467bd')
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title('Cross-lingual Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            wandb.log({"cross_lingual_comparison": wandb.Image(fig)})
            plt.close(fig)
            
    def log_model_artifact(self, 
                          model_path: str,
                          artifact_name: str,
                          artifact_type: str = "model",
                          description: str = None) -> None:
        """
        Log model as wandb artifact.
        
        Args:
            model_path: Path to the model file
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            description: Description of the artifact
        """
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description
        )
        
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
        
        print(f"Logged model artifact: {artifact_name}")
        
    def log_dataset_artifact(self, 
                           dataset_path: str,
                           artifact_name: str,
                           description: str = None) -> None:
        """
        Log dataset as wandb artifact.
        
        Args:
            dataset_path: Path to the dataset
            artifact_name: Name for the artifact
            description: Description of the dataset
        """
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=description
        )
        
        if os.path.isdir(dataset_path):
            artifact.add_dir(dataset_path)
        else:
            artifact.add_file(dataset_path)
            
        self.run.log_artifact(artifact)
        
        print(f"Logged dataset artifact: {artifact_name}")
        
    def log_results_artifact(self, 
                           results_data: Dict[str, Any],
                           artifact_name: str,
                           description: str = None) -> None:
        """
        Log experiment results as wandb artifact.
        
        Args:
            results_data: Dictionary containing results
            artifact_name: Name for the artifact
            description: Description of the results
        """
        # Save results to temporary file
        temp_path = f"/tmp/{artifact_name}.json"
        with open(temp_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        artifact = wandb.Artifact(
            name=artifact_name,
            type="results",
            description=description
        )
        
        artifact.add_file(temp_path)
        self.run.log_artifact(artifact)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        print(f"Logged results artifact: {artifact_name}")
        
    def create_custom_dashboard(self, 
                              dashboard_config: Dict[str, Any]) -> None:
        """
        Create custom wandb dashboard for gender bias analysis.
        
        Args:
            dashboard_config: Configuration for the dashboard
        """
        # Define custom charts for gender bias analysis
        custom_charts = [
            {
                "title": "Gender Bias Over Time",
                "type": "line",
                "metrics": ["bias_gender_bias_score", "bias_male_accuracy", "bias_female_accuracy"]
            },
            {
                "title": "Cross-lingual Comparison",
                "type": "bar",
                "metrics": ["english_bias_score", "arabic_bias_score"]
            },
            {
                "title": "Quality vs Bias Trade-off",
                "type": "scatter",
                "x_metric": "bias_gender_bias_score",
                "y_metric": "quality_bleu_score"
            },
            {
                "title": "Circuit Activation Patterns",
                "type": "heatmap",
                "metrics": ["circuit_*_mean_correlation"]
            }
        ]
        
        # Log dashboard configuration
        wandb.log({"dashboard_config": custom_charts})
        
    def finish_experiment(self, 
                         summary_metrics: Dict[str, float] = None) -> None:
        """
        Finish the wandb experiment and log summary.
        
        Args:
            summary_metrics: Final summary metrics
        """
        if summary_metrics:
            for key, value in summary_metrics.items():
                self.run.summary[key] = value
        
        # Log experiment completion
        self.run.summary["experiment_completed"] = True
        self.run.summary["completion_time"] = datetime.now().isoformat()
        
        wandb.finish()
        print("Experiment completed and logged to wandb")
        
    def create_sweep_config(self, 
                          parameter_ranges: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create wandb sweep configuration for hyperparameter optimization.
        
        Args:
            parameter_ranges: Dictionary defining parameter ranges
            
        Returns:
            Sweep configuration dictionary
        """
        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "bias_gender_bias_score",
                "goal": "minimize"
            },
            "parameters": parameter_ranges,
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 3
            }
        }
        
        return sweep_config
        
    def start_sweep(self, 
                   sweep_config: Dict[str, Any],
                   train_function: callable,
                   count: int = 10) -> str:
        """
        Start a wandb sweep for hyperparameter optimization.
        
        Args:
            sweep_config: Sweep configuration
            train_function: Training function to optimize
            count: Number of runs in the sweep
            
        Returns:
            Sweep ID
        """
        sweep_id = wandb.sweep(
            sweep_config,
            project=self.project_name,
            entity=self.entity
        )
        
        wandb.agent(sweep_id, train_function, count=count)
        
        return sweep_id



    def log_experiment_results(self, results: Dict[str, Any]) -> None:
        """
        Log complete experiment results to wandb.
        
        Args:
            results: Complete experiment results dictionary
        """
        # Log final metrics
        if 'final_evaluation' in results:
            final_eval = results['final_evaluation']
            if 'final_results' in final_eval:
                final_results = final_eval['final_results']
                
                # Log bias metrics
                if hasattr(final_results, 'bias_metrics'):
                    bias_metrics = final_results.bias_metrics
                    wandb.log({
                        "final_gender_gap": bias_metrics.gender_gap,
                        "final_male_accuracy": bias_metrics.male_accuracy,
                        "final_female_accuracy": bias_metrics.female_accuracy,
                        "final_demographic_parity": bias_metrics.demographic_parity,
                        "final_equalized_odds": bias_metrics.equalized_odds
                    })
                
                # Log quality metrics
                if hasattr(final_results, 'quality_metrics'):
                    quality_metrics = final_results.quality_metrics
                    wandb.log({
                        "final_bleu_4": quality_metrics.bleu_4,
                        "final_rouge_l": quality_metrics.rouge_l,
                        "final_meteor": quality_metrics.meteor,
                        "final_bertscore_f1": quality_metrics.bertscore_f1
                    })
        
        # Log results as artifact
        self.log_results_artifact(results, "complete_experiment_results")
        
        print("Complete experiment results logged to wandb")

def create_sample_experiment():
    """Create a sample experiment for demonstration."""
    # Initialize tracker
    tracker = WandbTracker(project_name="gender-bias-demo")
    
    # Sample configuration
    config = {
        "model_name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "learning_rate": 1e-5,
        "batch_size": 4,
        "num_epochs": 3,
        "dataset_size": 1000,
        "intervention_type": "activation_patching",
        "languages": ["english", "arabic"]
    }
    
    # Initialize experiment
    tracker.init_experiment(
        config=config,
        experiment_name="sample_gender_bias_experiment",
        notes="Demonstration of gender bias analysis with mechanistic interpretability"
    )
    
    # Simulate training metrics
    for epoch in range(3):
        for step in range(10):
            metrics = {
                "loss": 2.5 - (epoch * 10 + step) * 0.01,
                "accuracy": 0.6 + (epoch * 10 + step) * 0.005,
                "learning_rate": config["learning_rate"] * (0.9 ** epoch)
            }
            tracker.log_training_metrics(metrics, step=epoch*10 + step, epoch=epoch)
    
    # Sample bias analysis results
    bias_results = {
        "gender_metrics": {
            "male_accuracy": 0.85,
            "female_accuracy": 0.72,
            "neutral_accuracy": 0.68,
            "bias_gap": 0.13
        },
        "quality_metrics": {
            "bleu_score": 0.45,
            "rouge_score": 0.52,
            "meteor_score": 0.38
        }
    }
    
    tracker.log_bias_analysis(bias_results)
    
    # Sample cross-lingual results
    english_results = {"bias_score": 0.25, "quality_score": 0.78}
    arabic_results = {"bias_score": 0.18, "quality_score": 0.74}
    
    tracker.log_cross_lingual_comparison(english_results, arabic_results)
    
    # Finish experiment
    summary_metrics = {
        "final_bias_score": 0.15,
        "final_quality_score": 0.76,
        "bias_reduction": 0.62
    }
    
    tracker.finish_experiment(summary_metrics)
    
    return tracker


if __name__ == "__main__":
    # Create sample experiment
    tracker = create_sample_experiment()
    print("Sample wandb experiment completed!")

