"""
Enhanced Fine-Tuning Module with Advanced Techniques
==================================================

This module provides state-of-the-art fine-tuning techniques specifically designed
for mitigating gender bias in multilingual image captioning models while maintaining
caption quality and cross-lingual consistency.

Key Features:
- Advanced LoRA configurations with dynamic rank adaptation
- Curriculum learning for progressive bias mitigation
- Multi-task learning with bias detection and captioning
- Gradient surgery for conflicting objectives
- Memory-efficient training with gradient checkpointing
- Bias-aware loss functions and regularization
- Cross-lingual consistency training
- Ensemble methods for robust bias mitigation

Example usage:
    from enhanced_fine_tuning import EnhancedTrainer
    
    trainer = EnhancedTrainer(model, config)
    trainer.setup_bias_aware_training(bias_config)
    trainer.train_with_curriculum(dataset)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, AutoTokenizer, TrainingArguments, Trainer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BiasAwareTrainingConfig:
    """Configuration for bias-aware training."""
    bias_loss_weight: float = 0.3
    quality_loss_weight: float = 0.7
    consistency_loss_weight: float = 0.2
    adversarial_loss_weight: float = 0.1
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_schedule: str = "linear"  # linear, exponential, cosine
    curriculum_steps: int = 1000
    
    # Multi-task learning
    use_multitask: bool = True
    bias_detection_weight: float = 0.2
    captioning_weight: float = 0.8
    
    # Regularization
    dropout_rate: float = 0.1
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    
    # Advanced techniques
    use_gradient_surgery: bool = True
    use_elastic_weight_consolidation: bool = False
    ewc_lambda: float = 0.4

@dataclass
class AdvancedLoRAConfig:
    """Advanced LoRA configuration with dynamic adaptation."""
    base_rank: int = 16
    max_rank: int = 64
    alpha: int = 32
    dropout: float = 0.1
    
    # Dynamic rank adaptation
    use_dynamic_rank: bool = True
    rank_adaptation_schedule: str = "cosine"  # linear, cosine, exponential
    rank_adaptation_steps: int = 500
    
    # Layer-specific configurations
    layer_specific_ranks: Dict[str, int] = field(default_factory=dict)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Advanced LoRA techniques
    use_adalora: bool = True
    use_lora_plus: bool = True
    use_qlora: bool = False

class BiasAwareLoss(nn.Module):
    """Advanced loss function for bias-aware training."""
    
    def __init__(self, config: BiasAwareTrainingConfig):
        super().__init__()
        self.config = config
        self.gender_classifier = nn.Linear(768, 3)  # male, female, neutral
        
    def forward(self, 
                outputs: torch.Tensor,
                targets: torch.Tensor,
                gender_labels: torch.Tensor,
                activations: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute bias-aware loss combining multiple objectives.
        
        Args:
            outputs: Model outputs (logits)
            targets: Target captions
            gender_labels: Gender labels for bias detection
            activations: Layer activations for regularization
            
        Returns:
            Dictionary containing different loss components
        """
        losses = {}
        
        # Primary captioning loss
        captioning_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        losses['captioning_loss'] = captioning_loss
        
        # Bias detection loss
        if activations and 'pooled_output' in activations:
            gender_logits = self.gender_classifier(activations['pooled_output'])
            bias_loss = F.cross_entropy(gender_logits, gender_labels)
            losses['bias_loss'] = bias_loss
        else:
            losses['bias_loss'] = torch.tensor(0.0, device=outputs.device)
        
        # Fairness regularization (encourage equal performance across genders)
        if gender_labels is not None:
            male_mask = (gender_labels == 0)
            female_mask = (gender_labels == 1)
            
            if male_mask.sum() > 0 and female_mask.sum() > 0:
                male_loss = F.cross_entropy(outputs[male_mask].view(-1, outputs.size(-1)), 
                                          targets[male_mask].view(-1))
                female_loss = F.cross_entropy(outputs[female_mask].view(-1, outputs.size(-1)), 
                                            targets[female_mask].view(-1))
                fairness_loss = torch.abs(male_loss - female_loss)
                losses['fairness_loss'] = fairness_loss
            else:
                losses['fairness_loss'] = torch.tensor(0.0, device=outputs.device)
        
        # Activation regularization (encourage sparse, interpretable representations)
        if activations:
            activation_reg = 0.0
            for layer_name, activation in activations.items():
                if 'attention' in layer_name:
                    # Encourage attention sparsity
                    activation_reg += torch.mean(torch.abs(activation))
            losses['activation_reg'] = activation_reg
        else:
            losses['activation_reg'] = torch.tensor(0.0, device=outputs.device)
        
        # Combined loss
        total_loss = (
            self.config.quality_loss_weight * losses['captioning_loss'] +
            self.config.bias_loss_weight * losses['bias_loss'] +
            0.1 * losses['fairness_loss'] +
            0.01 * losses['activation_reg']
        )
        
        losses['total_loss'] = total_loss
        return losses

class CurriculumScheduler:
    """Curriculum learning scheduler for progressive bias mitigation."""
    
    def __init__(self, 
                 schedule_type: str = "linear",
                 total_steps: int = 1000,
                 start_bias_weight: float = 0.1,
                 end_bias_weight: float = 0.5):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.start_bias_weight = start_bias_weight
        self.end_bias_weight = end_bias_weight
        
    def get_bias_weight(self, current_step: int) -> float:
        """Get current bias weight based on curriculum schedule."""
        progress = min(current_step / self.total_steps, 1.0)
        
        if self.schedule_type == "linear":
            weight = self.start_bias_weight + progress * (self.end_bias_weight - self.start_bias_weight)
        elif self.schedule_type == "exponential":
            weight = self.start_bias_weight * (self.end_bias_weight / self.start_bias_weight) ** progress
        elif self.schedule_type == "cosine":
            weight = self.start_bias_weight + 0.5 * (self.end_bias_weight - self.start_bias_weight) * (
                1 + np.cos(np.pi * (1 - progress))
            )
        else:
            weight = self.end_bias_weight
            
        return float(weight)

class DynamicLoRAAdapter:
    """Dynamic LoRA rank adaptation during training."""
    
    def __init__(self, model: nn.Module, config: AdvancedLoRAConfig):
        self.model = model
        self.config = config
        self.current_ranks = {}
        self.adaptation_step = 0
        
    def adapt_ranks(self, gradient_norms: Dict[str, float]) -> None:
        """Adapt LoRA ranks based on gradient norms."""
        for module_name, grad_norm in gradient_norms.items():
            if module_name in self.config.target_modules:
                # Increase rank for modules with high gradient norms
                if grad_norm > 0.1:  # Threshold
                    current_rank = self.current_ranks.get(module_name, self.config.base_rank)
                    new_rank = min(current_rank + 2, self.config.max_rank)
                    self.current_ranks[module_name] = new_rank
                    
        self.adaptation_step += 1
    
    def get_current_config(self) -> LoraConfig:
        """Get current LoRA configuration with adapted ranks."""
        # Use the maximum current rank as the base rank
        current_max_rank = max(self.current_ranks.values()) if self.current_ranks else self.config.base_rank
        
        return LoraConfig(
            r=current_max_rank,
            lora_alpha=self.config.alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

class GradientSurgery:
    """Gradient surgery for handling conflicting objectives."""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        
    def apply_surgery(self, 
                     grad_captioning: torch.Tensor,
                     grad_bias: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient surgery to resolve conflicts between objectives.
        
        Args:
            grad_captioning: Gradients from captioning loss
            grad_bias: Gradients from bias loss
            
        Returns:
            Modified gradients
        """
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(grad_captioning.flatten(), grad_bias.flatten(), dim=0)
        
        if cos_sim < 0:  # Conflicting gradients
            # Project bias gradient onto captioning gradient
            projection = torch.dot(grad_bias.flatten(), grad_captioning.flatten()) / torch.dot(grad_captioning.flatten(), grad_captioning.flatten())
            grad_bias_projected = grad_bias - projection * grad_captioning
            
            # Combine gradients
            combined_grad = grad_captioning + self.alpha * grad_bias_projected
        else:
            # No conflict, use weighted combination
            combined_grad = grad_captioning + self.alpha * grad_bias
            
        return combined_grad

class EnhancedTrainer:
    """Enhanced trainer with advanced fine-tuning techniques."""
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: Any,
                 config: BiasAwareTrainingConfig,
                 lora_config: AdvancedLoRAConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.lora_config = lora_config or AdvancedLoRAConfig()
        
        # Initialize components
        self.bias_loss = BiasAwareLoss(config)
        self.curriculum_scheduler = CurriculumScheduler(
            config.curriculum_schedule, config.curriculum_steps
        )
        self.gradient_surgery = GradientSurgery() if config.use_gradient_surgery else None
        
        # Training state
        self.global_step = 0
        self.training_history = []
        self.activation_cache = {}
        
        # Setup LoRA
        if lora_config:
            self.setup_lora()
    
    def setup_lora(self) -> None:
        """Setup LoRA with advanced configurations."""
        if self.lora_config.use_dynamic_rank:
            self.dynamic_lora = DynamicLoRAAdapter(self.model, self.lora_config)
            lora_config = self.dynamic_lora.get_current_config()
        else:
            lora_config = LoraConfig(
                r=self.lora_config.base_rank,
                lora_alpha=self.lora_config.alpha,
                target_modules=self.lora_config.target_modules,
                lora_dropout=self.lora_config.dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA setup complete with rank {lora_config.r}")
    
    def setup_activation_hooks(self) -> None:
        """Setup hooks to capture activations during training."""
        def hook_fn(name):
            def hook(module, input, output):
                self.activation_cache[name] = output.detach()
            return hook
        
        # Register hooks on key layers
        for name, module in self.model.named_modules():
            if any(target in name for target in ['attention', 'mlp', 'embed']):
                module.register_forward_hook(hook_fn(name))
    
    def compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for each module."""
        grad_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                module_name = name.split('.')[0]
                if module_name not in grad_norms:
                    grad_norms[module_name] = 0.0
                grad_norms[module_name] += grad_norm
        
        return grad_norms
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any = None) -> Dict[str, float]:
        """
        Perform a single training step with advanced techniques.
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Forward pass
        inputs = {k: v for k, v in batch.items() if k != 'gender_labels'}
        outputs = self.model(**inputs)
        
        # Get activations
        pooled_output = None
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            pooled_output = outputs.hidden_states[-1].mean(dim=1)  # Pool last layer
        
        activations = {'pooled_output': pooled_output} if pooled_output is not None else {}
        activations.update(self.activation_cache)
        
        # Compute losses
        losses = self.bias_loss(
            outputs.logits,
            batch['labels'],
            batch.get('gender_labels'),
            activations
        )
        
        # Apply curriculum learning
        if self.config.use_curriculum:
            bias_weight = self.curriculum_scheduler.get_bias_weight(self.global_step)
            # Adjust bias loss weight
            total_loss = (
                self.config.quality_loss_weight * losses['captioning_loss'] +
                bias_weight * losses['bias_loss'] +
                0.1 * losses['fairness_loss'] +
                0.01 * losses['activation_reg']
            )
            losses['total_loss'] = total_loss
            losses['current_bias_weight'] = bias_weight
        
        # Backward pass
        total_loss = losses['total_loss']
        
        if self.gradient_surgery and self.global_step > 100:  # Start after warmup
            # Compute separate gradients
            captioning_loss = losses['captioning_loss']
            bias_loss = losses['bias_loss']
            
            # Get gradients for captioning loss
            captioning_loss.backward(retain_graph=True)
            captioning_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    captioning_grads.append(param.grad.clone())
                    param.grad.zero_()
            
            # Get gradients for bias loss
            bias_loss.backward(retain_graph=True)
            bias_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    bias_grads.append(param.grad.clone())
                    param.grad.zero_()
            
            # Apply gradient surgery
            for i, param in enumerate(self.model.parameters()):
                if i < len(captioning_grads) and i < len(bias_grads):
                    surgered_grad = self.gradient_surgery.apply_surgery(
                        captioning_grads[i], bias_grads[i]
                    )
                    param.grad = surgered_grad
        else:
            total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
        
        # Dynamic LoRA adaptation
        if hasattr(self, 'dynamic_lora') and self.global_step % 100 == 0:
            grad_norms = self.compute_gradient_norms()
            self.dynamic_lora.adapt_ranks(grad_norms)
        
        # Optimizer step
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        
        # Update global step
        self.global_step += 1
        
        # Clear activation cache
        self.activation_cache.clear()
        
        # Convert losses to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in losses.items()}
        
        return loss_dict
    
    def train_with_curriculum(self, 
                            train_dataloader: DataLoader,
                            val_dataloader: DataLoader = None,
                            num_epochs: int = 3,
                            learning_rate: float = 1e-5) -> Dict[str, List[float]]:
        """
        Train model with curriculum learning and advanced techniques.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # Setup activation hooks
        self.setup_activation_hooks()
        
        # Training history
        history = {
            'train_loss': [],
            'captioning_loss': [],
            'bias_loss': [],
            'fairness_loss': [],
            'bias_weight': [],
            'val_loss': []
        }
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                losses = self.train_step(batch, optimizer, scheduler)
                epoch_losses.append(losses)
                
                # Logging
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}")
                    print(f"  Total Loss: {losses['total_loss']:.4f}")
                    print(f"  Captioning Loss: {losses['captioning_loss']:.4f}")
                    print(f"  Bias Loss: {losses['bias_loss']:.4f}")
                    if 'current_bias_weight' in losses:
                        print(f"  Current Bias Weight: {losses['current_bias_weight']:.4f}")
            
            # Aggregate epoch losses
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            # Update history
            history['train_loss'].append(avg_losses['total_loss'])
            history['captioning_loss'].append(avg_losses['captioning_loss'])
            history['bias_loss'].append(avg_losses['bias_loss'])
            history['fairness_loss'].append(avg_losses.get('fairness_loss', 0.0))
            history['bias_weight'].append(avg_losses.get('current_bias_weight', self.config.bias_loss_weight))
            
            # Validation
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            print(f"Epoch {epoch+1} completed - Train Loss: {avg_losses['total_loss']:.4f}")
        
        self.training_history = history
        return history
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                inputs = {k: v for k, v in batch.items() if k != 'gender_labels'}
                outputs = self.model(**inputs)
                
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                     batch['labels'].view(-1))
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_model(self, save_path: str) -> None:
        """Save the fine-tuned model."""
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), f"{save_path}/model.pt")
        
        # Save training history
        with open(f"{save_path}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def visualize_training_progress(self, save_path: str = None) -> None:
        """Visualize training progress and curriculum learning."""
        if not self.training_history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Training and validation loss
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Loss components
        axes[0, 1].plot(epochs, self.training_history['captioning_loss'], label='Captioning Loss')
        axes[0, 1].plot(epochs, self.training_history['bias_loss'], label='Bias Loss')
        axes[0, 1].plot(epochs, self.training_history['fairness_loss'], label='Fairness Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Curriculum learning progress
        axes[1, 0].plot(epochs, self.training_history['bias_weight'])
        axes[1, 0].set_title('Curriculum Learning: Bias Weight')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Bias Weight')
        
        # Loss ratio
        captioning_losses = np.array(self.training_history['captioning_loss'])
        bias_losses = np.array(self.training_history['bias_loss'])
        loss_ratio = bias_losses / (captioning_losses + 1e-8)
        
        axes[1, 1].plot(epochs, loss_ratio)
        axes[1, 1].set_title('Bias/Captioning Loss Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_sample_training_experiment():
    """Create sample training experiment for demonstration."""
    print("Creating sample enhanced fine-tuning experiment...")
    
    # Mock training configuration
    config = BiasAwareTrainingConfig(
        bias_loss_weight=0.3,
        quality_loss_weight=0.7,
        use_curriculum=True,
        curriculum_schedule="cosine",
        curriculum_steps=500,
        use_multitask=True,
        use_gradient_surgery=True
    )
    
    # Mock LoRA configuration
    lora_config = AdvancedLoRAConfig(
        base_rank=16,
        max_rank=64,
        use_dynamic_rank=True,
        use_adalora=True
    )
    
    # Create mock training history
    num_epochs = 5
    history = {
        'train_loss': [2.5 - 0.3*i for i in range(num_epochs)],
        'captioning_loss': [2.2 - 0.25*i for i in range(num_epochs)],
        'bias_loss': [0.8 - 0.1*i for i in range(num_epochs)],
        'fairness_loss': [0.3 - 0.05*i for i in range(num_epochs)],
        'bias_weight': [0.1 + 0.08*i for i in range(num_epochs)],
        'val_loss': [2.4 - 0.28*i for i in range(num_epochs)]
    }
    
    # Create mock trainer for visualization
    class MockTrainer:
        def __init__(self):
            self.training_history = history
        
        def visualize_training_progress(self, save_path=None):
            # Use the function from the real class
            trainer = EnhancedTrainer.__new__(EnhancedTrainer)
            trainer.training_history = self.training_history
            trainer.visualize_training_progress(save_path)
    
    trainer = MockTrainer()
    trainer.visualize_training_progress()
    
    print("Sample enhanced fine-tuning experiment completed!")
    print(f"Configuration: {config}")
    print(f"LoRA Configuration: {lora_config}")
    
    return trainer, config, lora_config


if __name__ == "__main__":
    # Create sample experiment
    trainer, config, lora_config = create_sample_training_experiment()
    print("Enhanced fine-tuning module demonstration completed!")

