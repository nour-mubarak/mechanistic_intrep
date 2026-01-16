#!/usr/bin/env python3
"""
Multi-Model Adaptation Template
================================

Template for adapting the cross-lingual SAE analysis pipeline to new models.
Copy this file and modify for your specific model.

Supported model types:
- LLaMA-style (meta-llama/Llama-3.2-11B-Vision)
- Qwen-VL (Qwen/Qwen2-VL-7B-Instruct)
- Gemma (google/gemma-2-9b)
- CLIP-style (openai/clip-vit-large-patch14)
- PaLiGemma (google/paligemma-3b-pt-224) [Current]

Usage:
    python scripts/model_adaptation_template.py --model meta-llama/Llama-3.2-11B-Vision --test
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import argparse


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    # PaLiGemma (current model)
    "google/paligemma-3b-pt-224": {
        "d_model": 2048,
        "num_layers": 26,
        "hook_template": "model.model.layers.{layer_idx}",
        "dtype": torch.float32,
        "model_class": "PaliGemmaForConditionalGeneration",
        "processor_class": "PaliGemmaProcessor",
        "extraction_layers": [0, 3, 6, 9, 12, 15, 17],
    },
    
    # LLaMA 3.2 Vision
    "meta-llama/Llama-3.2-11B-Vision": {
        "d_model": 4096,
        "num_layers": 32,
        "hook_template": "model.layers.{layer_idx}",
        "dtype": torch.bfloat16,
        "model_class": "MllamaForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "extraction_layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
    },
    
    # Qwen2-VL
    "Qwen/Qwen2-VL-7B-Instruct": {
        "d_model": 3584,
        "num_layers": 28,
        "hook_template": "model.layers.{layer_idx}",
        "dtype": torch.bfloat16,
        "model_class": "Qwen2VLForConditionalGeneration",
        "processor_class": "Qwen2VLProcessor",
        "extraction_layers": [0, 4, 8, 12, 16, 20, 24, 27],
    },
    
    # Gemma 2
    "google/gemma-2-9b": {
        "d_model": 3584,
        "num_layers": 42,
        "hook_template": "model.layers.{layer_idx}",
        "dtype": torch.bfloat16,
        "model_class": "Gemma2ForCausalLM",
        "processor_class": "AutoTokenizer",
        "extraction_layers": [0, 6, 12, 18, 24, 30, 36, 41],
    },
    
    # CLIP ViT
    "openai/clip-vit-large-patch14": {
        "d_model": 1024,
        "num_layers": 24,
        "hook_template": "vision_model.encoder.layers.{layer_idx}",
        "dtype": torch.float32,
        "model_class": "CLIPModel",
        "processor_class": "CLIPProcessor",
        "extraction_layers": [0, 4, 8, 12, 16, 20, 23],
    },
    
    # LLaVA 1.5
    "llava-hf/llava-1.5-7b-hf": {
        "d_model": 4096,
        "num_layers": 32,
        "hook_template": "language_model.model.layers.{layer_idx}",
        "dtype": torch.float16,
        "model_class": "LlavaForConditionalGeneration",
        "processor_class": "LlavaProcessor",
        "extraction_layers": [0, 4, 8, 12, 16, 20, 24, 28, 31],
    },
}


# ============================================================================
# ACTIVATION HOOK
# ============================================================================

class ModelActivationHook:
    """Universal activation hook for different model architectures."""
    
    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name, self._infer_config())
        self.activations = {}
        self.hooks = []
        
    def _infer_config(self) -> dict:
        """Infer config for unknown models."""
        print(f"Warning: Unknown model {self.model_name}, inferring config...")
        # Try to infer from model structure
        config = {
            "d_model": 4096,  # Common default
            "num_layers": 32,
            "hook_template": "model.layers.{layer_idx}",
            "dtype": torch.float16,
        }
        
        # Try to find model config
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                config['d_model'] = self.model.config.hidden_size
            if hasattr(self.model.config, 'num_hidden_layers'):
                config['num_layers'] = self.model.config.num_hidden_layers
                
        return config
    
    def _get_hook_target(self, layer_idx: int) -> nn.Module:
        """Get the module to hook based on model architecture."""
        hook_path = self.config['hook_template'].format(layer_idx=layer_idx)
        
        module = self.model
        for attr in hook_path.split('.'):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module
    
    def _hook_fn(self, layer_idx: int) -> Callable:
        """Create hook function for a specific layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[layer_idx] = output.detach().cpu()
        return hook
    
    def register_hooks(self, layers: List[int]):
        """Register hooks for specified layers."""
        self.remove_hooks()
        
        for layer_idx in layers:
            try:
                target = self._get_hook_target(layer_idx)
                hook = target.register_forward_hook(self._hook_fn(layer_idx))
                self.hooks.append(hook)
                print(f"  Registered hook for layer {layer_idx}")
            except Exception as e:
                print(f"  Failed to register hook for layer {layer_idx}: {e}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Return captured activations."""
        return self.activations


# ============================================================================
# MODEL LOADER
# ============================================================================

def load_model_and_processor(model_name: str, device: str = "cuda"):
    """Load model and processor based on model type."""
    config = MODEL_CONFIGS.get(model_name, {})
    dtype = config.get('dtype', torch.float16)
    
    print(f"Loading {model_name}...")
    
    # Special handling for different model types
    if "paligemma" in model_name.lower():
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        processor = PaliGemmaProcessor.from_pretrained(model_name)
        
    elif "llama" in model_name.lower() and "vision" in model_name.lower():
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
    elif "qwen" in model_name.lower() and "vl" in model_name.lower():
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        processor = Qwen2VLProcessor.from_pretrained(model_name)
        
    elif "clip" in model_name.lower():
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        
    elif "llava" in model_name.lower():
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        processor = LlavaProcessor.from_pretrained(model_name)
        
    else:
        # Generic loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except:
            processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model.eval()
    print(f"Loaded {model_name} with dtype {dtype}")
    
    return model, processor


# ============================================================================
# SAE CONFIG GENERATOR
# ============================================================================

def generate_sae_config(model_name: str) -> dict:
    """Generate SAE configuration for a model."""
    config = MODEL_CONFIGS.get(model_name, {"d_model": 4096})
    
    d_model = config['d_model']
    expansion_factor = 8
    
    # Adjust expansion factor for very large models to save memory
    if d_model > 4096:
        expansion_factor = 4
    
    return {
        "d_model": d_model,
        "expansion_factor": expansion_factor,
        "d_hidden": d_model * expansion_factor,
        "l1_coefficient": 5e-4,
        "learning_rate": 3e-4,
        "batch_size": 256 if d_model <= 4096 else 128,
        "epochs": 50,
    }


def generate_config_yaml(model_name: str, output_path: str = None) -> str:
    """Generate a config.yaml for the model."""
    config = MODEL_CONFIGS.get(model_name, {})
    sae_config = generate_sae_config(model_name)
    
    yaml_content = f"""# Configuration for {model_name}
# Auto-generated by model_adaptation_template.py

model:
  name: "{model_name}"
  device: "cuda"
  dtype: "{str(config.get('dtype', torch.float16)).split('.')[-1]}"

sae:
  d_model: {sae_config['d_model']}
  expansion_factor: {sae_config['expansion_factor']}
  l1_coefficient: {sae_config['l1_coefficient']}
  learning_rate: {sae_config['learning_rate']}
  batch_size: {sae_config['batch_size']}
  epochs: {sae_config['epochs']}

layers:
  extraction: {config.get('extraction_layers', [0, 4, 8, 12, 16, 20])}
  analysis: {config.get('extraction_layers', [0, 4, 8, 12, 16, 20])}

languages:
  - arabic
  - english

data:
  images_dir: "data/raw/images"
  captions_file: "data/raw/captions.csv"
  processed_dir: "data/processed"

output:
  checkpoints_dir: "checkpoints"
  results_dir: "results"
"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        print(f"Config saved to {output_path}")
    
    return yaml_content


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_model_setup(model_name: str):
    """Test that a model can be loaded and hooked correctly."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    config = MODEL_CONFIGS.get(model_name, {})
    
    if not config:
        print(f"⚠️  No predefined config for {model_name}")
        print("   Will attempt to infer configuration...")
    
    # Print expected configuration
    print(f"\nExpected Configuration:")
    print(f"  d_model: {config.get('d_model', 'unknown')}")
    print(f"  num_layers: {config.get('num_layers', 'unknown')}")
    print(f"  hook_template: {config.get('hook_template', 'unknown')}")
    print(f"  dtype: {config.get('dtype', 'unknown')}")
    print(f"  extraction_layers: {config.get('extraction_layers', 'unknown')}")
    
    # Generate SAE config
    sae_config = generate_sae_config(model_name)
    print(f"\nSAE Configuration:")
    print(f"  d_model: {sae_config['d_model']}")
    print(f"  d_hidden: {sae_config['d_hidden']}")
    print(f"  expansion_factor: {sae_config['expansion_factor']}")
    
    # Try loading model (optional, requires GPU/memory)
    try_load = input("\nAttempt to load model? (requires GPU) [y/N]: ").lower() == 'y'
    
    if try_load:
        try:
            model, processor = load_model_and_processor(model_name)
            print("✅ Model loaded successfully!")
            
            # Test hook registration
            hook = ModelActivationHook(model, model_name)
            test_layers = config.get('extraction_layers', [0])[:2]
            hook.register_hooks(test_layers)
            print(f"✅ Hooks registered for layers {test_layers}")
            
            hook.remove_hooks()
            del model, processor
            torch.cuda.empty_cache()
            print("✅ Cleanup successful")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Generate config file
    config_path = f"configs/config_{model_name.split('/')[-1].lower()}.yaml"
    generate_config_yaml(model_name, config_path)
    print(f"\n✅ Generated config: {config_path}")
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model adaptation template")
    parser.add_argument("--model", type=str, required=True, help="Model name/path")
    parser.add_argument("--test", action="store_true", help="Run test for model")
    parser.add_argument("--generate-config", action="store_true", help="Generate config.yaml")
    parser.add_argument("--list-models", action="store_true", help="List supported models")
    args = parser.parse_args()
    
    if args.list_models:
        print("\nSupported Models:")
        print("-" * 60)
        for model_name, config in MODEL_CONFIGS.items():
            print(f"  {model_name}")
            print(f"    d_model: {config['d_model']}, layers: {config['num_layers']}")
        return
    
    if args.test:
        test_model_setup(args.model)
    
    if args.generate_config:
        config_path = f"configs/config_{args.model.split('/')[-1].lower()}.yaml"
        generate_config_yaml(args.model, config_path)


if __name__ == "__main__":
    main()
