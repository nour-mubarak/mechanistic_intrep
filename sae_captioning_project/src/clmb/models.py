"""
CLMB Model Registry
===================

Registry of small, memory-efficient vision-language models for bias analysis.
Focuses on models that fit in 28GB GPU memory (NCC constraint).

Model Categories:
1. English-Centric: PaLiGemma, LLaVA-small
2. Multilingual: Qwen-VL, mBLIP
3. Arabic-Native: Peacock, AraLLaVA
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoModel,
    BlipProcessor, 
    BlipForConditionalGeneration,
    LlavaForConditionalGeneration,
)

# Try to import Qwen model class (may not be available in older transformers)
try:
    from transformers import Qwen2VLForConditionalGeneration
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a vision-language model."""
    name: str
    hf_id: str
    category: str  # 'english', 'multilingual', 'arabic'
    vision_encoder: str
    language_model: str
    num_vision_layers: int
    num_language_layers: int
    hidden_size: int
    max_memory_gb: float
    supports_arabic: bool
    model_class: str
    processor_class: str = "AutoProcessor"
    notes: str = ""


# Registry of small, memory-efficient models
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    # ===== English-Centric Models =====
    "paligemma-3b": ModelConfig(
        name="PaLiGemma-3B",
        hf_id="google/paligemma-3b-pt-224",
        category="english",
        vision_encoder="SigLIP-So400m",
        language_model="Gemma-2B",
        num_vision_layers=27,
        num_language_layers=18,
        hidden_size=2048,
        max_memory_gb=12,
        supports_arabic=False,
        model_class="AutoModelForVision2Seq",
        notes="Currently running extraction"
    ),
    
    "llava-1.5-7b": ModelConfig(
        name="LLaVA-1.5-7B",
        hf_id="llava-hf/llava-1.5-7b-hf",
        category="english",
        vision_encoder="CLIP-ViT-L-336",
        language_model="Vicuna-7B",
        num_vision_layers=24,
        num_language_layers=32,
        hidden_size=4096,
        max_memory_gb=16,
        supports_arabic=False,
        model_class="LlavaForConditionalGeneration",
        notes="Popular baseline"
    ),
    
    # ===== Small/Efficient Models (Recommended for NCC) =====
    "blip2-opt-2.7b": ModelConfig(
        name="BLIP-2-OPT-2.7B",
        hf_id="Salesforce/blip2-opt-2.7b",
        category="english",
        vision_encoder="ViT-G/14",
        language_model="OPT-2.7B",
        num_vision_layers=39,
        num_language_layers=32,
        hidden_size=2560,
        max_memory_gb=10,
        supports_arabic=False,
        model_class="Blip2ForConditionalGeneration",
        notes="Memory efficient"
    ),
    
    "git-base": ModelConfig(
        name="GIT-Base",
        hf_id="microsoft/git-base",
        category="english",
        vision_encoder="CLIP-ViT-B",
        language_model="GPT-2",
        num_vision_layers=12,
        num_language_layers=6,
        hidden_size=768,
        max_memory_gb=2,
        supports_arabic=False,
        model_class="AutoModelForCausalLM",
        notes="Very small, good for testing"
    ),
    
    # ===== Multilingual Models =====
    "qwen-vl-chat": ModelConfig(
        name="Qwen-VL-Chat",
        hf_id="Qwen/Qwen-VL-Chat",
        category="multilingual",
        vision_encoder="ViT-G",
        language_model="Qwen-7B",
        num_vision_layers=48,
        num_language_layers=32,
        hidden_size=4096,
        max_memory_gb=18,
        supports_arabic=True,
        model_class="AutoModelForCausalLM",
        notes="Good Arabic via Chinese training"
    ),
    
    "mblip-mt0-xl": ModelConfig(
        name="mBLIP-mT0-XL",
        hf_id="Gregor/mblip-mt0-xl",
        category="multilingual",
        vision_encoder="ViT-G",
        language_model="mT0-XL",
        num_vision_layers=39,
        num_language_layers=24,
        hidden_size=2048,
        max_memory_gb=12,
        supports_arabic=True,
        model_class="Blip2ForConditionalGeneration",
        notes="96 languages including Arabic"
    ),
    
    "mblip-bloomz-7b": ModelConfig(
        name="mBLIP-BLOOMZ-7B",
        hf_id="Gregor/mblip-bloomz-7b",
        category="multilingual",
        vision_encoder="ViT-G",
        language_model="BLOOMZ-7B",
        num_vision_layers=39,
        num_language_layers=30,
        hidden_size=4096,
        max_memory_gb=18,
        supports_arabic=True,
        model_class="Blip2ForConditionalGeneration",
        notes="Strong multilingual"
    ),
    
    # ===== Arabic-Native Models =====
    "peacock-7b": ModelConfig(
        name="Peacock-7B",
        hf_id="UBC-NLP/Peacock",
        category="arabic",
        vision_encoder="CLIP-ViT-L",
        language_model="LLaMA-7B-Arabic",
        num_vision_layers=24,
        num_language_layers=32,
        hidden_size=4096,
        max_memory_gb=16,
        supports_arabic=True,
        model_class="LlavaForConditionalGeneration",
        notes="Arabic-native VLM from MBZUAI"
    ),
    
    "arabic-blip": ModelConfig(
        name="Arabic-BLIP",
        hf_id="omarsabri8756/blip-Arabic-flickr-8k",
        category="arabic",
        vision_encoder="ViT-B",
        language_model="BLIP-Arabic",
        num_vision_layers=12,
        num_language_layers=12,
        hidden_size=768,
        max_memory_gb=3,
        supports_arabic=True,
        model_class="BlipForConditionalGeneration",
        notes="Small Arabic captioning model"
    ),
}


class ModelRegistry:
    """Registry for loading and managing vision-language models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_processors: Dict[str, Any] = {}
    
    @staticmethod
    def list_models(category: Optional[str] = None, 
                    max_memory_gb: Optional[float] = None,
                    supports_arabic: Optional[bool] = None) -> List[str]:
        """List available models with optional filters."""
        models = []
        for name, config in MODEL_CONFIGS.items():
            if category and config.category != category:
                continue
            if max_memory_gb and config.max_memory_gb > max_memory_gb:
                continue
            if supports_arabic is not None and config.supports_arabic != supports_arabic:
                continue
            models.append(name)
        return models
    
    @staticmethod
    def get_config(model_name: str) -> ModelConfig:
        """Get configuration for a model."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        return MODEL_CONFIGS[model_name]
    
    def load_model(self, model_name: str, device: str = "cuda", 
                   dtype: torch.dtype = torch.float16) -> Tuple[Any, Any]:
        """Load a model and its processor."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_processors[model_name]
        
        config = self.get_config(model_name)
        logger.info(f"Loading {config.name} ({config.hf_id})...")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            config.hf_id, 
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        # Load model based on class
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
        }
        
        if config.model_class == "AutoModelForVision2Seq":
            model = AutoModelForVision2Seq.from_pretrained(config.hf_id, **model_kwargs)
        elif config.model_class == "LlavaForConditionalGeneration":
            model = LlavaForConditionalGeneration.from_pretrained(config.hf_id, **model_kwargs)
        elif config.model_class == "BlipForConditionalGeneration":
            model = BlipForConditionalGeneration.from_pretrained(config.hf_id, **model_kwargs)
        elif config.model_class == "Blip2ForConditionalGeneration":
            from transformers import Blip2ForConditionalGeneration
            model = Blip2ForConditionalGeneration.from_pretrained(config.hf_id, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.hf_id, **model_kwargs)
        
        model.eval()
        
        self.loaded_models[model_name] = model
        self.loaded_processors[model_name] = processor
        
        logger.info(f"Loaded {config.name} successfully")
        return model, processor
    
    def get_layer_modules(self, model_name: str, model: Any) -> Dict[str, List]:
        """Get vision and language layer modules for hook registration."""
        config = self.get_config(model_name)
        
        layers = {
            'vision': [],
            'language': [],
            'projection': None
        }
        
        # Model-specific layer access
        if 'paligemma' in model_name:
            if hasattr(model, 'vision_tower'):
                layers['vision'] = list(model.vision_tower.vision_model.encoder.layers)
            if hasattr(model, 'language_model'):
                if hasattr(model.language_model, 'model'):
                    layers['language'] = list(model.language_model.model.layers)
                elif hasattr(model.language_model, 'layers'):
                    layers['language'] = list(model.language_model.layers)
            if hasattr(model, 'multi_modal_projector'):
                layers['projection'] = model.multi_modal_projector
                
        elif 'llava' in model_name:
            if hasattr(model, 'vision_tower'):
                layers['vision'] = list(model.vision_tower.vision_model.encoder.layers)
            if hasattr(model, 'language_model'):
                layers['language'] = list(model.language_model.model.layers)
            if hasattr(model, 'multi_modal_projector'):
                layers['projection'] = model.multi_modal_projector
                
        elif 'blip' in model_name:
            if hasattr(model, 'vision_model'):
                layers['vision'] = list(model.vision_model.encoder.layers)
            if hasattr(model, 'text_decoder'):
                layers['language'] = list(model.text_decoder.bert.encoder.layer)
            if hasattr(model, 'qformer'):
                layers['projection'] = model.qformer
                
        elif 'qwen' in model_name:
            if hasattr(model, 'transformer'):
                # Qwen-VL has interleaved vision-language
                layers['language'] = list(model.transformer.h)
                
        return layers
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.loaded_processors[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded {model_name}")


def get_model_config(model_name: str) -> ModelConfig:
    """Convenience function to get model config."""
    return ModelRegistry.get_config(model_name)


# Alias for import consistency
MODEL_REGISTRY = MODEL_CONFIGS


def get_recommended_models(max_memory_gb: float = 28) -> Dict[str, List[str]]:
    """Get recommended models for each category within memory budget."""
    registry = ModelRegistry()
    
    return {
        'english': registry.list_models(category='english', max_memory_gb=max_memory_gb),
        'multilingual': registry.list_models(category='multilingual', max_memory_gb=max_memory_gb),
        'arabic': registry.list_models(category='arabic', max_memory_gb=max_memory_gb),
    }


if __name__ == "__main__":
    # Print available models
    print("Available Models for CLMB Analysis:")
    print("=" * 60)
    
    for category in ['english', 'multilingual', 'arabic']:
        print(f"\n{category.upper()} Models:")
        for name in ModelRegistry.list_models(category=category):
            config = get_model_config(name)
            arabic = "✓" if config.supports_arabic else "✗"
            print(f"  {name:20} | {config.max_memory_gb:5.1f}GB | Arabic: {arabic} | {config.notes}")
