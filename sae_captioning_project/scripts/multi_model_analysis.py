#!/usr/bin/env python3
"""
Multi-Model Comparative Bias Analysis Framework
================================================

Analyzes gender bias across multiple vision-language models using SAEs.

Models Supported:
- PaLiGemma (Google) - Currently implemented
- LLaVA-1.5/1.6 (LLaMA-based)
- Qwen-VL (Alibaba)
- InternVL (Shanghai AI Lab)
- Peacock (MBZUAI - Arabic)

Usage:
    python scripts/multi_model_analysis.py --model paligemma --config configs/config.yaml
    python scripts/multi_model_analysis.py --model llava --config configs/config.yaml
    python scripts/multi_model_analysis.py --model qwen-vl --config configs/config.yaml
"""

import torch
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import wandb
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a vision-language model."""
    name: str
    hf_id: str
    model_class: str
    processor_class: str
    vision_encoder_path: str  # Path to vision encoder layers
    language_model_path: str  # Path to language model layers
    num_vision_layers: int
    num_language_layers: int
    hidden_size: int
    arabic_support: str  # 'native', 'good', 'limited'
    requires_auth: bool = False


# Model Registry
MODEL_CONFIGS = {
    'paligemma': ModelConfig(
        name='PaLiGemma-3B',
        hf_id='google/paligemma-3b-pt-224',
        model_class='AutoModelForVision2Seq',
        processor_class='AutoProcessor',
        vision_encoder_path='vision_tower.vision_model.encoder.layers',
        language_model_path='language_model.model.layers',
        num_vision_layers=27,
        num_language_layers=18,
        hidden_size=2048,
        arabic_support='limited',
    ),
    'llava-1.5': ModelConfig(
        name='LLaVA-1.5-7B',
        hf_id='llava-hf/llava-1.5-7b-hf',
        model_class='LlavaForConditionalGeneration',
        processor_class='AutoProcessor',
        vision_encoder_path='vision_tower.vision_model.encoder.layers',
        language_model_path='language_model.model.layers',
        num_vision_layers=24,
        num_language_layers=32,
        hidden_size=4096,
        arabic_support='limited',
    ),
    'llava-1.6': ModelConfig(
        name='LLaVA-1.6-Mistral-7B',
        hf_id='llava-hf/llava-v1.6-mistral-7b-hf',
        model_class='LlavaNextForConditionalGeneration',
        processor_class='LlavaNextProcessor',
        vision_encoder_path='vision_tower.vision_model.encoder.layers',
        language_model_path='language_model.model.layers',
        num_vision_layers=24,
        num_language_layers=32,
        hidden_size=4096,
        arabic_support='limited',
    ),
    'qwen-vl': ModelConfig(
        name='Qwen-VL-Chat',
        hf_id='Qwen/Qwen-VL-Chat',
        model_class='AutoModelForCausalLM',
        processor_class='AutoTokenizer',
        vision_encoder_path='transformer.visual.transformer.resblocks',
        language_model_path='transformer.h',
        num_vision_layers=48,
        num_language_layers=32,
        hidden_size=4096,
        arabic_support='good',
        requires_auth=True,
    ),
    'internvl': ModelConfig(
        name='InternVL-Chat-V1.5',
        hf_id='OpenGVLab/InternVL-Chat-V1-5',
        model_class='AutoModel',
        processor_class='AutoTokenizer',
        vision_encoder_path='vision_model.encoder.layers',
        language_model_path='language_model.model.layers',
        num_vision_layers=48,
        num_language_layers=40,
        hidden_size=5120,
        arabic_support='good',
        requires_auth=True,
    ),
    'peacock': ModelConfig(
        name='Peacock-Arabic',
        hf_id='UBC-NLP/Peacock',
        model_class='AutoModelForCausalLM',
        processor_class='AutoProcessor',
        vision_encoder_path='vision_tower.vision_model.encoder.layers',
        language_model_path='model.layers',
        num_vision_layers=24,
        num_language_layers=32,
        hidden_size=4096,
        arabic_support='native',
    ),
    'mblip': ModelConfig(
        name='mBLIP-mT5-XL',
        hf_id='Gregor/mblip-mt0-xl',
        model_class='Blip2ForConditionalGeneration',
        processor_class='Blip2Processor',
        vision_encoder_path='vision_model.encoder.layers',
        language_model_path='language_model.encoder.block',
        num_vision_layers=39,
        num_language_layers=24,
        hidden_size=2048,
        arabic_support='good',
    ),
}


class VLMAnalyzer(ABC):
    """Abstract base class for vision-language model analysis."""
    
    def __init__(self, model_config: ModelConfig, device: str = 'cuda'):
        self.config = model_config
        self.device = device
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def get_vision_layers(self) -> List:
        """Get list of vision encoder layers."""
        pass
    
    @abstractmethod
    def get_language_layers(self) -> List:
        """Get list of language model layers."""
        pass
    
    @abstractmethod
    def extract_activations(self, images: List, prompts: List, layers: List) -> Dict:
        """Extract activations from specified layers."""
        pass
    
    def register_hooks(self, layers: List, layer_type: str = 'language') -> Tuple[List, Dict]:
        """Register forward hooks to capture activations."""
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                activations[name] = act.detach().cpu()
            return hook
        
        target_layers = self.get_language_layers() if layer_type == 'language' else self.get_vision_layers()
        
        for idx in layers:
            if idx < len(target_layers):
                layer = target_layers[idx]
                h = layer.register_forward_hook(make_hook(f'{layer_type}_layer_{idx}'))
                hooks.append(h)
        
        return hooks, activations


class PaLiGemmaAnalyzer(VLMAnalyzer):
    """Analyzer for PaLiGemma model."""
    
    def load_model(self):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        logger.info(f"Loading {self.config.name}...")
        self.processor = AutoProcessor.from_pretrained(self.config.hf_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.hf_id,
            torch_dtype=torch.float32,
            device_map='auto',
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"Model loaded: {self.config.name}")
    
    def get_vision_layers(self) -> List:
        if hasattr(self.model, 'vision_tower'):
            return list(self.model.vision_tower.vision_model.encoder.layers)
        return []
    
    def get_language_layers(self) -> List:
        if hasattr(self.model, 'language_model'):
            return list(self.model.language_model.model.layers)
        return []
    
    def extract_activations(self, images: List, prompts: List, layers: List, 
                           layer_type: str = 'language') -> Dict:
        hooks, activations = self.register_hooks(layers, layer_type)
        
        try:
            inputs = self.processor(images=images, text=prompts, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()
        
        return activations


class LLaVAAnalyzer(VLMAnalyzer):
    """Analyzer for LLaVA models."""
    
    def load_model(self):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        
        logger.info(f"Loading {self.config.name}...")
        self.processor = AutoProcessor.from_pretrained(self.config.hf_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config.hf_id,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        self.model.eval()
        logger.info(f"Model loaded: {self.config.name}")
    
    def get_vision_layers(self) -> List:
        if hasattr(self.model, 'vision_tower'):
            return list(self.model.vision_tower.vision_model.encoder.layers)
        return []
    
    def get_language_layers(self) -> List:
        if hasattr(self.model, 'language_model'):
            return list(self.model.language_model.model.layers)
        return []
    
    def extract_activations(self, images: List, prompts: List, layers: List,
                           layer_type: str = 'language') -> Dict:
        hooks, activations = self.register_hooks(layers, layer_type)
        
        try:
            # LLaVA uses conversation format
            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts[0]}]}]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=images[0], text=text, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()
        
        return activations


# Factory function
def create_analyzer(model_name: str, device: str = 'cuda') -> VLMAnalyzer:
    """Create appropriate analyzer for the model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    if model_name == 'paligemma':
        return PaLiGemmaAnalyzer(config, device)
    elif model_name in ['llava-1.5', 'llava-1.6']:
        return LLaVAAnalyzer(config, device)
    else:
        raise NotImplementedError(f"Analyzer for {model_name} not yet implemented")


def compare_models(
    models: List[str],
    dataset_path: Path,
    output_dir: Path,
    config: Dict,
    device: str = 'cuda'
) -> Dict:
    """
    Compare bias patterns across multiple models.
    
    Returns:
        Dictionary with comparative analysis results
    """
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            analyzer = create_analyzer(model_name, device)
            analyzer.load_model()
            
            model_config = MODEL_CONFIGS[model_name]
            
            results[model_name] = {
                'name': model_config.name,
                'arabic_support': model_config.arabic_support,
                'num_vision_layers': model_config.num_vision_layers,
                'num_language_layers': model_config.num_language_layers,
                'vision_analysis': {},
                'language_analysis': {},
            }
            
            # TODO: Add full analysis pipeline here
            
        except Exception as e:
            logger.error(f"Failed to analyze {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Bias Analysis')
    parser.add_argument('--model', type=str, default='paligemma',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to analyze')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available models')
    parser.add_argument('--layer-type', type=str, default='language',
                       choices=['vision', 'language', 'both'],
                       help='Which layers to analyze')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize WandB
    wandb.init(
        project=config.get('wandb', {}).get('project', 'sae-captioning-bias'),
        name=f"multi-model-{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            'model': args.model,
            'layer_type': args.layer_type,
            'compare_all': args.compare_all,
        },
        tags=['multi-model', args.model]
    )
    
    if args.compare_all:
        models = list(MODEL_CONFIGS.keys())
        results = compare_models(
            models=models,
            dataset_path=Path(config['paths']['processed_data']),
            output_dir=Path(config['paths']['results']) / 'multi_model',
            config=config,
        )
    else:
        # Single model analysis
        analyzer = create_analyzer(args.model)
        analyzer.load_model()
        
        logger.info(f"\nModel: {args.model}")
        logger.info(f"Vision layers: {len(analyzer.get_vision_layers())}")
        logger.info(f"Language layers: {len(analyzer.get_language_layers())}")
        logger.info(f"Arabic support: {MODEL_CONFIGS[args.model].arabic_support}")
    
    wandb.finish()
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
