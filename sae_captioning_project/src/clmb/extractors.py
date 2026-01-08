"""
Multi-Model Extractor
======================

Unified activation extraction across multiple vision-language models.
Handles different model architectures with a consistent interface.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
import gc

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArchitecture:
    """Description of a model's architecture for extraction."""
    name: str
    vision_encoder_path: str  # e.g., "vision_tower.vision_model.encoder.layers"
    projection_path: str  # e.g., "multi_modal_projector"
    language_model_path: str  # e.g., "language_model.model.layers"
    num_vision_layers: int
    num_language_layers: int
    hidden_dim: int
    uses_chat_template: bool = False


# Registry of supported model architectures
MODEL_ARCHITECTURES = {
    "paligemma": ModelArchitecture(
        name="PaLiGemma-3B",
        vision_encoder_path="vision_tower.vision_model.encoder.layers",
        projection_path="multi_modal_projector",
        language_model_path="language_model.model.layers",
        num_vision_layers=27,  # ViT-L
        num_language_layers=18,  # Gemma-2B
        hidden_dim=2048,
        uses_chat_template=False,
    ),
    "llava": ModelArchitecture(
        name="LLaVA-1.5-7B",
        vision_encoder_path="vision_tower.vision_model.encoder.layers",
        projection_path="multi_modal_projector.linear_1",
        language_model_path="language_model.model.layers",
        num_vision_layers=24,  # ViT
        num_language_layers=32,  # Llama
        hidden_dim=4096,
        uses_chat_template=True,
    ),
    "qwen-vl": ModelArchitecture(
        name="Qwen-VL-Chat",
        vision_encoder_path="transformer.visual.transformer.resblocks",
        projection_path="transformer.visual.proj",
        language_model_path="transformer.h",
        num_vision_layers=48,
        num_language_layers=32,
        hidden_dim=4096,
        uses_chat_template=True,
    ),
    "internvl": ModelArchitecture(
        name="InternVL-Chat-V1-5",
        vision_encoder_path="vision_model.encoder.layers",
        projection_path="mlp1",
        language_model_path="language_model.model.layers",
        num_vision_layers=24,
        num_language_layers=40,
        hidden_dim=5120,
        uses_chat_template=True,
    ),
    "mblip": ModelArchitecture(
        name="mBLIP",
        vision_encoder_path="vision_model.encoder.layers",
        projection_path="qformer",
        language_model_path="language_model.model.layers",
        num_vision_layers=24,
        num_language_layers=32,
        hidden_dim=4096,
        uses_chat_template=True,
    ),
}


class BaseModelExtractor(ABC):
    """Base class for model-specific extractors."""
    
    def __init__(
        self,
        model_name: str,
        architecture: ModelArchitecture,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.architecture = architecture
        self.device = device
        self.dtype = dtype
        
        self.model = None
        self.processor = None
        self.hooks: List[Any] = []
        self.activations: Dict[str, List[torch.Tensor]] = {}
    
    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def get_layer_module(self, component: str, layer_idx: int) -> nn.Module:
        """Get a specific layer module for hook registration."""
        pass
    
    def _register_hook(self, module: nn.Module, name: str):
        """Register a forward hook to capture activations."""
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                act = out[0]
            else:
                act = out
            
            # Store as CPU float32 to save GPU memory
            act = act.detach().cpu().float()
            
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(act)
        
        h = module.register_forward_hook(hook)
        self.hooks.append(h)
    
    def register_hooks(
        self,
        vision_layers: Optional[List[int]] = None,
        language_layers: Optional[List[int]] = None,
        include_projection: bool = True
    ):
        """Register hooks on specified layers."""
        self._clear_hooks()
        
        # Vision layers
        if vision_layers:
            for idx in vision_layers:
                if idx < self.architecture.num_vision_layers:
                    module = self.get_layer_module('vision', idx)
                    self._register_hook(module, f'vision_layer_{idx}')
        
        # Projection
        if include_projection:
            proj = self.get_layer_module('projection', 0)
            if proj is not None:
                self._register_hook(proj, 'projection')
        
        # Language layers
        if language_layers:
            for idx in language_layers:
                if idx < self.architecture.num_language_layers:
                    module = self.get_layer_module('language', idx)
                    self._register_hook(module, f'language_layer_{idx}')
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()
    
    def extract_activations(
        self,
        images: List[Any],
        texts: List[str],
        batch_size: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for a batch of image-text pairs.
        
        Returns:
            Dict mapping layer names to concatenated activations.
        """
        self.clear_activations()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Extracting"):
                batch_images = images[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                
                # Process inputs
                inputs = self.processor(
                    images=batch_images,
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                _ = self.model(**inputs)
                
                # Clear GPU memory
                torch.cuda.empty_cache()
        
        # Concatenate activations
        result = {}
        for name, acts in self.activations.items():
            if acts:
                result[name] = torch.cat(acts, dim=0)
        
        return result
    
    def cleanup(self):
        """Clean up resources."""
        self._clear_hooks()
        self.clear_activations()
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        gc.collect()
        torch.cuda.empty_cache()


class PaLiGemmaExtractor(BaseModelExtractor):
    """Extractor for PaLiGemma models."""
    
    def load_model(self):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        
        logger.info(f"Loading PaLiGemma: {self.model_name}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()
    
    def get_layer_module(self, component: str, layer_idx: int) -> nn.Module:
        if component == 'vision':
            return self.model.vision_tower.vision_model.encoder.layers[layer_idx]
        elif component == 'projection':
            return self.model.multi_modal_projector
        elif component == 'language':
            return self.model.language_model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown component: {component}")


class LLaVAExtractor(BaseModelExtractor):
    """Extractor for LLaVA models."""
    
    def load_model(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        logger.info(f"Loading LLaVA: {self.model_name}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
    
    def get_layer_module(self, component: str, layer_idx: int) -> nn.Module:
        if component == 'vision':
            return self.model.vision_tower.vision_model.encoder.layers[layer_idx]
        elif component == 'projection':
            return self.model.multi_modal_projector
        elif component == 'language':
            return self.model.language_model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown component: {component}")


class QwenVLExtractor(BaseModelExtractor):
    """Extractor for Qwen-VL models."""
    
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading Qwen-VL: {self.model_name}")
        
        self.processor = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
    
    def get_layer_module(self, component: str, layer_idx: int) -> nn.Module:
        if component == 'vision':
            return self.model.transformer.visual.transformer.resblocks[layer_idx]
        elif component == 'projection':
            return self.model.transformer.visual.proj
        elif component == 'language':
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unknown component: {component}")


def get_extractor(model_id: str, device: str = "cuda") -> BaseModelExtractor:
    """
    Factory function to get the appropriate extractor for a model.
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on
    
    Returns:
        Appropriate BaseModelExtractor subclass
    """
    model_id_lower = model_id.lower()
    
    if "paligemma" in model_id_lower:
        arch = MODEL_ARCHITECTURES["paligemma"]
        return PaLiGemmaExtractor(model_id, arch, device)
    
    elif "llava" in model_id_lower:
        arch = MODEL_ARCHITECTURES["llava"]
        return LLaVAExtractor(model_id, arch, device)
    
    elif "qwen" in model_id_lower and "vl" in model_id_lower:
        arch = MODEL_ARCHITECTURES["qwen-vl"]
        return QwenVLExtractor(model_id, arch, device)
    
    else:
        logger.warning(f"Unknown model type: {model_id}, using default PaLiGemma extractor")
        arch = MODEL_ARCHITECTURES["paligemma"]
        return PaLiGemmaExtractor(model_id, arch, device)


class MultiModelExtractor:
    """
    Extract activations from multiple models for comparative analysis.
    
    Ensures consistent layer sampling across different architectures.
    """
    
    def __init__(
        self,
        model_ids: List[str],
        output_dir: Path,
        device: str = "cuda"
    ):
        self.model_ids = model_ids
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
    
    def get_comparable_layers(
        self,
        model_id: str,
        num_layers: int = 6
    ) -> Tuple[List[int], List[int]]:
        """
        Get comparable layer indices for a model.
        
        Samples layers evenly to get representative activations.
        
        Returns:
            Tuple of (vision_layers, language_layers)
        """
        extractor = get_extractor(model_id, self.device)
        arch = extractor.architecture
        
        # Sample evenly across layers
        vision_layers = np.linspace(
            0, arch.num_vision_layers - 1, min(num_layers, arch.num_vision_layers)
        ).astype(int).tolist()
        
        language_layers = np.linspace(
            0, arch.num_language_layers - 1, min(num_layers, arch.num_language_layers)
        ).astype(int).tolist()
        
        extractor.cleanup()
        
        return vision_layers, language_layers
    
    def extract_all_models(
        self,
        images: List[Any],
        texts: List[str],
        languages: List[str] = ['english', 'arabic'],
        samples_per_language: int = 1000
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract activations from all registered models.
        
        Returns:
            Dict mapping model_id -> language -> layer_name -> activations
        """
        results = {}
        
        for model_id in self.model_ids:
            logger.info(f"\n{'='*50}")
            logger.info(f"Extracting from: {model_id}")
            logger.info(f"{'='*50}")
            
            try:
                extractor = get_extractor(model_id, self.device)
                extractor.load_model()
                
                # Get layer configuration
                vision_layers, language_layers = self.get_comparable_layers(model_id)
                extractor.register_hooks(vision_layers, language_layers, include_projection=True)
                
                # Extract for each language
                model_results = {}
                for lang in languages:
                    lang_texts = [t for t, l in zip(texts, languages) if l == lang][:samples_per_language]
                    lang_images = [i for i, l in zip(images, languages) if l == lang][:samples_per_language]
                    
                    activations = extractor.extract_activations(lang_images, lang_texts)
                    model_results[lang] = activations
                    
                    # Save to disk
                    save_path = self.output_dir / model_id.replace('/', '_') / lang
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    for layer_name, acts in activations.items():
                        torch.save(acts, save_path / f"{layer_name}.pt")
                
                results[model_id] = model_results
                extractor.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to extract from {model_id}: {e}")
                continue
        
        return results
