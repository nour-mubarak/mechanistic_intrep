Activation Extraction Module
============================

This module contains utilities to extract hidden state activations from a
transformer-based vision–language model such as LLaVA or CLIP. It uses the
PyTorch ``register_forward_hook`` mechanism to record the outputs of each
transformer layer during a forward pass. The extracted activations can then
be used for downstream analysis, including sparse autoencoder training,
causal tracing and bias localisation.

Example usage::

    from mechanistic_system.activation_extractor import ActivationExtractor

    extractor = ActivationExtractor(model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda")
    extractor.load_model()
    activations = extractor.extract_activations(
        prompt="Describe the person in the image.",
        image_path="/path/to/image.jpg"
    )
    extractor.save_activations("sample_001", activations)

The activations are stored as a dictionary where keys correspond to layer
identifiers (e.g. ``layer_0``, ``layer_1``) and the values are NumPy arrays
with shape ``(sequence_length, hidden_size)``. A sequence length can vary
depending on the input prompt and the particular model.

Note that this script does not perform any dataset-specific logic. It is
intended to be called from a higher-level orchestration script such as
``pipeline.py``. You should provide your own loop over the dataset and
decide which prompts/images to feed into the model.

"""

from __future__ import annotations

import os
import json
from typing import Dict, Optional, List, Union

import torch
import numpy as np

from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
