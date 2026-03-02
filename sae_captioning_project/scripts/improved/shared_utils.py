#!/usr/bin/env python3
"""
Shared Utilities for Improved Intervention Experiments
======================================================

Provides:
- Gender term counting (English, Arabic, with non-binary support)
- Length normalization (gender terms / total tokens)
- Per-image paired statistics (Δ_targeted vs Δ_random)
- Bootstrap confidence intervals
- SAE hook and model loading
- Unified results saving
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import random
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# ============================================================
# Gender Term Dictionaries
# ============================================================

ENGLISH_GENDER_TERMS = {
    'male': ['man', 'men', 'boy', 'boys', 'he', 'him', 'his', 'male', 'gentleman',
             'father', 'son', 'brother', 'husband', 'guy', 'guys'],
    'female': ['woman', 'women', 'girl', 'girls', 'she', 'her', 'hers', 'female',
               'lady', 'ladies', 'mother', 'daughter', 'sister', 'wife']
}

ARABIC_GENDER_TERMS = {
    'male': ['رجل', 'صبي', 'ولد', 'هو', 'أب', 'ابن', 'أخ', 'زوج', 'شاب', 'سيد'],
    'female': ['امرأة', 'فتاة', 'بنت', 'هي', 'أم', 'ابنة', 'أخت', 'زوجة', 'سيدة']
}

# Non-binary / inclusive terms (for future evaluation)
ENGLISH_NONBINARY_TERMS = {
    'nonbinary': ['they', 'them', 'their', 'person', 'people', 'individual',
                  'human', 'someone', 'child', 'kid', 'kids', 'baby', 'infant',
                  'adult', 'teenager', 'teen', 'elder', 'elderly']
}

# All gendered terms combined (for detection)
ALL_ENGLISH_GENDERED = ENGLISH_GENDER_TERMS['male'] + ENGLISH_GENDER_TERMS['female']


# ============================================================
# Gender Term Counting with Length Normalization
# ============================================================

def count_gender_terms(text, language='english'):
    """Count gender terms in a caption.
    
    Returns dict with 'male', 'female', 'total' counts, 
    'total_tokens' (word count), and 'normalized_rate' (gender terms / total tokens).
    """
    text_lower = text.lower()
    terms = ENGLISH_GENDER_TERMS if language == 'english' else ARABIC_GENDER_TERMS
    
    male_count = sum(1 for term in terms['male'] if term in text_lower)
    female_count = sum(1 for term in terms['female'] if term in text_lower)
    total_gender = male_count + female_count
    
    # Length normalization: count total words
    total_tokens = len(text_lower.split())
    normalized_rate = total_gender / max(total_tokens, 1)
    
    return {
        'male': male_count, 
        'female': female_count, 
        'total': total_gender,
        'total_tokens': total_tokens,
        'normalized_rate': normalized_rate
    }


def count_gender_terms_detailed(text, language='english'):
    """Count each individual gender term for detailed per-term analysis.
    
    Returns dict mapping each term -> count of captions containing it.
    """
    text_lower = text.lower()
    terms = ENGLISH_GENDER_TERMS if language == 'english' else ARABIC_GENDER_TERMS
    
    result = {}
    for category in ['male', 'female']:
        for term in terms[category]:
            if term in text_lower:
                result[term] = result.get(term, 0) + 1
    return result


def count_nonbinary_terms(text):
    """Count gender-neutral terms (future evaluation of non-binary inclusion)."""
    text_lower = text.lower()
    nb_count = sum(1 for term in ENGLISH_NONBINARY_TERMS['nonbinary'] if term in text_lower)
    total_tokens = len(text_lower.split())
    return {
        'nonbinary': nb_count,
        'total_tokens': total_tokens,
        'normalized_rate': nb_count / max(total_tokens, 1)
    }


def has_gender_term(text, language='english'):
    """Check if caption contains any gender term."""
    return count_gender_terms(text, language)['total'] > 0


# ============================================================
# Per-Image Paired Statistics
# ============================================================

def compute_per_image_deltas(baseline_captions, ablated_captions, language='english'):
    """Compute per-image Δ (change in gender terms) for paired statistics.
    
    Returns array of deltas: ablated_count - baseline_count for each image.
    """
    assert len(baseline_captions) == len(ablated_captions), \
        f"Mismatched lengths: {len(baseline_captions)} vs {len(ablated_captions)}"
    
    deltas = []
    normalized_deltas = []
    
    for base_cap, abl_cap in zip(baseline_captions, ablated_captions):
        base_text = base_cap['caption'] if isinstance(base_cap, dict) else base_cap
        abl_text = abl_cap['caption'] if isinstance(abl_cap, dict) else abl_cap
        
        base_counts = count_gender_terms(base_text, language)
        abl_counts = count_gender_terms(abl_text, language)
        
        # Raw delta
        delta = abl_counts['total'] - base_counts['total']
        deltas.append(delta)
        
        # Length-normalized delta
        norm_delta = abl_counts['normalized_rate'] - base_counts['normalized_rate']
        normalized_deltas.append(norm_delta)
    
    return np.array(deltas), np.array(normalized_deltas)


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, statistic=np.mean):
    """Compute bootstrap confidence interval for a statistic.
    
    Returns (point_estimate, ci_lower, ci_upper).
    """
    data = np.asarray(data)
    n = len(data)
    point_est = statistic(data)
    
    # Bootstrap resampling
    rng = np.random.RandomState(42)
    boot_stats = np.array([
        statistic(data[rng.randint(0, n, n)]) for _ in range(n_bootstrap)
    ])
    
    alpha = 1 - ci
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return float(point_est), float(ci_lower), float(ci_upper)


def compute_paired_statistics(baseline_caps, targeted_caps, random_caps_list, language='english'):
    """Compute full paired statistics: per-image Δ for targeted vs each random run.
    
    Args:
        baseline_caps: list of {'caption': str, 'image_path': str}
        targeted_caps: list of {'caption': str, 'image_path': str}
        random_caps_list: list of lists of {'caption': str, 'image_path': str} (one per random run)
    
    Returns comprehensive statistics dict.
    """
    # Targeted deltas
    targeted_deltas, targeted_norm_deltas = compute_per_image_deltas(
        baseline_caps, targeted_caps, language
    )
    
    # Random deltas (per run)
    all_random_deltas = []
    all_random_norm_deltas = []
    for random_caps in random_caps_list:
        rd, rnd = compute_per_image_deltas(baseline_caps, random_caps, language)
        all_random_deltas.append(rd)
        all_random_norm_deltas.append(rnd)
    
    # Mean random delta across runs (per image)
    mean_random_deltas = np.mean(all_random_deltas, axis=0)
    mean_random_norm_deltas = np.mean(all_random_norm_deltas, axis=0)
    
    # Difference: targeted - random (per image)
    diff_deltas = targeted_deltas - mean_random_deltas
    diff_norm_deltas = targeted_norm_deltas - mean_random_norm_deltas
    
    # Bootstrap CIs
    targ_mean, targ_ci_lo, targ_ci_hi = bootstrap_ci(targeted_deltas)
    rand_mean, rand_ci_lo, rand_ci_hi = bootstrap_ci(mean_random_deltas)
    diff_mean, diff_ci_lo, diff_ci_hi = bootstrap_ci(diff_deltas)
    
    # Normalized versions
    targ_norm_mean, targ_norm_ci_lo, targ_norm_ci_hi = bootstrap_ci(targeted_norm_deltas)
    diff_norm_mean, diff_norm_ci_lo, diff_norm_ci_hi = bootstrap_ci(diff_norm_deltas)
    
    # Wilcoxon signed-rank test (non-parametric paired test)
    from scipy.stats import wilcoxon
    try:
        w_stat, w_pval = wilcoxon(targeted_deltas, mean_random_deltas, alternative='less')
    except Exception:
        w_stat, w_pval = float('nan'), float('nan')
    
    return {
        'targeted_delta': {
            'mean': targ_mean,
            'ci_95': [targ_ci_lo, targ_ci_hi],
            'std': float(np.std(targeted_deltas)),
            'n': len(targeted_deltas)
        },
        'random_delta': {
            'mean': rand_mean,
            'ci_95': [rand_ci_lo, rand_ci_hi],
            'std': float(np.std(mean_random_deltas)),
            'n_runs': len(random_caps_list)
        },
        'difference_targeted_minus_random': {
            'mean': diff_mean,
            'ci_95': [diff_ci_lo, diff_ci_hi],
            'std': float(np.std(diff_deltas))
        },
        'normalized': {
            'targeted_rate_delta': {
                'mean': targ_norm_mean,
                'ci_95': [targ_norm_ci_lo, targ_norm_ci_hi]
            },
            'difference': {
                'mean': diff_norm_mean,
                'ci_95': [diff_norm_ci_lo, diff_norm_ci_hi]
            }
        },
        'wilcoxon_test': {
            'statistic': float(w_stat),
            'p_value': float(w_pval),
            'interpretation': 'targeted < random' if w_pval < 0.05 else 'no significant difference'
        }
    }


# ============================================================
# Aggregate Statistics with Length Normalization
# ============================================================

def compute_aggregate_stats(captions, language='english'):
    """Compute aggregate statistics for a set of captions including length normalization."""
    stats = {
        'n_images': len(captions),
        'male_terms': 0,
        'female_terms': 0,
        'total_gender_terms': 0,
        'total_tokens': 0,
        'captions_with_gender': 0,
        'per_term_counts': {},
        'nonbinary_terms': 0,
    }
    
    per_image_counts = []
    per_image_rates = []
    
    for cap_data in captions:
        text = cap_data['caption'] if isinstance(cap_data, dict) else cap_data
        
        counts = count_gender_terms(text, language)
        stats['male_terms'] += counts['male']
        stats['female_terms'] += counts['female']
        stats['total_gender_terms'] += counts['total']
        stats['total_tokens'] += counts['total_tokens']
        
        if counts['total'] > 0:
            stats['captions_with_gender'] += 1
        
        per_image_counts.append(counts['total'])
        per_image_rates.append(counts['normalized_rate'])
        
        # Per-term detail
        detail = count_gender_terms_detailed(text, language)
        for term, cnt in detail.items():
            stats['per_term_counts'][term] = stats['per_term_counts'].get(term, 0) + cnt
        
        # Non-binary terms
        nb = count_nonbinary_terms(text)
        stats['nonbinary_terms'] += nb['nonbinary']
    
    # Length-normalized gender rate
    stats['gender_rate'] = stats['total_gender_terms'] / max(stats['total_tokens'], 1)
    stats['gender_rate_per_caption_mean'] = float(np.mean(per_image_rates))
    stats['gender_rate_per_caption_std'] = float(np.std(per_image_rates))
    stats['gender_count_per_caption_mean'] = float(np.mean(per_image_counts))
    stats['gender_count_per_caption_std'] = float(np.std(per_image_counts))
    
    return stats


# ============================================================
# SAE Hook (supports multiple architectures)
# ============================================================

class SAEHook:
    """Hook to intercept and modify activations through SAE with optional feature ablation."""
    
    def __init__(self, sae, ablate_features=None, ablation_value=0.0):
        self.sae = sae
        self.ablate_features = ablate_features if ablate_features is not None else []
        self.ablation_value = ablation_value
        
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_acts = hidden_states.view(-1, hidden_dim)
        
        with torch.no_grad():
            pre_acts = flat_acts @ self.sae.encoder.weight.T + self.sae.encoder.bias
            sae_acts = torch.relu(pre_acts)
            
            if len(self.ablate_features) > 0:
                sae_acts[:, self.ablate_features] = self.ablation_value
            
            reconstructed = sae_acts @ self.sae.decoder.weight.T + self.sae.decoder.bias
        
        modified = reconstructed.view(batch_size, seq_len, hidden_dim)
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


class SimpleSAE(nn.Module):
    """Simple SAE module matching checkpoint format."""
    def __init__(self, d_model, n_features):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)


# ============================================================
# Model Loading (Multi-model support)
# ============================================================

def load_model_and_sae(model_name, layer, device='cuda'):
    """Load VLM and corresponding trained SAE.
    
    Args:
        model_name: one of 'paligemma', 'qwen2vl', 'llama32vision'
        layer: layer index to load SAE for
        device: torch device
    
    Returns:
        (model, processor, sae, n_features, target_layer, generate_fn)
    """
    base_dir = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')
    
    if model_name == 'paligemma':
        return _load_paligemma(layer, device, base_dir)
    elif model_name == 'qwen2vl':
        return _load_qwen2vl(layer, device, base_dir)
    elif model_name == 'llama32vision':
        return _load_llama32vision(layer, device, base_dir)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'paligemma', 'qwen2vl', or 'llama32vision'")


def _load_paligemma(layer, device, base_dir):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    
    print("Loading PaLiGemma-3B...")
    model_id = "google/paligemma-3b-mix-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    
    sae_path = base_dir / f'checkpoints/saes/sae_english_layer_{layer}.pt'
    sae, n_features = _load_sae(sae_path, device)
    target_layer = model.language_model.layers[layer]
    
    def generate_fn(image, prompt="Caption:"):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        if "Caption:" in caption:
            caption = caption.replace("Caption:", '').strip()
        return caption
    
    return model, processor, sae, n_features, target_layer, generate_fn


def _load_qwen2vl(layer, device, base_dir):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    
    print("Loading Qwen2-VL-7B-Instruct...")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    
    sae_path = base_dir / f'checkpoints/qwen2vl/saes/qwen2vl_sae_english_layer_{layer}.pt'
    sae, n_features = _load_sae(sae_path, device)
    # Qwen2-VL: layers are at model.model.language_model.layers
    target_layer = model.model.language_model.layers[layer]
    
    def generate_fn(image, prompt="Describe this image in English:"):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    
    return model, processor, sae, n_features, target_layer, generate_fn


def _load_llama32vision(layer, device, base_dir):
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    
    print("Loading Llama-3.2-Vision-11B-Instruct...")
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    
    sae_path = base_dir / f'checkpoints/llama32vision/saes/llama32vision_sae_english_layer{layer}.pt'
    sae, n_features = _load_sae(sae_path, device)
    # Llama-3.2-Vision: model.language_model returns MllamaTextModel which has .layers directly
    target_layer = model.language_model.layers[layer]
    
    def generate_fn(image, prompt="Describe this image:"):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        # Strip the conversation prefix
        if "assistant" in caption.lower():
            caption = caption.split("assistant")[-1].strip()
        return caption.strip()
    
    return model, processor, sae, n_features, target_layer, generate_fn


def _load_sae(sae_path, device):
    """Load SAE checkpoint (handles both PaLiGemma/Qwen2-VL and Llama formats)."""
    print(f"Loading SAE from {sae_path}...")
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")
    
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    d_model = checkpoint['d_model']
    
    # Handle different checkpoint formats
    if 'd_hidden' in checkpoint:
        n_features = checkpoint['d_hidden']
    else:
        # Llama SAEs store expansion_factor instead of d_hidden
        n_features = d_model * checkpoint['expansion_factor']
    print(f"SAE dimensions: d_model={d_model}, n_features={n_features}")
    
    sae = SimpleSAE(d_model, n_features).to(device).to(torch.bfloat16)
    
    # Handle different state dict keys
    if 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['state_dict'])
    else:
        raise KeyError(f"No recognized state dict key in checkpoint. Keys: {list(checkpoint.keys())}")
    sae.eval()
    
    return sae, n_features


# ============================================================
# Feature Loading (Multi-model support)
# ============================================================

def load_top_gender_features(layer=9, k=100, model_name='paligemma'):
    """Load top-k gender-associated features from analysis results.
    
    Supports multiple sources:
    1. PaLiGemma: feature_interpretation_results.json or cross_lingual_results.json
    2. Llama-3.2-Vision: gender_features embedded in SAE checkpoint
    3. Qwen2-VL: computed on-the-fly from activation data with gender labels
    """
    base_dir = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')
    
    # --- Source 1: PaLiGemma feature interpretation results ---
    if model_name == 'paligemma':
        results_path = base_dir / 'results/feature_interpretation/feature_interpretation_results.json'
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            for layer_data in data:
                if layer_data['layer'] == layer:
                    male_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['male_associated'][:k//2]]
                    female_features = [f['feature_id'] for f in layer_data['arabic']['top_features']['female_associated'][:k//2]]
                    if len(male_features) + len(female_features) > 0:
                        print(f"Loaded {len(male_features)+len(female_features)} gender features from feature_interpretation_results.json")
                        return (male_features + female_features)[:k]
        
        # Fallback: cross-lingual results
        cross_path = base_dir / 'results/proper_cross_lingual/cross_lingual_results.json'
        if cross_path.exists():
            with open(cross_path) as f:
                data = json.load(f)
            layer_key = str(layer)
            if layer_key in data:
                layer_data = data[layer_key]
                if 'arabic' in layer_data and 'top_features' in layer_data['arabic']:
                    features = layer_data['arabic']['top_features']
                    if isinstance(features, dict):
                        male = features.get('male_associated', [])[:k//2]
                        female = features.get('female_associated', [])[:k//2]
                        result = [f['feature_id'] if isinstance(f, dict) else f for f in male + female][:k]
                        if result:
                            print(f"Loaded {len(result)} gender features from cross_lingual_results.json")
                            return result
    
    # --- Source 2: Llama-3.2-Vision gender features from SAE checkpoint ---
    if model_name == 'llama32vision':
        sae_path = base_dir / f'checkpoints/llama32vision/saes/llama32vision_sae_english_layer{layer}.pt'
        if sae_path.exists():
            checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
            if 'gender_features' in checkpoint:
                gf = checkpoint['gender_features']
                male = gf.get('male_features', [])[:k//2]
                female = gf.get('female_features', [])[:k//2]
                combined = list(set(male + female))[:k]
                print(f"Loaded {len(combined)} gender features from Llama SAE checkpoint")
                return combined
    
    # --- Source 3: Qwen2-VL — compute from activation data ---
    if model_name == 'qwen2vl':
        features = _compute_gender_features_from_activations(
            base_dir, layer, k, model_name='qwen2vl'
        )
        if features:
            return features
    
    # --- Generic fallback: compute from activation data if available ---
    features = _compute_gender_features_from_activations(
        base_dir, layer, k, model_name=model_name
    )
    if features:
        return features
    
    print(f"WARNING: Could not load features for {model_name} layer {layer}, using random features")
    return list(range(k))


def _compute_gender_features_from_activations(base_dir, layer, k, model_name):
    """Compute top gender-associated SAE features from activation data.
    
    Uses stored activations with gender labels to find features with
    highest differential activation between male and female samples.
    """
    # Find activation file
    act_paths = {
        'qwen2vl': base_dir / f'checkpoints/qwen2vl/layer_checkpoints/qwen2vl_layer_{layer}_english.pt',
    }
    act_path = act_paths.get(model_name)
    if act_path is None or not act_path.exists():
        return None
    
    # Find corresponding SAE
    sae_paths = {
        'qwen2vl': base_dir / f'checkpoints/qwen2vl/saes/qwen2vl_sae_english_layer_{layer}.pt',
    }
    sae_path = sae_paths.get(model_name)
    if sae_path is None or not sae_path.exists():
        return None
    
    print(f"Computing gender features for {model_name} layer {layer} from activation data...")
    
    # Load activations and labels
    act_data = torch.load(act_path, map_location='cpu', weights_only=False)
    activations = act_data['activations'].float()
    genders = act_data['genders']
    
    # Load SAE
    sae_checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
    d_model = sae_checkpoint['d_model']
    n_features = sae_checkpoint.get('d_hidden', d_model * sae_checkpoint.get('expansion_factor', 8))
    sae = SimpleSAE(d_model, n_features)
    if 'model_state_dict' in sae_checkpoint:
        sae.load_state_dict(sae_checkpoint['model_state_dict'])
    elif 'state_dict' in sae_checkpoint:
        sae.load_state_dict(sae_checkpoint['state_dict'])
    sae.eval()
    
    # Get SAE encodings
    with torch.no_grad():
        encoded = torch.relu(sae.encoder(activations))  # [N, n_features]
    
    # Split by gender
    male_mask = torch.tensor([g == 'male' for g in genders])
    female_mask = torch.tensor([g == 'female' for g in genders])
    
    if male_mask.sum() == 0 or female_mask.sum() == 0:
        print(f"  Warning: insufficient gender labels (male={male_mask.sum()}, female={female_mask.sum()})")
        return None
    
    male_mean = encoded[male_mask].mean(dim=0)
    female_mean = encoded[female_mask].mean(dim=0)
    
    # Features with highest differential activation
    diff = (male_mean - female_mean).abs()
    top_features = diff.argsort(descending=True)[:k].tolist()
    
    print(f"  Computed {len(top_features)} gender features ({male_mask.sum()} male, {female_mask.sum()} female samples)")
    return top_features


# ============================================================
# Image Loading
# ============================================================

def load_test_images(n_images=500, seed=42):
    """Load test images, shuffled with seed for reproducibility.
    
    Loads from 8,093 available images in data/raw/images/.
    """
    from PIL import Image
    
    data_dir = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/data/raw/images')
    if not data_dir.exists():
        # Fallback
        data_dir = Path('/home2/jmsk62/project/mechanistic_intrep/sae_captioning_project/data/raw/images')
    
    img_files = sorted(data_dir.glob('*.jpg'))
    print(f"Found {len(img_files)} total images")
    
    # Shuffle deterministically
    rng = random.Random(seed)
    img_files = list(img_files)
    rng.shuffle(img_files)
    
    images = []
    image_paths = []
    
    for img_path in img_files[:n_images * 2]:  # try extra in case of errors
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            image_paths.append(str(img_path))
            if len(images) >= n_images:
                break
        except Exception as e:
            continue
    
    print(f"Loaded {len(images)} images for experiment")
    return images, image_paths


# ============================================================
# Results Saving
# ============================================================

def save_results(results, output_path, pretty=True):
    """Save results as JSON, handling non-serializable types."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2 if pretty else None, default=str)
    print(f"Results saved to: {output_path}")
