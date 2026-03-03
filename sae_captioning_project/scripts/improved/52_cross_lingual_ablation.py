#!/usr/bin/env python3
"""
52_cross_lingual_ablation.py — Cross-Lingual Causal Intervention
=================================================================

Justification:
  Our paper claims near-zero cross-lingual feature overlap (CLBAS < 0.015),
  but this is correlational (feature overlap analysis), not causal.
  A reviewer will ask: "If you ablate English gender features, does it
  actually leave Arabic gender language untouched?"

  This experiment provides the causal evidence by running a 2×2 design:
    Feature source (EN vs AR) × Caption language (EN vs AR)
  
  Expected results if CLBAS ≈ 0 (features are language-specific):
    - Same-language ablation (EN→EN, AR→AR): significant effect
    - Cross-language ablation (EN→AR, AR→EN): minimal/no effect

Conditions:
  1. EN→EN: English SAE, English gender features, generate English captions
             (this is the existing experiment — we reuse the baseline+targeted)
  2. EN→AR: English SAE, English gender features, generate ARABIC captions
             (cross-lingual transfer test)
  3. AR→EN: Arabic SAE, Arabic gender features, generate ENGLISH captions
             (cross-lingual transfer test, reverse)  
  4. AR→AR: Arabic SAE, Arabic gender features, generate ARABIC captions
             (within-language Arabic control)

Models:
  - Qwen2-VL-7B (natively multilingual, strong Arabic support)
  - PaLiGemma-3B (can generate Arabic via prompt)

Usage:
  python scripts/improved/52_cross_lingual_ablation.py \
      --model qwen2vl --layer 12 --n_images 500 --n_random_runs 10

  python scripts/improved/52_cross_lingual_ablation.py \
      --model paligemma --layer 9 --n_images 500 --n_random_runs 10
"""

import os
import sys
import json
import torch
import numpy as np
import random
import argparse
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from shared_utils import (
    count_gender_terms, count_gender_terms_detailed,
    has_gender_term, compute_per_image_deltas, bootstrap_ci,
    compute_paired_statistics, compute_aggregate_stats,
    SAEHook, SimpleSAE, load_test_images, save_results,
    ARABIC_GENDER_TERMS, ENGLISH_GENDER_TERMS
)


# ============================================================
# Arabic Gender Feature Loading
# ============================================================

def load_arabic_gender_features(model_name, layer, k=100):
    """Load top-k Arabic gender features from stored data.
    
    Justification: Arabic features are identified independently from Arabic 
    activation data using the Arabic SAE — they are NOT translated English features.
    
    Strategy (memory-efficient):
      1. PaLiGemma: use pre-computed feature_stats CSV (no activation loading)
      2. Llama: use pre-computed gender_features JSON
      3. Qwen2-VL: compute from activation data (smaller activations)
    """
    import csv
    base_dir = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')
    
    # --- Strategy 1: Pre-computed feature stats CSV (PaLiGemma) ---
    # These files have |gender_diff| per feature, no need to load activations
    csv_path = base_dir / f'results/feature_stats_layer_{layer}_arabic.csv'
    if csv_path.exists() and model_name == 'paligemma':
        print(f"  Loading Arabic features from pre-computed stats: {csv_path.name}")
        features = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    features.append((int(row['feature_idx']), abs(float(row['gender_diff']))))
                except (ValueError, KeyError):
                    continue
        features.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in features[:k]]
        print(f"    Selected {len(top_features)} Arabic gender features from CSV")
        return top_features
    
    # --- Strategy 2: Pre-computed JSON (Llama) ---
    if model_name == 'llama32vision':
        json_path = base_dir / f'checkpoints/llama32vision/saes/llama32vision_gender_features_arabic_layer{layer}.json'
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            male = data.get('male_features', [])[:k//2]
            female = data.get('female_features', [])[:k//2]
            combined = list(set(male + female))[:k]
            print(f"  Loaded {len(combined)} Arabic gender features from {json_path.name}")
            return combined
    
    # --- Strategy 3: Compute from activation data (Qwen2-VL) ---
    act_paths = {
        'qwen2vl': base_dir / f'checkpoints/qwen2vl/layer_checkpoints/qwen2vl_layer_{layer}_arabic.pt',
        'paligemma': base_dir / f'checkpoints/full_layers_ncc/layer_checkpoints/layer_{layer}_arabic.pt',
    }
    sae_paths = {
        'qwen2vl': base_dir / f'checkpoints/qwen2vl/saes/qwen2vl_sae_arabic_layer_{layer}.pt',
        'paligemma': base_dir / f'checkpoints/saes/sae_arabic_layer_{layer}.pt',
    }
    
    act_path = act_paths.get(model_name)
    sae_path = sae_paths.get(model_name)
    
    if act_path is None or sae_path is None:
        raise ValueError(f"No Arabic data paths for model {model_name}")
    if not act_path.exists():
        raise FileNotFoundError(f"Arabic activation data not found: {act_path}")
    if not sae_path.exists():
        raise FileNotFoundError(f"Arabic SAE not found: {sae_path}")
    
    print(f"  Computing Arabic gender features from {act_path.name}...")
    
    # Load activations + labels
    act_data = torch.load(act_path, map_location='cpu', weights_only=False)
    activations = act_data['activations'].float()
    genders = act_data['genders']
    
    # Load Arabic SAE
    sae_checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
    d_model = sae_checkpoint['d_model']
    n_features = sae_checkpoint.get('d_hidden', d_model * sae_checkpoint.get('expansion_factor', 8))
    
    sae = SimpleSAE(d_model, n_features)
    if 'model_state_dict' in sae_checkpoint:
        sae.load_state_dict(sae_checkpoint['model_state_dict'])
    elif 'state_dict' in sae_checkpoint:
        sae.load_state_dict(sae_checkpoint['state_dict'])
    sae.eval()
    
    # Encode and compute differential activation
    with torch.no_grad():
        encoded = torch.relu(sae.encoder(activations))
    
    male_mask = torch.tensor([g == 'male' for g in genders])
    female_mask = torch.tensor([g == 'female' for g in genders])
    
    print(f"    Arabic data: {male_mask.sum().item()} male, {female_mask.sum().item()} female")
    
    male_mean = encoded[male_mask].mean(dim=0)
    female_mean = encoded[female_mask].mean(dim=0)
    
    diff = (male_mean - female_mean).abs()
    top_features = diff.argsort(descending=True)[:k].tolist()
    
    print(f"    Selected {len(top_features)} Arabic gender features")
    
    # Free memory immediately
    del act_data, activations, genders, sae_checkpoint, sae, encoded
    del male_mask, female_mask, male_mean, female_mean, diff
    gc.collect()
    
    return top_features


def load_english_gender_features(model_name, layer, k=100):
    """Load top-k English gender features (delegates to shared_utils, with CSV fallback).
    
    Justification: These are the same features used in the existing EN→EN experiments.
    """
    import csv
    base_dir = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')
    
    # Try shared_utils first
    try:
        from shared_utils import load_top_gender_features
        features = load_top_gender_features(layer, k, model_name)[:k]
        if features and features != list(range(k)):  # not the fallback random
            print(f"  Loaded {len(features)} English gender features via shared_utils")
            return features
    except Exception as e:
        print(f"  shared_utils fallback: {e}")
    
    # Fallback: pre-computed CSV
    csv_path = base_dir / f'results/feature_stats_layer_{layer}_english.csv'
    if csv_path.exists() and model_name == 'paligemma':
        print(f"  Loading English features from pre-computed stats: {csv_path.name}")
        features = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    features.append((int(row['feature_idx']), abs(float(row['gender_diff']))))
                except (ValueError, KeyError):
                    continue
        features.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in features[:k]]
        print(f"    Selected {len(top_features)} English gender features from CSV")
        return top_features
    
    raise RuntimeError(f"Could not load English gender features for {model_name} layer {layer}")


# ============================================================
# Model Loading with Bilingual Generation
# ============================================================

def load_model_bilingual(model_name, layer, device='cuda'):
    """Load model with English AND Arabic SAEs + bilingual generate functions.
    
    Returns:
        model, processor, english_sae, arabic_sae, n_features_en, n_features_ar,
        target_layer_en, target_layer_ar, generate_en_fn, generate_ar_fn
    
    Justification: We need the same model to generate in both languages.
    The English and Arabic SAEs are trained on different activation data (from 
    different language prompts), so they capture language-specific features.
    Both SAEs hook into the SAME decoder layer.
    """
    base_dir = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')
    
    if model_name == 'qwen2vl':
        return _load_qwen2vl_bilingual(layer, device, base_dir)
    elif model_name == 'paligemma':
        return _load_paligemma_bilingual(layer, device, base_dir)
    else:
        raise ValueError(f"Cross-lingual ablation not supported for {model_name}")


def _load_sae_from_path(sae_path, device):
    """Load SAE from checkpoint path."""
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE not found: {sae_path}")
    
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    d_model = checkpoint['d_model']
    
    if 'd_hidden' in checkpoint:
        n_features = checkpoint['d_hidden']
    else:
        n_features = d_model * checkpoint.get('expansion_factor', 8)
    
    sae = SimpleSAE(d_model, n_features).to(device).to(torch.bfloat16)
    
    if 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['state_dict'])
    sae.eval()
    
    return sae, n_features


def _load_qwen2vl_bilingual(layer, device, base_dir):
    """Load Qwen2-VL with bilingual generation support."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    
    print("Loading Qwen2-VL-7B-Instruct (bilingual)...")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    
    # English SAE (same as existing experiment)
    en_sae_path = base_dir / f'checkpoints/qwen2vl/saes/qwen2vl_sae_english_layer_{layer}.pt'
    print(f"  Loading English SAE: {en_sae_path.name}")
    en_sae, n_features_en = _load_sae_from_path(en_sae_path, device)
    
    # Arabic SAE
    ar_sae_path = base_dir / f'checkpoints/qwen2vl/saes/qwen2vl_sae_arabic_layer_{layer}.pt'
    print(f"  Loading Arabic SAE: {ar_sae_path.name}")
    ar_sae, n_features_ar = _load_sae_from_path(ar_sae_path, device)
    
    # Both hook into the same decoder layer
    target_layer = model.model.language_model.layers[layer]
    
    def generate_en(image, prompt="Describe this image in English:"):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    
    def generate_ar(image, prompt="صف هذه الصورة بالعربية:"):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    
    return (model, processor, en_sae, ar_sae, n_features_en, n_features_ar,
            target_layer, target_layer, generate_en, generate_ar)


def _load_paligemma_bilingual(layer, device, base_dir):
    """Load PaLiGemma with bilingual generation support."""
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    
    print("Loading PaLiGemma-3B (bilingual)...")
    model_id = "google/paligemma-3b-mix-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    
    # English SAE
    en_sae_path = base_dir / f'checkpoints/saes/sae_english_layer_{layer}.pt'
    print(f"  Loading English SAE: {en_sae_path.name}")
    en_sae, n_features_en = _load_sae_from_path(en_sae_path, device)
    
    # Arabic SAE
    ar_sae_path = base_dir / f'checkpoints/saes/sae_arabic_layer_{layer}.pt'
    print(f"  Loading Arabic SAE: {ar_sae_path.name}")
    ar_sae, n_features_ar = _load_sae_from_path(ar_sae_path, device)
    
    target_layer = model.language_model.layers[layer]
    
    def generate_en(image, prompt="Caption:"):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        if "Caption:" in caption:
            caption = caption.replace("Caption:", '').strip()
        return caption
    
    def generate_ar(image, prompt="وصف الصورة بالعربية:"):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        # Strip prompt prefix
        for prefix in ["وصف الصورة بالعربية:", "Caption:"]:
            if prefix in caption:
                caption = caption.split(prefix)[-1].strip()
        return caption
    
    return (model, processor, en_sae, ar_sae, n_features_en, n_features_ar,
            target_layer, target_layer, generate_en, generate_ar)


# ============================================================
# Experiment Runner
# ============================================================

def run_condition(generate_fn, sae, target_layer, images, image_paths,
                  ablate_features, language, label, n_random_runs=10):
    """Run a single cross-lingual condition (baseline + targeted + random).
    
    Args:
        generate_fn: function that generates captions in the target language
        sae: the SAE to use for ablation (English or Arabic)
        target_layer: the model layer to hook
        ablate_features: list of feature indices to ablate
        language: 'english' or 'arabic' — for counting gender terms
        label: descriptive label for this condition
        n_random_runs: number of random ablation control runs
    
    Returns:
        dict with baseline, targeted, random results + statistics
    """
    n_features = sae.encoder.weight.shape[0]  # total features in SAE
    
    print(f"\n{'='*60}")
    print(f"CONDITION: {label}")
    print(f"  Language: {language} | Features: {len(ablate_features)} | SAE dims: {n_features}")
    print(f"{'='*60}")
    
    # --- Baseline (no ablation) ---
    print(f"\n  [1/3] Baseline ({language} captions, no hook)...")
    baseline_caps = []
    for image, path in tqdm(zip(images, image_paths), total=len(images), desc=f"{label} baseline"):
        caption = generate_fn(image)
        baseline_caps.append({'image_path': path, 'caption': caption})
    baseline_stats = compute_aggregate_stats(baseline_caps, language)
    print(f"    Baseline: {baseline_stats['total_gender_terms']} gender terms in {baseline_stats['n_images']} images")
    
    # --- Targeted ablation ---
    print(f"\n  [2/3] Targeted ablation ({len(ablate_features)} features)...")
    hook = SAEHook(sae, ablate_features=ablate_features)
    handle = target_layer.register_forward_hook(hook)
    
    targeted_caps = []
    try:
        for image, path in tqdm(zip(images, image_paths), total=len(images), desc=f"{label} targeted"):
            caption = generate_fn(image)
            targeted_caps.append({'image_path': path, 'caption': caption})
    finally:
        handle.remove()
    targeted_stats = compute_aggregate_stats(targeted_caps, language)
    print(f"    Targeted: {targeted_stats['total_gender_terms']} gender terms")
    
    # --- Random ablation controls ---
    print(f"\n  [3/3] Random ablation ({n_random_runs} runs)...")
    available_features = list(set(range(n_features)) - set(ablate_features))
    k = len(ablate_features)
    
    random_caps_list = []
    random_stats_list = []
    for run_idx in range(n_random_runs):
        rng = random.Random(42 + run_idx * 1000)
        random_features = rng.sample(available_features, min(k, len(available_features)))
        
        hook = SAEHook(sae, ablate_features=random_features)
        handle = target_layer.register_forward_hook(hook)
        
        caps = []
        try:
            for image, path in tqdm(zip(images, image_paths), total=len(images),
                                     desc=f"{label} random {run_idx+1}/{n_random_runs}"):
                caption = generate_fn(image)
                caps.append({'image_path': path, 'caption': caption})
        finally:
            handle.remove()
        
        stats = compute_aggregate_stats(caps, language)
        random_caps_list.append(caps)
        random_stats_list.append(stats)
        print(f"    Run {run_idx+1}: {stats['total_gender_terms']} gender terms")
    
    # --- Compute paired statistics ---
    paired_stats = compute_paired_statistics(
        baseline_caps, targeted_caps, random_caps_list, language
    )
    
    baseline_total = baseline_stats['total_gender_terms']
    targeted_total = targeted_stats['total_gender_terms']
    targeted_change_pct = (targeted_total - baseline_total) / max(baseline_total, 1) * 100
    
    random_totals = [s['total_gender_terms'] for s in random_stats_list]
    random_changes = [(t - baseline_total) / max(baseline_total, 1) * 100 for t in random_totals]
    random_mean_pct = float(np.mean(random_changes))
    random_std_pct = float(np.std(random_changes))
    
    effect_ratio = abs(targeted_change_pct) / max(abs(random_mean_pct), 0.001)
    
    result = {
        'label': label,
        'language': language,
        'n_images': len(images),
        'n_features_ablated': len(ablate_features),
        'n_random_runs': n_random_runs,
        'baseline_gender_terms': baseline_total,
        'targeted_gender_terms': targeted_total,
        'targeted_change_pct': targeted_change_pct,
        'random_change_mean_pct': random_mean_pct,
        'random_change_std_pct': random_std_pct,
        'effect_specificity_pct': targeted_change_pct - random_mean_pct,
        'effect_ratio': effect_ratio,
        'paired_statistics': paired_stats,
        'baseline_stats': baseline_stats,
        'targeted_stats': targeted_stats,
        'random_stats_list': random_stats_list,
        'baseline_per_term': baseline_stats['per_term_counts'],
        'targeted_per_term': targeted_stats['per_term_counts'],
        # Sample captions for verification
        'sample_captions': {
            'baseline': [c['caption'] for c in baseline_caps[:5]],
            'targeted': [c['caption'] for c in targeted_caps[:5]],
        }
    }
    
    # Summary print
    print(f"\n  === {label} Summary ===")
    print(f"    Baseline: {baseline_total} gender terms")
    print(f"    Targeted: {targeted_total} ({targeted_change_pct:+.1f}%)")
    print(f"    Random:   mean {random_mean_pct:+.1f}% ± {random_std_pct:.1f}%")
    print(f"    Specificity: {targeted_change_pct - random_mean_pct:+.1f} pp")
    print(f"    Ratio: {effect_ratio:.1f}×")
    ci = paired_stats['difference_targeted_minus_random']['ci_95']
    print(f"    Bootstrap CI (targeted-random): [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    sig = "SIGNIFICANT" if ci[0] > 0 or ci[1] < 0 else "not significant"
    print(f"    Significance: {sig}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Lingual Ablation Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cross-lingual ablation: tests whether gender features are language-specific.

Conditions:
  EN→EN: English features ablated, English captions generated  (within-language)
  EN→AR: English features ablated, Arabic captions generated   (cross-lingual)
  AR→EN: Arabic features ablated, English captions generated   (cross-lingual)
  AR→AR: Arabic features ablated, Arabic captions generated    (within-language)

Expected: Within-language conditions show significant effects;
          Cross-language conditions show minimal effects.
        """
    )
    parser.add_argument('--model', type=str, required=True,
                       choices=['qwen2vl', 'paligemma'],
                       help='Model to use (must support bilingual generation)')
    parser.add_argument('--layer', type=int, required=True,
                       help='Layer to intervene on (e.g., 12 for qwen2vl, 9 for paligemma)')
    parser.add_argument('--k', type=int, default=100,
                       help='Number of features to ablate')
    parser.add_argument('--n_images', type=int, default=500,
                       help='Number of test images')
    parser.add_argument('--n_random_runs', type=int, default=10,
                       help='Random ablation runs per condition (10 for efficiency)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--conditions', nargs='+',
                       default=['EN_EN', 'EN_AR', 'AR_EN', 'AR_AR'],
                       help='Which conditions to run (default: all 4)')
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Output directory
    base = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project/results/cross_lingual_ablation')
    output_dir = base / f'{args.model}_L{args.layer}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'experiment': 'cross_lingual_ablation',
        'model': args.model,
        'layer': args.layer,
        'k': args.k,
        'n_images': args.n_images,
        'n_random_runs': args.n_random_runs,
        'conditions': args.conditions,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'justification': (
            'Tests whether gender bias features are language-specific by ablating '
            'English gender features during Arabic generation (and vice versa). '
            'If CLBAS ≈ 0, cross-lingual ablation should have minimal effect.'
        )
    }
    
    print("=" * 70)
    print("CROSS-LINGUAL ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  k: {args.k}")
    print(f"  Images: {args.n_images}")
    print(f"  Random runs: {args.n_random_runs}")
    print(f"  Conditions: {args.conditions}")
    print("=" * 70)
    
    # Load images
    images, image_paths = load_test_images(args.n_images, seed=args.seed)
    
    # Load gender features BEFORE loading VLM (to avoid OOM)
    # The feature loading may require Arabic activation data which is large
    print("\n--- Loading Gender Features (before VLM to save memory) ---")
    en_features = load_english_gender_features(args.model, args.layer, args.k)
    ar_features = load_arabic_gender_features(args.model, args.layer, args.k)
    gc.collect()
    
    # Check overlap (should be near-zero if CLBAS ≈ 0)
    overlap = set(en_features) & set(ar_features)
    print(f"\n  EN-AR feature overlap: {len(overlap)}/{args.k} ({100*len(overlap)/args.k:.1f}%)")
    print(f"  Jaccard index: {len(overlap)/len(set(en_features)|set(ar_features)):.3f}")
    
    config['feature_overlap'] = {
        'overlap_count': len(overlap),
        'overlap_pct': 100 * len(overlap) / args.k,
        'jaccard': len(overlap) / len(set(en_features) | set(ar_features)),
        'en_features_sample': en_features[:10],
        'ar_features_sample': ar_features[:10],
    }
    
    # Now load the VLM with bilingual support
    print("\n--- Loading VLM ---")
    (model, processor, en_sae, ar_sae, n_features_en, n_features_ar,
     target_layer_en, target_layer_ar, generate_en, generate_ar) = \
        load_model_bilingual(args.model, args.layer, args.device)
    
    # ============================================================
    # Run all conditions
    # ============================================================
    results = {'config': config, 'conditions': {}}
    
    # Condition 1: EN→EN (English features, English captions)
    if 'EN_EN' in args.conditions:
        results['conditions']['EN_EN'] = run_condition(
            generate_fn=generate_en,
            sae=en_sae,
            target_layer=target_layer_en,
            images=images,
            image_paths=image_paths,
            ablate_features=en_features,
            language='english',
            label='EN→EN (within-language)',
            n_random_runs=args.n_random_runs
        )
        gc.collect()
        torch.cuda.empty_cache()
    
    # Condition 2: EN→AR (English features, Arabic captions)
    if 'EN_AR' in args.conditions:
        results['conditions']['EN_AR'] = run_condition(
            generate_fn=generate_ar,
            sae=en_sae,
            target_layer=target_layer_en,  # Same layer, English SAE
            images=images,
            image_paths=image_paths,
            ablate_features=en_features,
            language='arabic',
            label='EN→AR (cross-lingual)',
            n_random_runs=args.n_random_runs
        )
        gc.collect()
        torch.cuda.empty_cache()
    
    # Condition 3: AR→EN (Arabic features, English captions)
    if 'AR_EN' in args.conditions:
        results['conditions']['AR_EN'] = run_condition(
            generate_fn=generate_en,
            sae=ar_sae,
            target_layer=target_layer_ar,  # Same layer, Arabic SAE
            images=images,
            image_paths=image_paths,
            ablate_features=ar_features,
            language='english',
            label='AR→EN (cross-lingual)',
            n_random_runs=args.n_random_runs
        )
        gc.collect()
        torch.cuda.empty_cache()
    
    # Condition 4: AR→AR (Arabic features, Arabic captions)
    if 'AR_AR' in args.conditions:
        results['conditions']['AR_AR'] = run_condition(
            generate_fn=generate_ar,
            sae=ar_sae,
            target_layer=target_layer_ar,  # Same layer, Arabic SAE
            images=images,
            image_paths=image_paths,
            ablate_features=ar_features,
            language='arabic',
            label='AR→AR (within-language)',
            n_random_runs=args.n_random_runs
        )
        gc.collect()
        torch.cuda.empty_cache()
    
    # ============================================================
    # Cross-lingual summary
    # ============================================================
    print("\n" + "=" * 70)
    print("CROSS-LINGUAL ABLATION SUMMARY")
    print("=" * 70)
    
    summary_table = []
    for cond_key, cond_data in results['conditions'].items():
        ci = cond_data['paired_statistics']['difference_targeted_minus_random']['ci_95']
        sig = ci[0] > 0 or ci[1] < 0
        summary_table.append({
            'condition': cond_key,
            'label': cond_data['label'],
            'baseline': cond_data['baseline_gender_terms'],
            'targeted_change': f"{cond_data['targeted_change_pct']:+.1f}%",
            'random_change': f"{cond_data['random_change_mean_pct']:+.1f}% ± {cond_data['random_change_std_pct']:.1f}%",
            'specificity': f"{cond_data['effect_specificity_pct']:+.1f} pp",
            'ratio': f"{cond_data['effect_ratio']:.1f}×",
            'ci': f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            'significant': sig
        })
    
    print(f"\n{'Condition':<25} {'Change':<12} {'Random':<18} {'Spec.':<10} {'Ratio':<8} {'CI':<24} {'Sig?'}")
    print("-" * 100)
    for row in summary_table:
        sig_str = "✓ YES" if row['significant'] else "✗ no"
        print(f"{row['condition']:<25} {row['targeted_change']:<12} {row['random_change']:<18} "
              f"{row['specificity']:<10} {row['ratio']:<8} {row['ci']:<24} {sig_str}")
    
    # Interpretation
    print(f"\n--- Interpretation ---")
    within = [r for r in summary_table if r['condition'] in ['EN_EN', 'AR_AR']]
    cross = [r for r in summary_table if r['condition'] in ['EN_AR', 'AR_EN']]
    
    n_within_sig = sum(1 for r in within if r['significant'])
    n_cross_sig = sum(1 for r in cross if r['significant'])
    
    if n_within_sig > 0 and n_cross_sig == 0:
        print("  ✓ LANGUAGE-SPECIFIC: Within-language ablation is significant,")
        print("    cross-lingual ablation is NOT → features are language-specific.")
        print("    This causally confirms the CLBAS ≈ 0 finding.")
    elif n_within_sig > 0 and n_cross_sig > 0:
        print("  ⚠ PARTIAL TRANSFER: Both within and cross-lingual effects detected.")
        print("    Some gender features may be shared across languages.")
    elif n_within_sig == 0:
        print("  ⚠ NO WITHIN-LANGUAGE EFFECT detected — may need more images or")
        print("    the Arabic SAE quality may be insufficient.")
    
    results['summary_table'] = summary_table
    results['interpretation'] = {
        'within_language_significant': n_within_sig,
        'cross_lingual_significant': n_cross_sig,
        'features_are_language_specific': n_within_sig > 0 and n_cross_sig == 0,
    }
    
    # Save results
    save_results(results, output_dir / 'cross_lingual_ablation_results.json')
    
    # Save lightweight summary
    lightweight = {
        'config': config,
        'summary_table': summary_table,
        'interpretation': results['interpretation'],
        'feature_overlap': config['feature_overlap'],
    }
    for cond_key, cond_data in results['conditions'].items():
        lightweight[cond_key] = {
            'baseline_gender_terms': cond_data['baseline_gender_terms'],
            'targeted_gender_terms': cond_data['targeted_gender_terms'],
            'targeted_change_pct': cond_data['targeted_change_pct'],
            'random_change_mean_pct': cond_data['random_change_mean_pct'],
            'random_change_std_pct': cond_data['random_change_std_pct'],
            'effect_specificity_pct': cond_data['effect_specificity_pct'],
            'effect_ratio': cond_data['effect_ratio'],
            'paired_statistics': cond_data['paired_statistics'],
            'sample_captions': cond_data['sample_captions'],
        }
    save_results(lightweight, output_dir / 'summary_only.json')
    
    print(f"\nResults saved to: {output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
