#!/usr/bin/env python3
"""
49_k_sensitivity_sweep.py
========================

Runs a matched-random k-sensitivity sweep for SAE feature ablation.

Goal
----
Provide reviewer-proof evidence for whether k=100 is sufficient (and whether
larger/smaller k values change conclusions), using the SAME design as the
main intervention setup:
- one shared baseline per image set
- targeted ablation for each k
- matched random controls for each k
- per-image paired bootstrap CI for (targeted - random)

Default sweep:
  k in {25, 50, 100, 200}

Usage examples
--------------
# PaLiGemma L9, publication-like settings
python scripts/improved/49_k_sensitivity_sweep.py \
  --model paligemma --layer 9 --k_values 25 50 100 200 \
  --n_images 500 --n_random_runs 25

# Faster smoke run
python scripts/improved/49_k_sensitivity_sweep.py \
  --model paligemma --layer 9 --k_values 25 50 100 \
  --n_images 40 --n_random_runs 3
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Local shared utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    SAEHook,
    load_model_and_sae,
    load_test_images,
    compute_aggregate_stats,
    compute_paired_statistics,
    save_results,
)


BASE_DIR = Path('/home2/jmsk62/mechanistic_intrep/sae_captioning_project')


def _activation_path_for(model_name: str, layer: int) -> Optional[Path]:
    if model_name == 'paligemma':
        return BASE_DIR / f'checkpoints/full_layers_ncc/layer_checkpoints/layer_{layer}_english.pt'
    if model_name == 'qwen2vl':
        return BASE_DIR / f'checkpoints/qwen2vl/layer_checkpoints/qwen2vl_layer_{layer}_english.pt'
    return None


def _rank_gender_features_from_activations(
    model_name: str,
    layer: int,
    sae,
    max_k: int,
    n_features: int,
):
    """Rank features by |male_mean - female_mean| from stored activations.

    Returns
    -------
    ranked_features: list[int]
    metadata: dict
    """
    act_path = _activation_path_for(model_name, layer)
    if act_path is None or not act_path.exists():
        return [], {
            'source': 'activations',
            'ok': False,
            'reason': f'activation file missing for {model_name} layer {layer}',
            'path': str(act_path) if act_path else None,
        }

    print(f"Loading activations for ranking: {act_path}")
    act_data = torch.load(act_path, map_location='cpu', weights_only=False)
    activations = act_data.get('activations')
    genders = act_data.get('genders')

    if activations is None or genders is None:
        return [], {
            'source': 'activations',
            'ok': False,
            'reason': 'activations or genders missing in checkpoint',
            'path': str(act_path),
        }

    activations = activations.float()
    genders = list(genders)

    male_mask = torch.tensor([g == 'male' for g in genders], dtype=torch.bool)
    female_mask = torch.tensor([g == 'female' for g in genders], dtype=torch.bool)

    n_male = int(male_mask.sum().item())
    n_female = int(female_mask.sum().item())

    if n_male == 0 or n_female == 0:
        return [], {
            'source': 'activations',
            'ok': False,
            'reason': f'insufficient labels male={n_male}, female={n_female}',
            'path': str(act_path),
        }

    # Encode with SAE encoder on CPU for ranking
    sae_cpu = sae.to('cpu').float().eval()
    with torch.no_grad():
        encoded = torch.relu(sae_cpu.encoder(activations))  # [N, n_features]

    male_mean = encoded[male_mask].mean(dim=0)
    female_mean = encoded[female_mask].mean(dim=0)
    diff = (male_mean - female_mean).abs()

    top = diff.argsort(descending=True)[: min(max_k, n_features)].tolist()
    return top, {
        'source': 'activations',
        'ok': True,
        'path': str(act_path),
        'n_samples': int(activations.shape[0]),
        'n_male': n_male,
        'n_female': n_female,
        'max_k_requested': int(max_k),
        'n_ranked_returned': int(len(top)),
    }


def _rank_gender_features_from_llama_checkpoint(layer: int, max_k: int):
    """Fallback ranking source for llama checkpoints with stored gender features."""
    sae_path = BASE_DIR / f'checkpoints/llama32vision/saes/llama32vision_sae_english_layer{layer}.pt'
    if not sae_path.exists():
        return [], {
            'source': 'llama_checkpoint',
            'ok': False,
            'reason': 'checkpoint missing',
            'path': str(sae_path),
        }

    ckpt = torch.load(sae_path, map_location='cpu', weights_only=False)
    if 'gender_features' not in ckpt:
        return [], {
            'source': 'llama_checkpoint',
            'ok': False,
            'reason': 'gender_features missing in checkpoint',
            'path': str(sae_path),
        }

    gf = ckpt['gender_features']
    male = gf.get('male_features', [])
    female = gf.get('female_features', [])

    ranked = []
    seen = set()
    # Interleave male/female for balance
    for i in range(max(len(male), len(female))):
        if i < len(male) and male[i] not in seen:
            ranked.append(int(male[i]))
            seen.add(int(male[i]))
        if i < len(female) and female[i] not in seen:
            ranked.append(int(female[i]))
            seen.add(int(female[i]))
        if len(ranked) >= max_k:
            break

    return ranked[:max_k], {
        'source': 'llama_checkpoint',
        'ok': len(ranked) > 0,
        'path': str(sae_path),
        'n_ranked_returned': int(len(ranked)),
        'n_male_raw': int(len(male)),
        'n_female_raw': int(len(female)),
    }


def run_condition_with_hook(generate_fn, target_layer, sae, images, image_paths, ablate_features, desc):
    hook = SAEHook(sae, ablate_features=ablate_features)
    handle = target_layer.register_forward_hook(hook)
    caps = []
    try:
        for image, p in tqdm(zip(images, image_paths), total=len(images), desc=desc):
            caps.append({'image_path': p, 'caption': generate_fn(image)})
    finally:
        handle.remove()
    return caps


def run_baseline(generate_fn, images, image_paths):
    caps = []
    for image, p in tqdm(zip(images, image_paths), total=len(images), desc='baseline'):
        caps.append({'image_path': p, 'caption': generate_fn(image)})
    return caps


def markdown_from_results(results: dict) -> str:
    cfg = results['config']
    lines = []
    lines.append('# k-Sensitivity Sweep Results')
    lines.append('')
    lines.append('## Configuration')
    lines.append('')
    lines.append(f"- Model: **{cfg['model']}**")
    lines.append(f"- Layer: **{cfg['layer']}**")
    lines.append(f"- Images: **{cfg['n_images']}**")
    lines.append(f"- Random runs per k: **{cfg['n_random_runs']}**")
    lines.append(f"- k values: **{cfg['k_values']}**")
    lines.append(f"- Feature ranking source: **{results['feature_ranking'].get('source')}**")
    lines.append('')

    lines.append('## Main table')
    lines.append('')
    lines.append('| k | baseline terms | targeted terms | targeted change (%) | random mean (%) | random sd (%) | specificity (pp) | paired Δ CI95 |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---|')

    for row in results['k_sweep']:
        ci = row['paired_statistics']['difference_targeted_minus_random']['ci_95']
        ci_txt = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(
            f"| {row['k']} | {row['baseline_gender_terms']} | {row['targeted_gender_terms']} | "
            f"{row['targeted_change_pct']:+.2f} | {row['random_change_mean_pct']:+.2f} | "
            f"{row['random_change_std_pct']:.2f} | {row['effect_specificity_pp']:+.2f} | {ci_txt} |"
        )

    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    lines.append('- Publication-safe claim: fixed-k and k-sweep targeted-vs-random specificity if paired CI excludes 0 for each k.')
    lines.append('- Avoid claiming global k-optimality unless trend is monotonic and stable across seeds/models/layers.')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Matched-random k-sensitivity sweep for SAE interventions')
    parser.add_argument('--model', type=str, default='paligemma', choices=['paligemma', 'qwen2vl', 'llama32vision'])
    parser.add_argument('--layer', type=int, default=9)
    parser.add_argument('--k_values', type=int, nargs='+', default=[25, 50, 100, 200])
    parser.add_argument('--n_images', type=int, default=500)
    parser.add_argument('--n_random_runs', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = str(
            BASE_DIR / 'results' / 'k_sensitivity' / f"{args.model}_L{args.layer}"
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('K-SENSITIVITY SWEEP')
    print('=' * 70)
    print(f"model={args.model} layer={args.layer} k_values={args.k_values}")
    print(f"n_images={args.n_images} n_random_runs={args.n_random_runs}")

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda but CUDA is not available. "
            "Use --device cpu or --device auto."
        )

    if device == 'cpu':
        print("WARNING: running on CPU; this will be very slow for large models.")

    images, image_paths = load_test_images(args.n_images, seed=args.seed)
    model, processor, sae, n_features, target_layer, generate_fn = load_model_and_sae(
        args.model, args.layer, device
    )

    max_k = max(args.k_values)

    # Rank features once, then prefix-slice by k for a proper nested sweep.
    ranked_features = []
    ranking_meta = {}

    if args.model in ('paligemma', 'qwen2vl'):
        ranked_features, ranking_meta = _rank_gender_features_from_activations(
            args.model, args.layer, sae, max_k, n_features
        )
    elif args.model == 'llama32vision':
        ranked_features, ranking_meta = _rank_gender_features_from_llama_checkpoint(args.layer, max_k)

    if len(ranked_features) < max_k:
        raise RuntimeError(
            f"Could not obtain enough ranked features for max_k={max_k}. "
            f"Only got {len(ranked_features)}. Meta: {ranking_meta}"
        )

    # move SAE back to requested device/dtype for generation hooks
    sae = sae.to(device).to(torch.bfloat16).eval()

    baseline_caps = run_baseline(generate_fn, images, image_paths)
    baseline_stats = compute_aggregate_stats(baseline_caps)

    k_rows = []
    for k in args.k_values:
        print('\n' + '-' * 70)
        print(f'k={k}')
        print('-' * 70)

        targeted_features = ranked_features[:k]
        targeted_caps = run_condition_with_hook(
            generate_fn, target_layer, sae, images, image_paths,
            targeted_features, desc=f'targeted_k{k}'
        )
        targeted_stats = compute_aggregate_stats(targeted_caps)

        # matched random controls for this k
        available = list(set(range(n_features)) - set(targeted_features))
        random_caps_list = []
        random_stats_list = []
        random_changes = []

        for ridx in range(args.n_random_runs):
            rng = random.Random(args.seed + 10_000 * k + ridx)
            random_features = rng.sample(available, k)
            rcaps = run_condition_with_hook(
                generate_fn, target_layer, sae, images, image_paths,
                random_features, desc=f'random_k{k}_{ridx+1}/{args.n_random_runs}'
            )
            rstats = compute_aggregate_stats(rcaps)
            random_caps_list.append(rcaps)
            random_stats_list.append(rstats)

            base_total = baseline_stats['total_gender_terms']
            rchg = (rstats['total_gender_terms'] - base_total) / max(base_total, 1) * 100
            random_changes.append(float(rchg))

        base_total = baseline_stats['total_gender_terms']
        targ_total = targeted_stats['total_gender_terms']
        targ_change = (targ_total - base_total) / max(base_total, 1) * 100

        paired = compute_paired_statistics(baseline_caps, targeted_caps, random_caps_list)

        row = {
            'k': int(k),
            'baseline_gender_terms': int(base_total),
            'targeted_gender_terms': int(targ_total),
            'targeted_change_pct': float(targ_change),
            'random_change_mean_pct': float(np.mean(random_changes)),
            'random_change_std_pct': float(np.std(random_changes)),
            'effect_specificity_pp': float(targ_change - np.mean(random_changes)),
            'paired_statistics': paired,
            'targeted_features_head': list(map(int, targeted_features[:20])),
        }
        k_rows.append(row)

        print(
            f"k={k}: targeted={row['targeted_change_pct']:+.2f}% | "
            f"random={row['random_change_mean_pct']:+.2f}% ± {row['random_change_std_pct']:.2f}% | "
            f"spec={row['effect_specificity_pp']:+.2f} pp"
        )

    results = {
        'config': {
            'model': args.model,
            'layer': args.layer,
            'k_values': [int(k) for k in args.k_values],
            'n_images': int(args.n_images),
            'n_random_runs': int(args.n_random_runs),
            'seed': int(args.seed),
            'timestamp': datetime.now().isoformat(),
        },
        'feature_ranking': ranking_meta,
        'k_sweep': k_rows,
    }

    save_results(results, out_dir / 'k_sensitivity_results.json')

    md = markdown_from_results(results)
    with open(out_dir / 'k_sensitivity_summary.md', 'w') as f:
        f.write(md)

    print('\nSaved:')
    print(f"- {out_dir / 'k_sensitivity_results.json'}")
    print(f"- {out_dir / 'k_sensitivity_summary.md'}")


if __name__ == '__main__':
    main()
