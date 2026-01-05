#!/usr/bin/env python3
"""Consolidate results from all layer analysis runs."""

import json
import re
from pathlib import Path

# Parse each layer's log file to extract results
log_dir = Path('logs')
results = {'timestamp': None, 'layers': {}}

for layer in [10, 14, 18, 22]:
    log_file = log_dir / f'analysis_layer_{layer}_optimized.log'

    if not log_file.exists():
        print(f"Warning: Log file not found for layer {layer}")
        continue

    print(f"Processing layer {layer}...")

    with open(log_file, 'r') as f:
        log_content = f.read()

    # Extract statistics
    layer_data = {
        'english_stats_summary': {},
        'arabic_stats_summary': {},
        'gender_bias': {'english': {}, 'arabic': {}},
        'prisma_analysis': {},
        'visualizations': {}
    }

    # Extract L0 and dead features
    en_l0_match = re.search(r'English L0: ([\d.]+)', log_content)
    ar_l0_match = re.search(r'Arabic L0: ([\d.]+)', log_content)
    en_dead_match = re.search(r'English.*Dead features: (\d+)', log_content)
    ar_dead_match = re.search(r'Arabic.*Dead features: (\d+)', log_content)

    if en_l0_match:
        layer_data['english_stats_summary']['l0_per_sample'] = float(en_l0_match.group(1))
    if ar_l0_match:
        layer_data['arabic_stats_summary']['l0_per_sample'] = float(ar_l0_match.group(1))
    if en_dead_match:
        layer_data['english_stats_summary']['dead_features'] = int(en_dead_match.group(1))
    if ar_dead_match:
        layer_data['arabic_stats_summary']['dead_features'] = int(ar_dead_match.group(1))

    # Extract ranks
    en_rank_match = re.search(r'Effective rank \(EN\): (\d+)', log_content)
    ar_rank_match = re.search(r'Effective rank \(AR\): (\d+)', log_content)

    if en_rank_match and ar_rank_match:
        layer_data['prisma_analysis']['factored_matrix'] = {
            'english_rank': int(en_rank_match.group(1)),
            'arabic_rank': int(ar_rank_match.group(1)),
            'rank_difference': abs(int(en_rank_match.group(1)) - int(ar_rank_match.group(1)))
        }

    # Extract alignment
    alignment_match = re.search(r'Feature alignment: ([\d.]+)%', log_content)
    if alignment_match:
        layer_data['prisma_analysis']['feature_alignment'] = {
            'alignment_ratio': float(alignment_match.group(1)) / 100.0
        }

    # Extract gender correlation
    gender_corr_match = re.search(r'Gender bias correlation: ([\d.]+)', log_content)
    if gender_corr_match:
        layer_data['prisma_analysis']['gender_correlation'] = {
            'gender_bias_correlation': float(gender_corr_match.group(1))
        }

    # Add visualization paths
    layer_data['visualizations'] = {
        'statistics': f'visualizations/layer_{layer}_comprehensive_statistics.png',
        'gender_bias': f'visualizations/layer_{layer}_gender_biased_features.png',
        'embeddings': f'visualizations/layer_{layer}_feature_embeddings.png'
    }

    # Note: Top biased features would need to be extracted from saved analysis results
    # For now, we'll add placeholders
    layer_data['gender_bias']['english'] = {
        'top_male_biased': [],
        'top_female_biased': []
    }
    layer_data['gender_bias']['arabic'] = {
        'top_male_biased': [],
        'top_female_biased': []
    }

    results['layers'][str(layer)] = layer_data

# Try to load actual gender bias features from W&B or previous runs
# Check if there's any layer-specific result file
for layer in [10, 14, 18, 22]:
    layer_result_file = Path(f'results/layer_{layer}_results.json')
    if layer_result_file.exists():
        print(f"Loading detailed results for layer {layer}")
        with open(layer_result_file) as f:
            layer_details = json.load(f)
        if 'gender_bias' in layer_details:
            results['layers'][str(layer)]['gender_bias'] = layer_details['gender_bias']

# Save consolidated results
output_file = Path('results/comprehensive_analysis_results_all_layers.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nConsolidated results saved to: {output_file}")
print(f"Layers included: {list(results['layers'].keys())}")
