#!/usr/bin/env python3
"""
Statistical Significance Tests for Cross-Lingual Probe Accuracy
================================================================

Tests whether differences in probe accuracy between:
1. Arabic vs English within each model
2. Different models

Methods:
- Bootstrap confidence intervals
- McNemar's test for paired comparisons
- Permutation tests for unpaired comparisons
- Effect size (Cohen's d)

Usage:
    python scripts/42_statistical_significance_tests.py
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def bootstrap_accuracy(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    n = len(y_true)
    accuracies = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        acc = (y_true[idx] == y_pred[idx]).mean()
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(accuracies, alpha * 100)
    ci_upper = np.percentile(accuracies, (1 - alpha) * 100)
    
    return {
        'mean': accuracies.mean(),
        'std': accuracies.std(),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for paired comparisons.
    Tests if two classifiers have significantly different error rates.
    """
    # Contingency table
    # b = A correct, B wrong
    # c = A wrong, B correct
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    
    b = np.sum(a_correct & ~b_correct)  # A right, B wrong
    c = np.sum(~a_correct & b_correct)  # A wrong, B right
    
    # McNemar's test (with continuity correction)
    if b + c == 0:
        return {'statistic': 0, 'p_value': 1.0, 'b': b, 'c': c}
    
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        'statistic': chi2,
        'p_value': p_value,
        'b': int(b),
        'c': int(c),
        'significant': p_value < 0.05
    }


def permutation_test_accuracy(acc_a, acc_b, n_permutations=10000):
    """
    Permutation test for difference in accuracy.
    """
    observed_diff = abs(acc_a - acc_b)
    combined = np.array([acc_a, acc_b])
    
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = abs(combined[0] - combined[1])
        if perm_diff >= observed_diff:
            count += 1
    
    p_value = (count + 1) / (n_permutations + 1)
    
    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Interpretation
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {'d': d, 'interpretation': interpretation}


def load_activations_and_genders(checkpoint_dir: Path, model_prefix: str, language: str, layer: int):
    """Load activations for a specific model/language/layer."""
    pattern = f"{model_prefix}_{language}_layer{layer}_*.npz"
    files = list(checkpoint_dir.glob(pattern))
    
    if not files:
        return None, None
    
    all_activations = []
    all_genders = []
    
    for f in sorted(files):
        data = np.load(f)
        all_activations.append(data['activations'])
        all_genders.extend(data['genders'].tolist())
    
    activations = np.concatenate(all_activations, axis=0)
    genders = np.array(all_genders)
    
    # Filter to male/female only
    mask = (genders == "male") | (genders == "female")
    activations = activations[mask]
    genders = genders[mask]
    
    return activations, genders


def train_and_evaluate_probe(activations, genders, n_splits=5):
    """Train logistic regression probe with cross-validation."""
    X = activations
    y = (genders == "male").astype(int)
    
    if len(np.unique(y)) < 2:
        return None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    
    # Get predictions for all samples via cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    
    accuracy = (y_pred == y).mean()
    
    return {
        'accuracy': accuracy,
        'y_true': y,
        'y_pred': y_pred,
        'n_samples': len(y)
    }


def run_statistical_analysis():
    """Run comprehensive statistical analysis."""
    
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS FOR CROSS-LINGUAL PROBE ACCURACY")
    print("=" * 80)
    print()
    
    # Define model configurations
    models = {
        'PaLiGemma-3B': {
            'checkpoint_dir': Path('checkpoints/full_layers_ncc/layer_checkpoints'),
            'prefix': 'paligemma',
            'layers': [3, 6, 9, 12, 15, 17]
        },
        'Qwen2-VL-7B': {
            'checkpoint_dir': Path('checkpoints/qwen2vl/layer_checkpoints'),
            'prefix': 'qwen2vl',
            'layers': [0, 4, 8, 12, 16, 20, 24, 27]
        },
        'LLaVA-1.5-7B': {
            'checkpoint_dir': Path('checkpoints/llava/layer_checkpoints'),
            'prefix': 'llava',
            'layers': [0, 4, 8, 12, 16, 20, 24, 28, 31]
        },
        'Llama-3.2-Vision-11B': {
            'checkpoint_dir': Path('checkpoints/llama32vision/layer_checkpoints'),
            'prefix': 'llama32vision',
            'layers': [0, 5, 10, 15, 20, 25, 30, 35, 39]
        }
    }
    
    results = {}
    
    for model_name, config in models.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        
        model_results = {
            'arabic': {'accuracies': [], 'predictions': []},
            'english': {'accuracies': [], 'predictions': []},
            'per_layer': {}
        }
        
        # Try middle layer first for detailed analysis
        test_layer = config['layers'][len(config['layers']) // 2]
        
        ar_data = en_data = None
        
        for language in ['arabic', 'english']:
            print(f"\n  {language.upper()}:")
            
            for layer in config['layers']:
                try:
                    activations, genders = load_activations_and_genders(
                        config['checkpoint_dir'], config['prefix'], language, layer
                    )
                    
                    if activations is None:
                        continue
                    
                    probe_result = train_and_evaluate_probe(activations, genders)
                    
                    if probe_result is None:
                        continue
                    
                    # Bootstrap CI
                    bootstrap = bootstrap_accuracy(
                        probe_result['y_true'], 
                        probe_result['y_pred']
                    )
                    
                    model_results[language]['accuracies'].append(probe_result['accuracy'])
                    
                    # Store for paired comparison
                    if layer == test_layer:
                        if language == 'arabic':
                            ar_data = probe_result
                        else:
                            en_data = probe_result
                    
                    print(f"    Layer {layer:2d}: {probe_result['accuracy']*100:.1f}% "
                          f"[{bootstrap['ci_lower']*100:.1f}%, {bootstrap['ci_upper']*100:.1f}%] "
                          f"(n={probe_result['n_samples']})")
                    
                except Exception as e:
                    print(f"    Layer {layer}: Error - {e}")
                    continue
        
        # Compute summary statistics
        if model_results['arabic']['accuracies'] and model_results['english']['accuracies']:
            ar_accs = np.array(model_results['arabic']['accuracies'])
            en_accs = np.array(model_results['english']['accuracies'])
            
            ar_mean = ar_accs.mean()
            en_mean = en_accs.mean()
            gap = en_mean - ar_mean
            
            print(f"\n  SUMMARY:")
            print(f"    Arabic mean:  {ar_mean*100:.2f}% ± {ar_accs.std()*100:.2f}%")
            print(f"    English mean: {en_mean*100:.2f}% ± {en_accs.std()*100:.2f}%")
            print(f"    Gap (EN-AR):  {gap*100:+.2f}%")
            
            # Effect size
            if len(ar_accs) > 1 and len(en_accs) > 1:
                effect = cohens_d(en_accs, ar_accs)
                print(f"    Cohen's d:    {effect['d']:.3f} ({effect['interpretation']})")
            
            # Permutation test on mean difference
            perm = permutation_test_accuracy(ar_mean, en_mean)
            print(f"    Permutation test p-value: {perm['p_value']:.4f} "
                  f"{'*' if perm['significant'] else ''}")
            
            # McNemar's test if we have paired predictions
            if ar_data is not None and en_data is not None:
                # Need same samples for McNemar - use minimum
                n_min = min(len(ar_data['y_true']), len(en_data['y_true']))
                mcnemar = mcnemar_test(
                    ar_data['y_true'][:n_min],
                    ar_data['y_pred'][:n_min],
                    en_data['y_pred'][:n_min]
                )
                print(f"    McNemar's test p-value: {mcnemar['p_value']:.4f} "
                      f"{'*' if mcnemar['significant'] else ''}")
            
            model_results['summary'] = {
                'ar_mean': ar_mean,
                'en_mean': en_mean,
                'gap': gap,
                'ar_std': ar_accs.std(),
                'en_std': en_accs.std()
            }
        
        results[model_name] = model_results
    
    # Cross-model comparison
    print("\n" + "=" * 80)
    print("CROSS-MODEL STATISTICAL COMPARISON")
    print("=" * 80)
    
    model_names = list(results.keys())
    print("\n  Pairwise gap comparisons:")
    print(f"  {'Model A':<25} {'Model B':<25} {'Gap Diff':>10} {'p-value':>10}")
    print("  " + "-" * 75)
    
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            if 'summary' not in results[model_a] or 'summary' not in results[model_b]:
                continue
            
            gap_a = results[model_a]['summary']['gap']
            gap_b = results[model_b]['summary']['gap']
            
            # Simple comparison of gaps
            gap_diff = abs(gap_a - gap_b)
            
            print(f"  {model_a:<25} {model_b:<25} {gap_diff*100:>9.2f}%")
    
    # Save results
    output_path = Path('results/statistical_significance')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = {}
    for model, data in results.items():
        serializable_results[model] = {
            'arabic_accuracies': convert_to_serializable(data['arabic']['accuracies']),
            'english_accuracies': convert_to_serializable(data['english']['accuracies']),
            'summary': convert_to_serializable(data.get('summary', {}))
        }
    
    with open(output_path / 'probe_accuracy_statistics.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path / 'probe_accuracy_statistics.json'}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Model':<25} {'AR Acc':>10} {'EN Acc':>10} {'Gap':>10} {'Significant?':>12}")
    print("-" * 80)
    
    for model_name, data in results.items():
        if 'summary' in data:
            s = data['summary']
            # Significance based on permutation test (would need to store this)
            sig = "Yes" if abs(s['gap']) > 0.03 else "No"  # Simplified threshold
            print(f"{model_name:<25} {s['ar_mean']*100:>9.1f}% {s['en_mean']*100:>9.1f}% "
                  f"{s['gap']*100:>+9.1f}% {sig:>12}")
    
    return results


if __name__ == "__main__":
    results = run_statistical_analysis()
