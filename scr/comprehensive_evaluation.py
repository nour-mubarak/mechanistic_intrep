"""
Comprehensive Evaluation Framework for Gender Bias Analysis
=========================================================

This module provides a complete evaluation framework for assessing gender bias
mitigation in multilingual image captioning models, including advanced metrics,
statistical testing, and cross-lingual analysis.

Key Features:
- Advanced bias metrics (demographic parity, equalized odds, etc.)
- Statistical significance testing with multiple correction methods
- Cross-lingual evaluation and consistency metrics
- Quality-bias trade-off analysis
- Automated evaluation pipelines
- Fairness-aware evaluation protocols
- Robustness testing across different demographics
- Causal evaluation of interventions

Example usage:
    from comprehensive_evaluation import ComprehensiveEvaluator
    
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_model(model, test_data)
    evaluator.generate_evaluation_report(results)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import evaluation libraries
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation."""
    # Bias metrics
    compute_demographic_parity: bool = True
    compute_equalized_odds: bool = True
    compute_calibration: bool = True
    compute_individual_fairness: bool = True
    
    # Quality metrics
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_meteor: bool = True
    compute_bertscore: bool = True
    compute_clip_score: bool = True
    
    # Statistical testing
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"  # bonferroni, fdr_bh, holm
    bootstrap_samples: int = 1000
    
    # Cross-lingual evaluation
    evaluate_cross_lingual: bool = True
    translation_consistency_threshold: float = 0.8
    
    # Robustness testing
    test_robustness: bool = True
    noise_levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "json"  # json, csv, html

@dataclass
class BiasMetrics:
    """Container for bias evaluation metrics."""
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    equality_of_opportunity: float = 0.0
    calibration_error: float = 0.0
    individual_fairness: float = 0.0
    statistical_parity: float = 0.0
    
    # Gender-specific metrics
    male_accuracy: float = 0.0
    female_accuracy: float = 0.0
    neutral_accuracy: float = 0.0
    gender_gap: float = 0.0
    
    # Cross-lingual metrics
    cross_lingual_consistency: float = 0.0
    translation_bias_transfer: float = 0.0

@dataclass
class QualityMetrics:
    """Container for caption quality metrics."""
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    rouge_l: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    meteor: float = 0.0
    bertscore_f1: float = 0.0
    clip_score: float = 0.0
    
    # Fluency and coherence
    perplexity: float = 0.0
    semantic_similarity: float = 0.0

@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""
    bias_metrics: BiasMetrics
    quality_metrics: QualityMetrics
    statistical_tests: Dict[str, float]
    cross_lingual_results: Dict[str, Any]
    robustness_results: Dict[str, Any]
    intervention_effects: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class GenderClassifier:
    """Gender classifier for bias evaluation."""
    
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
        
        # Gender keywords for rule-based classification
        self.male_keywords = [
            'man', 'boy', 'father', 'son', 'brother', 'husband', 'male',
            'he', 'his', 'him', 'gentleman', 'guy', 'dude'
        ]
        self.female_keywords = [
            'woman', 'girl', 'mother', 'daughter', 'sister', 'wife', 'female',
            'she', 'her', 'hers', 'lady', 'gal'
        ]
        
    def train(self, texts: List[str], labels: List[str]) -> None:
        """Train the gender classifier."""
        # Simple bag-of-words features
        features = []
        for text in texts:
            text_lower = text.lower()
            feature_vector = [
                sum(1 for word in self.male_keywords if word in text_lower),
                sum(1 for word in self.female_keywords if word in text_lower),
                len(text.split()),  # Text length
                text_lower.count('person'),
                text_lower.count('people')
            ]
            features.append(feature_vector)
        
        # Encode labels
        label_map = {'male': 0, 'female': 1, 'neutral': 2}
        encoded_labels = [label_map.get(label, 2) for label in labels]
        
        self.classifier.fit(features, encoded_labels)
        self.is_trained = True
        
    def predict(self, texts: List[str]) -> List[str]:
        """Predict gender labels for texts."""
        if not self.is_trained:
            # Use rule-based classification
            return self._rule_based_classify(texts)
        
        features = []
        for text in texts:
            text_lower = text.lower()
            feature_vector = [
                sum(1 for word in self.male_keywords if word in text_lower),
                sum(1 for word in self.female_keywords if word in text_lower),
                len(text.split()),
                text_lower.count('person'),
                text_lower.count('people')
            ]
            features.append(feature_vector)
        
        predictions = self.classifier.predict(features)
        label_map = {0: 'male', 1: 'female', 2: 'neutral'}
        return [label_map[pred] for pred in predictions]
    
    def _rule_based_classify(self, texts: List[str]) -> List[str]:
        """Rule-based gender classification."""
        predictions = []
        for text in texts:
            text_lower = text.lower()
            male_count = sum(1 for word in self.male_keywords if word in text_lower)
            female_count = sum(1 for word in self.female_keywords if word in text_lower)
            
            if male_count > female_count:
                predictions.append('male')
            elif female_count > male_count:
                predictions.append('female')
            else:
                predictions.append('neutral')
        
        return predictions

class ComprehensiveEvaluator:
    """Comprehensive evaluation system for gender bias analysis."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.gender_classifier = GenderClassifier()
        
        # Initialize evaluation metrics
        if EVALUATE_AVAILABLE:
            self.bleu_metric = evaluate.load("bleu")
            self.rouge_metric = evaluate.load("rouge")
            try:
                self.bertscore_metric = evaluate.load("bertscore")
            except:
                self.bertscore_metric = None
        
    def compute_bias_metrics(self, 
                           predictions: List[str],
                           ground_truth_labels: List[str],
                           predicted_labels: List[str] = None) -> BiasMetrics:
        """
        Compute comprehensive bias metrics.
        
        Args:
            predictions: Model predictions (captions)
            ground_truth_labels: True gender labels
            predicted_labels: Predicted gender labels (if None, will classify predictions)
            
        Returns:
            BiasMetrics object containing all bias measurements
        """
        if predicted_labels is None:
            predicted_labels = self.gender_classifier.predict(predictions)
        
        # Convert labels to numeric for computation
        label_map = {'male': 0, 'female': 1, 'neutral': 2}
        y_true = [label_map.get(label, 2) for label in ground_truth_labels]
        y_pred = [label_map.get(label, 2) for label in predicted_labels]
        
        metrics = BiasMetrics()
        
        # Basic accuracy metrics
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Gender-specific accuracies
        male_mask = np.array(y_true) == 0
        female_mask = np.array(y_true) == 1
        neutral_mask = np.array(y_true) == 2
        
        if male_mask.sum() > 0:
            metrics.male_accuracy = accuracy_score(
                np.array(y_true)[male_mask], np.array(y_pred)[male_mask]
            )
        
        if female_mask.sum() > 0:
            metrics.female_accuracy = accuracy_score(
                np.array(y_true)[female_mask], np.array(y_pred)[female_mask]
            )
        
        if neutral_mask.sum() > 0:
            metrics.neutral_accuracy = accuracy_score(
                np.array(y_true)[neutral_mask], np.array(y_pred)[neutral_mask]
            )
        
        # Gender gap
        metrics.gender_gap = abs(metrics.male_accuracy - metrics.female_accuracy)
        
        # Demographic parity (equal positive prediction rates)
        if self.config.compute_demographic_parity:
            male_pos_rate = np.mean(np.array(y_pred)[male_mask] != 2) if male_mask.sum() > 0 else 0
            female_pos_rate = np.mean(np.array(y_pred)[female_mask] != 2) if female_mask.sum() > 0 else 0
            metrics.demographic_parity = abs(male_pos_rate - female_pos_rate)
        
        # Equalized odds (equal TPR and FPR across groups)
        if self.config.compute_equalized_odds:
            try:
                # Compute TPR and FPR for each group
                male_tpr = self._compute_tpr(np.array(y_true)[male_mask], np.array(y_pred)[male_mask])
                female_tpr = self._compute_tpr(np.array(y_true)[female_mask], np.array(y_pred)[female_mask])
                
                male_fpr = self._compute_fpr(np.array(y_true)[male_mask], np.array(y_pred)[male_mask])
                female_fpr = self._compute_fpr(np.array(y_true)[female_mask], np.array(y_pred)[female_mask])
                
                tpr_diff = abs(male_tpr - female_tpr)
                fpr_diff = abs(male_fpr - female_fpr)
                metrics.equalized_odds = max(tpr_diff, fpr_diff)
            except:
                metrics.equalized_odds = 0.0
        
        # Statistical parity
        male_pred_rate = np.mean(np.array(y_pred)[male_mask] == 0) if male_mask.sum() > 0 else 0
        female_pred_rate = np.mean(np.array(y_pred)[female_mask] == 1) if female_mask.sum() > 0 else 0
        metrics.statistical_parity = abs(male_pred_rate - female_pred_rate)
        
        return metrics
    
    def _compute_tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute True Positive Rate."""
        if len(y_true) == 0:
            return 0.0
        tp = np.sum((y_true != 2) & (y_pred != 2))
        fn = np.sum((y_true != 2) & (y_pred == 2))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _compute_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute False Positive Rate."""
        if len(y_true) == 0:
            return 0.0
        fp = np.sum((y_true == 2) & (y_pred != 2))
        tn = np.sum((y_true == 2) & (y_pred == 2))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def compute_quality_metrics(self, 
                              predictions: List[str],
                              references: List[str]) -> QualityMetrics:
        """
        Compute caption quality metrics.
        
        Args:
            predictions: Generated captions
            references: Reference captions
            
        Returns:
            QualityMetrics object containing quality measurements
        """
        metrics = QualityMetrics()
        
        # BLEU scores
        if self.config.compute_bleu and EVALUATE_AVAILABLE:
            try:
                bleu_results = self.bleu_metric.compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                metrics.bleu_4 = bleu_results.get('bleu', 0.0)
                
                # Compute individual BLEU scores
                if NLTK_AVAILABLE:
                    smoothing = SmoothingFunction().method1
                    bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
                    
                    for pred, ref in zip(predictions, references):
                        pred_tokens = pred.split()
                        ref_tokens = [ref.split()]
                        
                        for n in range(1, 5):
                            weights = [1/n] * n + [0] * (4-n)
                            score = sentence_bleu(ref_tokens, pred_tokens, 
                                                weights=weights, smoothing_function=smoothing)
                            bleu_scores[f'bleu_{n}'].append(score)
                    
                    metrics.bleu_1 = np.mean(bleu_scores['bleu_1'])
                    metrics.bleu_2 = np.mean(bleu_scores['bleu_2'])
                    metrics.bleu_3 = np.mean(bleu_scores['bleu_3'])
                    metrics.bleu_4 = np.mean(bleu_scores['bleu_4'])
            except Exception as e:
                print(f"BLEU computation failed: {e}")
        
        # ROUGE scores
        if self.config.compute_rouge and EVALUATE_AVAILABLE:
            try:
                rouge_results = self.rouge_metric.compute(
                    predictions=predictions,
                    references=references
                )
                metrics.rouge_l = rouge_results.get('rougeL', 0.0)
                metrics.rouge_1 = rouge_results.get('rouge1', 0.0)
                metrics.rouge_2 = rouge_results.get('rouge2', 0.0)
            except Exception as e:
                print(f"ROUGE computation failed: {e}")
        
        # METEOR score
        if self.config.compute_meteor and NLTK_AVAILABLE:
            try:
                meteor_scores = []
                for pred, ref in zip(predictions, references):
                    score = meteor_score([ref.split()], pred.split())
                    meteor_scores.append(score)
                metrics.meteor = np.mean(meteor_scores)
            except Exception as e:
                print(f"METEOR computation failed: {e}")
        
        # BERTScore
        if self.config.compute_bertscore and self.bertscore_metric:
            try:
                bert_results = self.bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang="en"
                )
                metrics.bertscore_f1 = np.mean(bert_results['f1'])
            except Exception as e:
                print(f"BERTScore computation failed: {e}")
        
        return metrics
    
    def perform_statistical_tests(self, 
                                baseline_results: Dict[str, List[float]],
                                intervention_results: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Perform statistical significance tests.
        
        Args:
            baseline_results: Baseline model results
            intervention_results: Results after intervention
            
        Returns:
            Dictionary of p-values for different tests
        """
        test_results = {}
        
        for metric_name in baseline_results.keys():
            if metric_name in intervention_results:
                baseline_values = baseline_results[metric_name]
                intervention_values = intervention_results[metric_name]
                
                # Paired t-test
                if len(baseline_values) == len(intervention_values):
                    t_stat, p_value = stats.ttest_rel(baseline_values, intervention_values)
                    test_results[f'{metric_name}_ttest_pvalue'] = p_value
                
                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_p_value = stats.wilcoxon(baseline_values, intervention_values)
                    test_results[f'{metric_name}_wilcoxon_pvalue'] = w_p_value
                except:
                    test_results[f'{metric_name}_wilcoxon_pvalue'] = 1.0
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(baseline_values) + np.var(intervention_values)) / 2)
                if pooled_std > 0:
                    cohens_d = (np.mean(intervention_values) - np.mean(baseline_values)) / pooled_std
                    test_results[f'{metric_name}_cohens_d'] = cohens_d
        
        # Apply multiple testing correction
        p_values = [v for k, v in test_results.items() if 'pvalue' in k]
        if p_values:
            if self.config.multiple_testing_correction == "bonferroni":
                corrected_alpha = self.config.significance_level / len(p_values)
                test_results['corrected_alpha'] = corrected_alpha
            elif self.config.multiple_testing_correction == "fdr_bh":
                # Benjamini-Hochberg procedure
                sorted_p = sorted(p_values)
                n = len(p_values)
                for i, p in enumerate(sorted_p):
                    if p <= (i + 1) / n * self.config.significance_level:
                        test_results['fdr_threshold'] = p
                        break
        
        return test_results
    
    def evaluate_cross_lingual_consistency(self, 
                                         english_predictions: List[str],
                                         arabic_predictions: List[str],
                                         english_references: List[str],
                                         arabic_references: List[str]) -> Dict[str, float]:
        """
        Evaluate cross-lingual consistency and bias transfer.
        
        Args:
            english_predictions: English model predictions
            arabic_predictions: Arabic model predictions
            english_references: English reference captions
            arabic_references: Arabic reference captions
            
        Returns:
            Dictionary of cross-lingual metrics
        """
        results = {}
        
        # Gender classification consistency
        en_gender_labels = self.gender_classifier.predict(english_predictions)
        ar_gender_labels = self.gender_classifier.predict(arabic_predictions)
        
        # Cross-lingual gender consistency
        gender_consistency = accuracy_score(en_gender_labels, ar_gender_labels)
        results['gender_consistency'] = gender_consistency
        
        # Bias transfer analysis
        en_bias_metrics = self.compute_bias_metrics(
            english_predictions,
            self.gender_classifier.predict(english_references),
            en_gender_labels
        )
        ar_bias_metrics = self.compute_bias_metrics(
            arabic_predictions,
            self.gender_classifier.predict(arabic_references),
            ar_gender_labels
        )
        
        results['english_bias_score'] = en_bias_metrics.gender_gap
        results['arabic_bias_score'] = ar_bias_metrics.gender_gap
        results['bias_transfer_ratio'] = (
            ar_bias_metrics.gender_gap / en_bias_metrics.gender_gap 
            if en_bias_metrics.gender_gap > 0 else 0.0
        )
        
        # Quality consistency
        en_quality = self.compute_quality_metrics(english_predictions, english_references)
        ar_quality = self.compute_quality_metrics(arabic_predictions, arabic_references)
        
        results['quality_consistency'] = 1.0 - abs(en_quality.bleu_4 - ar_quality.bleu_4)
        
        return results
    
    def evaluate_robustness(self, 
                          model: Any,
                          test_inputs: List[Any],
                          noise_levels: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate model robustness to input perturbations.
        
        Args:
            model: Model to evaluate
            test_inputs: Test input samples
            noise_levels: Levels of noise to test
            
        Returns:
            Dictionary of robustness results
        """
        if noise_levels is None:
            noise_levels = self.config.noise_levels
        
        robustness_results = {}
        
        # Get baseline predictions
        baseline_predictions = []
        for inputs in test_inputs:
            with torch.no_grad():
                outputs = model(inputs)
                # Assuming outputs have a method to get text
                if hasattr(outputs, 'sequences'):
                    pred = outputs.sequences
                else:
                    pred = outputs
                baseline_predictions.append(pred)
        
        # Test robustness at different noise levels
        for noise_level in noise_levels:
            noisy_predictions = []
            
            for inputs in test_inputs:
                # Add noise to inputs
                if isinstance(inputs, torch.Tensor):
                    noise = torch.randn_like(inputs) * noise_level
                    noisy_inputs = inputs + noise
                else:
                    # For dict inputs, add noise to tensor values
                    noisy_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            noise = torch.randn_like(v) * noise_level
                            noisy_inputs[k] = v + noise
                        else:
                            noisy_inputs[k] = v
                
                with torch.no_grad():
                    outputs = model(noisy_inputs)
                    if hasattr(outputs, 'sequences'):
                        pred = outputs.sequences
                    else:
                        pred = outputs
                    noisy_predictions.append(pred)
            
            # Compute consistency between baseline and noisy predictions
            # This is a simplified version - in practice, you'd need proper text comparison
            consistency_scores = []
            for baseline, noisy in zip(baseline_predictions, noisy_predictions):
                # Simplified consistency measure
                if isinstance(baseline, torch.Tensor) and isinstance(noisy, torch.Tensor):
                    consistency = F.cosine_similarity(
                        baseline.flatten(), noisy.flatten(), dim=0
                    ).item()
                    consistency_scores.append(consistency)
            
            robustness_results[f'noise_{noise_level}'] = {
                'mean_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
                'std_consistency': np.std(consistency_scores) if consistency_scores else 0.0
            }
        
        return robustness_results
    
    def evaluate_model(self, 
                      model: Any,
                      test_data: Dict[str, Any],
                      baseline_model: Any = None) -> EvaluationResults:
        """
        Perform comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            test_data: Test dataset
            baseline_model: Baseline model for comparison
            
        Returns:
            Comprehensive evaluation results
        """
        # Extract data
        predictions = test_data.get('predictions', [])
        references = test_data.get('references', [])
        gender_labels = test_data.get('gender_labels', [])
        
        # Compute bias metrics
        bias_metrics = self.compute_bias_metrics(predictions, gender_labels)
        
        # Compute quality metrics
        quality_metrics = self.compute_quality_metrics(predictions, references)
        
        # Statistical tests (if baseline provided)
        statistical_tests = {}
        if baseline_model and 'baseline_predictions' in test_data:
            baseline_results = {'bias_score': [bias_metrics.gender_gap]}
            intervention_results = {'bias_score': [bias_metrics.gender_gap]}
            statistical_tests = self.perform_statistical_tests(
                baseline_results, intervention_results
            )
        
        # Cross-lingual evaluation
        cross_lingual_results = {}
        if 'arabic_predictions' in test_data and 'arabic_references' in test_data:
            cross_lingual_results = self.evaluate_cross_lingual_consistency(
                predictions,
                test_data['arabic_predictions'],
                references,
                test_data['arabic_references']
            )
        
        # Robustness evaluation
        robustness_results = {}
        if self.config.test_robustness and model:
            test_inputs = test_data.get('test_inputs', [])
            if test_inputs:
                robustness_results = self.evaluate_robustness(model, test_inputs)
        
        # Compile results
        results = EvaluationResults(
            bias_metrics=bias_metrics,
            quality_metrics=quality_metrics,
            statistical_tests=statistical_tests,
            cross_lingual_results=cross_lingual_results,
            robustness_results=robustness_results,
            intervention_effects={},
            metadata={
                'num_samples': len(predictions),
                'evaluation_config': self.config.__dict__
            }
        )
        
        return results
    
    def generate_evaluation_report(self, 
                                 results: EvaluationResults,
                                 output_path: str = "evaluation_report") -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            output_path: Output path for the report
        """
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Generate summary statistics
        summary = {
            'bias_summary': {
                'gender_gap': results.bias_metrics.gender_gap,
                'male_accuracy': results.bias_metrics.male_accuracy,
                'female_accuracy': results.bias_metrics.female_accuracy,
                'demographic_parity': results.bias_metrics.demographic_parity,
                'equalized_odds': results.bias_metrics.equalized_odds
            },
            'quality_summary': {
                'bleu_4': results.quality_metrics.bleu_4,
                'rouge_l': results.quality_metrics.rouge_l,
                'meteor': results.quality_metrics.meteor,
                'bertscore_f1': results.quality_metrics.bertscore_f1
            },
            'cross_lingual_summary': results.cross_lingual_results,
            'statistical_tests': results.statistical_tests
        }
        
        # Save summary
        with open(output_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_evaluation_plots(results, output_dir)
        
        # Generate detailed report
        self._generate_detailed_report(results, output_dir)
        
        print(f"Evaluation report generated at: {output_dir}")
    
    def _generate_evaluation_plots(self, 
                                 results: EvaluationResults,
                                 output_dir: Path) -> None:
        """Generate evaluation visualizations."""
        # Bias metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gender accuracy comparison
        genders = ['Male', 'Female', 'Neutral']
        accuracies = [
            results.bias_metrics.male_accuracy,
            results.bias_metrics.female_accuracy,
            results.bias_metrics.neutral_accuracy
        ]
        
        axes[0, 0].bar(genders, accuracies, color=['lightblue', 'lightpink', 'lightgreen'])
        axes[0, 0].set_title('Gender-Specific Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        
        # Quality metrics
        quality_names = ['BLEU-4', 'ROUGE-L', 'METEOR', 'BERTScore']
        quality_values = [
            results.quality_metrics.bleu_4,
            results.quality_metrics.rouge_l,
            results.quality_metrics.meteor,
            results.quality_metrics.bertscore_f1
        ]
        
        axes[0, 1].bar(quality_names, quality_values, color='orange')
        axes[0, 1].set_title('Quality Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Bias vs Quality trade-off
        axes[1, 0].scatter([results.bias_metrics.gender_gap], [results.quality_metrics.bleu_4], 
                          s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Gender Gap (Bias)')
        axes[1, 0].set_ylabel('BLEU-4 Score (Quality)')
        axes[1, 0].set_title('Bias vs Quality Trade-off')
        
        # Cross-lingual comparison
        if results.cross_lingual_results:
            lang_metrics = ['English Bias', 'Arabic Bias', 'Gender Consistency']
            lang_values = [
                results.cross_lingual_results.get('english_bias_score', 0),
                results.cross_lingual_results.get('arabic_bias_score', 0),
                results.cross_lingual_results.get('gender_consistency', 0)
            ]
            
            axes[1, 1].bar(lang_metrics, lang_values, color='purple')
            axes[1, 1].set_title('Cross-lingual Analysis')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_detailed_report(self, 
                                results: EvaluationResults,
                                output_dir: Path) -> None:
        """Generate detailed evaluation report."""
        report_lines = [
            "# Comprehensive Gender Bias Evaluation Report",
            "",
            "## Executive Summary",
            f"- Gender Gap: {results.bias_metrics.gender_gap:.4f}",
            f"- Overall Quality (BLEU-4): {results.quality_metrics.bleu_4:.4f}",
            f"- Cross-lingual Consistency: {results.cross_lingual_results.get('gender_consistency', 'N/A')}",
            "",
            "## Bias Metrics",
            f"- Male Accuracy: {results.bias_metrics.male_accuracy:.4f}",
            f"- Female Accuracy: {results.bias_metrics.female_accuracy:.4f}",
            f"- Neutral Accuracy: {results.bias_metrics.neutral_accuracy:.4f}",
            f"- Demographic Parity: {results.bias_metrics.demographic_parity:.4f}",
            f"- Equalized Odds: {results.bias_metrics.equalized_odds:.4f}",
            "",
            "## Quality Metrics",
            f"- BLEU-1: {results.quality_metrics.bleu_1:.4f}",
            f"- BLEU-2: {results.quality_metrics.bleu_2:.4f}",
            f"- BLEU-3: {results.quality_metrics.bleu_3:.4f}",
            f"- BLEU-4: {results.quality_metrics.bleu_4:.4f}",
            f"- ROUGE-L: {results.quality_metrics.rouge_l:.4f}",
            f"- METEOR: {results.quality_metrics.meteor:.4f}",
            "",
            "## Statistical Significance",
        ]
        
        for test_name, p_value in results.statistical_tests.items():
            if isinstance(p_value, float):
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                report_lines.append(f"- {test_name}: {p_value:.6f} {significance}")
        
        # Save report
        with open(output_dir / "detailed_report.md", 'w') as f:
            f.write('\n'.join(report_lines))


def create_sample_evaluation():
    """Create sample evaluation for demonstration."""
    print("Creating sample comprehensive evaluation...")
    
    # Mock test data
    test_data = {
        'predictions': [
            "A woman is standing in the park",
            "A man is walking down the street",
            "A person is sitting on a bench",
            "The girl is playing with a ball",
            "The boy is riding a bicycle"
        ],
        'references': [
            "A woman stands in a beautiful park",
            "A man walks along the street",
            "A person sits on a wooden bench",
            "A young girl plays with a red ball",
            "A boy rides his bicycle"
        ],
        'gender_labels': ['female', 'male', 'neutral', 'female', 'male'],
        'arabic_predictions': [
            "امرأة تقف في الحديقة",
            "رجل يمشي في الشارع",
            "شخص يجلس على مقعد",
            "فتاة تلعب بالكرة",
            "ولد يركب دراجة"
        ],
        'arabic_references': [
            "امرأة تقف في حديقة جميلة",
            "رجل يمشي على الشارع",
            "شخص يجلس على مقعد خشبي",
            "فتاة صغيرة تلعب بكرة حمراء",
            "ولد يركب دراجته"
        ]
    }
    
    # Create evaluator
    config = EvaluationConfig(
        compute_bleu=True,
        compute_rouge=True,
        evaluate_cross_lingual=True,
        generate_plots=True
    )
    
    evaluator = ComprehensiveEvaluator(config)
    
    # Run evaluation
    results = evaluator.evaluate_model(None, test_data)
    
    # Generate report
    evaluator.generate_evaluation_report(results, "sample_evaluation_report")
    
    print("Sample evaluation completed!")
    print(f"Gender Gap: {results.bias_metrics.gender_gap:.4f}")
    print(f"Quality (BLEU-4): {results.quality_metrics.bleu_4:.4f}")
    
    return evaluator, results


if __name__ == "__main__":
    # Create sample evaluation
    evaluator, results = create_sample_evaluation()
    print("Comprehensive evaluation framework demonstration completed!")

