# API Documentation

This document provides a detailed overview of the modules and classes in the Enhanced Mechanistic Interpretability System. It is intended for developers and researchers who want to understand, extend, or customize the system.



'''
## `real_experiment.py`

This module orchestrates the entire gender bias analysis experiment. It integrates all other components of the system to provide a seamless, end-to-end pipeline.

### `GenderBiasExperiment`

This is the main class that manages the experiment.

#### `__init__(self, config: ExperimentConfig = None)`

*   **Description:** Initializes the experiment with a given configuration.
*   **Arguments:**
    *   `config` (ExperimentConfig, optional): An `ExperimentConfig` object containing the experiment settings. If not provided, a default configuration is used.

#### `setup_experiment(self)`

*   **Description:** Sets up the experiment environment, including initializing WandB, loading the model and processors, and preparing the datasets.

#### `run_complete_experiment(self)`

*   **Description:** Runs the complete experimental pipeline, from baseline evaluation to final reporting.
*   **Returns:** A dictionary containing the complete results of the experiment.

#### `generate_final_report(self, results: Dict[str, Any])`

*   **Description:** Generates a comprehensive final report, including an executive summary and an interactive dashboard.
*   **Arguments:**
    *   `results` (dict): The results dictionary returned by `run_complete_experiment()`.
'''


'''
## `comprehensive_evaluation.py`

This module provides a robust framework for evaluating gender bias and caption quality in multilingual models.

### `ComprehensiveEvaluator`

This class implements a wide range of evaluation metrics and statistical tests.

#### `__init__(self, config: EvaluationConfig = None)`

*   **Description:** Initializes the evaluator with a given configuration.
*   **Arguments:**
    *   `config` (EvaluationConfig, optional): An `EvaluationConfig` object specifying which metrics to compute.

#### `evaluate_model(self, model: Any, test_data: Dict[str, Any], baseline_model: Any = None)`

*   **Description:** Performs a comprehensive evaluation of a given model.
*   **Arguments:**
    *   `model` (Any): The model to be evaluated.
    *   `test_data` (dict): A dictionary containing the test data, including predictions, references, and gender labels.
    *   `baseline_model` (Any, optional): A baseline model for comparison.
*   **Returns:** An `EvaluationResults` object containing the detailed evaluation results.

#### `generate_evaluation_report(self, results: EvaluationResults, output_path: str = "evaluation_report")`

*   **Description:** Generates a detailed evaluation report with visualizations.
*   **Arguments:**
    *   `results` (EvaluationResults): The evaluation results to be reported.
    *   `output_path` (str, optional): The directory where the report will be saved.
'''



## `enhanced_fine_tuning.py`

This module implements advanced fine-tuning techniques for bias mitigation.

### `EnhancedTrainer`

This class provides a sophisticated training loop with features like curriculum learning and bias-aware loss functions.

#### `__init__(self, model: nn.Module, tokenizer: Any, config: BiasAwareTrainingConfig, lora_config: AdvancedLoRAConfig = None)`

*   **Description:** Initializes the enhanced trainer.
*   **Arguments:**
    *   `model` (nn.Module): The model to be fine-tuned.
    *   `tokenizer` (Any): The tokenizer for the model.
    *   `config` (BiasAwareTrainingConfig): The configuration for bias-aware training.
    *   `lora_config` (AdvancedLoRAConfig, optional): The configuration for LoRA.

#### `train_with_curriculum(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None, num_epochs: int = 3, learning_rate: float = 1e-5)`

*   **Description:** Trains the model using a curriculum learning strategy.
*   **Arguments:**
    *   `train_dataloader` (DataLoader): The data loader for the training set.
    *   `val_dataloader` (DataLoader, optional): The data loader for the validation set.
    *   `num_epochs` (int, optional): The number of training epochs.
    *   `learning_rate` (float, optional): The learning rate for the optimizer.
*   **Returns:** A dictionary containing the training history.




## `circuit_discovery.py`

This module provides tools for discovering and analyzing neural circuits related to gender bias.

### `CircuitAnalyzer`

This class implements methods for identifying and visualizing gender bias circuits.

#### `__init__(self, model: nn.Module, tokenizer: Any = None)`

*   **Description:** Initializes the circuit analyzer.
*   **Arguments:**
    *   `model` (nn.Module): The model to be analyzed.
    *   `tokenizer` (Any, optional): The tokenizer for the model.

#### `discover_gender_bias_circuits(self, attention_analysis: Dict[str, List[Dict]], method: str = "attention_flow")`

*   **Description:** Discovers gender bias circuits using attention analysis.
*   **Arguments:**
    *   `attention_analysis` (dict): A dictionary containing attention patterns for different gender groups.
    *   `method` (str, optional): The method to be used for circuit discovery.
*   **Returns:** A dictionary containing the discovered circuits.




## `interventions.py`

This module implements advanced causal intervention techniques for mechanistic interpretability.

### `InterventionEngine`

This class provides a suite of tools for performing causal interventions on the model.

#### `__init__(self, model: nn.Module, tokenizer: Any = None, device: str = "auto")`

*   **Description:** Initializes the intervention engine.
*   **Arguments:**
    *   `model` (nn.Module): The model to be intervened on.
    *   `tokenizer` (Any, optional): The tokenizer for the model.
    *   `device` (str, optional): The device to be used for computations.

#### `run_intervention_experiment(self, config: InterventionConfig, test_inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]], baseline_inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None)`

*   **Description:** Runs a complete intervention experiment.
*   **Arguments:**
    *   `config` (InterventionConfig): The configuration for the intervention.
    *   `test_inputs` (list): The test inputs for the experiment.
    *   `baseline_inputs` (list, optional): The baseline inputs for comparison.
*   **Returns:** An `InterventionEffect` object containing the results of the intervention.


