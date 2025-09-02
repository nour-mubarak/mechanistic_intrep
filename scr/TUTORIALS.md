# Tutorials and Usage Examples

This document provides tutorials and usage examples to help you get started with the Enhanced Mechanistic Interpretability System.




## Tutorial 1: Running the Full Experiment

This tutorial will guide you through the process of running the complete gender bias analysis experiment.

### Step 1: Prepare Your Environment

Ensure that you have installed all the required dependencies as described in the `README.md` file.

### Step 2: Run the Experiment Script

The easiest way to run the full experiment is to use the `real_experiment.py` script. Open a terminal and run the following command:

```bash
python -m enhanced_mechanistic_system.real_experiment
```

This will execute the entire pipeline, including:

1.  **Data Preparation:** A 1000-sample gender-balanced dataset will be created.
2.  **Baseline Evaluation:** The baseline model will be evaluated for gender bias and caption quality.
3.  **Mechanistic Analysis:** The system will discover and analyze the neural circuits related to gender bias.
4.  **Bias Interventions:** A series of causal interventions will be performed to test hypotheses about the identified circuits.
5.  **Enhanced Fine-Tuning:** The model will be fine-tuned using a curriculum learning strategy to mitigate bias.
6.  **Final Evaluation:** The fine-tuned model will be evaluated and compared to the baseline.

### Step 3: Analyze the Results

After the experiment is complete, a `gender_bias_experiment_results` directory will be created. This directory contains all the results, reports, and visualizations generated during the experiment.

*   **`comprehensive_dashboard.html`:** An interactive dashboard for exploring the results.
*   **`executive_summary.md`:** A high-level summary of the experiment and its key findings.
*   **`baseline_vs_final_comparison.json`:** A detailed comparison of the baseline and fine-tuned models.
*   **`mechanistic_analysis.json`:** The results of the circuit discovery and analysis.
*   **`intervention_results.json`:** The results of the causal intervention experiments.




## Tutorial 2: Customizing the Experiment

This tutorial will show you how to customize the experiment by modifying the configuration.

### Step 1: Open the Configuration File

The main configuration for the experiment is located in `enhanced_mechanistic_system/configs/experiment_config.yaml`. Open this file in a text editor.

### Step 2: Modify the Configuration

You can change various aspects of the experiment by modifying the values in this file. For example, you can:

*   **Change the number of samples:**

    ```yaml
    total_samples: 2000
    ```

*   **Use a different base model:**

    ```yaml
    base_model_name: "Salesforce/blip-image-captioning-large"
    ```

*   **Adjust the fine-tuning parameters:**

    ```yaml
    num_epochs: 5
    learning_rate: 2e-5
    ```

### Step 3: Run the Experiment

After you have saved your changes to the configuration file, run the experiment script as before:

```bash
python -m enhanced_mechanistic_system.real_experiment
```

The experiment will now run with your custom settings.


