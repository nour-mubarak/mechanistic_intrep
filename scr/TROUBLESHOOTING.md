
# Troubleshooting Guide

This guide provides solutions to common issues you might encounter while using the Enhanced Mechanistic Interpretability System.

## Common Issues and Solutions

### 1. `ModuleNotFoundError`

**Problem:** You see an error like `ModuleNotFoundError: No module named 'wandb'` or similar.

**Solution:** This usually means that a required Python package is not installed. Ensure you have installed all dependencies by running:

```bash
pip install -r requirements.txt
```

If the problem persists, check your Python environment to make sure you are using the correct interpreter.

### 2. GPU Memory Issues

**Problem:** You encounter `CUDA out of memory` errors during training or inference.

**Solution:** This indicates that your GPU does not have enough memory to process the current batch size or model. Try the following:

*   **Reduce `batch_size`:** In `ExperimentConfig` in `real_experiment.py`, decrease the `batch_size`.
*   **Increase `gradient_accumulation_steps`:** This allows you to simulate a larger batch size with less memory. Increase `gradient_accumulation_steps` in `ExperimentConfig`.
*   **Use a smaller model:** If possible, switch to a smaller `base_model_name` in `ExperimentConfig`.
*   **Enable `use_qlora`:** If your model supports it, set `use_qlora: True` in `AdvancedLoRAConfig` in `enhanced_fine_tuning.py` for quantized LoRA.

### 3. Data Loading Errors

**Problem:** The experiment fails during data loading, especially with image files.

**Solution:**

*   **Check image paths:** Ensure that the image paths in your dataset (or the sample dataset in `real_experiment.py`) are correct and accessible.
*   **Verify image format:** Make sure the images are in a supported format (e.g., JPEG, PNG) and are not corrupted.
*   **Internet connectivity:** If using remote image URLs, ensure you have a stable internet connection.

### 4. WandB Integration Issues

**Problem:** WandB logging is not working, or you see authentication errors.

**Solution:**

*   **Login to WandB:** Ensure you are logged in to your WandB account. Run `wandb login` in your terminal and follow the prompts.
*   **Check API Key:** Verify that your WandB API key is correctly configured.
*   **Internet connection:** WandB requires an active internet connection to log data.
*   **Set `use_wandb` to `True`:** In `ExperimentConfig` in `real_experiment.py`, ensure `use_wandb` is set to `True`.

### 5. Unexpected Bias or Quality Results

**Problem:** The bias mitigation techniques do not seem to be effective, or caption quality degrades significantly.

**Solution:**

*   **Review dataset:** Ensure your dataset accurately reflects the biases you are trying to mitigate. The quality and diversity of your data are crucial.
*   **Adjust hyperparameters:** Experiment with different `learning_rate`, `num_epochs`, and bias-related weights in `BiasAwareTrainingConfig`.
*   **Inspect visualizations:** Use the generated plots and dashboards to identify where the model is failing or where bias is still present.
*   **Analyze mechanistic insights:** Revisit the circuit discovery and intervention results to gain deeper insights into the model's behavior.

If you encounter an issue not listed here, please refer to the detailed documentation or open an issue on the GitHub repository.

