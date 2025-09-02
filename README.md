# Mechanistic Bias Analysis System

This package provides a complete implementation for studying and mitigating
gender bias in multilingual image captioning models using techniques from
mechanistic interpretability. It was developed in response to research
questions about **localising** and **reducing** gender bias in Arabic/English
vision–language models such as LLaVA. The design is modular so that
individual stages can be run independently or as part of a full pipeline.

## Components

| Module | Description |
| --- | --- |
| `activation_extractor.py` | Extract hidden-layer activations from a Hugging Face transformer model by registering forward hooks. Supports LLaVA and CLIP models. |
| `sae_training.py` | Implements a simple Sparse Autoencoder (SAE) and a training function to discover latent circuits in pooled activations. |
| `fine_tuning.py` | Provides a dataset class and training routine to fine‑tune a language model on gender-swapped caption data using parameter efficient LoRA. |
| `evaluation.py` | Contains helper functions to infer gender from captions and compute fairness and quality metrics (BLEU, ROUGE). |
| `pipeline.py` | Orchestrates the entire workflow: activation extraction, SAE training, bias‑mitigation fine‑tuning and evaluation. |

## Usage

1. **Prepare your dataset**. You should have a directory containing:
   * An `image_caption.csv` file with columns `image`, `en_caption`, `ar_caption`.
   * An `images/` subdirectory with the images referenced in the CSV. If
     images are stored elsewhere, adjust the `image_path` logic in
     `pipeline.py`.

2. **Run activation extraction** to capture hidden representations. For
   example:

   ```bash
   python -m mechanistic_system.pipeline \
       --dataset-dir /path/to/your/dataset \
       --activations-dir ./activations \
       --run-extraction \
       --num-samples 100
   ```

   This will extract activations for the first 100 examples and save them
   under `./activations/sample_XXXXX/` as `.npy` files with an `info.json`
   manifest.

3. **Train a sparse autoencoder** on the pooled activations:

   ```bash
   python -m mechanistic_system.pipeline \
       --activations-dir ./activations \
       --run-sae-training \
       --sae-dict-size 8192 \
       --sae-sparsity 0.001 \
       --sae-epochs 5
   ```

4. **Fine‑tune the model** on gender-swapped prompts and targets for
   bias mitigation. This will save the tuned model in `./fine_tuned_model`:

   ```bash
   python -m mechanistic_system.pipeline \
       --dataset-dir /path/to/your/dataset \
       --fine-tune-output ./fine_tuned_model \
       --run-fine-tuning \
       --num-samples 200
   ```

   The script uses parameter‑efficient LoRA and a simple prompt/target
   construction. Modify `build_fine_tune_dataset` in `pipeline.py` to better
   suit your data.

5. **Evaluate baseline vs. tuned model** using gender classification and
   caption quality metrics:

   ```bash
   python -m mechanistic_system.pipeline \
       --dataset-dir /path/to/your/dataset \
       --fine-tune-output ./fine_tuned_model \
       --run-evaluation \
       --num-samples 50
   ```

   This will print metrics such as overall gender classification accuracy,
   male/female accuracy and bias gap, as well as BLEU and ROUGE-L scores if
   the `evaluate` library is installed.

## Background

*Activation patching* is a causal intervention technique in which
representations computed on a *clean* input are injected into a *corrupted*
input to see how much of the correct behaviour can be recovered【207162601214896†L563-L607】. This helps to localise circuits responsible for a particular behaviour inside a model. By extracting hidden activations and training a sparse autoencoder, we can discover a rich set of directions in the representation space that correspond to specific concepts【207162601214896†L765-L778】. Fine‑tuning the model on gender-swapped data encourages it to produce gender‑neutral descriptions, and evaluating with gender‑classification metrics allows us to quantify residual bias.

For more details on mechanistic interpretability techniques, see the
Understanding LLMs course notes【207162601214896†L563-L607】【207162601214896†L765-L778】.
