# Mechanistic Interpretability (Arabic Image Captioning) – Mini Pipeline

## 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r mechanistic/requirements.txt
```

Edit `mechanistic/config.yaml` with your paths:
- `images_dir: /home2/jmsk62/project/mechanistic_intrep/dataset/images`
- `captions_csv: /home2/jmsk62/project/mechanistic_intrep/dataset/image_caption.csv`

## 1) Filter (male/female/mixed)
```bash
bash mechanistic/scripts/run_filter.sh
# outputs -> ${out_root}/filtered/*.csv
```

## 2) Activation extraction (+ token spans & runs index)
```bash
bash mechanistic/scripts/run_extract.sh
# outputs -> ${out_root}/acts/*.npz and runs_index.csv
```

## 3) Train SAE on small windows (choose layer in config `sae.layer`)
```bash
bash mechanistic/scripts/run_sae.sh
# outputs -> ${out_root}/sae/sae_<layer>_k<k>.pt and gender_top_features.csv
```

## 4) Causal tests
- **Direct patching**: replaces hidden-state window with baseline
- **Latent patching**: enc->mask top gender features->dec->inject window
```bash
bash mechanistic/scripts/run_causal.sh
```

## 5) Metrics & Plots
```bash
bash mechanistic/scripts/run_metrics.sh
python -m mechanistic.metrics.plot_causal_results   --config mechanistic/config.yaml   --before_csv ${out_root}/metrics/bias_metrics_runs_before.csv   --after_csv  ${out_root}/metrics/bias_metrics_runs_after.csv
```
