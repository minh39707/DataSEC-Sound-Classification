# DataSEC — Environmental Sound Classification under Acoustic Noise

ATI Course Project 2026, Track 3. 22-class environmental sound classification on runtime log-Mel spectrograms,
with a robustness study under test-time acoustic corruptions.

- Dataset: **DataSEC**, Zenodo DOI [10.5281/zenodo.17033970](https://doi.org/10.5281/zenodo.17033970)
  (pinned; license CC-BY-NC-SA-4.0, raw audio is **not** redistributed in this repo)
- Paper: Fredianelli et al., *Scientific Data* 12 (2025), doi:10.1038/s41597-025-05991-w

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # for a CUDA build of torch, install it first from pytorch.org
```

Training was run on Google Colab (free tier, one T4 16 GB); data preparation and analysis also run on CPU.

## Reproduce

```bash
# 1. data (6.4 GB zip, MD5-verified, resumable parallel download)
python scripts/download_data.py --out data/raw
python scripts/build_index.py          # -> data/index.csv (+ exact_duplicates.csv)
python scripts/make_splits.py          # -> splits/{train,val,test}.csv  (already committed; do not regenerate)
python scripts/cache_waveforms.py      # -> data/cache_22k/*.npy  (22.05 kHz int16 audio; no spectrograms)

# 2. train (checkpoint chosen on validation macro-F1; writes val/test prediction CSVs + JSON log)
python -m src.train --model small_cnn    --seed 42
python -m src.train --model mobilenetv3s --seed 42
python -m src.train --model crnn         --seed 42

# 3. re-evaluate a saved checkpoint
python -m src.evaluate --model crnn --seed 42 --split test

# 4. analysis (no retraining; reads data/ and results/)
python scripts/eda.py                  # part A: dataset statistics and figures -> results/*/eda_*
python scripts/make_baseline_table.py  # part D: baseline table -> results/tables/midterm_baselines.csv
python scripts/plot_metrics.py         # part D: per-class F1 + confusion matrices -> results/figures/
```

Environment overrides: `ATI_CONFIG=<yaml>` selects another config, `ATI_RESULTS=<dir>` redirects all outputs.

## Feature pipeline (computed at runtime, see `configs/default.yaml`)

wav (44.1 kHz) → resample 22.05 kHz (soxr HQ, done once by `scripts/cache_waveforms.py`; the cache holds 1-D audio only) →
5.0 s crop/pad (train: random, eval: center) →
STFT (n_fft 1024, Hann, hop 512) → 64 Slaney Mel bands (20 Hz – 11.025 kHz) → 10·log10, 80 dB floor per clip →
standardize with the train-set mean/std → input 1 × 64 × 216.

## Models

| name | description | params | MACs / clip |
|---|---|---|---|
| `small_cnn` | 4 × [Conv3×3-BN-ReLU-MaxPool], freq-mean + time max/mean pooling | 0.39 M | 198 M |
| `mobilenetv3s` | torchvision MobileNetV3-Small, ImageNet weights, 1→3 channel replication | 1.54 M | 18 M |
| `crnn` | 4 conv blocks pooling mainly in frequency → BiGRU(128) → time pooling | 0.87 M | 487 M |

Training: AdamW, cosine schedule with warmup, class-weighted cross-entropy, AMP on GPU, early stopping on val macro-F1.

## Results (Midterm, seed 42, clean test set)

Dataset analysis (part A, with the waveform / STFT / log-Mel theory): [docs/dataset_analysis.md](docs/dataset_analysis.md).

Baselines ([results/tables/midterm_baselines.csv](results/tables/midterm_baselines.csv)); checkpoints selected on
validation macro-F1, test reported once:

| model | val macro-F1 | test acc | test macro-F1 | params | MACs / clip | latency (ms / clip) | train time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `small_cnn` | 0.856 | 0.939 | **0.884** | 0.39 M | 198 M | 1.32 | 5.0 |
| `mobilenetv3s` | **0.882** | 0.925 | 0.853 | 1.54 M | 18 M | 5.78 | 4.0 |
| `crnn` | 0.803 | 0.902 | 0.815 | 0.87 M | 487 M | 2.19 | 2.1 |

Latency is the median of 200 batch-1 runs on a T4, including log-Mel extraction. Per-class F1:
[results/tables/test_per_class_f1.csv](results/tables/test_per_class_f1.csv),
[results/figures/test_per_class_f1.png](results/figures/test_per_class_f1.png). Row-normalized confusion matrices:
`results/figures/confusion_{small_cnn,mobilenetv3s,crnn}.png` (raw counts are in the JSON logs).
These are single-seed results; seeds 43 and 44 are planned before drawing model-ranking conclusions.

Environment: Colab T4, Python 3.13, torch 2.11.0+cu130, torchaudio 2.11.0, scikit-learn 1.6.1; seed 42.
Each JSON log records the exact package versions, GPU, config and split md5.

| path | content |
|---|---|
| `splits/` | fixed split IDs + `split_meta.json` (seed, method, per-class counts) |
| `results/predictions/{model}_seed{s}_{split}_{perturbation}_{severity}.csv` | `sample_id, true_class, pred_class, probabilities, duration, model, split, perturbation, severity, seed` |
| `results/logs/{model}_seed{s}.json` | metrics, per-class F1, confusion matrix, params, MACs, latency, training time and history, config, package versions |
| `results/checkpoints/` | model weights (not committed) |
| `results/tables/`, `results/figures/` | `eda_*` (part A), baseline / per-class / confusion outputs (part D), `midterm_e*` (Midterm questions) |

## Repository layout

```
configs/    default.yaml, classes.json
splits/     train.csv, val.csv, test.csv, split_meta.json   (fixed; never regenerate)
src/        features.py, data.py, train.py, evaluate.py, metrics.py, utils.py, models/
scripts/    data prep: download_data.py, build_index.py, make_splits.py, cache_waveforms.py
            analysis:  eda.py, make_baseline_table.py, plot_metrics.py
docs/       dataset_analysis.md
results/    predictions/, logs/, tables/, figures/, logmel_stats.json
```
