# How to run the project end-to-end

This is the single guide that walks the whole pipeline on your machine,
from a fresh terminal to a populated `reports/` folder with every Phase-1
result. It assumes the work in this repo is already in place — what's
left is just executing the commands in order.

> If anything below errors out, send me the exact line and the error.
> Each step writes its outputs to disk, so a failure halfway through
> never loses earlier work.

---

## 0. One-time setup

```bash
cd ~/Desktop/S2026/IDS/IDS_Project/multi-model-ids-project
conda activate multi-model-ids
```

You should see `(multi-model-ids)` at the start of the prompt.

A quick sanity check that the code is healthy:

```bash
pytest tests/                # ~30-60 seconds; should report all green
```

If `pytest` is missing, run `pip install pytest pytest-cov` inside the
active env once and retry.

---

## 1. Part 5 — Feature engineering

### 1a. Filter ranking (Part 5.1) — *fast: ~5–15 minutes each*

```bash
python scripts/run_filter_selection.py                    # multiclass
python scripts/run_filter_selection.py --target label     # binary
```

Outputs land in `reports/feature_selection/`.

### 1b. RFA (Part 5.2) — *slow: ~1–3 hours*

```bash
python scripts/run_rfa.py                                  # both variants
```

If you want to run them separately in two terminals:

```bash
python scripts/run_rfa.py --variant rf
python scripts/run_rfa.py --variant svm
```

For very long runs, append `> rfa.log 2>&1 &` to run in the background
and free the terminal.

### 1c. Five-condition ablation (Part 5.4) — *fast: ~5–15 minutes each*

The ablation needs the filter and RFA rankings from 1a / 1b. Once those
files exist:

```bash
python scripts/run_ablation.py                             # multiclass
python scripts/run_ablation.py --task binary               # binary
```

Outputs in `reports/ablation/` — CSV, comparisons, and a markdown summary.

---

## 2. Part 6 — Phase-1 models

Every model uses the **same command shape**. Pick the model with
`--model` and the task with `--task`. Each call trains the protocol's
five seeds and writes the per-seed artefacts under
`reports/runs/<model>/seed_<n>/`.

```bash
# Interpretable rule-based baseline (depth-5 decision tree).
# Fast: ~5 minutes total.
python scripts/run_model.py --model rule_based

# Random Forest. Fast: ~5-20 minutes total (5 seeds).
python scripts/run_model.py --model random_forest

# SVM (uses the 30k stratified subsample internally per protocol §9).
# Medium: ~30-90 minutes total.
python scripts/run_model.py --model svm

# XGBoost. Fast-medium: ~5-30 minutes total.
python scripts/run_model.py --model xgboost

# 1-D CNN (Keras). Needs a GPU to be reasonable; works on CPU but slow.
# With GPU: pass --gpu so memory-growth is enabled before TF allocates.
python scripts/run_model.py --model cnn_1d --gpu

# LSTM (Keras). Same caveats as the CNN.
python scripts/run_model.py --model lstm --gpu
```

Run them on the **binary task** too once you have the multiclass set:

```bash
for m in rule_based random_forest svm xgboost cnn_1d lstm; do
  python scripts/run_model.py --model "$m" --task binary
done
```

### Tuning a model without losing the headline run (`--tag` + `--hparam`)

The runner accepts two extra flags:

* `--tag NAME` — adds a subfolder under `<save_dir>/<model>/<task>/`, so a
  tuned variant never overwrites the existing headline result.
* `--hparam KEY=VAL` — repeatable; overrides any hyperparameter exposed
  by the model wrapper. Values are parsed with `yaml.safe_load` so
  `learning_rate=5e-4` becomes a float, `bidirectional=true` becomes a
  bool, etc.

For example, a tuned CNN with a lower learning rate, more epochs, and
more patience lands at
`reports/runs/cnn_1d/multiclass/tuned_lr5e4_ep150/seed_*/`, leaving the
original `reports/runs/cnn_1d/multiclass/seed_*/` untouched:

```bash
python scripts/run_model.py --model cnn_1d --gpu \
  --tag tuned_lr5e4_ep150 \
  --hparam learning_rate=5e-4 --hparam max_epochs=150 \
  --hparam early_stopping_patience=15
```

The DL configurations worth trying first (since the headline DL run
under-performs) are documented below.

Per-seed artefacts (every model × every seed):

* `model.joblib` (or `model.keras` for DL)
* `predictions_val.parquet` / `predictions_test.parquet`
* `metrics.json`
* `confusion_matrix_val.csv` / `confusion_matrix_test.csv`
* `metadata.json` (seed, n_features, runtime, timestamp, task)
* For the two DL models: a `tensorboard/` subfolder with the live
  training curves (loss, accuracy, val loss, val accuracy).

### Watching the DL training in TensorBoard

While `cnn_1d` or `lstm` is running (or after), open a second terminal
and launch the dashboard:

```bash
cd ~/Desktop/S2026/IDS/IDS_Project/multi-model-ids-project
conda activate multi-model-ids
make tensorboard
```

Then open <http://localhost:6006> in a browser. Every model × every
seed appears as its own run (named after the path
`reports/runs/<model>/seed_<n>/tensorboard`), so you can compare seeds
or models side-by-side on the same axes. `make tensorboard` is just a
shortcut for `tensorboard --logdir=reports/runs --port=6006` -- you can
also run that command directly if you don't have `make` installed.

---

## 2b. Tuning the deep-learning models

The headline DL run uses the default hyperparameters from
`OneDCNNConfig` / `LSTMConfig`. On UNSW-NB15 those gave 1D-CNN
multiclass macro-F1 around 0.14 and binary around 0.69 — much weaker
than the tree models and below what these architectures *should*
reach. The most likely cause is training-config (LR too high,
patience too short, model converging into a class-collapse minimum)
rather than architectural unfitness, and the recommended next
experiments below test that. Each run takes 5–25 minutes on the GTX
1050.

All four runs together (~1 hour) plus four for binary (~1 hour) give
a tuned-variant column you can put next to the headline column.

```bash
# Lower learning rate + more epochs + more patience -- the main fix
python scripts/run_model.py --model cnn_1d --gpu --task multiclass \
  --tag tuned_lr5e4_ep150 \
  --hparam learning_rate=5e-4 --hparam max_epochs=150 \
  --hparam early_stopping_patience=15

python scripts/run_model.py --model cnn_1d --gpu --task binary \
  --tag tuned_lr5e4_ep150 \
  --hparam learning_rate=5e-4 --hparam max_epochs=150 \
  --hparam early_stopping_patience=15

python scripts/run_model.py --model lstm --gpu --task multiclass \
  --tag tuned_lr5e4_ep150_bi \
  --hparam learning_rate=5e-4 --hparam max_epochs=150 \
  --hparam early_stopping_patience=15 --hparam bidirectional=true

python scripts/run_model.py --model lstm --gpu --task binary \
  --tag tuned_lr5e4_ep150_bi \
  --hparam learning_rate=5e-4 --hparam max_epochs=150 \
  --hparam early_stopping_patience=15 --hparam bidirectional=true
```

If those still underperform, the next things to try (one at a time,
each as its own `--tag`) are: a larger CNN (`conv_blocks=3
filters_per_block=128`), a smaller `batch_size=128` (gradients
estimated with fewer rare-class samples per batch can be
counter-productive), a longer LSTM window (`window_size=16`), or
switching the optimizer (`optimizer=adamw`). Each one gets its own
tag and folder so you accumulate variants without losing anything.

If the macro-F1 numbers still don't budge after these, the conclusion
becomes a defensible finding for the write-up: deep architectures
genuinely under-perform on UNSW-NB15's flow-summary representation.

---

## 3. Quick reference — what you should see when each step finishes

| Step | Files that appear |
|---|---|
| Filter | `reports/feature_selection/filter_ranking_*.csv`, `filter_selected_top30_*.json` |
| RFA | `reports/feature_selection/rfa_ranking_*.csv`, `rfa_selection_path_*.{csv,png}`, `rfa_hyperparams_*.json` |
| Ablation | `reports/ablation/ablation_per_seed_*.csv`, `ablation_summary_*.{csv,md}`, `ablation_comparisons_*.csv` |
| Models | `reports/runs/<model>/seed_<n>/` with the seven files above |

---

## 4. Total expected runtime (one laptop with GPU)

| Stage | Time |
|---|---|
| Tests | < 1 min |
| Filter (both tasks) | ~10-30 min |
| RFA (both variants) | ~1-3 hours |
| Ablation (both tasks) | ~10-30 min |
| Models multiclass (all six) | ~1.5-3 hours |
| Models binary (all six) | ~1.5-3 hours |
| **Total** | **~5-9 hours**, mostly background-able |

Most of the time is in RFA and the deep-learning models. Both are
back-groundable, so a typical pattern is: kick off RFA in the morning,
run filter/ablation/classical models while RFA runs, run the DL models
overnight.

---

## 5. After everything is done

You will have:

* the canonical feature rankings on disk (both filter and RFA variants);
* the five-condition ablation table with statistical comparisons;
* every Phase-1 model's per-seed metrics, predictions, and confusion
  matrices.

That is the complete Part 5 + Part 6 deliverable. The next stage is
**Part 7 — Phase 2 (deep CNN-LSTM, autoencoder, Kitsune / DeepLog
baselines, CIC-IDS2017 extension)**, which we'll plan together when
you're ready.

---

## 6. Troubleshooting

* `conda: command not found` — open a fresh terminal window.
* `could not find conda environment: multi-model-ids` — once, run
  `make env` (or `conda env create -f environment.yml`).
* `No module named pytest` — `pip install pytest pytest-cov` inside
  the active env.
* `No module named tensorflow` — `pip install tensorflow` inside the
  active env (the conda env should already include it).
* DL script complains about GPU memory — drop the `--gpu` flag (runs on
  CPU; slower) or lower `batch_size` in `configs/hparam_spaces/cnn_1d.yaml`
  / `lstm.yaml`.
