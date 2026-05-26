# Experimental Protocol — Multi-Model Intrusion Detection

**Status:** frozen 2026-05-26 (Task 2.2 deliverable).
**Authority:** this protocol fixes the rules every experiment in this project
must follow. The point of freezing it now — *before* any model is trained — is
to guarantee that comparisons are fair and that nothing can be tuned after
seeing results. Amendments after the freeze date require a dated, justified
entry in the Change log (§13). Changes that affect already-collected results
require the affected runs to be re-executed.

---

## 1. Scope

This document is the single source of truth for **how** models are trained,
evaluated, and compared in this project. It covers data and splits (§2),
preprocessing (§3), class-imbalance handling per model family (§4), tasks and
labels (§5), metrics (§6), the validation scheme (§7), statistical rigour (§8),
hyperparameter search (§9), the feature-engineering ablation (§10), the
semantic feature groups used in the RQ1 analysis (§11), reporting and
reproducibility (§12), and the change log (§13).

What is **not** in scope: project management, dataset acquisition, EDA, and
the engineering of any specific model. Those are documented elsewhere
(`Master_Task_Breakdown.md`, the EDA report, model-specific source files).

---

## 2. Data and splits

**Source.** UNSW-NB15 partitioned benchmark (`data/raw/UNSW-NB15/`), the two
files pooled, deduplicated on all feature+label columns with the `id` column
removed. **94,928** of **257,673** pooled rows are exact duplicates (36.8%);
the working dataset is **162,745 unique records**.

**Split.** Stratified **70 / 15 / 15** on the 10-way `attack_cat` label.
Stored after the preprocessing pipeline at:

| File | Use |
|---|---|
| `data/processed/train.parquet` | model fitting and hyperparameter selection |
| `data/processed/val.parquet`   | model selection only — never used for fitting |
| `data/processed/test.parquet`  | final held-out evaluation, locked (§7) |

**Seed.** All randomness in splitting, sampling and initialisation defaults to
`src.utils.reproducibility.DEFAULT_SEED = 42`. Multi-seed runs (§8) use the
fixed list `[42, 43, 44, 45, 46]`.

**Lock.** The split is produced **once** by `scripts/run_preprocessing.py` and
re-loaded by every experiment. Re-splitting inside a run is forbidden.

---

## 3. Preprocessing — identical across all model families

Every preprocessing step that *learns* a value (encoding vocabulary, scaling
statistics, imputation medians) is fitted on the **training split only** and
applied unchanged to validation and test. Train-only fitting is the leakage
safeguard that makes every score in this project defensible.

Steps, in order:

1. Drop `id` (row-counter; no detection signal).
2. Drop exact-duplicate rows from the pooled dataset (94,928 rows).
3. Stratified 70/15/15 split on `attack_cat` (§2).
4. Median imputation (fitted on train only).
5. One-hot encoding of `proto`, `service`, `state`; rare protocol values are
   bucketed before encoding so the column count stays bounded. Vocabulary
   fitted on train only; categories that appear in val/test but not in train
   are mapped to the rare-bucket column.
6. **Scaling — the only model-dependent branch:**
   - **Tree models** (Random Forest, XGBoost, rule-based): unscaled. Tree
     splits are scale-invariant; standardising adds no information and
     complicates feature-importance interpretation.
   - **Neural networks** (1D-CNN, LSTM, CNN-LSTM, autoencoder): z-score
     standardised, with mean and std fitted on train only.
7. Sliding-window builder (recurrent models only): consecutive flow records
   into fixed-length overlapping windows.

The fitted pipeline is pickled to `data/processed/preprocessor.joblib` and
re-loaded by every model run, so the *same* transformations are applied
across experiments.

---

## 4. Class-imbalance handling — per model family

After deduplication the largest class (Normal, 85,722) outnumbers the smallest
(Worms, 171) by ~501×. The handling rule below fixes which technique each
model family uses for its **headline** run. Three tools are available
(`src/data/imbalance_handling.py`).

| Model family | Headline-run method | Rationale |
|---|---|---|
| Random Forest | inverse-frequency class weights | sklearn `class_weight="balanced"`; no synthetic data |
| SVM | inverse-frequency class weights | sklearn `class_weight="balanced"`; the RBF kernel is sensitive to synthetic interpolation |
| XGBoost | sample weights (multiclass) / `scale_pos_weight` (binary) | native support; SMOTE on one-hot features is noisy |
| 1D-CNN | inverse-frequency class weights in the loss | preferred over SMOTE for stability |
| LSTM | inverse-frequency class weights | preserves temporal order (SMOTE on windows breaks it) |
| CNN-LSTM (Phase 2) | inverse-frequency class weights | as LSTM |
| Autoencoder (Phase 2) | n/a — trained on Normal only | anomaly score, not classification |
| Rule-based | n/a (no learning) | — |

**Ablation.** Per proposal §5.1 and WBS 4.3.4, each model family is *also* run
with **plain SMOTE** and **rare-class augmentation (BorderlineSMOTE)** to
quantify whether oversampling beats class weighting on the rarest classes.
The headline row uses class weights; the ablation rows are clearly labelled.

**Inviolable rule.** Resampling is applied to the **training fold only** —
never to validation or test. Validation and test keep the natural class
distribution.

---

## 5. Tasks and labels

Two parallel tasks, run on the same split:

1. **Binary.** `label ∈ {0, 1}` where 0 = Normal, 1 = any attack.
2. **Multiclass.** `attack_cat ∈ {Normal, Analysis, Backdoor, DoS, Exploits,
   Fuzzers, Generic, Reconnaissance, Shellcode, Worms}`.

Every model is evaluated on both tasks. Binary answers "does the model detect
*anything*"; multiclass is required for RQ1 (which feature groups predict
which attacks) and RQ2 (per-class difficulty and FPR).

---

## 6. Metrics

### 6.1 Primary metrics — headline tables

| Task | Primary metrics |
|---|---|
| Binary | macro-F1, PR-AUC, AUROC, balanced accuracy |
| Multiclass | macro-F1, balanced accuracy, **per-class recall**, **per-class F1** |

Plain accuracy is **not** a primary metric. Under the binary class balance
(~53/47 post-dedup) it is easy, and under the multiclass imbalance (~500×) it
is misleading — a model that ignores Worms entirely still scores ≥99%
multiclass accuracy. Macro-F1 averages per-class F1 with equal weight, so a
model that ignores a rare class is penalised. PR-AUC is preferred over AUROC
under imbalance because the precision–recall curve does not collapse to the
prevalence baseline when the negative class dominates.

### 6.2 Secondary metrics — supplementary tables

Reported for every model:

- Confusion matrix (per task, both val and test)
- Per-class precision, recall, F1, support
- Matthews correlation coefficient (MCC) — balanced single-number summary
- Per-class false-positive rate (for RQ2)
- Inference latency per record (ms) — needed for the trade-off analysis
  (Part 10.3)

### 6.3 The no-peek rule

Only the **validation set** is used for tuning, early stopping and model
selection. The **test set is touched exactly once per model**: a single
evaluation pass that produces the headline numbers. Re-tuning after seeing
test results violates the protocol; any result obtained that way must be
discarded and the model retrained.

---

## 7. Validation scheme

The 15% validation set serves four purposes, and only these:

1. Hyperparameter selection (search runs against val macro-F1; §9).
2. Early stopping (deep models monitor val loss / val macro-F1).
3. Model checkpointing (best-on-val checkpoint kept; final epoch is not).
4. Architecture-within-family selection (e.g. picking the LSTM hidden width).

The 15% test set is **held out** until all hyperparameters are frozen for a
given model. Test-set evaluation happens once and produces the headline
numbers.

5-fold cross-validation is **not** used in this project. The held-out
validation split serves the same purpose with far less compute and is
sufficient given the size of the training pool (n_train ≈ 113,920 unique
records). 5-seed runs (§8) carry the variance estimation that CV would
otherwise provide.

---

## 8. Statistical rigour

Single-seed numbers are **not** reported as comparisons. Every comparison in
this project follows the rules below.

**Multi-seed runs.** Each model is trained with **5 seeds** (`42, 43, 44, 45, 46`).
Reported numbers are mean ± standard deviation across seeds.

**Confidence intervals.** 95% bootstrap CIs are computed on the per-seed
metric distributions (10,000 resamples).

**Paired comparisons.** When comparing two models A and B, the comparison is
*paired* across seeds: A and B see the same five seeds, the same splits, and
the same preprocessing artefacts. The reported test is the **paired Wilcoxon
signed-rank test** on per-seed macro-F1 — non-parametric, robust to a sample
of 5.

**Multiple-comparisons correction.** When more than two models are compared in
the same family of tests (e.g. six Phase-1 models on multiclass macro-F1),
p-values are corrected with **Holm–Bonferroni**.

**Effect size.** A statistically significant difference is only worth
reporting if the effect is meaningful. Headline tables include the macro-F1
**delta** alongside the p-value; deltas under 0.005 are not claimed even if
significant.

---

## 9. Hyperparameter search

**Policy** (per Decisions-log item 4): decide per model. Grid search for small
search spaces; Bayesian optimisation (Optuna, TPE sampler) for large or
expensive ones.

| Model | Method | Budget | Selection target |
|---|---|---|---|
| Random Forest | grid | full grid | val macro-F1 |
| SVM | grid (small; trains slowest) | full grid | val macro-F1 |
| XGBoost | Optuna (TPE) | 50 trials | val macro-F1 |
| 1D-CNN | Optuna (TPE) | 30 trials | val macro-F1 |
| LSTM | Optuna (TPE) | 30 trials | val macro-F1 |
| CNN-LSTM (Phase 2) | Optuna (TPE) | 30 trials | val macro-F1 |
| Autoencoder (Phase 2) | Optuna (TPE) | 30 trials | val reconstruction error |

Each search runs on a single seed (42) for efficiency; the chosen
hyperparameters then go into the 5-seed evaluation (§8).

**Search spaces.** The exact ranges searched for each model are checked-in as
version-controlled YAML at `configs/hparam_spaces/<model>.yaml`. The YAML
files are part of the experimental record — any change to a search space is a
protocol amendment (§13).

**SVM tractability.** SVM-RBF on the full 113k training pool is prohibitively
slow. SVM runs (both the classifier and the SVM cost-function RFA in §10) use
a **stratified subsample of the training set capped at 30,000 records**
(seeded; size logged in the run record). Validation and test are full. This
matches the proposal's §4.3 SVM size cap.

---

## 10. Feature-engineering ablation protocol

Resolves the §6 decisions from the 2026-05-22 supervision meeting (see
`docs/rfa_bigram_spec.md` §§6–7 and the briefing's §6.4).

### 10.1 The five conditions

| # | Condition | Selector | Downstream classifier |
|---|---|---|---|
| 1 | Baseline | all 190 model-ready features | Random Forest |
| 2 | Filter only | Extra-Trees importance ∩ mutual information (consensus, proposal §5.1) | Random Forest |
| 3 | RFA — original (Hamed, Dara & Kremer 2018) | SVM-RBF cost-function rank (thesis Eq. 3.10, Algorithm 2) | Random Forest |
| 4 | RFA — proposal's variant | Random Forest + val macro-F1 gain, early stopping | Random Forest |
| 5 | RFA + flow-pair features | best RFA from (3) vs (4); plus pairwise differences/ratios from the sliding window | Random Forest |

The shared downstream classifier (Random Forest) is what makes the comparison
clean: any difference between conditions is attributable to the **selector**
or the **features**, not to the classifier.

### 10.2 RFA truncation rule (shared by 3 and 4)

Both RFA variants produce a full ranking. For the ablation, the chosen `k` is
the smallest k such that further additions stop improving val macro-F1 by
more than 0.001 over a patience window of 3 additions. The chosen `k` and the
full ranking are both saved.

### 10.3 Citation discipline

- **Condition 3** is cited as faithful to Hamed, Dara & Kremer (2018) and the
  open thesis Algorithm 2.
- **Condition 4** is cited as *"in the spirit of Hamed et al. (2018)"* — a
  forward, add-one wrapper with a contemporary RF/F1 evaluator. Not presented
  as the original algorithm.
- **Condition 5**'s flow-pair features are cited as a *flow-temporal
  construction inspired by* the Hamed payload-bigram technique — not as the
  bigram technique itself. (The original payload-bigram cannot run on
  UNSW-NB15, which has no payload data.)

### 10.4 Statistical comparison for the ablation

Conditions 1–5 are compared pairwise under §8: 5 seeds each, paired Wilcoxon
on per-seed macro-F1, Holm–Bonferroni correction across the
10 pairwise comparisons.

---

## 11. Semantic feature groups (for the RQ1 analysis)

The 41 non-label UNSW-NB15 features fall into five canonical groups (Moustafa
& Slay 2015; Al-Daweri et al. 2020 taxonomy):

1. **Basic flow** — duration, protocol identity, byte / packet counts, rates.
2. **Content** — sequence numbers, TCP window sizes, mean payload sizes.
3. **Time** — inter-arrival times, jitter, round-trip.
4. **General-purpose** — TTL, FTP-login indicators, HTTP method indicators.
5. **Connection-tracking (`ct_*`)** — counts of recent connections to the
   same destination/service.

**RQ1 asks:** which feature groups predict which attack types? The analysis
trains one Random Forest per single-group feature subset and compares
per-class recall across groups. Group membership is fixed at
`configs/feature_groups.yaml` (created in Part 5); any change is a protocol
amendment (§13).

---

## 12. Reporting and reproducibility

### 12.1 Report format (per model)

Every model's reporting block contains:

1. Architecture / configuration summary.
2. Chosen hyperparameters (with a link to the YAML search-space file).
3. Binary results table — primary metrics, mean ± std over 5 seeds, 95% CI.
4. Multiclass results table — primary + per-class recall and F1.
5. Confusion matrices (validation and test).
6. Per-class precision/recall/F1/support.
7. Inference latency per record.
8. The random seed list and the environment hash
   (Python version + `requirements.txt` hash).

### 12.2 Reproducibility artefacts per run

For each model × seed, the following are written under
`reports/runs/<model>/<seed>/`:

- model checkpoint (best on val)
- training log (TensorBoard for deep models)
- predictions on val and test (parquet)
- metrics JSON
- the full config snapshot used (not a diff)
- the random seed
- the git commit hash

### 12.3 Environment

Python 3.10, the project's pinned conda env (`environment.yml`), TensorFlow
2.16–2.17 / Keras 3, scikit-learn ≥ 1.4, XGBoost ≥ 2.0, imbalanced-learn
≥ 0.12. The Docker image (`Dockerfile`) and the Apptainer recipe
(`apptainer.def`) are the canonical environment definitions for the HPC
and external reproducibility.

---

## 13. Change log

Any amendment after the freeze date requires a dated entry here with a
one-line justification.

| Date | Change | Justification |
|---|---|---|
| 2026-05-26 | Protocol frozen (Task 2.2 deliverable). | Reflects the proposal, the WBS, and the 2026-05-22 supervision-meeting decisions. |
