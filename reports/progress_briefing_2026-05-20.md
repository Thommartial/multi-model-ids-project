# Project Progress Briefing — Multi-Model Intrusion Detection

**Prepared for:** the supervision meeting with Prof. Rozita Dara
**Student:** Thomas Martial Ekwelle Epalle (1406946) · CIS\*6560 · MCTI, University of Guelph
**Date:** 20 May 2026

This briefing summarises everything completed so far, explains the reasoning
behind each step, states clearly how the approved proposal has been followed
and where it has been adjusted, and gives an honest outlook — including whether
the project is worth continuing and whether the results could be published. It
also raises two method decisions (Section 6) that need your direction: the
project is deliberately paused at that point until the meeting.

---

## 1. Status at a glance

- **On schedule.** The proposal's timeline allots 13–26 May to dataset
  acquisition, EDA, and the preprocessing pipeline. All three are finished
  (20 May) — the project is on track, slightly ahead.
- **Completed:** project infrastructure; dataset acquisition and verification;
  a nine-analysis exploratory data analysis (EDA); the complete preprocessing,
  sampling, and class-imbalance pipeline; and the literature/method-grounding
  step — studying the original RFA + bigram source against the proposal.
- **Paused, pending this meeting:** the experimental-protocol freeze and all
  feature-engineering and model-implementation work. The method-grounding step
  surfaced two discrepancies between the proposal and the original RFA + bigram
  paper that need your direction before the feature-engineering design can be
  fixed (Section 6). Work is intentionally held here until the meeting.
- **Code:** version-controlled and public at
  `github.com/Thommartial/multi-model-ids-project`, MIT-licensed, with a
  reproducible environment and continuous-integration checks.

The project remains structured in the proposal's two phases. Everything done so
far is **Phase 1** (the guaranteed deliverable) groundwork.

---

## 2. What has been built

### 2.1 Project infrastructure

A reproducible engineering foundation: a structured Git repository; a Conda
environment pinning Python 3.10 and the full scientific stack; Docker and an
Apptainer recipe (for the university HPC); a global random-seed module so every
experiment is exactly repeatable; a typed configuration system so no parameter
is hard-coded; logging; and an automated CI pipeline that lint-checks and tests
the code on every change.

### 2.2 Dataset acquisition and EDA

The UNSW-NB15 benchmark was downloaded, integrity-checked against SHA-256
checksums, and verified to be the genuine dataset. A nine-part EDA module then
analysed it — class distribution, missing/invalid values, feature statistics,
correlations, per-attack patterns, dimensionality (PCA/t-SNE), outliers, and
statistical tests — producing a Markdown report and eight publication-quality
figures (`reports/eda/`).

### 2.3 The data pipeline

A leakage-safe pipeline takes the raw records to model-ready data — cleaning,
the stratified train/validation/test split, categorical encoding, scaling, a
sequence-window builder for the recurrent models, stratified subsampling, and
three class-imbalance tools. The details and rationale are in Section 3.

### 2.4 Literature and method grounding

The original source for the project's feature-engineering method — Hamed, Dara
& Kremer (2018), "Network intrusion detection system based on recursive feature
addition and bigram technique" — was located and studied in full. The journal
article is paywalled, but the first author's PhD thesis (open access via the
University of Guelph Atrium repository) gives the complete algorithms and was
used as the primary source. Reading it carefully against the proposal produced
an important finding — two points where the proposal's description of the method
differs materially from the published method. This is set out in Section 6, and
a full written specification has been added to the repository at
`docs/rfa_bigram_spec.md`.

---

## 3. The data transformations — what each means and why

Machine-learning results are only as trustworthy as the data preparation behind
them. Each transformation below has a specific purpose and a specific safeguard.

**1. Pooling the two dataset partitions.** UNSW-NB15 ships as a training file
and a testing file. They were combined into one pool because this project makes
its *own* stratified split (per the proposal); the dataset's pre-made split is
not used.

**2. Dropping the `id` column.** `id` is just a 1, 2, 3… row counter, not a
network measurement. It carries no detection signal, and keeping it would let a
model "cheat" by memorising row numbers. It was removed.

**3. Deduplication — the most consequential step.** Once the `id` was removed,
**94,928 of 257,673 records (36.8%) were found to be exact duplicates** —
identical across every feature and label. They were removed. *Why this matters:*
if the same record appears in both the training and test sets, a model can
memorise it, and the test score then measures memory, not detection — this is
"train/test leakage". Removing duplicates is required for an honest evaluation;
the proposal mandates it (§5.1), and published work confirms the problem is real
and large in UNSW-NB15 (Zoghi & Serpen, 2024, report 42.24% duplicates in its
training set). The working dataset is now **162,745 unique records**.

**4. Stratified 70/15/15 split.** The data is split into training (fit the
models), validation (tune them), and test (one final, untouched estimate of
real performance). The split is *stratified* on the 10-way attack category, so
every class — including Worms, with only 171 records — appears in all three
splits in the same proportion. Verified: the attack rate is exactly 47.33% in
each split.

**5. One-hot encoding of categorical features.** Three features (`proto`,
`service`, `state`) are text labels; models need numbers. One-hot encoding turns
each category into its own 0/1 indicator column. Rare protocol values are
bucketed together so the ~130 distinct protocols do not explode the feature
count. Result: 190 model-ready features.

**6. z-score standardisation (neural networks only).** Many features have wildly
different scales and heavy skew. Neural networks train poorly when inputs are on
very different scales, so their inputs are standardised to mean 0, standard
deviation 1. Tree models (Random Forest, XGBoost) are unaffected by scale and
use the raw values — exactly the dual treatment the proposal specifies (§5.1).

**7. Median imputation.** Any missing value is filled with the training-set
median. UNSW-NB15 has none, but this keeps the pipeline robust for the messier
CIC-IDS2017 dataset in Phase 2.

**8. The sliding-window builder.** The LSTM and CNN-LSTM models consume
*sequences* of records, not single records. A builder turns consecutive records
into overlapping fixed-length windows for them.

**9. Stratified subsampling.** A tool to draw a smaller, class-proportional
sample — used for the robustness experiments and to cap the SVM's training size.

**10. Class-imbalance handling.** Three tools to stop models from ignoring rare
attacks: inverse-frequency *class weights* (no synthetic data), *SMOTE*
oversampling, and *rare-class augmentation*.

**The safeguard running through all of it:** every step that *learns* something
from the data — the encoding vocabulary, the scaling statistics, the imputation
medians — is fitted on the **training split only** and then applied unchanged to
validation and test. Fitting on the whole dataset would leak information and
inflate the reported scores. Likewise, class-imbalance resampling is applied to
the training fold only. This discipline is what makes the eventual results
defensible.

---

## 4. How the proposal was followed — and what was adjusted

### Followed as written

The two-phase structure; UNSW-NB15 as the primary benchmark; the 70/15/15
stratified split; duplicate and missing-value removal; the dual scaling policy
(z-score for networks, unscaled for trees); one-hot categorical encoding;
Python 3.10 with TensorFlow/Keras, scikit-learn and TensorBoard; a public,
MIT-licensed, reproducible GitHub repository; and the planned model families,
metrics, and statistical-rigour approach (these are upcoming, not yet built).

### Adjusted — with rationale (please review these)

1. **Dataset distribution.** The proposal describes UNSW-NB15 as ~2.5M records
   with 49 features and ~87% normal traffic — that is the *full* release. The
   project uses the **partitioned train/test benchmark** (257,673 records, 45
   features), which is the version the great majority of published IDS studies
   use, making the results comparable. Its balance is ~36% normal / 64% attack;
   after deduplication it is ~53% / 47%.

2. **Deduplication effect.** Removing duplicates is *following* the proposal,
   but the magnitude (36.8%) and its effect — the dataset shrank to 162,745
   records and the class balance shifted — are large enough to flag explicitly.

3. **TensorFlow version.** The proposal names "TensorFlow/Keras" without a
   version. A current TensorFlow (2.17) is used; an older pin proposed in
   planning had an unresolvable dependency conflict with modern libraries.

4. **Subsampling targets.** The proposal (§4.3) planned 200k–400k subsamples to
   keep training feasible. After deduplication the whole training split is only
   ~114k records, so subsampling is unnecessary for the main experiments; it is
   retained for the robustness sweep and the SVM size cap.

5. **Environment tool.** Conda/Miniconda is used (the student's existing setup)
   rather than a plain virtual environment.

6. **Hyperparameter search.** Decided per model — grid search for small search
   spaces, revisited for the expensive networks — rather than uniform grid
   search.

7. **Data validation.** A lightweight, transparent pandas-based validation is
   used instead of the heavier `great-expectations` library originally listed.

8. **Execution order.** The dataset/EDA/preprocessing work was done before the
   literature-grounding step, because EDA findings usefully inform the
   methodology. The literature-grounding step is now complete and still
   precedes feature engineering, as planned.

None of these changes the scientific question, the model set, or the two-phase
scope.

---

## 5. Key findings from the data so far

- **Heavy duplication, unevenly spread.** 36.8% of records were duplicates,
  concentrated in the Generic and DoS attack types (Generic fell from ~58,900 to
  7,599 records, DoS from ~16,400 to 5,500) — these attacks produce highly
  repetitive flow records.
- **Severe class imbalance.** After deduplication the largest class outnumbers
  the smallest by ~501× (Normal 85,722 vs Worms 171). This is the central
  challenge the project's research questions address.
- **The data is high-dimensional.** The first two principal components capture
  only ~34% of the variance — there is no simple low-dimensional rule;
  the models will have real work to do.
- **Features are skewed and redundant.** 28 of 39 numeric features are heavily
  skewed (motivating the scaling policy); 18 feature pairs are almost perfectly
  correlated (motivating the feature-selection study in Part 5).
- **Connection-tracking features look most discriminative.** An ANOVA test
  ranked the connection-count features (`ct_dst_sport_ltm`, `ct_srv_dst`, …) as
  the strongest separators of attack types — a useful early signal for the
  feature-engineering work.

---

## 6. Method-grounding finding: RFA and the bigram technique

The proposal's feature-engineering study (§5.1) is built on **Recursive Feature
Addition (RFA)** and a **bigram technique**, both attributed to Hamed, Dara &
Kremer (2018) — Prof. Dara's own work. Before implementing them, the original
method was studied in full (Section 2.4). This surfaced two points where the
proposal's description differs materially from the published method. Neither is
a mistake to be "corrected" silently — both are design decisions that need your
direction, because you co-invented the original technique.

### 6.1 What the original paper does

**RFA** is a *forward* feature-selection method: it starts from an empty feature
set and adds features one at a time, keeping the most useful. In the original
paper, the "usefulness" of a candidate feature is measured by a **Support Vector
Machine** — specifically, by how much that feature changes the SVM's internal
cost function. RFA continues until *every* feature has been ranked; the output
is a complete ranking, not a shortlist.

**The bigram technique** in the original paper operates on the **raw bytes of a
packet's payload** — the actual content of network traffic, treated like text.
It counts every distinct two-character sequence ("bigram"), much as one might
count letter pairs in a document. It was designed to catch stealthy attacks
whose signature is hidden in the payload content, and it was applied to a
dataset (ISCX) that *contains* payload data.

### 6.2 What the proposal describes

The proposal describes **RFA** with a **Random Forest** as the evaluator (not an
SVM), scoring candidate features by the gain in **validation F1**, and stopping
early when improvement stalls (rather than ranking every feature).

The proposal describes the **bigram technique** as **pairing consecutive flow
records** within a sliding window to encode state-transition patterns — not as
character pairs in packet payloads.

### 6.3 The two discrepancies — and why they matter

**Discrepancy 1 — RFA's engine.** The proposal's RFA (Random Forest + F1 gain +
early stopping) is RFA *in spirit* — a forward, add-one-feature method — but it
is not the original algorithm (SVM + cost-function change + full ranking). The
proposal's version is a reasonable, standard, defensible method and it runs fine
on this project's data. The only real issue is honesty of citation: it should be
described as "a forward feature-addition method in the spirit of Hamed, Dara &
Kremer (2018)", not presented as their exact algorithm. **Decision needed:** is
the Random-Forest version acceptable as the project's RFA, or should the
original SVM version be implemented for fidelity — or both, run as a comparison?

**Discrepancy 2 — the bigram technique (the more serious one).** The proposal's
"bigram" (pairs of flow records) and the original "bigram" (character pairs in
packet payloads) are *fundamentally different techniques that happen to share a
name*. This matters for a concrete, unavoidable reason: **the partitioned
UNSW-NB15 benchmark contains no packet payloads at all** — only summary
statistics about each network flow. The original payload-bigram technique
therefore *cannot be run on this dataset*. The proposal's flow-record-pairing
idea is a separate, new construction. **Decision needed:** (a) reframe the
flow-record pairing honestly as an *adaptation inspired by* the bigram
technique, not the bigram technique itself; (b) add a payload-bearing dataset so
the genuine technique can be replicated; or (c) drop the "bigram" framing
altogether.

A full written specification of both methods — the original algorithms, the
proposal's variants, a side-by-side comparison table, and a working
implementation spec — is in the repository at `docs/rfa_bigram_spec.md`, ready
to walk through in the meeting.

---

## 7. What to expect going forward

- **Immediately after this meeting — protocol freeze:** with the RFA + bigram
  questions resolved, freeze the experimental protocol — the exact, written-down
  rules every model will be evaluated under (splits, seeds, metrics, validation
  scheme), fixed once so every comparison is fair and nothing can be tuned after
  results are seen.
- **Feature engineering:** implement and compare filter selection, RFA, and the
  flow-pair ("bigram") features — in whatever form the meeting settles on.
- **Phase 1 models:** rule-based IDS, Random Forest, SVM, XGBoost, 1D-CNN, and
  LSTM, all evaluated under the unified protocol on binary and multiclass tasks,
  with the class-level analysis answering RQ1 and RQ2.
- **Phase 2 (conditional):** CNN-LSTM, autoencoder, the Kitsune/DeepLog
  baselines, CIC-IDS2017, and the noise-robustness experiments.

A realistic expectation: the models will reach strong binary-classification
performance, but the **rare attack classes (Worms, Shellcode, Analysis,
Backdoor) will be genuinely hard** — and *characterising why* is the
contribution, not a number on a leaderboard.

---

## 8. Honest assessment

### Is the project worth continuing?

**Yes — clearly.** The project is well-scoped, the engineering and data pipeline
are sound, and the methodology is rigorous (leakage-safe, reproducible,
statistically careful). It is on track to deliver a complete, defensible MCTI
thesis: a controlled comparison of IDS model families plus the class-level
analysis, backed by a fully reproducible codebase. Notably, the rigour already
applied — proper deduplication, strict train/test separation, fixed seeds —
exceeds what many published IDS papers actually do. The learning value is also
high: it builds genuine end-to-end fluency in ML-for-security.

### Could the results be publishable?

This is worth an honest answer. A plain "we compared several models on
UNSW-NB15 and here are the F1 scores" would **not** be publishable — that has
been done many times. Publishability depends entirely on the angle, and there
are two credible ones:

1. **A rigorous re-evaluation.** Many UNSW-NB15 studies report inflated results
   precisely because they do not remove duplicate records, so identical samples
   leak between train and test. A paper that quantifies "results with vs without
   deduplication and leakage control" is a genuine, useful contribution — and
   this project is already set up to produce exactly that comparison.
2. **The class-level analysis (RQ1/RQ2).** *Which feature groups predict which
   attacks, and why do rare classes stay hard* is a more original question than
   another accuracy table — and it extends the supervisor's own RFA + bigram
   work to a contemporary dataset.

Realistically: a workshop paper, a student symposium, or a mid-tier journal/
conference is plausible **if** the work leans into rigour and the class-level
analysis; a top-tier venue would need more methodological novelty. Prof. Dara is
the right person to judge the venue and feasibility — this is a key point to
discuss. And it is worth remembering that an MCTI thesis does **not** need to be
published to be a success; publishability is an upside, not a requirement.

---

## 9. Points to raise with Prof. Dara

The first two are **blocking decisions** — the feature-engineering work cannot
be correctly designed until they are settled, and the project is paused until
they are. The rest are confirmations on work already done and forward-planning
items.

### Blocking — method decisions (see Section 6 and `docs/rfa_bigram_spec.md`)

1. **RFA's engine.** Is the Random-Forest / validation-F1 forward-selection
   version acceptable as the project's RFA (cited honestly as "in the spirit
   of" the original), or should the original SVM cost-function RFA be
   implemented for fidelity — or both, run as a comparison?
2. **The bigram technique.** The original payload-bigram method cannot run on
   UNSW-NB15 (the partitioned benchmark has no payload data). Which way forward:
   (a) reframe the flow-record pairing as an adaptation inspired by the bigram
   technique; (b) add a payload-bearing dataset so the genuine technique can be
   replicated; or (c) drop the "bigram" framing? This is your technique, so your
   steer matters most here.

### Confirmations on work already done

3. Confirm the use of the **partitioned UNSW-NB15 benchmark** (vs the full
   release) and that the ~87% / size figures in the proposal refer to the full
   release.
4. Note the **36.8% deduplication** and the resulting dataset size/balance.
5. Confirm the **TensorFlow-version** and **conda** adjustments are acceptable.
6. Agree the **hyperparameter-search** approach (per-model).

### Planning and outlook

7. Discuss the **publication angle** — is the rigour / class-level framing worth
   targeting a venue, and if so which?
8. Request **University of Guelph HPC access** for the Phase 2 deep-learning
   runs (worth starting early — approval can take time).
9. Agree the **end-of-June checkpoint** for the Phase 1 → Phase 2 go/no-go
   decision.
