# Filter-based feature-selection outputs

Outputs of `scripts/run_filter_selection.py` — the **filter** condition of
the experimental protocol's feature-engineering ablation
(`docs/experimental_protocol.md` §10.1 condition 2 / Part 5.1).

The filter is the consensus of two complementary rankings, fitted on the
**training fold only**:

- Extra-Trees impurity importance (`class_weight="balanced"`)
- Mutual information with the label (`mutual_info_classif`, stratified
  subsample for tractability)

The consensus rank is the mean of the two per-method ranks; lower is better.

## Files

| File | Meaning |
|---|---|
| `filter_ranking_<target>.csv` | Every input feature, both per-method scores and ranks, and the consensus rank — sorted best first. |
| `filter_selected_top<k>_<target>_<mode>.json` | The top-k feature names selected at that k / mode. |

## Preview vs canonical runs

The rankings committed here were produced with **reduced parameters** to
keep the verification run fast (`--et-trees 50 --mi-sample 5000`). This is
enough to verify the pipeline works end-to-end but does not give the
ranking the project will use in the headline ablation.

To regenerate the **canonical** ranking — Extra-Trees with 400 estimators
and MI on a 30,000-row stratified subsample — run with defaults:

```bash
python scripts/run_filter_selection.py
```

For the binary task instead of multiclass:

```bash
python scripts/run_filter_selection.py --target label
```

Both targets are run before the Part 5.4 ablation.

## Reproducibility

Both methods are seeded with `DEFAULT_SEED = 42` (see
`src.utils.reproducibility`). Re-running the script with the same data and
the same parameters reproduces the ranking exactly.

---

## RFA outputs (Part 5.2)

`scripts/run_rfa.py` produces the **RFA** condition of the ablation
(`docs/experimental_protocol.md` §10.1 conditions 3 and 4). Two variants,
both per the supervisor decisions recorded 2026-05-22:

- The **proposal's variant** — Random Forest evaluator, validation macro-F1,
  patience-based early stopping; multiclass target `attack_cat`.
- The **original** (Hamed, Dara & Kremer 2018) — SVM-RBF cost-function
  approximation, binary target `label`. The SVM uses a 5,000-row stratified
  subsample by default (the kernel matrix is *O(N²)*); the first feature is
  selected by mutual information as the empty-set initialisation, since
  Algorithm 2 does not specify it.

To generate the canonical RFA artefacts on your machine:

```bash
# Both variants. Takes 1-3 hours on a laptop.
python scripts/run_rfa.py

# Just the RF variant, capped at 30 features (faster — useful for a demo).
python scripts/run_rfa.py --variant rf --max-features 30

# Just the SVM variant (binary target).
python scripts/run_rfa.py --variant svm
```

Each run writes a `rfa_ranking_<tag>.csv`, `rfa_selection_path_<tag>.csv`,
`rfa_hyperparams_<tag>.json`, and `rfa_selection_path_<tag>.png` under this
folder.

### Known limitation — SVM RFA on sparse features

The Guyon–Weston-style cost-function approximation
(`½αᵀ(H − H₍₊ᵢ₎)α`) is computed on the support vectors of the current SVM.
When the candidate feature is a sparse one-hot column (most rows = 0), the
pairwise differences on the support vectors are mostly zero and the kernel
update collapses to identity, so the score saturates at zero. At small
subsamples (≤ 2,000) most one-hot columns end up in this state and the
tail of the ranking is essentially arbitrary. The 5,000-row default is the
smallest subsample at which the approximation is robust on UNSW-NB15's
one-hot encoding; do not run below that.

