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
