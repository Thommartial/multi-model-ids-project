# Feature-engineering ablation outputs

Outputs of `scripts/run_ablation.py` — the **five-condition** comparison
from `docs/experimental_protocol.md` §10.1, run under the protocol's
§8 statistical rigour.

## The conditions (all with a Random Forest downstream classifier)

| # | Condition | Feature source |
|---|---|---|
| 1 | `baseline_all_features` | every column after preprocessing |
| 2 | `filter_top{k}` | top-k from the consensus filter (`scripts/run_filter_selection.py`) |
| 3 | `rfa_svm_original` | top-k from the SVM cost-function RFA (Hamed et al. 2018) |
| 4 | `rfa_rf_proposal` | top-k from the RF / val-macro-F1 RFA (the proposal's variant) |
| 5 | `rfa_plus_flow_pair` | the better of (3) vs (4) + flow-pair (`diff_` + `ratio_`) features |

Conditions 3, 4, and 5 are **skipped automatically** if the corresponding
ranking files do not exist in `reports/feature_selection/`. The runner
prints a warning and proceeds with whichever conditions it can run.

## Files written per task

For `--task multiclass` (default) and again for `--task binary`:

| File | Contents |
|---|---|
| `ablation_per_seed_<task>.csv` | one row per (condition, seed) with every val/test metric |
| `ablation_summary_<task>.csv` | one row per condition: mean ± std + 95% bootstrap CI on the primary metric |
| `ablation_comparisons_<task>.csv` | pairwise paired Wilcoxon + Holm–Bonferroni p-values + claim flag |
| `ablation_summary_<task>.md` | human-readable markdown summary |

## How to generate the canonical run

The ablation expects the upstream rankings to exist first:

```bash
# (one-off, slow) RFA rankings -- once these are on disk, all five
#                                  conditions become available
python scripts/run_rfa.py

# (fast) the ablation itself: 5 conditions x 5 seeds, RF defaults
python scripts/run_ablation.py
python scripts/run_ablation.py --task binary
```

You can also run a faster preview with fewer seeds while debugging:

```bash
python scripts/run_ablation.py --seeds 42 43 --n-estimators 50 --max-depth 10
```

## Reading the comparisons table

* `delta_a_minus_b` = mean test-macro-F1 of condition A minus condition B.
* `p_wilcoxon` = paired Wilcoxon signed-rank test on the per-seed metric.
* `p_holm` = Holm–Bonferroni-corrected p-value across the full pairwise
  family.
* `claim` = `True` iff `p_holm < 0.05` **and** `|delta| >= 0.005`
  (the effect-size floor from protocol §8).

A "win" is only claimed when **both** gates pass. A statistically
significant but tiny improvement is not claimed; neither is a large
nominal improvement that does not survive multiple-comparisons
correction.

## Preview / canonical

This folder ships empty in the repo: ablation runs depend on the RFA
rankings (which live on your machine, not in git, by design — they are
the output of a 1–3-hour local job). Run `scripts/run_ablation.py`
locally to populate this folder.
