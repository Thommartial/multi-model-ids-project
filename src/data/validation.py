"""Lightweight data-quality validation for the raw UNSW-NB15 files.

Before any analysis or modelling, the raw data is checked against a set
of expectations -- correct schema, valid label values, known attack
categories, no missing labels. Catching a malformed file here prevents
silent errors much later in the pipeline.

Note: the plan originally named the ``great-expectations`` library for
this step. Plain, explicit pandas checks are used instead -- they are
easier to read, audit, and defend, and avoid a heavy dependency for what
is a short, well-defined set of rules.
"""

from __future__ import annotations

import pandas as pd

EXPECTED_N_COLUMNS = 45
KEY_COLUMNS = ["dur", "proto", "service", "state", "attack_cat", "label"]
ATTACK_CATEGORIES = {
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
}


def validate_dataset(df: pd.DataFrame, name: str = "dataset") -> dict[str, bool]:
    """Run data-quality checks on a UNSW-NB15 DataFrame.

    Returns a mapping of check name -> pass/fail, and prints a short
    report. A failing check signals a malformed or unexpected file.
    """
    checks: dict[str, bool] = {
        "column_count_is_45": df.shape[1] == EXPECTED_N_COLUMNS,
        "key_columns_present": all(c in df.columns for c in KEY_COLUMNS),
        "label_is_binary": set(df["label"].dropna().unique()) <= {0, 1},
        "label_has_no_missing": bool(df["label"].notna().all()),
        "attack_cat_values_known": (set(df["attack_cat"].dropna().unique()) <= ATTACK_CATEGORIES),
        "no_fully_empty_columns": not bool(df.isna().all().any()),
        "row_count_positive": len(df) > 0,
    }

    print(f"[validation] {name}:")
    for check, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {check}")
    if not all(checks.values()):
        print(f"[validation] {name}: SOME CHECKS FAILED")
    return checks
