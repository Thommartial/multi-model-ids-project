#!/usr/bin/env python
"""Run the complete EDA pipeline on the UNSW-NB15 dataset.

Usage:
    python scripts/run_eda.py

Outputs:
    reports/eda/figures/*.png      -- analysis figures
    reports/eda/eda_report_*.md    -- the comprehensive Markdown report
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when this file is run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_unsw_nb15, verify_checksums  # noqa: E402
from src.data.preprocessor import Preprocessor  # noqa: E402
from src.data.validation import validate_dataset  # noqa: E402
from src.eda.complete_eda import EDAnalyzer  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("UNSW-NB15 -- Exploratory Data Analysis")
    print("=" * 60)

    verify_checksums()
    raw = load_unsw_nb15(combine=True)
    validate_dataset(raw, "pooled UNSW-NB15 (raw)")

    # Deduplicate first, so the EDA describes exactly the data the models
    # use -- the preprocessing pipeline removes these same duplicates.
    n_raw = len(raw)
    df = Preprocessor().clean(raw)
    n_duplicates = n_raw - len(df)

    eda = EDAnalyzer(output_dir="reports/eda")
    eda.basic_info(df, "UNSW-NB15", n_raw=n_raw, n_duplicates=n_duplicates)
    eda.class_distribution_analysis(df)
    eda.missing_value_analysis(df)
    eda.feature_statistics_analysis(df)
    eda.correlation_analysis(df)
    eda.attack_pattern_analysis(df)
    eda.dimensionality_visualization(df)
    eda.outlier_detection_analysis(df)
    eda.statistical_tests(df)
    eda.generate_comprehensive_eda_report(df, "UNSW-NB15")

    print("=" * 60)
    print("EDA complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
