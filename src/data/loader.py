"""Dataset loading for the Multi-Model IDS project.

Provides clean, integrity-checked access to the UNSW-NB15 partitioned
benchmark (training + testing CSV files) under ``data/raw/UNSW-NB15/``.
The project later re-splits the pooled records into its own stratified
70/15/15 train/validation/test sets (Part 4), so this module simply
loads the raw records and verifies them.

Provenance
----------
UNSW-NB15 was created by Moustafa & Slay (2015) at the Australian Centre
for Cyber Security. Authoritative source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

The partitioned benchmark used here has 175,341 training and 82,332
testing records (257,673 in total), 45 columns, and labels spanning
normal traffic plus nine attack categories. For automated, reproducible
fetching, :func:`download_unsw_nb15` pulls the published CSV files from
Hugging Face dataset mirrors.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/UNSW-NB15")
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"

# Hugging Face mirrors of the published UNSW-NB15 partitioned CSV files.
DOWNLOAD_URLS = {
    TRAIN_FILE: (
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/"
        "resolve/main/UNSW_NB15_training-set.csv"
    ),
    TEST_FILE: (
        "https://huggingface.co/datasets/dileepa0011/unsw-nb15/"
        "resolve/main/UNSW_NB15_testing-set.csv"
    ),
}

# SHA-256 checksums of the verified raw files (integrity guard).
CHECKSUMS = {
    TRAIN_FILE: "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    TEST_FILE: "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}

# Expected record counts (quick sanity check).
EXPECTED_ROWS = {TRAIN_FILE: 175_341, TEST_FILE: 82_332}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_unsw_nb15(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    """Download the UNSW-NB15 partitioned CSV files if they are absent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOAD_URLS.items():
        dest = raw_dir / name
        if dest.exists() and not force:
            print(f"[loader] {name}: already present")
            continue
        print(f"[loader] downloading {name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        print(f"[loader] saved {name} ({dest.stat().st_size / 1e6:.1f} MB)")


def verify_checksums(raw_dir: Path = RAW_DIR) -> bool:
    """Check the raw files against their known SHA-256 checksums."""
    ok = True
    for name, expected in CHECKSUMS.items():
        path = raw_dir / name
        if not path.exists():
            print(f"[loader] missing: {path}")
            ok = False
            continue
        if _sha256(path) == expected:
            print(f"[loader] {name}: checksum OK")
        else:
            print(f"[loader] {name}: checksum MISMATCH")
            ok = False
    return ok


def load_partition(which: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load a single partition. ``which`` is 'train' or 'test'."""
    if which not in ("train", "test"):
        raise ValueError("which must be 'train' or 'test'")
    name = TRAIN_FILE if which == "train" else TEST_FILE
    return pd.read_csv(raw_dir / name)


def load_unsw_nb15(raw_dir: Path = RAW_DIR, combine: bool = True) -> pd.DataFrame:
    """Load the UNSW-NB15 partitioned benchmark.

    With ``combine=True`` (default) the training and testing partitions
    are pooled into one DataFrame, ready for the project's own stratified
    70/15/15 split. With ``combine=False`` only the training partition is
    returned.
    """
    train = load_partition("train", raw_dir)
    if not combine:
        return train
    test = load_partition("test", raw_dir)
    return pd.concat([train, test], ignore_index=True)
