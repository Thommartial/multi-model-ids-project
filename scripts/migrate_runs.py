"""One-shot migration: ``reports/runs/<model>/seed_*/`` → ``reports/runs/<model>/<task>/seed_*/``.

Reads each seed folder's ``metadata.json`` for its ``task`` value and moves
the folder into the new task-aware layout.  Idempotent (skips folders
already in the new layout).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path("reports/runs")


def migrate(root: Path = ROOT) -> int:
    if not root.is_dir():
        print(f"[migrate] {root} does not exist; nothing to do.")
        return 0

    moved = 0
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # Skip if every immediate child is already a task subfolder
        # (multiclass / binary), i.e. the model is already migrated.
        children = [p for p in model_dir.iterdir() if p.is_dir()]
        seed_dirs = [p for p in children if p.name.startswith("seed_")]
        if not seed_dirs:
            print(f"[migrate] {model_dir}: already migrated (no top-level seed_* dirs)")
            continue

        for seed_dir in seed_dirs:
            meta_path = seed_dir / "metadata.json"
            if not meta_path.exists():
                print(f"[migrate] {seed_dir}: no metadata.json -- skipping")
                continue
            with open(meta_path) as fh:
                task = json.load(fh).get("task")
            if task not in {"multiclass", "binary"}:
                print(f"[migrate] {seed_dir}: unknown task {task!r} -- skipping")
                continue
            target = model_dir / task / seed_dir.name
            if target.exists():
                print(f"[migrate] {target} already exists -- skipping {seed_dir}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(seed_dir), str(target))
            print(f"[migrate] moved {seed_dir} -> {target}")
            moved += 1

    print()
    print(f"[migrate] done; {moved} seed folder(s) moved.")
    return moved


if __name__ == "__main__":
    sys.exit(0 if migrate() >= 0 else 1)
