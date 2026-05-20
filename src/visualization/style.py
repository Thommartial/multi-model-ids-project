"""Shared plotting style for academic-quality figures.

A single call to :func:`set_academic_style` configures Matplotlib and
Seaborn for publication-grade output. It is used by both the EDA figures
(Part 3) and the publication figures (Part 13) so that every figure in
the project shares one consistent, professional look.

Parameter choices and why they matter for an academic report:

* **Colour-blind-safe palette** -- Seaborn's ``colorblind`` palette is
  distinguishable by readers with the common forms of colour-vision
  deficiency; many journals now require this.
* **300 DPI export** -- the standard minimum resolution for print
  figures; 120 DPI screen DPI keeps interactive work comfortable.
* **"paper" context** -- Seaborn scales fonts and line widths for a
  printed page rather than a screen.
* **Restrained grid and thin axes** -- keeps the ink-to-data ratio low
  so the data, not the chart furniture, draws the eye.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn's colour-blind-safe qualitative palette (10 distinct colours).
ACADEMIC_PALETTE = "colorblind"


def set_academic_style() -> None:
    """Configure Matplotlib + Seaborn for publication-quality figures."""
    sns.set_theme(context="paper", style="whitegrid", palette=ACADEMIC_PALETTE)
    plt.rcParams.update(
        {
            "savefig.dpi": 300,  # publication-resolution export
            "figure.dpi": 120,  # comfortable on screen
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.30,
            "grid.linewidth": 0.6,
        }
    )
