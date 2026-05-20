"""Exploratory Data Analysis for the Multi-Model IDS project.

The :class:`EDAnalyzer` runs nine complementary analyses on a network-
intrusion dataset and writes a comprehensive Markdown report plus
figures. EDA is the step where we *understand* the data before touching
a model: how imbalanced the classes are, where data-quality problems
hide, which features look discriminative, and how separable the attack
types are. Those findings shape every later decision -- preprocessing,
feature engineering, sampling, and the choice of evaluation metrics.

The analyser is **resumable**: each finished analysis is cached to
``reports/eda/_findings.json``, so the pipeline can be re-run safely and
will skip work that is already done.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.visualization.style import set_academic_style  # noqa: E402

# Columns that are identifiers/targets, not predictive features.
NON_FEATURE_COLS = ["id", "label", "attack_cat"]
CATEGORICAL_COLS = ["proto", "service", "state"]
# Interpretable features used for the per-attack pattern comparison.
PATTERN_FEATURES = [
    "dur",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sttl",
    "ct_state_ttl",
    "ct_srv_src",
    "smean",
]


class EDAnalyzer:
    """Run a full, resumable EDA pass and emit figures + a Markdown report."""

    def __init__(self, output_dir: str = "reports/eda"):
        set_academic_style()
        self.report_dir = Path(output_dir)
        self.fig_dir = self.report_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self._findings_file = self.report_dir / "_findings.json"
        if self._findings_file.exists():
            self.findings = json.loads(self._findings_file.read_text())
        else:
            self.findings = {}

    # ----- helpers ---------------------------------------------------
    def _numeric_features(self, df: pd.DataFrame) -> list[str]:
        return [
            c
            for c in df.columns
            if c not in NON_FEATURE_COLS
            and c not in CATEGORICAL_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    def _save(self, fig, name: str) -> str:
        fig.savefig(self.fig_dir / name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return name

    def _persist(self) -> None:
        self._findings_file.write_text(json.dumps(self.findings, indent=2))

    def _done(self, key: str) -> bool:
        if key in self.findings:
            print(f"[eda] {key} -- cached, skipping")
            return True
        return False

    # ----- 1. basic information -------------------------------------
    def basic_info(self, df: pd.DataFrame, name: str = "UNSW-NB15") -> None:
        key = "1. Basic information"
        if self._done(key):
            return
        mem = df.memory_usage(deep=True).sum() / 1e6
        # Every row carries a unique `id`, so duplicates only become
        # visible once that identifier column is excluded.
        dup_cols = [c for c in df.columns if c != "id"]
        dups = int(df[dup_cols].duplicated().sum())
        n_num = len(self._numeric_features(df))
        self.findings[key] = (
            f"- **Records:** {df.shape[0]:,}\n"
            f"- **Columns:** {df.shape[1]} "
            f"({n_num} numeric features, {len(CATEGORICAL_COLS)} categorical "
            f"features, 3 identifier/target columns)\n"
            f"- **In-memory size:** {mem:.1f} MB\n"
            f"- **Duplicate records** (identical across every field except "
            f"the `id` index): {dups:,} ({100 * dups / len(df):.1f}%) -- "
            f"removed during preprocessing to prevent train/test leakage\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 2. class distribution ------------------------------------
    def class_distribution_analysis(self, df: pd.DataFrame) -> None:
        key = "2. Class distribution"
        if self._done(key):
            return
        binary = df["label"].value_counts().sort_index()
        multi = df["attack_cat"].value_counts()
        imbalance = multi.max() / multi.min()

        lines = ["| Attack category | Records | Share |", "|---|---:|---:|"]
        for cat, cnt in multi.items():
            lines.append(f"| {cat} | {cnt:,} | {100 * cnt / len(df):.2f}% |")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        b = binary.rename({0: "Benign", 1: "Attack"})
        sns.barplot(
            x=list(b.index),
            y=list(b.values),
            hue=list(b.index),
            palette=["#4c72b0", "#c44e52"],
            legend=False,
            ax=axes[0],
        )
        axes[0].set_title("Binary label distribution")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Records")
        sns.barplot(
            x=list(multi.index),
            y=list(multi.values),
            hue=list(multi.index),
            legend=False,
            ax=axes[1],
        )
        axes[1].set_yscale("log")
        axes[1].set_title("Attack-category distribution (log scale)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Records (log scale)")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fname = self._save(fig, "01_class_distribution.png")

        self.findings[key] = (
            f"Binary labels -- benign: {binary.get(0, 0):,} "
            f"({100 * binary.get(0, 0) / len(df):.1f}%), "
            f"attack: {binary.get(1, 0):,} "
            f"({100 * binary.get(1, 0) / len(df):.1f}%).\n\n"
            f"Multiclass imbalance ratio (largest / smallest class): "
            f"**{imbalance:,.0f}x**.\n\n" + "\n".join(lines) + "\n\n"
            f"![class distribution](figures/{fname})\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 3. missing / invalid values ------------------------------
    def missing_value_analysis(self, df: pd.DataFrame) -> None:
        key = "3. Missing and invalid values"
        if self._done(key):
            return
        nan_cols = int((df.isna().sum() > 0).sum())
        inf_total = sum(int(np.isinf(df[c]).sum()) for c in self._numeric_features(df))
        dash = {
            c: int((df[c].astype(str) == "-").sum())
            for c in CATEGORICAL_COLS
            if c in df.columns and (df[c].astype(str) == "-").any()
        }
        self.findings[key] = (
            f"- Columns containing NaN values: {nan_cols}\n"
            f"- Infinite values across numeric features: {inf_total:,}\n"
            f"- Placeholder '-' values in categoricals: "
            f"{', '.join(f'{k}={v:,}' for k, v in dash.items()) or 'none'}\n\n"
            f"UNSW-NB15's published partitions are already cleaned; the '-' "
            f"placeholder in `service` simply means 'no application protocol' "
            f"and is treated as its own category during preprocessing.\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 4. feature statistics ------------------------------------
    def feature_statistics_analysis(self, df: pd.DataFrame) -> None:
        key = "4. Feature statistics"
        if self._done(key):
            return
        feats = self._numeric_features(df)
        skew = df[feats].skew().abs().sort_values(ascending=False)
        n_skewed = int((skew > 2).sum())

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(x=skew.head(20).values, y=skew.head(20).index, ax=ax, color="#4c72b0")
        ax.set_title("Top 20 features by absolute skewness")
        ax.set_xlabel("|skewness|")
        fig.tight_layout()
        fname = self._save(fig, "02_feature_skewness.png")

        self.findings[key] = (
            f"Of {len(feats)} numeric features, **{n_skewed}** are highly "
            f"skewed (|skew| > 2). Heavy skew means raw feature scales are "
            f"very uneven -- this is why neural-network inputs need z-score "
            f"standardisation while tree models can be left unscaled.\n\n"
            f"![feature skewness](figures/{fname})\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 5. correlation -------------------------------------------
    def correlation_analysis(self, df: pd.DataFrame) -> None:
        key = "5. Feature correlation"
        if self._done(key):
            return
        feats = self._numeric_features(df)
        corr = df[feats].corr()

        pairs = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                r = corr.iloc[i, j]
                if abs(r) > 0.90:
                    pairs.append((feats[i], feats[j], r))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)

        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6}, ax=ax)
        ax.set_title("Numeric feature correlation matrix")
        fig.tight_layout()
        fname = self._save(fig, "03_correlation_matrix.png")

        listed = "\n".join(f"  - {a} ~ {b}: r = {r:.3f}" for a, b, r in pairs[:10])
        self.findings[key] = (
            f"**{len(pairs)}** feature pairs are highly correlated (|r| > 0.90). "
            f"Redundant features add cost and noise without adding information "
            f"-- direct motivation for the feature-selection work in Part 5.\n\n"
            f"Most-correlated pairs:\n{listed or '  - none'}\n\n"
            f"![correlation matrix](figures/{fname})\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 6. attack patterns ---------------------------------------
    def attack_pattern_analysis(self, df: pd.DataFrame) -> None:
        key = "6. Attack-category patterns"
        if self._done(key):
            return
        feats = [f for f in PATTERN_FEATURES if f in df.columns]
        means = df.groupby("attack_cat")[feats].mean()
        znorm = (means - means.mean()) / means.std(ddof=0)

        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(
            znorm, cmap="coolwarm", center=0, annot=True, fmt=".1f", cbar_kws={"shrink": 0.7}, ax=ax
        )
        ax.set_title("Per-attack feature signatures (z-normalised feature means)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Attack category")
        fig.tight_layout()
        fname = self._save(fig, "04_attack_patterns.png")

        self.findings[key] = (
            "Each attack category has a distinct 'signature' across key flow "
            "features (z-normalised means below). Where two categories share a "
            "similar signature, a classifier will tend to confuse them.\n\n"
            f"![attack patterns](figures/{fname})\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 7. dimensionality ----------------------------------------
    def dimensionality_visualization(self, df: pd.DataFrame, sample: int = 5000) -> None:
        key = "7. Dimensionality and separability"
        if self._done(key):
            return
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        feats = self._numeric_features(df)
        samp = df.sample(n=min(sample, len(df)), random_state=42)
        x = StandardScaler().fit_transform(samp[feats].replace([np.inf, -np.inf], np.nan).fillna(0))
        cats = samp["attack_cat"].to_numpy()

        pca = PCA(n_components=2, random_state=42)
        xp = pca.fit_transform(x)
        evr = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(x=xp[:, 0], y=xp[:, 1], hue=cats, s=12, alpha=0.6, ax=ax)
        ax.set_title(f"PCA projection ({len(samp):,}-record sample)")
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        f1 = self._save(fig, "05_pca_projection.png")

        ts_n = min(1500, len(samp))
        xt = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(
            x[:ts_n]
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(x=xt[:, 0], y=xt[:, 1], hue=cats[:ts_n], s=14, alpha=0.7, ax=ax)
        ax.set_title(f"t-SNE projection ({ts_n:,}-record sample)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        f2 = self._save(fig, "06_tsne_projection.png")

        self.findings[key] = (
            f"The first two principal components capture only "
            f"{evr[:2].sum() * 100:.1f}% of total variance -- the data is "
            f"genuinely high-dimensional, so no trivial 2-D rule separates the "
            f"classes. PCA and t-SNE both show normal traffic and the larger "
            f"attack types forming visible structure, while the rare classes "
            f"scatter thinly -- a visual preview of why they are hard to "
            f"detect.\n\n"
            f"![PCA](figures/{f1})\n\n![t-SNE](figures/{f2})\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 8. outliers ----------------------------------------------
    def outlier_detection_analysis(self, df: pd.DataFrame) -> None:
        key = "8. Outlier analysis"
        if self._done(key):
            return
        feats = self._numeric_features(df)
        q1 = df[feats].quantile(0.25)
        q3 = df[feats].quantile(0.75)
        iqr = q3 - q1
        frac = {}
        for c in feats:
            if iqr[c] == 0:
                frac[c] = 0.0
                continue
            lo, hi = q1[c] - 1.5 * iqr[c], q3[c] + 1.5 * iqr[c]
            frac[c] = float(((df[c] < lo) | (df[c] > hi)).mean())
        s = pd.Series(frac).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(x=s.head(20).values * 100, y=s.head(20).index, ax=ax, color="#dd8452")
        ax.set_title("Top 20 features by IQR-outlier fraction")
        ax.set_xlabel("Records flagged as outliers (%)")
        fig.tight_layout()
        fname = self._save(fig, "07_outlier_fraction.png")

        heavy = int((s > 0.10).sum())
        self.findings[key] = (
            f"By the 1.5x-IQR rule, **{heavy}** features flag more than 10% of "
            f"records as outliers. In network traffic these 'outliers' are "
            f"usually genuine extreme behaviour (bursts, scans), not errors -- "
            f"so they are kept, but they reinforce the need for robust "
            f"scaling.\n\n"
            f"![outlier fraction](figures/{fname})\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- 9. statistical tests -------------------------------------
    def statistical_tests(self, df: pd.DataFrame) -> None:
        key = "9. Statistical tests (ANOVA)"
        if self._done(key):
            return
        from sklearn.feature_selection import f_classif

        feats = self._numeric_features(df)
        x = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        f_stat, _ = f_classif(x, df["attack_cat"])
        s = (
            pd.Series(f_stat, index=feats)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(x=s.head(20).values, y=s.head(20).index, ax=ax, color="#55a868")
        ax.set_title("Top 20 features by ANOVA F-statistic (attack-category separation)")
        ax.set_xlabel("ANOVA F-statistic")
        fig.tight_layout()
        fname = self._save(fig, "08_anova_discriminative.png")

        top = ", ".join(s.head(8).index)
        self.findings[key] = (
            "A one-way ANOVA per feature measures how strongly its mean "
            "differs across attack categories -- a high F-statistic means the "
            "feature separates attack types well.\n\n"
            f"Most discriminative features: **{top}**.\n\n"
            f"![ANOVA discriminative features](figures/{fname})\n"
        )
        self._persist()
        print(f"[eda] {key} done")

    # ----- report ----------------------------------------------------
    def generate_comprehensive_eda_report(self, df: pd.DataFrame, name: str) -> Path | None:
        expected = 9
        if len(self.findings) < expected:
            missing = expected - len(self.findings)
            print(
                f"[eda] report not written -- {missing} analysis section(s) "
                f"still pending; re-run scripts/run_eda.py"
            )
            return None

        today = _dt.date.today().isoformat()
        parts = [
            f"# Exploratory Data Analysis - {name}",
            "",
            f"*Multi-Model IDS project - generated {today} by `scripts/run_eda.py`.*",
            "",
            "This report is produced automatically from the raw data. It "
            "summarises nine analyses that together describe the dataset and "
            "motivate the preprocessing, feature-engineering, sampling, and "
            "evaluation choices made later in the project.",
            "",
        ]
        for section in sorted(self.findings):
            parts += [f"## {section}", "", self.findings[section], ""]
        parts += [
            "## Implications for the project",
            "",
            "- **Severe class imbalance** drives the use of stratified "
            "sampling, SMOTE / class-weighted loss, and imbalance-aware "
            "metrics (macro-F1, PR-AUC, per-class recall) rather than plain "
            "accuracy.",
            "- **Heavy feature skew and outliers** mean neural-network inputs "
            "must be z-score standardised; tree models can stay unscaled.",
            "- **Highly correlated feature pairs** confirm redundancy and "
            "motivate the feature-selection comparison in Part 5.",
            "- **Low 2-D variance and overlapping rare classes** preview RQ2: "
            "some attack categories are intrinsically hard to detect.",
            "",
        ]
        out = self.report_dir / f"eda_report_{name}.md"
        out.write_text("\n".join(parts), encoding="utf-8")
        print(f"[eda] report -> {out}")
        return out
