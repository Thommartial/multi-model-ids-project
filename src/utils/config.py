"""Central experiment configuration.

Every tunable parameter for the project lives here, in one typed place,
instead of being hard-coded throughout the codebase. Configuration is
expressed as dataclasses -- so an IDE and the type-checker can catch
mistakes -- and can be loaded from / saved to YAML files in
``experiments/configs/``.

Why this matters: an experiment is only reproducible if you know exactly
which settings produced it. Keeping every parameter in one versioned
file means each result can be traced back to the precise configuration
behind it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

# Default location of the baseline configuration file.
DEFAULT_CONFIG_PATH = Path("experiments/configs/default.yaml")


@dataclass
class DataConfig:
    """Dataset selection, split ratios and stratified sample sizes."""

    dataset: str = "UNSW-NB15"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    subsampled_dir: str = "data/subsampled"
    # Stratified train/validation/test split (proposal Section 5.1)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Stratified sample sizes. After deduplication the training split is
    # ~114k records, so the binary/multiclass caps are effectively no-ops
    # (the full split is used); the robustness sweep uses a subsample.
    sample_size_binary: int = 120_000
    sample_size_multiclass: int = 120_000
    sample_size_robustness: int = 75_000

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {total:.3f}")


@dataclass
class FeatureConfig:
    """Feature-engineering strategy settings (proposal Section 5.1)."""

    methods: list[str] = field(default_factory=lambda: ["filter", "rfa", "rfa_bigram"])
    n_filter_features: int = 20
    rfa_evaluator: str = "random_forest"
    bigram_window: int = 10
    sequence_window_min: int = 5
    sequence_window_max: int = 20


@dataclass
class ModelConfig:
    """Which models to run in each phase."""

    phase1: list[str] = field(
        default_factory=lambda: [
            "rule_based",
            "random_forest",
            "svm",
            "xgboost",
            "cnn1d",
            "lstm",
        ]
    )
    phase2: list[str] = field(
        default_factory=lambda: ["cnn_lstm", "autoencoder", "kitsune", "deeplog"]
    )
    # Cap RBF-SVM training set; larger subsets use SGD (proposal risk register).
    svm_max_records: int = 50_000


@dataclass
class TrainingConfig:
    """Training and statistical-repetition settings (proposal Section 6.3)."""

    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 101112])
    cv_folds: int = 5
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    hyperparameter_search: str = "grid"  # decided per model (decisions log)


@dataclass
class EvalConfig:
    """Evaluation metric and statistical-rigour settings."""

    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    primary_binary_metric: str = "pr_auc"
    primary_multiclass_metric: str = "macro_f1"


@dataclass
class ExperimentConfig:
    """Top-level configuration aggregating every section."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load a configuration from a YAML file."""
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            features=FeatureConfig(**raw.get("features", {})),
            models=ModelConfig(**raw.get("models", {})),
            training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save this configuration to a YAML file."""
        with open(path, "w") as handle:
            yaml.safe_dump(asdict(self), handle, sort_keys=False)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Load the experiment configuration, falling back to built-in defaults."""
    path = Path(path)
    if path.exists():
        return ExperimentConfig.from_yaml(path)
    return ExperimentConfig()
