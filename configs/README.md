# `configs/` — protocol-frozen experimental configuration

Version-controlled configuration files that the experimental protocol
(`docs/experimental_protocol.md`) treats as part of the experimental record.

- `hparam_spaces/<model>.yaml` — the hyperparameter search space for each model
  family. Frozen 2026-05-26. Any change requires a §13 amendment entry in the
  protocol document and a re-run of any results that depended on the previous
  space.
- `feature_groups.yaml` — the semantic feature groups used in the RQ1
  analysis (per-group classifiers, per-class recall comparison).
