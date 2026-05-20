# Multi-Model Intrusion Detection System

Comparative feature engineering and class-level performance analysis of rule-based,
classical machine-learning, and deep-learning intrusion detection (IDS) models,
evaluated under a single unified experimental framework.

**CIS\*6560 Research Project** — Master of Cybersecurity and Threat Intelligence,
University of Guelph.

- **Student:** Thomas Martial Ekwelle Epalle (1406946)
- **Supervisor:** Prof. Rozita Dara

## Datasets

- **UNSW-NB15** — primary benchmark (Phase 1)
- **CIC-IDS2017** — extension (Phase 2)

## Project status

Under active development. The full work plan is in `Master_Task_Breakdown.md`
(parent folder). A complete setup-and-reproduction guide will be added later
(Task 15.1).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Core implementation: data, EDA, models, evaluation, visualization, utils |
| `data/` | Datasets — versioned with DVC, not git |
| `experiments/` | Configs, results, checkpoints, logs |
| `dashboard/` | Streamlit progress + results dashboard |
| `notebooks/` | Exploratory development notebooks |
| `reports/` | Figures, tables, weekly progress reports |
| `references/` | Cited papers (PDFs) |
| `tests/` | Unit tests |

## License

MIT — see [`LICENSE`](LICENSE).
