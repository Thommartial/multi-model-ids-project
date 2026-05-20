# Exploratory Data Analysis - UNSW-NB15

*Multi-Model IDS project - generated 2026-05-20 by `scripts/run_eda.py`.*

This report is produced automatically from the raw data. It summarises nine analyses that together describe the dataset and motivate the preprocessing, feature-engineering, sampling, and evaluation choices made later in the project.

## 1. Basic information

- **Raw dataset (as delivered):** 257,673 records
- **Duplicate records removed:** 94,928 (36.8%) -- exact duplicates once the `id` index is excluded; removed to prevent train/test leakage (proposal Section 5.1)
- **Records analysed below (after deduplication):** 162,745
- **Features:** 39 numeric + 3 categorical (the `id` index and the `label` / `attack_cat` targets are excluded)
- **In-memory size:** 59.8 MB


## 2. Class distribution

Binary labels -- benign: 85,722 (52.7%), attack: 77,023 (47.3%).

Multiclass imbalance ratio (largest / smallest class): **501x**.

| Attack category | Records | Share |
|---|---:|---:|
| Normal | 85,722 | 52.67% |
| Exploits | 27,434 | 16.86% |
| Fuzzers | 20,960 | 12.88% |
| Reconnaissance | 9,991 | 6.14% |
| Generic | 7,599 | 4.67% |
| DoS | 5,500 | 3.38% |
| Analysis | 2,032 | 1.25% |
| Backdoor | 1,880 | 1.16% |
| Shellcode | 1,456 | 0.89% |
| Worms | 171 | 0.11% |

![class distribution](figures/01_class_distribution.png)


## 3. Missing and invalid values

- Columns containing NaN values: 0
- Infinite values across numeric features: 0
- Placeholder '-' values in categoricals: service=101,287

UNSW-NB15's published partitions are already cleaned; the '-' placeholder in `service` simply means 'no application protocol' and is treated as its own category during preprocessing.


## 4. Feature statistics

Of 39 numeric features, **32** are highly skewed (|skew| > 2). Heavy skew means raw feature scales are very uneven -- this is why neural-network inputs need z-score standardisation while tree models can be left unscaled.

![feature skewness](figures/02_feature_skewness.png)


## 5. Feature correlation

**12** feature pairs are highly correlated (|r| > 0.90). Redundant features add cost and noise without adding information -- direct motivation for the feature-selection work in Part 5.

Most-correlated pairs:
  - is_ftp_login ~ ct_ftp_cmd: r = 0.999
  - dbytes ~ dloss: r = 0.997
  - sbytes ~ sloss: r = 0.997
  - dpkts ~ dloss: r = 0.980
  - swin ~ dwin: r = 0.975
  - dpkts ~ dbytes: r = 0.974
  - spkts ~ sloss: r = 0.974
  - spkts ~ sbytes: r = 0.966
  - sinpkt ~ is_sm_ips_ports: r = 0.938
  - tcprtt ~ synack: r = 0.937

![correlation matrix](figures/03_correlation_matrix.png)


## 6. Attack-category patterns

Each attack category has a distinct 'signature' across key flow features (z-normalised means below). Where two categories share a similar signature, a classifier will tend to confuse them.

![attack patterns](figures/04_attack_patterns.png)


## 7. Dimensionality and separability

The first two principal components capture only 29.1% of total variance -- the data is genuinely high-dimensional, so no trivial 2-D rule separates the classes. PCA and t-SNE both show normal traffic and the larger attack types forming visible structure, while the rare classes scatter thinly -- a visual preview of why they are hard to detect.

![PCA](figures/05_pca_projection.png)

![t-SNE](figures/06_tsne_projection.png)


## 8. Outlier analysis

By the 1.5x-IQR rule, **11** features flag more than 10% of records as outliers. In network traffic these 'outliers' are usually genuine extreme behaviour (bursts, scans), not errors -- so they are kept, but they reinforce the need for robust scaling.

![outlier fraction](figures/07_outlier_fraction.png)


## 9. Statistical tests (ANOVA)

A one-way ANOVA per feature measures how strongly its mean differs across attack categories -- a high F-statistic means the feature separates attack types well.

Most discriminative features: **ct_dst_sport_ltm, sttl, ct_src_dport_ltm, ct_srv_dst, ct_srv_src, ct_dst_src_ltm, dttl, ct_dst_ltm**.

![ANOVA discriminative features](figures/08_anova_discriminative.png)


## Implications for the project

- **Severe class imbalance** drives the use of stratified sampling, SMOTE / class-weighted loss, and imbalance-aware metrics (macro-F1, PR-AUC, per-class recall) rather than plain accuracy.
- **Heavy feature skew and outliers** mean neural-network inputs must be z-score standardised; tree models can stay unscaled.
- **Highly correlated feature pairs** confirm redundancy and motivate the feature-selection comparison in Part 5.
- **Low 2-D variance and overlapping rare classes** preview RQ2: some attack categories are intrinsically hard to detect.
