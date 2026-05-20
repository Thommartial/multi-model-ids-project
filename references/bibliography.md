# Bibliography — key works

The methodologically load-bearing references for the project. Each entry notes
what the project uses it for and whether the PDF is held locally in
`references/` (the PDFs are gitignored — kept for reading and citing, not
redistributed through the public repository). The proposal's full citation list
includes further survey and model papers; this file collects the works the
project's *methods* directly depend on.

## Dataset

- **Moustafa, N., & Slay, J. (2015).** UNSW-NB15: a comprehensive data set for
  network intrusion detection systems. *Military Communications and Information
  Systems Conference (MilCIS)*, IEEE. — The benchmark dataset.
  *Access: IEEE (paywalled) — obtain via the UoG library.*

## Methodology the project builds on

- **Hamed, T., Dara, R., & Kremer, S. C. (2018).** Network intrusion detection
  system based on recursive feature addition and bigram technique. *Computers &
  Security*, Elsevier. — The RFA + bigram method (see `docs/rfa_bigram_spec.md`).
  *Access: Elsevier (paywalled). The first author's PhD thesis, which covers the
  same methods in full, is open access and held locally:*
  `references/Hamed_RFA_UoGuelph_atrium.pdf`.

## Data-quality / duplicate evidence

- **Zoghi, Z., & Serpen, G. (2024).** UNSW-NB15 computer security dataset:
  analysis through visualization. *Security and Privacy*, Wiley. — Documents
  ~42% duplicate records in UNSW-NB15.
  *Access: open (arXiv:2101.05067) — `references/Zoghi_Serpen_2021_UNSW-NB15_analysis.pdf`.*
- **An analysis of the KDD99 and UNSW-NB15 datasets for the intrusion detection
  system (2020).** *Symmetry*, 12(10):1666, MDPI. — Independent confirmation of
  the duplicates; a five-group feature taxonomy for UNSW-NB15.
  *Access: open access (MDPI) — download from the article page; full author
  list on that page.*
- **Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. (2009).** A detailed
  analysis of the KDD CUP 99 data set. *IEEE CISDA*. — The classic precedent:
  duplicate records inflate IDS results; motivated the NSL-KDD dataset.
  *Access: IEEE (paywalled).*

## SOTA baselines (Phase 2)

- **Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018).** Kitsune: an
  ensemble of autoencoders for online network intrusion detection. *NDSS*.
  *Access: open (arXiv:1802.09089) — `references/Mirsky2018_Kitsune.pdf`.*
- **Du, M., Li, F., Zheng, G., & Srikumar, V. (2017).** DeepLog: anomaly
  detection and diagnosis from system logs through deep learning. *ACM CCS*.
  *Access: ACM (paywalled).*

## Reproducibility (methodology references)

- **Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013).** Ten simple
  rules for reproducible computational research. *PLOS Computational Biology*.
  *Access: open access (PLOS).*
- **Boettiger, C. (2015).** An introduction to Docker for reproducible research.
  *ACM SIGOPS Operating Systems Review*.
  *Access: open (arXiv:1410.0846) — `references/Boettiger2015_Docker_reproducible_research.pdf`.*

## Notes

- Paywalled papers are best obtained through the University of Guelph library.
- The Hamed, Dara & Kremer (2018) paper is best obtained directly from
  Prof. Dara (a co-author); the open thesis covers the same methods in full.
- The proposal cites additional survey and deep-learning IDS papers
  (2023–2026); those remain in the proposal's own `references.bib`, collated by
  the student.
