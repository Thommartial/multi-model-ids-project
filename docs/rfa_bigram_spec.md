# RFA + Bigram Method Specification

**Task 2.1 deliverable — the precise specification the feature-engineering
implementation (Part 5) must follow.**

> **Headline finding.** Grounding the method in the original source revealed
> that the proposal's descriptions of *both* RFA and the *bigram technique*
> differ materially from the actual Hamed, Dara & Kremer (2018) method. These
> differences are not cosmetic — they change what gets implemented. They must
> be resolved with Prof. Dara (a co-author of the original) before Part 5.

---

## 1. Source

- **Hamed, T., Dara, R., & Kremer, S. C. (2018).** "Network intrusion detection
  system based on recursive feature addition and bigram technique."
  *Computers & Security* — Elsevier (paywalled; obtain via the library or from
  Prof. Dara).
- **Hamed, T. (PhD thesis).** "Recursive Feature Addition: a Novel Feature
  Selection Technique." University of Guelph — **open access** via the UoG
  Atrium repository; downloaded to `references/`. The thesis gives the full
  algorithms (Algorithm 2 for RFA; Algorithms 3–4 for the bigram technique) and
  is the basis for the specification below.

---

## 2. The original RFA (as published)

RFA is an **embedded, forward** feature-selection method (the opposite of
Recursive Feature Elimination, which works backwards).

- It uses a **Support Vector Machine (RBF kernel)** as its core ranker.
- It starts from an **empty** feature set and adds **one feature per
  iteration**.
- The score for a candidate feature is the **change it causes in the SVM cost
  function** — derived from the SVM's `alpha` vector and support vectors
  (thesis Eq. 3.10: `DJ = ½αᵀHα − ½αᵀH₍₊ᵢ₎α`). The feature producing the
  largest change is added.
- It continues until **all features are ranked** — the output is a full
  ranking from most to least important, not a truncated subset. The stopping
  criterion is simply "all features ranked."

**Original RFA — algorithm:**

```
Input: dataset D, feature set F (size N)
S <- empty ranked list
while |S| < N:
    train SVM (RBF kernel) on the features currently in S
    obtain the alpha vector and support vectors
    for each remaining feature i in F:
        compute ranking coefficient DJ(i)   # change in SVM cost function
    f <- feature with the maximum DJ
    append f to S; remove f from F
Output: ranked feature list S
```

## 3. The original bigram technique (as published)

The bigram technique operates on **packet payload content** — the raw bytes of
the payload, treated like text.

- **Dictionary generation:** collect every distinct 2-character substring
  (bigram) that appears across all payload strings.
- **Feature-vector extraction:** for each payload, count how many times each
  dictionary bigram occurs — a bag-of-bigrams count vector. Order within the
  payload is *not* preserved.
- This produces a **very high-dimensional** feature space (thousands of bigram
  features), so a filter method (CFS) is used to pre-rank before RFA.
- Purpose: catch **stealthy attacks** whose signature is in the payload bytes.

It was applied to the **ISCX** dataset, which *contains payload content*.

---

## 4. What the proposal describes

- **RFA (proposal §5.1):** "Starting from an empty feature set, RFA greedily
  adds whichever candidate feature yields the greatest incremental gain in
  validation **F1**, using a shallow **Random Forest** as the evaluator, and
  terminates when further additions produce no meaningful improvement."
- **Bigram (proposal §5.1):** "Bigram-based features will be constructed by
  pairing **consecutive flow records** within each sliding window, encoding
  state-transition patterns."

---

## 5. The discrepancies — to resolve with Prof. Dara

| Aspect | Original (Hamed et al. 2018) | Proposal's description |
|---|---|---|
| RFA evaluator | SVM, RBF kernel | shallow Random Forest |
| RFA scoring criterion | change in the SVM cost function | gain in validation F1 |
| RFA stopping rule | rank **all** features | stop on "no meaningful improvement" |
| "Bigram" operates on | packet **payload bytes** (character 2-grams) | pairs of **consecutive flow records** |
| Bigram output | bag-of-bigrams count vectors | flow-pair / state-transition features |
| Data requirement | payload content (e.g. ISCX) | flow features only |

**Two things follow:**

1. **The proposal's RFA is RFA *in spirit* — a forward, add-one-feature
   wrapper — but not algorithmically identical** to Hamed et al.'s SVM
   cost-function method. It is a reasonable, defensible standard wrapper. It is
   implementable on UNSW-NB15. It should simply be **cited honestly** — e.g.
   "a forward feature-addition method in the spirit of Hamed, Dara & Kremer
   (2018)" — not presented as their exact algorithm.

2. **The proposal's "bigram" and the original "bigram" are fundamentally
   different techniques that share only the name.** The original needs packet
   payload bytes; **the UNSW-NB15 partitioned benchmark has no payload
   content** — only flow-summary statistics — so the original payload-bigram
   technique *cannot be applied to it at all*. The proposal's "pairs of
   consecutive flow records" is a separate idea. This is the more serious gap
   and the one that most needs Prof. Dara's input, since she co-invented the
   original technique.

---

## 6. Working specification for Part 5 (confirmed with Prof. Dara, 2026-05-22)

Both discrepancies were discussed in the 22 May 2026 supervision meeting and
resolved. The feature-engineering study implements:

- **Filter selection.** Extra-Trees importance + mutual information, consensus
  of the two (proposal §5.1) — no discrepancy.
- **RFA — original (Hamed, Dara & Kremer 2018).** Forward selection from an
  empty set; at each step add the feature whose addition produces the largest
  change in the **SVM (RBF kernel) cost function** (thesis Eq. 3.10:
  `DJ = ½αᵀHα − ½αᵀH₍₊ᵢ₎α`); rank all features. Implementation follows the
  thesis Algorithm 2.
- **RFA — proposal's variant.** Forward selection from an empty set; at each
  step add the feature giving the largest gain in validation **macro-F1** of a
  shallow Random Forest; stop when the gain over a small patience window falls
  below a threshold.
- **Flow-pair features (the proposal's "bigram").** Pair each flow record with
  its predecessor within a sliding window and derive difference / ratio /
  concatenation features. Documented and cited as a **flow-temporal feature
  construction inspired by** — not identical to — the Hamed et al. payload-
  bigram technique. The original payload-bigram cannot run on UNSW-NB15; Phase
  1 does not introduce a payload-bearing dataset.

The ablation then compares: baseline, filter-only, RFA (original SVM),
RFA (proposal RF/F1), and RFA + flow-pair features. The two RFA variants are
evaluated with the same downstream classifier on the same split so the
comparison is clean.

---

## 7. Decisions resolved with Prof. Dara (2026-05-22)

1. **RFA engine — implement both, as a comparison.** The published SVM
   cost-function RFA (Hamed, Dara & Kremer 2018) and the proposal's
   RF / validation-F1 forward-selection RFA are both implemented; the ablation
   compares them. This is itself a contribution: a side-by-side of the original
   RFA against a contemporary RF/F1 variant on UNSW-NB15.
2. **Bigram framing — reframe flow-pair features as an adaptation.** The
   proposal's pairing of consecutive flow records is kept, cited honestly as
   *inspired by, not identical to* the Hamed payload-bigram technique. No new
   payload-bearing dataset is added for Phase 1.

Items deferred to a later meeting (not blocking next steps): publication angle,
HPC access for Phase 2, and the end-of-June Phase-1 → Phase-2 checkpoint.
