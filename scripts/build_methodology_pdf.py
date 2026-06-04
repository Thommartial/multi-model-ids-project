"""Build the pedagogical Methodology, Results & Analysis PDF (reportlab Platypus).

Run from the repo root after the analysis figures exist:
    python scripts/build_methodology_pdf.py
Reads reports/figures/ and reports/eda/figures/, writes docs/Methodology_Results_Analysis.pdf.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Image, Table, TableStyle, HRFlowable)
from reportlab.lib.utils import ImageReader

REPO = "."
FIG = REPO + "/reports/figures"
EDA = REPO + "/reports/eda/figures"
OUT = REPO + "/docs/Methodology_Results_Analysis.pdf"
os.makedirs(REPO + "/docs", exist_ok=True)

INK = colors.HexColor("#16233b"); ACC = colors.HexColor("#2563c9")
SUB = colors.HexColor("#54637e"); LINE = colors.HexColor("#ccd6e6")
BOXB = colors.HexColor("#eef3fb"); BOXY = colors.HexColor("#fff5e0")

ss = getSampleStyleSheet()
def style(name, **kw):
    kw.setdefault("parent", ss["Normal"])
    return ParagraphStyle(name, **kw)
TITLE = style("t", fontName="Helvetica-Bold", fontSize=26, textColor=INK, leading=30)
SUBT  = style("s", fontName="Helvetica", fontSize=12.5, textColor=SUB, leading=17)
H1    = style("h1", fontName="Helvetica-Bold", fontSize=16, textColor=ACC, leading=20, spaceBefore=18, spaceAfter=6)
H2    = style("h2", fontName="Helvetica-Bold", fontSize=12.5, textColor=INK, leading=16, spaceBefore=11, spaceAfter=3)
BODY  = style("b", fontName="Helvetica", fontSize=10.3, textColor=INK, leading=15.2, alignment=TA_JUSTIFY, spaceAfter=6)
BUL   = style("bul", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=2)
CAP   = style("c", fontName="Helvetica-Oblique", fontSize=8.8, textColor=SUB, leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
CALL  = style("call", fontName="Helvetica", fontSize=10, textColor=INK, leading=14.5, backColor=BOXB, borderColor=ACC, borderWidth=0, borderPadding=9, spaceBefore=4, spaceAfter=10, leftIndent=2, rightIndent=2)
CHECK = style("chk", parent=CALL, backColor=BOXY)
TOCI  = style("toc", fontName="Helvetica", fontSize=10.6, textColor=INK, leading=18)

S = []
def P(t, st=BODY): S.append(Paragraph(t, st))
def gap(h=6): S.append(Spacer(1, h))
def bullets(items, st=BUL):
    for it in items: S.append(Paragraph(it, st, bulletText=u"•"))
    gap(4)
def why(t): S.append(Paragraph("<b>Why this matters.</b> " + t, CALL))
def check(t): S.append(Paragraph("<b>Check your understanding.</b> " + t, CHECK))
def rule(): S.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceBefore=4, spaceAfter=8))

def fig(path, caption, maxw=470, maxh=560):
    if not os.path.exists(path):
        P("[figure missing: %s]" % os.path.basename(path), CAP); return
    iw, ih = ImageReader(path).getSize(); ar = ih / float(iw)
    w = maxw; h = w * ar
    if h > maxh: h = maxh; w = h / ar
    im = Image(path, width=w, height=h); im.hAlign = "CENTER"
    S.append(im); S.append(Paragraph(caption, CAP))

def table(data, colw, header=True):
    t = Table(data, colWidths=colw, hAlign="CENTER")
    sty = [("FONT",(0,0),(-1,-1),"Helvetica",8.6),("TEXTCOLOR",(0,0),(-1,-1),INK),
           ("GRID",(0,0),(-1,-1),0.4,LINE),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
           ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
           ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5)]
    if header:
        sty += [("BACKGROUND",(0,0),(-1,0),ACC),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONT",(0,0),(-1,0),"Helvetica-Bold",8.8)]
    t.setStyle(TableStyle(sty)); S.append(t); gap(10)

# ===================== TITLE =====================
P("Multi-Model Intrusion Detection on UNSW-NB15", TITLE)
gap(4)
P("Methodology, Results and Analysis &mdash; a pedagogical walkthrough", SUBT)
rule()
P("This document explains every stage of the project from first principles: what each "
  "transformation does, why it is done, what the alternatives were, what the result means, "
  "and where its limits lie. It is written so that you can fully own and defend the work. "
  "All numbers are taken directly from the project's run artifacts; figures are the exact "
  "ones produced by the analysis scripts.", BODY)
P("<b>Project:</b> CIS*6560 Research Project, MCTI, University of Guelph. "
  "<b>Dataset:</b> UNSW-NB15 (partitioned set). <b>Protocol:</b> 5 random seeds, identical "
  "pipeline across all model families.", BODY)
gap(6)

# ===================== CONTENTS =====================
P("Contents", H1)
toc = ["1. The big picture", "2. Data integrity and ingestion", "3. Exploratory data analysis (EDA)",
       "4. Preprocessing pipeline", "5. Class-imbalance handling", "6. Feature engineering",
       "7. Models and architectures", "8. Evaluation protocol and headline results",
       "9. RQ1 &mdash; feature-engineering strategy and feature groups",
       "10. RQ2 &mdash; which classes stay hard, and why", "11. Limitations and honest scope",
       "12. Explain this project in three minutes", "13. Glossary"]
for t in toc: P(t, TOCI)
S.append(PageBreak())

# ===================== 1. BIG PICTURE =====================
P("1. The big picture", H1)
P("An <b>intrusion detection system (IDS)</b> watches network traffic and flags activity that "
  "looks like an attack. This project does not chase a single best detector; its contribution is "
  "a <b>fair, like-for-like comparison</b> of three families of models under one identical "
  "pipeline: simple <i>rule-based</i> logic, <i>classical machine learning</i> (Random Forest, "
  "SVM, XGBoost), and <i>deep learning</i> (a 1-D CNN and an LSTM). Every model sees exactly the "
  "same data splits, preprocessing, and metrics, so any difference in results is attributable to "
  "the model, not to inconsistent treatment.", BODY)
P("Two research questions drive the analysis:", BODY)
bullets(["<b>RQ1:</b> Which feature-engineering strategy, and which kinds of features, best "
         "support detection of each attack type?",
         "<b>RQ2:</b> Which attack classes remain hard to detect regardless of model, and "
         "<i>why</i> (false positives and class confusion)?"])
P("The data flows through eight stages, shown in the map below: an integrity check, exploratory "
  "analysis, preprocessing, imbalance handling, feature engineering, model training, evaluation, "
  "and finally the visual analysis that answers the two questions. The rest of this document "
  "walks each stage in that order.", BODY)
fig(FIG + "/data_pipeline_mindmap.png", "Figure 1. The full data journey, stage by stage, with a one-line rationale per step.", maxw=430, maxh=600)
why("A comparison is only meaningful if it is fair. Fixing one pipeline and varying only the model "
    "is what lets us make defensible claims like \"trees beat deep nets here\" without the result "
    "being an artifact of uneven preprocessing.")
check("In your own words, why is it important that all six models share the same train/validation/test "
      "split and the same metrics?")
S.append(PageBreak())

# ===================== 2. INTEGRITY =====================
P("2. Data integrity and ingestion", H1)
P("<b>What it is.</b> Before any analysis, each downloaded data file is checked against a stored "
  "<b>SHA-256 checksum</b> &mdash; a 64-character fingerprint computed from the file's bytes. If a "
  "single byte changed (corrupted download, wrong version, tampering), the fingerprint would not "
  "match and the pipeline stops.", BODY)
P("<b>Why it matters.</b> Every downstream number depends on the exact input bytes. A silent "
  "corruption would invalidate the entire study, and you would not know. The checksum makes the "
  "starting point reproducible and verifiable by anyone who re-runs the project.", BODY)
P("<b>What we load.</b> The project uses the UNSW-NB15 <i>partitioned</i> distribution: a training "
  "partition of 175,341 records and a test partition of 82,332, pooled to 257,673 records with 45 "
  "columns. We pool them so we can apply our own controlled, stratified split rather than inherit "
  "the dataset authors' split. A lightweight <b>schema and range validation</b> then rejects wrong "
  "data types and impossible values (for example a negative byte count) before they can poison the "
  "statistics.", BODY)
check("If two people download the dataset on different days and get the same SHA-256 value, what "
      "does that guarantee, and what does it not guarantee?")

# ===================== 3. EDA =====================
P("3. Exploratory data analysis (EDA)", H1)
P("<b>What it is.</b> EDA is the \"get to know your data\" phase: nine analyses that profile the "
  "class balance, the feature distributions, the correlations between features, and how separable "
  "the attack classes are. It produces no model; it produces understanding that shapes every later "
  "decision.", BODY)
fig(EDA + "/01_class_distribution.png", "Figure 2. Class distribution. The dataset is dominated by Normal traffic and a few large attack types; several attack classes are extremely rare.", maxw=420, maxh=300)
P("The single most important finding is <b>severe class imbalance</b>. In the test split, Normal "
  "traffic has 12,859 records while Worms has only 25, and Analysis, Backdoor and Shellcode each "
  "have only a few hundred. This imbalance is the central difficulty of the whole project: a model "
  "can score high \"accuracy\" simply by ignoring the rare classes, so we must choose metrics and "
  "training tricks that refuse to let it cheat (Sections 5 and 8).", BODY)
P("Correlation and skewness analyses reveal two more things that directly drive preprocessing: many "
  "features are <b>heavily right-skewed</b> with enormous values (byte counts and load rates reach "
  "into the billions), and some features are redundant. The PCA and t-SNE projections (below) "
  "compress the 40-plus features into two dimensions so we can see, by eye, that some attack types "
  "form tight separable clusters while others overlap heavily with Normal traffic &mdash; an early "
  "warning of which classes will be hard.", BODY)
fig(EDA + "/05_pca_projection.png", "Figure 3. PCA projection. Linear structure: some classes separate, others overlap, hinting at intrinsic difficulty.", maxw=360, maxh=280)
why("EDA is where you earn the right to make modeling choices. The imbalance finding justifies "
    "class weights and macro-F1; the skew finding justifies the log transform for neural nets; the "
    "overlap seen in PCA/t-SNE foreshadows the RQ2 \"hard classes\" result.")
check("Why can a classifier reach 90%+ raw accuracy on this dataset while being almost useless at "
      "catching Worms?")
S.append(PageBreak())

# ===================== 4. PREPROCESSING =====================
P("4. Preprocessing pipeline", H1)
P("Preprocessing turns raw, messy records into clean numerical matrices, while obeying one rule "
  "above all: <b>no leakage</b>. Anything learned from the data (medians, scaling statistics, "
  "encodings) is computed on the <b>training fold only</b> and then applied to validation and test. "
  "If we fit those on the full dataset, the model would indirectly \"see\" the test set and our "
  "scores would be optimistically wrong.", BODY)
P("<b>Cleaning.</b> We drop the artificial <i>id</i> column (an index, not a signal), remove "
  "duplicate records, and neutralize infinities. Deduplication is significant here: 94,928 records "
  "(36.8% of the pool) are exact duplicates. Leaving them in would let the model memorize repeated "
  "rows and would inflate scores. Missing values are filled with the <b>median</b> of each feature "
  "computed on training data only (the median is robust to the heavy skew we found in EDA).", BODY)
P("<b>Encoding.</b> Three columns are categorical text (protocol, service, connection state). "
  "Models need numbers, so we <b>one-hot encode</b> them: each category becomes its own 0/1 column. "
  "Rare categories are bucketed together to avoid a blow-up of near-empty columns. After encoding, "
  "the feature count grows to 190.", BODY)
P("<b>Splitting and scaling.</b> We make a <b>70/15/15 stratified split</b> into train, validation "
  "and test, stratified on the attack label so every class appears in the same proportion in each "
  "fold. Scaling uses a <b>dual treatment</b>, and understanding why is important: tree models "
  "(Random Forest, XGBoost, the rule tree) are <i>scale-invariant</i> &mdash; they split on "
  "thresholds, so multiplying a feature by a million changes nothing &mdash; and they are fed the "
  "raw, unscaled features. Neural networks are the opposite: they do gradient descent and require "
  "inputs on a comparable, modest scale. They receive <b>z-scored</b> features (each feature "
  "shifted and rescaled to mean 0, standard deviation 1). A separate <b>sliding-window</b> builder "
  "stacks consecutive records into short sequences for the LSTM.", BODY)
why("The dual scaling is not an accident; it is the difference between a working deep model and a "
    "broken one. Section 7 shows that feeding raw billion-scale features to the networks collapses "
    "them, and that fixing the scaling is what rescued them.")
check("We compute the median for imputation on the training fold only, then reuse it on the test "
      "fold. What specific error would we be committing if we instead computed it over all data?")

# ===================== 5. IMBALANCE =====================
P("5. Class-imbalance handling", H1)
P("Because rare classes are easy to ignore, we counter the imbalance in three complementary ways, "
  "all applied to the <b>training fold only</b> (resampling the test set would fake the results):", BODY)
bullets([
  "<b>Inverse-frequency class weights:</b> the loss function is told that a mistake on a rare class "
  "costs more than a mistake on a common one, in proportion to how rare the class is. This is the "
  "lightest-touch fix and is used by every model family.",
  "<b>SMOTE and its variants:</b> synthesize new, plausible minority-class examples by interpolating "
  "between real ones, rather than just copying. This gives the model more rare-class signal to learn "
  "from. (Borderline variants were noted to leave the ultra-rare Worms class almost untouched, a "
  "documented limitation.)",
  "<b>Stratified subsampling:</b> used for the RBF-SVM, whose training cost grows roughly with the "
  "square of the number of records; we cap it at 30,000 stratified rows so it remains tractable."])
check("Why must SMOTE be applied after the train/test split and only to the training fold?")

# ===================== 6. FEATURE ENGINEERING =====================
P("6. Feature engineering", H1)
P("Feature engineering asks: do we need all 190 features, and can a cleverly chosen subset do "
  "better? Four strategies are compared, all judged under one fixed Random Forest so the comparison "
  "isolates the <i>features</i>, not the model:", BODY)
bullets([
  "<b>Filter (consensus):</b> rank features two ways &mdash; by mutual information (how much a feature "
  "tells us about the label) and by Extra-Trees importance &mdash; and keep the features both methods "
  "agree on. Fast, model-light.",
  "<b>Recursive Feature Addition (RFA):</b> start from nothing and greedily add the feature that most "
  "improves validation score, stopping when it stops helping. Two variants are implemented: the "
  "original SVM cost-function form (Hamed, Dara and Kremer 2018) and a Random-Forest / validation-F1 "
  "form.",
  "<b>Flow-pair (bigram) features:</b> pair each record with its predecessor to capture short-range "
  "structure. This is cited honestly as <i>inspired by</i> the original payload-bigram technique, not "
  "identical to it (the partitioned dataset has no payloads).",
  "<b>Baseline:</b> all 190 features, no selection."])
P("The ablation result is decisive and slightly counter-intuitive: <b>fewer, better-chosen features "
  "win</b>. The 14-feature RFA selection scores the highest multiclass macro-F1 (0.631), beating the "
  "full 190-feature baseline (0.530) and even the 394-feature flow-pair set (0.504), which is the "
  "<i>worst</i>. More features added noise and redundancy; targeted selection added signal and was "
  "also far cheaper to train.", BODY)
fig(FIG + "/rq1_ablation_strategy.png", "Figure 4. Feature-engineering strategy comparison (fixed Random Forest, multiclass). RFA with 14 features wins; piling on features hurts.", maxw=430, maxh=300)
why("This is a real, publishable finding and a classic lesson: dimensionality is not free. The best "
    "strategy is the one that discards the most while keeping the signal.")
S.append(PageBreak())

# ===================== 7. MODELS =====================
P("7. Models and architectures", H1)
P("All six models implement the same interface and are driven by the same harness, so they are "
  "trained and scored identically. Briefly, in increasing complexity:", BODY)
bullets([
  "<b>Rule-based:</b> a depth-5 decision tree whose paths read as human rules (\"if TTL is X and "
  "bytes are Y then ...\"). It is the interpretable baseline &mdash; the floor everything else must beat.",
  "<b>Random Forest:</b> hundreds of decision trees voting; class-balanced; fed unscaled features.",
  "<b>SVM (RBF kernel):</b> finds a curved separating boundary; trained on a 30,000-row stratified "
  "subsample with its own internal scaler.",
  "<b>XGBoost:</b> gradient-boosted trees that fix each other's mistakes in sequence; handles missing "
  "values natively; the strongest classical model here.",
  "<b>1-D CNN:</b> Conv1D + BatchNorm + ReLU + MaxPool blocks, then global average pooling, a dense "
  "layer, and a softmax. Convolution scans the feature vector for local patterns.",
  "<b>LSTM:</b> a recurrent network over short windows of consecutive records."])
P("<b>The deep-learning rescue (a key story).</b> Initially the CNN and LSTM performed terribly &mdash; "
  "multiclass macro-F1 around 0.13, <i>below</i> the rule-based baseline. The cause was not the "
  "architecture but the <b>inputs</b>: the networks were being fed raw, unscaled features with values "
  "up to roughly six billion, which makes gradient-based training diverge. The fix had two parts. "
  "First, give the networks a proper input transform: a signed <b>log1p</b> on the heavy-tailed "
  "columns (to compress the enormous range) followed by z-scoring, fitted on training data only. "
  "Second, tune the optimizer: a smaller learning rate (0.0005), dropout 0.3, a learning-rate "
  "schedule that halves the rate when progress stalls, and early stopping that restores the best "
  "epoch. After these changes the deep models train cleanly and land in a sensible range (Section 8).", BODY)
why("The lesson is that \"deep learning doesn't work here\" is often really \"the inputs were wrong.\" "
    "Diagnosing the scaling bug, rather than abandoning the models, is what produced honest, "
    "comparable deep-learning numbers.")
check("Why are tree models unaffected by unscaled billion-valued features while neural networks are "
      "crippled by them?")

# ===================== 8. EVALUATION =====================
P("8. Evaluation protocol and headline results", H1)
P("Every model is trained with <b>five random seeds</b> (42 to 46) and we report the mean and "
  "standard deviation, because a single run can be lucky. The metrics are chosen for an imbalanced "
  "problem:", BODY)
bullets([
  "<b>Macro-F1:</b> the F1 score (harmonic mean of precision and recall) computed per class and then "
  "averaged with equal weight. Equal weight is the point: a rare class counts as much as a common one, "
  "so the model cannot hide poor rare-class performance.",
  "<b>Balanced accuracy:</b> the average per-class recall &mdash; again immune to imbalance.",
  "<b>AUROC and PR-AUC (binary):</b> threshold-free quality; PR-AUC is the more honest of the two "
  "under heavy imbalance.",
  "<b>Bootstrap 95% confidence intervals:</b> we resample results thousands of times to estimate how "
  "uncertain each number is.",
  "<b>Train vs test scoring:</b> we score the training fold too, so the gap between train and test "
  "exposes overfitting."])
P("The headline test macro-F1 scores (mean over 5 seeds):", BODY)
table([
  ["Model", "Binary F1", "Multiclass F1", "Family"],
  ["Rule-based", "0.864", "0.360", "interpretable"],
  ["SVM (RBF)", "0.859", "0.389", "classical"],
  ["Random Forest", "0.919", "0.534", "classical"],
  ["XGBoost", "0.925", "0.611", "classical"],
  ["1-D CNN (tuned)", "0.905", "0.434", "deep"],
  ["LSTM (tuned)", "0.894", "0.447", "deep"],
], [150, 90, 110, 110])
P("Two clear stories. On the <b>binary</b> task (attack vs normal) everything works well; XGBoost "
  "leads at 0.925 and even the simple rule tree reaches 0.864. The <b>multiclass</b> task (naming "
  "which of ten attack types) is much harder for everyone, and here the tree ensembles clearly lead, "
  "with the deep models competitive but behind. On tabular flow features this is the expected outcome "
  "&mdash; gradient-boosted trees are simply strong on this kind of data.", BODY)
P("The training curves below confirm the deep models are now <b>well-trained, not overfit</b>: "
  "training and validation loss descend together, with validation at or below training (a "
  "consequence of dropout). The generalization-gap chart then makes a subtler point.", BODY)
fig(FIG + "/dl_training_curves.png", "Figure 5. Deep-model train vs validation curves (mean +/- std over 5 seeds). Curves track together; no diverging gap = no overfitting.", maxw=430, maxh=560)
fig(FIG + "/train_test_gap.png", "Figure 6. Train-vs-test macro-F1 gap. Tree ensembles win on test score but overfit multiclass (RF gap 0.31, XGBoost 0.19); the CNN barely overfits at all.", maxw=470, maxh=320)
P("The gap chart reveals an honesty point the leaderboard hides: <b>Random Forest and XGBoost buy "
  "much of their multiclass lead through overfitting</b> &mdash; RF scores 0.843 on train but 0.534 "
  "on test (a 0.31 gap), XGBoost 0.797 vs 0.611. The 1-D CNN, by contrast, scores about 0.438 on both "
  "(essentially zero gap): it generalizes almost perfectly, just at a lower absolute level. This "
  "interpretability-versus-accuracy-versus-honesty trade-off is exactly the kind of nuance the RQ "
  "analysis is meant to surface.", BODY)
check("XGBoost has the best multiclass test score but a 0.19 train-test gap; the CNN scores lower but "
      "has a ~0 gap. Which would you trust more on genuinely new traffic, and why?")
S.append(PageBreak())

# ===================== 9. RQ1 =====================
P("9. RQ1 &mdash; feature-engineering strategy and feature groups", H1)
P("RQ1 has two halves. The first &mdash; which <i>strategy</i> wins &mdash; was answered in Section 6: "
  "RFA with 14 features. The second half asks which <i>kinds</i> of features matter, and for which "
  "attack types. Features are grouped into five semantic families (basic flow statistics, content, "
  "timing, general-purpose, and connection-tracking), and we measure each group's importance per "
  "class by <b>permutation importance</b>: shuffle a group's columns, see how much that class's F1 "
  "drops. A big drop means the class depends on that group.", BODY)
P("First, the winning RFA selection is dominated by <b>basic flow</b> features &mdash; 10 of its 14 "
  "features are byte counts, packet rates, TTLs and protocol/service indicators &mdash; with one "
  "feature each from the timing, content, general-purpose and connection-tracking groups. A handful "
  "of basic volume statistics carry most of the multiclass signal.", BODY)
fig(FIG + "/rq1_rfa_group_composition.png", "Figure 7. Semantic groups inside the 14-feature RFA selection. Basic flow statistics dominate.", maxw=360, maxh=260)
fig(FIG + "/rq1_feature_group_importance.png", "Figure 8. Per-class feature-group importance (permutation: F1 drop when a group is shuffled). Darker = more essential.", maxw=420, maxh=330)
P("The heatmap (Figure 8) gives the per-class answer and two striking results:", BODY)
bullets([
  "<b>Basic flow dominates almost everywhere.</b> Generic (0.88 drop) and Reconnaissance (0.75) are "
  "essentially defined by their flow-volume statistics; remove them and detection collapses.",
  "<b>Shellcode and Worms need more than flow.</b> Shellcode relies on basic flow (0.60) <i>and</i> "
  "connection-tracking (0.43) together; Worms is the only class with no single dominant group &mdash; "
  "its importance is spread thinly across all five (0.26 to 0.38). That diffuse signal, plus its tiny "
  "sample size, is precisely why Worms is hard.",
  "<b>A negative result that explains a confusion.</b> For Analysis and Backdoor the connection-"
  "tracking group has <i>negative</i> importance (-0.20 and -0.10): shuffling those features actually "
  "<i>improves</i> their F1. Those features actively cause Analysis and Backdoor to be mistaken for "
  "each other &mdash; a direct, evidence-backed bridge to RQ2."])
why("RQ1 turns \"which features matter\" from a guess into a measured, per-class answer. The negative-"
    "importance finding is the kind of non-obvious result that makes a comparative study worth "
    "publishing.")
check("If a feature group has negative permutation importance for a class, what is that telling you "
      "about the features and that class?")
S.append(PageBreak())

# ===================== 10. RQ2 =====================
P("10. RQ2 &mdash; which classes stay hard, and why", H1)
P("RQ2 asks which attack types remain hard <i>regardless of model</i>, and diagnoses the cause. We "
  "average each class's F1 across all six models to get a difficulty ranking, then read the confusion "
  "structure to explain it.", BODY)
fig(FIG + "/rq2_class_difficulty.png", "Figure 9. Class difficulty (mean F1 across all six models; n = test support). Difficulty is not simply about rarity.", maxw=440, maxh=300)
fig(FIG + "/rq2_per_class_f1_heatmap.png", "Figure 10. Per-class F1 for every model. The hard classes (left) stay red across the whole row of models.", maxw=470, maxh=250)
P("The difficulty order, hardest first, is Backdoor (0.13), Analysis (0.15), DoS (0.17), Worms (0.20) "
  "and Shellcode (0.36); the easy end is Generic, Normal and Exploits. The heatmap shows these hard "
  "classes are red across <i>every</i> model column &mdash; the difficulty is intrinsic to the data, "
  "not a quirk of one model. That is the core RQ2 answer.", BODY)
P("<b>Why they are hard</b> is the more interesting part, and it is not just rarity:", BODY)
bullets([
  "<b>Analysis and Backdoor are mutually confused.</b> 49.5% of Backdoor records are predicted as "
  "Analysis and 35.1% the other way. In UNSW-NB15 these two attack types are behaviorally almost "
  "identical, and (from RQ1) their connection-tracking features make it worse.",
  "<b>Shellcode is a precision problem, not a recall problem.</b> Most models actually <i>recall</i> "
  "Shellcode well, but Reconnaissance records leak into it (13.6%), dragging its precision and "
  "therefore its F1 down.",
  "<b>Worms is diffuse and tiny.</b> Only 25 test records, frequently predicted as Exploits (25.7%), "
  "with no single feature group to anchor it &mdash; a genuinely hard case for any model.",
  "<b>DoS overlaps with Exploits</b> (31.1% confusion), reflecting real similarity in flow behavior."])
P("Most-confused pairs (fraction of the true class sent to the wrong label), top five:", BODY)
table([
  ["True class", "Predicted as", "Fraction"],
  ["Backdoor", "Analysis", "49.5%"],
  ["Analysis", "Backdoor", "35.1%"],
  ["DoS", "Exploits", "31.1%"],
  ["Worms", "Exploits", "25.7%"],
  ["Reconnaissance", "Shellcode", "13.6%"],
], [150, 150, 90])
why("RQ2 converts a vague worry (\"some classes are hard\") into specific, actionable diagnoses: which "
    "classes, how hard, and the exact confusions responsible. That is what a security team would need "
    "to decide where to invest.")
check("Shellcode has high recall but low F1. Explain, in terms of false positives, why a high-recall "
      "class can still score poorly.")
S.append(PageBreak())

# ===================== 11. LIMITATIONS =====================
P("11. Limitations and honest scope", H1)
P("Owning the project means owning its limits. The following are stated plainly and were design "
  "decisions, not oversights:", BODY)
bullets([
  "<b>The LSTM's sequences are not truly temporal.</b> The partitioned UNSW-NB15 set does not preserve "
  "packet timestamps or flow order, so the LSTM's sliding window is over feature-space neighbors, not "
  "real time. The recurrent results should be read with that caveat; genuine temporal modeling is "
  "future work on a timestamped dataset.",
  "<b>Tree ensembles overfit on multiclass.</b> Their headline lead is partly memorization (Section 8), "
  "which the train-test gap chart documents rather than hides.",
  "<b>RBF-SVM is subsampled</b> to 30,000 rows for tractability, so its numbers reflect that subset, "
  "not the full training pool.",
  "<b>Flow-pair features are an adaptation,</b> cited as inspired by the original payload-bigram method, "
  "not equivalent to it.",
  "<b>Scope is Phase 1:</b> one dataset (UNSW-NB15), six models. Cross-dataset validation (CIC-IDS2017), "
  "anomaly detectors, and noise-robustness testing are deliberately deferred."])
check("A reviewer says \"your LSTM result proves recurrent models work for IDS.\" Why is that claim too "
      "strong given this setup?")

# ===================== 12. THREE MINUTES =====================
P("12. Explain this project in three minutes", H1)
P("\"I built a fair comparison of rule-based, classical, and deep-learning intrusion detectors on "
  "UNSW-NB15, all trained through one identical, leakage-safe pipeline and evaluated over five seeds "
  "with imbalance-aware metrics. Binary attack-vs-normal detection is easy &mdash; everything scores "
  "around 0.86 to 0.93 macro-F1. Naming the specific attack type is much harder: gradient-boosted trees "
  "lead (XGBoost 0.61), the deep models are competitive (0.43 to 0.45) once I fixed an input-scaling "
  "bug that had been crippling them, and the simple baselines trail. Feature engineering matters: a "
  "14-feature RFA selection beats using all 190 features, so less is more. The hardest classes "
  "(Backdoor, Analysis, DoS, Worms, Shellcode) are hard for <i>every</i> model, mostly because of "
  "mutual confusion &mdash; Backdoor and Analysis are nearly indistinguishable in this data &mdash; and "
  "I can show, via permutation importance, exactly which feature groups drive or sabotage each class. "
  "The trees win on score but overfit; the CNN generalizes better than its score suggests.\"", BODY)

# ===================== 13. GLOSSARY =====================
P("13. Glossary", H1)
gl = [
 ("SHA-256", "A cryptographic fingerprint of a file; any change to the bytes changes the fingerprint."),
 ("Leakage", "When information from validation/test influences training, inflating scores dishonestly."),
 ("Stratified split", "A split that preserves each class's proportion in every fold."),
 ("One-hot encoding", "Turning a categorical value into a set of 0/1 indicator columns."),
 ("z-score", "Rescaling a feature to mean 0 and standard deviation 1."),
 ("log1p", "The transform log(1+x); compresses huge values so heavy-tailed features behave."),
 ("Class weights", "Telling the loss that errors on rare classes cost more."),
 ("SMOTE", "Synthesizing new minority-class samples by interpolation rather than copying."),
 ("RFA", "Recursive Feature Addition: greedily add the most helpful feature until it stops helping."),
 ("Permutation importance", "Shuffle a feature/group and measure the score drop to gauge its value."),
 ("Macro-F1", "Per-class F1 averaged with equal weight, so rare classes count fully."),
 ("Balanced accuracy", "Average per-class recall; immune to class imbalance."),
 ("AUROC / PR-AUC", "Threshold-free quality scores; PR-AUC is preferred under heavy imbalance."),
 ("Bootstrap CI", "Uncertainty estimate from resampling results many times."),
 ("Overfitting", "Learning the training data too literally; shows up as a large train-test gap."),
 ("Dropout", "Randomly disabling units during training to regularize a neural network."),
]
for term, d in gl:
    P("<b>%s.</b> %s" % (term, d), style("g", parent=BODY, spaceAfter=3))

doc = SimpleDocTemplate(OUT, pagesize=letter, topMargin=54, bottomMargin=48,
                        leftMargin=56, rightMargin=56,
                        title="Methodology, Results and Analysis - Multi-Model IDS")
def footer(c, d):
    c.setFont("Helvetica", 8); c.setFillColor(SUB)
    c.drawCentredString(letter[0]/2, 30, "Multi-Model IDS on UNSW-NB15  -  Methodology, Results and Analysis  -  page %d" % d.page)
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("WROTE", OUT, os.path.getsize(OUT)//1024, "KB")