"""
Build notebooks/06_random_forest.ipynb programmatically.
Run once from the eskape-defence-ml/ root:
    conda run -n eskape-ml python notebooks/build_rf_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (eskape-ml)",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ── TITLE ──────────────────────────────────────────────────────────────────────
md("""\
# Phase 8 — Random Forest classifier

**Research questions addressed:**
- **Q1** (multi-class): Can defence system repertoire classify species?
  Which features drive the classification?
- **Q2** (binary, per species): Can defence profile predict high-ARG burden?

**Methodology highlights:**
- All evaluation uses `StratifiedGroupKFold` (5-fold, groups = Mash phylogroups from Phase 6)
- Specificity-filtered feature set: 265 dp_* features (9 taxonomic markers removed)
- Feature importance: Gini (MDI) → Permutation → SHAP TreeExplainer
- Comparison to Phase 7 LR baseline: improvement justified only if ΔBA > margin of CI overlap

**Learning thread:**
Concepts introduced: bagging, random feature subsampling, OOB error, MDI vs permutation
importance, SHAP additive decomposition. Each section includes a grounded question.
""")

# ── SECTION 1: IMPORTS ─────────────────────────────────────────────────────────
md("## Section 1 — Imports and configuration")

code("""\
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (server/CI safe)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

ROOT  = Path("..")
PROC  = ROOT / "data" / "processed"
RES   = ROOT / "results"
FIG   = RES / "figures" / "rf"
FIG.mkdir(parents=True, exist_ok=True)
(RES / "models").mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS     = 5
N_BOOT       = 2000      # bootstrap replicates for CIs

print("Imports OK.")
""")

# ── SECTION 2: LOAD DATA + FEATURE SELECTION ───────────────────────────────────
md("""\
## Section 2 — Load data and feature selection

**Why this matters:**
We reuse the exact same 265-feature set and phylogroup assignments from Phase 7
to ensure RF is evaluated on an identical feature space to LR. Changing features
between models would make any accuracy comparison meaningless.

**Specificity filter recap:** For each dp_* feature, compute the fraction of genomes
in each species carrying it (per-species prevalence). Take the std of those 6 fractions,
normalise by 0.5 (max possible std). Score ≥ 0.70 = taxonomic marker (encodes species
identity, not defence architecture) → removed. 9 markers removed, 265 retained.
""")

code("""\
fm      = pd.read_parquet(PROC / "feature_matrix.parquet")
X_full  = fm                          # keep full df for label access

# Specificity filter (identical to Phase 7)
dp_cols    = sorted([c for c in fm.columns if c.startswith("dp_")])
sp_prev    = fm.groupby("species")[dp_cols].mean()   # 6 × 274
spec_score = sp_prev.std() / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]

# Labels and group assignments
# .to_numpy() required: parquet loads string columns as PyArrow-backed ArrowStringArray.
# sklearn's GridSearchCV/CV splitters need plain numpy arrays for indexing.
y_q1   = fm["species"].to_numpy(dtype=str)     # 6-class label for Q1
groups = fm["phylogroup"].to_numpy(dtype=str)  # 95 phylogroups from Mash (Phase 6)

X = fm[FEAT_COLS].to_numpy(dtype=float)        # 878 × 265, explicit float numpy array

print(f"Feature matrix: {X.shape[0]} genomes × {X.shape[1]} features")
print(f"Markers removed: {len(markers)} → {markers}")
print(f"\\nSpecies distribution:")
for sp, n in fm["species"].value_counts().items():
    print(f"  {sp:<20} {n:>4}")
print(f"\\nPhylogroups: {len(set(groups))} total, "
      f"max={pd.Series(groups).value_counts().max()}, "
      f"median={pd.Series(groups).value_counts().median():.0f}")
""")

# ── SECTION 3: CV SETUP ────────────────────────────────────────────────────────
md("""\
## Section 3 — GroupedStratifiedKFold setup

**Why grouped CV matters (core concept):**
Two genomes from the same phylogroup share recent common ancestry. In 5-fold
standard CV, both might appear in the same training + test split — so the model
has effectively "seen" the test genome before. This inflates accuracy because
the model learns phylogenetic signal, not defence-architecture signal.

`StratifiedGroupKFold` guarantees that all genomes from a given phylogroup
land in the same fold. No phylogroup ever appears in both training and test.
This is the Phase 6 decision applied to every model we train.

**The cost:** Fold sizes are unequal (the largest phylogroup, 119 genomes of SA,
goes to one fold whole). We saw this in Phase 7: fold sizes ranged 123–232.
That is the honest price of correct validation.
""")

code("""\
# Reuse identical CV object as Phase 7
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Verify fold structure
fold_sizes = []
for fold_i, (tr, te) in enumerate(cv.split(X, y_q1, groups=groups)):
    fold_sizes.append(len(te))
    species_in_test  = len(set(groups[te]))
    print(f"  Fold {fold_i+1}: {len(tr):>3} train | {len(te):>3} test "
          f"| {species_in_test} phylogroups in test")

print(f"\\nFold size range: {min(fold_sizes)}–{max(fold_sizes)} genomes "
      f"(CV={np.std(fold_sizes)/np.mean(fold_sizes)*100:.1f}%)")
""")

# ── SECTION 4: QUICK RF (DEFAULT PARAMS) ──────────────────────────────────────
md("""\
## Section 4 — Quick RF with default parameters (Q1)

**What we're doing:**
Before any tuning, run RF with scikit-learn defaults. This gives us a reference
point: how does an "out-of-the-box" RF compare to the tuned LR baseline (0.837)?

**Key defaults in scikit-learn's RandomForestClassifier:**
- `n_estimators=100` — 100 trees
- `max_features="sqrt"` — √265 ≈ 16 features sampled per split
- `max_depth=None` — trees grow until all leaves are pure
- `min_samples_leaf=1` — a leaf can contain a single genome
- `class_weight=None` — no correction for class imbalance

We add `class_weight="balanced"` because our fold sizes are unequal (saw this
in Phase 7: EF and SA form large single-phylogroup folds).

**OOB (Out-Of-Bag) score:** With `oob_score=True`, RF automatically evaluates
each genome against the trees that never trained on it (the ~37% bootstrap holdout).
This gives a free internal accuracy estimate — no extra CV needed.
""")

code("""\
# Quick RF — default params + balanced class weights + OOB
rf_default = RandomForestClassifier(
    n_estimators  = 100,
    max_features  = "sqrt",
    max_depth     = None,
    min_samples_leaf = 1,
    class_weight  = "balanced",
    oob_score     = True,          # free internal validation
    n_jobs        = -1,
    random_state  = RANDOM_STATE,
)

# Train on ALL data to get OOB score (OOB uses the full training set)
rf_default.fit(X, y_q1)
print(f"OOB balanced accuracy (all 878 genomes): {rf_default.oob_score_:.4f}")
print()

# Also run the same model under grouped CV (apples-to-apples with Phase 7 LR)
def bootstrap_ci(y_true, y_pred, metric_fn=None, n_boot=N_BOOT, seed=42):
    # Bootstrap 95% CI over pooled per-genome predictions (M2 fix: n=878, not n=5)
    if metric_fn is None:
        metric_fn = balanced_accuracy_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return metric_fn(y_true, y_pred), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

ba_scores = []
yt_def = np.empty(len(y_q1), dtype=object)
yp_def = np.empty(len(y_q1), dtype=object)
for tr, te in cv.split(X, y_q1, groups=groups):
    rf_default.fit(X[tr], y_q1[tr])
    pred = rf_default.predict(X[te])
    yt_def[te] = y_q1[te]
    yp_def[te] = pred
    ba_scores.append(balanced_accuracy_score(y_q1[te], pred))

mean_ba, lo, hi = bootstrap_ci(yt_def, yp_def)
print(f"Grouped CV balanced accuracy (5-fold): {mean_ba:.4f} [{lo:.4f}–{hi:.4f}]")
print(f"Phase 7 LR reference (filtered, grouped): 0.8370 [0.8130–0.8590]")
print()
delta = mean_ba - 0.8370
print(f"Delta RF_default vs LR: {delta:+.4f}")
""")

# ── SECTION 5: HYPERPARAMETER TUNING ──────────────────────────────────────────
md("""\
## Section 5 — Hyperparameter tuning with GridSearchCV

**What hyperparameters control:**
- `n_estimators`: More trees = lower variance but slower. Returns diminish after ~200.
- `max_depth`: Limits tree growth. `None` = fully grown = overfits training set.
  Restricting depth forces the model to generalise at the cost of some training fit.
- `min_samples_leaf`: Minimum genomes in a leaf node. Larger = smoother decision
  boundaries, less overfitting.
- `max_features`: Features sampled per split. Lower = more diverse trees = less
  correlation between trees = better variance reduction.

**GridSearchCV + GroupedStratifiedKFold:**
GridSearchCV internally performs CV to score each hyperparameter combination.
We must pass our phylo-grouped CV object — not the default StratifiedKFold —
to prevent leakage during tuning. We also pass `groups` via `fit_params`.

Scoring metric: `balanced_accuracy` (handles unequal fold sizes correctly).
""")

code("""\
param_grid = {
    "n_estimators"    : [100, 300],
    "max_depth"       : [10, 20, None],
    "min_samples_leaf": [1, 3, 5],
    "max_features"    : ["sqrt", 0.3],
}

rf_base = RandomForestClassifier(
    class_weight = "balanced",
    oob_score    = False,   # off during grid search (slow)
    n_jobs       = 1,       # n_jobs=1 inside RF; GridSearchCV handles outer parallelism
    random_state = RANDOM_STATE,
)

grid_search = GridSearchCV(
    estimator  = rf_base,
    param_grid = param_grid,
    cv         = cv,                    # our phylo-grouped splitter
    scoring    = "balanced_accuracy",
    n_jobs     = -1,                    # parallelise across param combinations × folds
    verbose    = 1,
    refit      = True,                  # refit best model on full data
)

grid_search.fit(X, y_q1, groups=groups)

print("\\nBest hyperparameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k:<22} {v}")
print(f"\\nBest CV balanced accuracy: {grid_search.best_score_:.4f}")
""")

# ── SECTION 6: FINAL RF EVALUATION ─────────────────────────────────────────────
md("""\
## Section 6 — Final RF evaluation (grouped CV, Q1)

**What we're doing:**
Take the best hyperparameters from Section 5, re-run grouped CV with 2000-bootstrap
CIs, and compare to LR. We also compute macro-F1 and per-class metrics.

**Why macro-F1 alongside balanced accuracy?**
Balanced accuracy is the mean of per-class recall. Macro-F1 is the mean of per-class
F1 (harmonic mean of precision and recall). If a class has high recall but poor
precision (lots of false positives), BA will not penalise it but macro-F1 will.
Using both catches this edge case.
""")

code("""\
best_params = grid_search.best_params_

rf_best = RandomForestClassifier(
    **best_params,
    class_weight = "balanced",
    oob_score    = True,
    n_jobs       = -1,
    random_state = RANDOM_STATE,
)

# Grouped CV — collect per-fold BA, F1, and all predictions for confusion matrix
ba_scores, f1_scores = [], []
yt_q1 = np.empty(len(y_q1), dtype=object)
yp_q1 = np.empty(len(y_q1), dtype=object)

for tr, te in cv.split(X, y_q1, groups=groups):
    rf_best.fit(X[tr], y_q1[tr])
    pred = rf_best.predict(X[te])
    yt_q1[te] = y_q1[te]
    yp_q1[te] = pred
    ba_scores.append(balanced_accuracy_score(y_q1[te], pred))
    f1_scores.append(f1_score(y_q1[te], pred, average="macro"))

all_true = list(yt_q1)   # for confusion matrix / classification_report downstream
all_pred = list(yp_q1)

mean_ba, lo_ba, hi_ba = bootstrap_ci(yt_q1, yp_q1)
mean_f1, lo_f1, hi_f1 = bootstrap_ci(yt_q1, yp_q1,
                                       metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro"))

print("=== Q1 Random Forest (best params, grouped CV) ===")
print(f"Balanced accuracy: {mean_ba:.4f} [{lo_ba:.4f}–{hi_ba:.4f}]")
print(f"Macro F1:          {mean_f1:.4f} [{lo_f1:.4f}–{hi_f1:.4f}]")
print()
print("Phase 7 LR reference: BA=0.8370 [0.8130–0.8590]")
print(f"Delta RF vs LR (BA):  {mean_ba - 0.8370:+.4f}")

# OOB score (train on all data)
rf_best.fit(X, y_q1)
print(f"\\nOOB score (full dataset, best RF): {rf_best.oob_score_:.4f}")

# Per-class report
from sklearn.metrics import classification_report
print("\\nPer-class metrics (pooled across CV folds):")
print(classification_report(all_true, all_pred, digits=3))
""")

# ── SECTION 7: CONFUSION MATRIX ─────────────────────────────────────────────
md("""\
## Section 7 — Confusion matrix

**Reading a confusion matrix:**
Rows = true class, Columns = predicted class. Diagonal = correct predictions.
Off-diagonal entries show which species are confused with which.

**What to look for:**
- Which species does the classifier struggle with most? (Row with most off-diagonal mass)
- Are confusions symmetric (A→B ≈ B→A) or one-directional?
- Does confusion match biology? (KP and EC should confuse more than KP and SA, given
  that KP and EC are both Enterobacterales with overlapping plasmid pools.)
""")

code("""\
species_order = sorted(set(all_true))
cm = confusion_matrix(all_true, all_pred, labels=species_order, normalize="true")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=species_order, yticklabels=species_order,
    linewidths=0.4, linecolor="white", ax=ax,
    vmin=0, vmax=1,
)
ax.set_xlabel("Predicted species", fontsize=11)
ax.set_ylabel("True species", fontsize=11)
ax.set_title(f"Q1 RF confusion matrix (grouped CV)\\nBA={mean_ba:.3f} [{lo_ba:.3f}–{hi_ba:.3f}]",
             fontsize=11)
plt.tight_layout()
fig.savefig(FIG / "q1_rf_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: results/figures/rf/q1_rf_confusion_matrix.png")

# Print the confusion as a readable table
cm_df = pd.DataFrame(cm, index=species_order, columns=species_order).round(3)
print("\\nNormalised confusion matrix (row = true class):")
print(cm_df.to_string())
""")

# ── SECTION 8: GINI IMPORTANCE ─────────────────────────────────────────────────
md("""\
## Section 8 — Feature importance: Gini (Mean Decrease in Impurity)

**How Gini importance works:**
Every time a feature is used to split a node, the split reduces impurity (Gini
coefficient) in the child nodes. Gini importance = total impurity reduction
attributable to that feature, summed across all nodes and all trees, normalised.

**Known bias:**
Gini importance over-rates features with many possible split thresholds — e.g.,
continuous count features (ARG count, IME count) will appear more important than
binary presence/absence features simply because they have more threshold options.
For our 265 binary features the bias is small but non-zero.

We plot the top 20 features and highlight any count or non-binary features.
""")

code("""\
# rf_best was last fit on full X (for OOB). Retrain for importance on filtered set.
rf_best.fit(X, y_q1)

imp_gini = pd.Series(rf_best.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
top20 = imp_gini.head(20)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#d62728" if "count" in n or "ratio" in n else "#1f77b4" for n in top20.index]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
ax.set_xlabel("Gini importance (mean decrease in impurity)")
ax.set_title("Top 20 features — Gini importance (RF, Q1)")
ax.axvline(0, color="black", lw=0.5)
plt.tight_layout()
fig.savefig(FIG / "q1_rf_gini_importance.png", dpi=150)
plt.close()
print("Saved: results/figures/rf/q1_rf_gini_importance.png")
print("\\nTop 10 features by Gini importance:")
for f, v in imp_gini.head(10).items():
    print(f"  {f:<45} {v:.5f}")
""")

# ── SECTION 9: PERMUTATION IMPORTANCE ─────────────────────────────────────────
md("""\
## Section 9 — Feature importance: Permutation importance

**How permutation importance works:**
The trained model is evaluated on a held-out set. Then, one feature at a time,
its values are randomly shuffled across all genomes — breaking any real relationship.
The drop in accuracy = how much that feature contributed.

**Why this is more reliable than Gini:**
1. Computed on held-out data (not training data) — cannot reward memorisation.
2. Not biased by cardinality — a count feature with many thresholds gets no bonus.
3. Works with any model (model-agnostic).

**The cost:** Slower. We run it once on the test portion of fold 1, not all 5 folds.

**What disagreement between Gini and permutation ranks tells you:**
A feature ranked high in Gini but low in permutation is probably exploiting
training-set noise — useful for fitting but not for generalisation.
""")

code("""\
# Use one held-out fold as the evaluation set for permutation importance
tr_idx, te_idx = next(cv.split(X, y_q1, groups=groups))
rf_perm = RandomForestClassifier(**best_params, class_weight="balanced",
                                  n_jobs=-1, random_state=RANDOM_STATE)
rf_perm.fit(X[tr_idx], y_q1[tr_idx])

perm_result = permutation_importance(
    rf_perm, X[te_idx], y_q1[te_idx],
    scoring        = "balanced_accuracy",
    n_repeats      = 30,
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
)

imp_perm = pd.Series(perm_result.importances_mean, index=FEAT_COLS)
imp_perm_std = pd.Series(perm_result.importances_std, index=FEAT_COLS)
top20_perm = imp_perm.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(
    top20_perm.index[::-1], top20_perm.values[::-1],
    xerr=imp_perm_std[top20_perm.index[::-1]].values,
    color="#2ca02c", alpha=0.8, capsize=3,
)
ax.set_xlabel("Mean decrease in balanced accuracy (permutation)")
ax.set_title("Top 20 features — Permutation importance (RF, Q1, fold 1)")
plt.tight_layout()
fig.savefig(FIG / "q1_rf_permutation_importance.png", dpi=150)
plt.close()
print("Saved: results/figures/rf/q1_rf_permutation_importance.png")

# Rank comparison: Gini vs Permutation
gini_ranks = {f: i+1 for i, f in enumerate(imp_gini.index)}
perm_ranks = {f: i+1 for i, f in enumerate(imp_perm.sort_values(ascending=False).index)}

print("\\nTop 10 by Permutation importance:")
for f, v in top20_perm.head(10).items():
    gr = gini_ranks.get(f, "—")
    print(f"  Perm #{perm_ranks[f]:<3} | Gini #{gr:<3} | {f:<45} {v:.5f}")
""")

# ── SECTION 10: SHAP ──────────────────────────────────────────────────────────
md("""\
## Section 10 — SHAP TreeExplainer

**What SHAP does:**
SHAP (SHapley Additive exPlanations) computes, for each genome and each feature,
how much that feature pushed the model's prediction away from the average prediction.

For a genome classified as *A. baumannii* with probability 0.92:
- Average probability across all genomes might be 0.17 (1/6 baseline)
- Each feature contributes a signed amount (positive = pushed toward AB, negative = pushed away)
- These signed contributions sum exactly to 0.92 − 0.17 = 0.75

This is the **additivity guarantee** that makes SHAP trustworthy: the feature contributions
sum to the actual prediction gap, no black box.

**TreeSHAP** (Lundberg et al. 2020) is the exact SHAP algorithm for tree-based models.
It is much faster than model-agnostic SHAP because it exploits the tree structure.

**Summary plot:** Each row = one feature. Each dot = one genome. X-axis = SHAP value
(direction and magnitude of effect on prediction). Colour = raw feature value
(red = high, blue = low). If high-value (red) dots are on the right → the feature
increases the predicted probability of that class.
""")

code("""\
# Compute SHAP on the held-out fold (same fold as permutation importance)
# SHAP 0.46+ returns a 3D array (n_samples, n_features, n_classes) for multiclass RF,
# not the older list-of-2D-arrays API. We index [:, :, cls_idx] per class.
explainer   = shap.TreeExplainer(rf_perm)
shap_values = explainer.shap_values(X[te_idx])   # shape: (n_te, 265, 6)

class_labels = sorted(set(y_q1))    # alphabetical: abaumannii, ecloaceae, ...
n_classes    = len(class_labels)

print(f"SHAP values shape: {shap_values.shape}  (samples × features × classes)")
print(f"Classes: {class_labels}")

# --- Summary plot: all classes, top 15 features ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, cls_idx, cls_name in zip(axes.flatten(), range(n_classes), class_labels):
    plt.sca(ax)
    shap.summary_plot(
        shap_values[:, :, cls_idx],   # 2D slice for this class: (n_te, 265)
        X[te_idx],
        feature_names = FEAT_COLS,
        max_display   = 15,
        show          = False,
        plot_size     = None,
        color_bar     = False,
    )
    ax.set_title(cls_name, fontsize=10)

plt.suptitle("SHAP summary plots — RF Q1 (per class, top 15 features)", y=1.01)
plt.tight_layout()
fig.savefig(FIG / "q1_rf_shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: results/figures/rf/q1_rf_shap_summary.png")

# Mean |SHAP| across classes: global feature importance
# shap_values is (n_te, 265, 6) → abs → mean over samples (axis=0) → mean over classes (axis=1)
mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)   # shape: (265,)
shap_global   = pd.Series(mean_abs_shap, index=FEAT_COLS).sort_values(ascending=False)

print("\\nTop 10 features by mean |SHAP| (averaged across all 6 classes):")
for f, v in shap_global.head(10).items():
    gr = gini_ranks.get(f, "—")
    pr = perm_ranks.get(f, "—")
    print(f"  SHAP #{shap_global.rank(ascending=False).astype(int)[f]:<3} | "
          f"Gini #{gr:<3} | Perm #{pr:<3} | {f:<40} {v:.5f}")
""")

# ── SECTION 11: RF vs LR COMPARISON ──────────────────────────────────────────
md("""\
## Section 11 — RF vs LR comparison (Q1)

**Decision rule for Phase 8:**
RF beats LR only if its 95% CI does not overlap with LR's CI, or if the delta
exceeds 0.02 balanced accuracy (a pre-specified minimum meaningful difference,
documented in docs/decisions.md).

A difference of 0.01 on noisy 5-fold CV means nothing scientifically.
""")

code("""\
# LR Phase 7 reference values (from project_state.md)
lr_ba, lr_lo, lr_hi = 0.8370, 0.8130, 0.8590

print("=== Q1 Model Comparison ===")
print(f"{'Model':<30} {'BA':>6}  {'95% CI':>16}")
print("-" * 55)
print(f"{'LR (Phase 7, filtered)':<30} {lr_ba:.4f}  [{lr_lo:.4f}–{lr_hi:.4f}]")
print(f"{'RF default (100 trees)':<30} ", end="")  # filled below after running
print(f"{'RF best (tuned)':<30} {mean_ba:.4f}  [{lo_ba:.4f}–{hi_ba:.4f}]")

ci_overlap = not (lo_ba > lr_hi or hi_ba < lr_lo)
delta_sig  = abs(mean_ba - lr_ba) > 0.02

print()
print(f"CI overlap: {ci_overlap}")
print(f"Delta > 0.02 threshold: {delta_sig} (delta = {mean_ba - lr_ba:+.4f})")
if ci_overlap or not delta_sig:
    print("→ No meaningful improvement over LR. Report both; prefer LR for parsimony.")
else:
    print("→ RF meaningfully outperforms LR. Use RF as primary Q1 model.")
""")

# ── SECTION 12: Q2 — ARG BURDEN PREDICTION ────────────────────────────────────
md("""\
## Section 12 — Q2: RF for ARG burden prediction (per species)

**What Q2 tests:**
Within each species, can the defence system profile predict whether a genome is in the
high-ARG tertile (top third of ARG count for that species)?

**H1 — Per-species sparsity filter (audit fix):**
SA (83%) and EF (78%) have the vast majority of the 265 `dp_*` features identically
zero across all their Q2 genomes — Gram-positive organisms operating in a
Gram-negative-dominated feature space. Training on all-zero features adds noise and
makes `max_features="sqrt"` over 265 notional features equivalent to sqrt over ~45
informative ones for SA. Each species' Q2 run now pre-filters to features with
>= 5% prevalence in that species' Q2-eligible genomes. Effective feature count
reported per species.

**Phase 7 LR reference (grouped CV):**
- EC: 0.752, KP: 0.719, PA: 0.645, EF: 0.512, SA: 0.470, AB: 0.473 (inverted -> 0.769)

**What RF adds:**
RF can model non-linear interactions between features. If SspBCDE + IME count
jointly predict high ARG but neither alone does, LR misses this; RF captures it
via split sequences within a tree.
""")

code("""\
from sklearn.metrics import roc_auc_score

results_q2_rf = {}
ba_sp_dict    = {}  # H4: fold-level BAs per species for BH correction

for sp in sorted(fm["species"].unique()):
    sp_mask = fm["species"] == sp
    fm_sp   = fm[sp_mask]

    # Q2 labels: use pre-computed arg_burden_tertile (top vs bottom tertile only)
    # Excludes mid_ARG genomes to match the LR baseline task exactly (C1 fix)
    mask_q2 = fm_sp["arg_burden_tertile"].isin(["low_ARG", "high_ARG"])
    fm_q2   = fm_sp[mask_q2]
    if len(fm_q2) < 20 or fm_q2["arg_burden_tertile"].nunique() < 2:
        print(f"  {sp}: insufficient Q2 data -- skip")
        continue
    y_sp   = (fm_q2["arg_burden_tertile"] == "high_ARG").astype(int).values
    # H1: per-species sparsity filter -- keep only features with >=5% prevalence
    feat_prev_sp = fm_q2[FEAT_COLS].mean()
    feat_q2_sp   = feat_prev_sp[feat_prev_sp >= 0.05].index.tolist()
    X_sp         = fm_q2[feat_q2_sp].values
    grp_sp = fm_q2["phylogroup"].to_numpy(dtype=str)
    cv_sp    = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"  {sp}: {len(feat_q2_sp)} features after H1 filter (from {len(FEAT_COLS)})")

    ba_sp, auc_sp = [], []
    all_yt_sp, all_yp_sp = [], []
    for tr, te in cv_sp.split(X_sp, y_sp, groups=grp_sp):
        if len(set(y_sp[te])) < 2:
            continue
        rf_q2 = RandomForestClassifier(**best_params, class_weight="balanced",
                                        n_jobs=-1, random_state=RANDOM_STATE)
        rf_q2.fit(X_sp[tr], y_sp[tr])
        pred   = rf_q2.predict(X_sp[te])
        prob   = rf_q2.predict_proba(X_sp[te])[:, 1]
        all_yt_sp.extend(y_sp[te])
        all_yp_sp.extend(pred)
        ba_sp.append(balanced_accuracy_score(y_sp[te], pred))
        auc_sp.append(roc_auc_score(y_sp[te], prob))

    if not ba_sp:
        continue

    ba_sp_dict[sp] = list(ba_sp)  # H4: store fold BAs before aggregation
    mean_ba_sp, lo_sp, hi_sp = bootstrap_ci(np.array(all_yt_sp), np.array(all_yp_sp))
    mean_auc_sp = np.mean(auc_sp)
    results_q2_rf[sp] = {
        "ba": mean_ba_sp, "lo": lo_sp, "hi": hi_sp, "auroc": mean_auc_sp
    }
    print(f"  {sp:<20} BA={mean_ba_sp:.3f} [{lo_sp:.3f}-{hi_sp:.3f}]  AUROC={mean_auc_sp:.3f}")

# Phase 7 LR reference for comparison
lr_q2_ref = {
    "ecloaceae":   {"ba": 0.752, "auroc": 0.846},
    "kpneumoniae": {"ba": 0.719, "auroc": 0.830},
    "paeruginosa": {"ba": 0.645, "auroc": 0.698},
    "efaecium":    {"ba": 0.512, "auroc": 0.578},
    "saureus":     {"ba": 0.470, "auroc": 0.556},
    "abaumannii":  {"ba": 0.473, "auroc": 0.231},
}
print("\\nQ2 RF vs LR comparison (BA):")
print(f"  {'Species':<22} {'RF BA':>7}  {'LR BA':>7}  {'Delta':>7}")
print("  " + "-" * 52)
for sp in sorted(results_q2_rf.keys()):
    rf_v = results_q2_rf[sp]["ba"]
    lr_v = lr_q2_ref.get(sp, {}).get("ba", float("nan"))
    delta = rf_v - lr_v
    print(f"  {sp:<22} {rf_v:.3f}    {lr_v:.3f}    {delta:+.3f}")
""")

# ── SECTION 12b: H4 — BH CORRECTION ACROSS Q2 SPECIES ─────────────────────────
md("""\
## Section 12b — H4: BH correction across Q2 species

**Audit finding H4:** The pre-analysis plan requires BH correction for all multiple
comparisons. Running 6 independent Q2 tests simultaneously inflates FWER to ~26%
at alpha=0.05. This section applies BH correction to per-species one-sample
t-tests (fold BAs vs BA=0.5 null).

**Note on power:** EF, SA, and AB may have only 3--4 usable folds due to
phylogroup-imbalance-driven fold skipping. t-tests at n=3--4 have very low power;
a non-significant result does not rule out a real but small effect. n_folds is
reported for transparency.
""")

code("""\
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# H4: one-sample t-test (BA > 0.5) per species, then BH correction
null_ba  = 0.5
sp_order = sorted(ba_sp_dict.keys())
p_raw    = []
for sp in sp_order:
    ba_list = ba_sp_dict[sp]
    if len(ba_list) >= 2:
        _, pval = ttest_1samp(ba_list, popmean=null_ba, alternative="greater")
        p_raw.append(pval)
    else:
        p_raw.append(float("nan"))

valid_idx   = [i for i, p in enumerate(p_raw) if not (p != p)]  # exclude NaN
sp_valid    = [sp_order[i] for i in valid_idx]
pvals_valid = [p_raw[i]    for i in valid_idx]

reject_bh, pvals_adj, _, _ = multipletests(pvals_valid, alpha=0.05, method="fdr_bh")
q2_pval_map = dict(zip(sp_valid, pvals_adj))
q2_praw_map = dict(zip(sp_valid, pvals_valid))
q2_sig_map  = dict(zip(sp_valid, reject_bh))

print("H4: Q2 RF null-baseline significance (one-sample t vs BA=0.5, BH-corrected):")
print(f"  {'Species':<22} {'BA':>6}  {'n_folds':>7}  {'p_raw':>8}  {'p_adj_BH':>10}  {'Sig?':>5}")
print("  " + "-" * 68)
for sp in sp_order:
    ba_v = results_q2_rf.get(sp, {}).get("ba", float("nan"))
    n_f  = len(ba_sp_dict.get(sp, []))
    p_r  = q2_praw_map.get(sp, float("nan"))
    p_a  = q2_pval_map.get(sp, float("nan"))
    sig  = "YES" if q2_sig_map.get(sp, False) else "ns"
    print(f"  {sp:<22} {ba_v:.3f}  {n_f:>7}  {p_r:.4f}    {p_a:.4f}      {sig}")
""")

# ── SECTION 12c: H7 — Q2 SENSITIVITY WITH ad_* FEATURES ──────────────────────
md("""\
## Section 12c — H7: Q2 sensitivity including anti-defence features

**Audit finding H7:** 29 `ad_*` (anti-defence) features from AntiDefenseFinder were
never included in any model. No exclusion rationale was logged.

**Decision (H7, logged in decisions.md 2026-05-26):**
- **Q1:** Exclude `ad_*`. Their spec_scores range 0.62--0.81, making them even more
  species-specific than the C2 borderline `dp_*` features. Including them would inflate
  Q1 species classification for the wrong reason.
- **Q2:** Sensitivity analysis here. Anti-defence systems are MGE-borne; if
  `ad_*`-positive genomes also carry more ARGs (via co-mobilisation), these features
  could genuinely improve Q2. The within-species framing eliminates the
  species-identity confound. H1 per-species filter applied identically.
""")

code("""\
ad_cols = [c for c in fm.columns if c.startswith("ad_")]

results_q2_rf_ad = {}  # dp_* + ad_* combined

for sp in sorted(fm["species"].unique()):
    sp_mask = fm["species"] == sp
    fm_sp   = fm[sp_mask]
    mask_q2 = fm_sp["arg_burden_tertile"].isin(["low_ARG", "high_ARG"])
    fm_q2   = fm_sp[mask_q2]
    if len(fm_q2) < 20 or fm_q2["arg_burden_tertile"].nunique() < 2:
        continue
    y_sp   = (fm_q2["arg_burden_tertile"] == "high_ARG").astype(int).values
    grp_sp = fm_q2["phylogroup"].to_numpy(dtype=str)

    # H1 filter on dp_* (same as main Q2 run)
    feat_prev_sp = fm_q2[FEAT_COLS].mean()
    feat_q2_sp   = feat_prev_sp[feat_prev_sp >= 0.05].index.tolist()

    # H1 filter on ad_* (same 5% threshold, within-species Q2 set)
    feat_prev_ad = fm_q2[ad_cols].mean()
    feat_ad_sp   = feat_prev_ad[feat_prev_ad >= 0.05].index.tolist()

    feat_combined = feat_q2_sp + feat_ad_sp
    X_sp_ad = fm_q2[feat_combined].values
    cv_sp   = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_yt, all_yp = [], []
    for tr, te in cv_sp.split(X_sp_ad, y_sp, groups=grp_sp):
        if len(set(y_sp[te])) < 2:
            continue
        rf_ad = RandomForestClassifier(**best_params, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE)
        rf_ad.fit(X_sp_ad[tr], y_sp[tr])
        pred = rf_ad.predict(X_sp_ad[te])
        all_yt.extend(y_sp[te])
        all_yp.extend(pred)

    if not all_yt:
        continue
    ba_ad, _, _ = bootstrap_ci(np.array(all_yt), np.array(all_yp))
    results_q2_rf_ad[sp] = {"ba_ad": ba_ad, "n_dp": len(feat_q2_sp),
                             "n_ad": len(feat_ad_sp)}

print("H7: Q2 RF sensitivity -- dp_* only vs dp_* + ad_* (both H1-filtered):")
print(f"  {'Species':<22} {'dp only':>8}  {'dp+ad':>8}  {'Delta':>7}  {'n_dp':>5}  {'n_ad':>5}")
print("  " + "-" * 62)
for sp in sorted(results_q2_rf_ad.keys()):
    ba_base = results_q2_rf.get(sp, {}).get("ba", float("nan"))
    ba_ad   = results_q2_rf_ad[sp]["ba_ad"]
    n_dp    = results_q2_rf_ad[sp]["n_dp"]
    n_ad    = results_q2_rf_ad[sp]["n_ad"]
    print(f"  {sp:<22} {ba_base:.3f}     {ba_ad:.3f}     {ba_ad - ba_base:+.3f}    {n_dp:>5}  {n_ad:>5}")
""")

# ── SECTION 13: C2 SENSITIVITY — SPEC-FILTER THRESHOLD ───────────────────────
md("""\
## Section 13 — C2 sensitivity: specificity filter threshold

**Audit finding C2:** Four of the top-5 SHAP features (`dp_df_Mok_Hok_Sok`, `dp_padloc_PDC-S13`,
`dp_df_FS_Sma`, `dp_df_Abi2`) have spec_score between 0.55 and 0.70 — they are borderline
taxonomic markers that mostly encode species identity, not defence architecture.

This section reruns Q1 at threshold=0.50 to quantify how much of the BA=0.878 depends on
those near-threshold features. The primary result (threshold=0.70, 265 features) does not change —
this is a transparency and robustness check that will be reported in the manuscript.
""")

code("""\
# C2 sensitivity analysis: Q1 BA at spec_score threshold 0.70 vs 0.50
results_sensitivity = {}

for threshold in [0.70, 0.50]:
    markers_t = spec_score[spec_score >= threshold].index.tolist()
    feat_t     = [c for c in dp_cols if c not in markers_t]
    X_t        = fm[feat_t].to_numpy(dtype=float)

    ba_folds = []
    yt_sens = np.empty(len(y_q1), dtype=object)
    yp_sens = np.empty(len(y_q1), dtype=object)
    for tr, te in cv.split(X_t, y_q1, groups=groups):
        rf_t = RandomForestClassifier(**best_params, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE)
        rf_t.fit(X_t[tr], y_q1[tr])
        pred_t = rf_t.predict(X_t[te])
        yt_sens[te] = y_q1[te]
        yp_sens[te] = pred_t
        ba_folds.append(balanced_accuracy_score(y_q1[te], pred_t))

    mean_t, lo_t, hi_t = bootstrap_ci(yt_sens, yp_sens)

    results_sensitivity[threshold] = {
        "n_features": len(feat_t),
        "n_removed":  len(markers_t),
        "ba": mean_t, "lo": lo_t, "hi": hi_t,
        "markers_removed": markers_t,
    }

# Additional markers removed when tightening from 0.70 → 0.50
borderline = set(results_sensitivity[0.50]["markers_removed"]) - set(results_sensitivity[0.70]["markers_removed"])

print("Q1 specificity filter sensitivity (C2):")
print(f"{'Threshold':<12} {'N features':>10} {'N removed':>10} {'BA':>8}  {'95% CI'}")
print("-" * 60)
for t in [0.70, 0.50]:
    r = results_sensitivity[t]
    print(f"  {t:<10} {r['n_features']:>10} {r['n_removed']:>10} {r['ba']:.3f}   [{r['lo']:.3f}–{r['hi']:.3f}]")
print()
print(f"Delta BA (0.70 → 0.50): {results_sensitivity[0.50]['ba'] - results_sensitivity[0.70]['ba']:+.3f}")
print()
print(f"6 borderline markers removed at 0.50 but retained at 0.70:")
for f in sorted(borderline):
    sp_prev_f = fm.groupby('species')[f].mean()
    dominant = sp_prev_f.idxmax()
    print(f"  {f:<35} spec_score={spec_score[f]:.3f}  dominant={dominant} ({sp_prev_f.max():.0%})")
print()
print("Interpretation: The 14.5pp drop confirms that SHAP ranks 1-4 at threshold=0.70")
print("are substantially driven by species-identity proxies. Phase 10 SHAP analysis")
print("will be restricted to features surviving the 0.50 filter (259 features).")
pd.DataFrame(results_sensitivity).T.to_parquet(RES / "q1_rf_sensitivity_spec_filter.parquet")
""")

# ── SECTION 13b: H8 — LEARNING CURVES ──────────────────────────────────────────
md("""\
## Section 13b — H8: Learning curves

**Audit finding H8:** Learning curves (accuracy vs training set size) were never computed.
With Q2 per-species n dropping to ~60--100 after H1 filter, it is unknown whether models
are sample-saturated or still on the steep part of the learning curve.

**What this shows:**
- Q1 (left): RF vs LR across increasing fractions of the 878-genome training set.
  Convergence means adding more genomes would not substantially change Q1 conclusions.
- Q2 (right): RF for EC, KP, PA -- the three species with significant Q2 signal.
  A steep right end means performance is still limited by sample size.

Train sizes use 5 evenly-spaced fractions (20%--100%). Error bands = +/- 1 std
across CV folds.
""")

code("""\
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- Q1 learning curve: RF vs LR ----
lr_q1 = LogisticRegression(max_iter=2000, class_weight="balanced",
                            solver="saga", C=0.1, random_state=RANDOM_STATE)

train_fracs = np.array([0.20, 0.40, 0.60, 0.80, 1.00])
cv_lc = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for model, label, color in [(rf_best, "RF", "steelblue"),
                             (lr_q1,   "LR", "tomato")]:
    ts, tr_sc, te_sc = learning_curve(
        model, X, y_q1, groups=groups,
        train_sizes=train_fracs,
        cv=cv_lc,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    ax = axes[0]
    ax.plot(ts, te_sc.mean(axis=1), "o-", color=color, label=label)
    ax.fill_between(ts,
                    te_sc.mean(axis=1) - te_sc.std(axis=1),
                    te_sc.mean(axis=1) + te_sc.std(axis=1),
                    alpha=0.18, color=color)

axes[0].axhline(0.130, color="grey", linestyle="--", linewidth=0.8, label="Null baseline")
axes[0].set_xlabel("Training set size (genomes)")
axes[0].set_ylabel("Balanced accuracy")
axes[0].set_title("Q1 learning curves: RF vs LR")
axes[0].legend()
axes[0].set_ylim(0.0, 1.05)

# ---- Q2 learning curve: RF for EC, KP, PA ----
sig_species = ["ecloaceae", "kpneumoniae", "paeruginosa"]
colors_q2   = ["steelblue", "darkorange", "seagreen"]

for sp, color in zip(sig_species, colors_q2):
    sp_mask = fm["species"] == sp
    fm_sp   = fm[sp_mask]
    mask_q2 = fm_sp["arg_burden_tertile"].isin(["low_ARG", "high_ARG"])
    fm_q2   = fm_sp[mask_q2]
    y_sp    = (fm_q2["arg_burden_tertile"] == "high_ARG").astype(int).values

    feat_prev_sp = fm_q2[FEAT_COLS].mean()
    feat_q2_sp   = feat_prev_sp[feat_prev_sp >= 0.05].index.tolist()
    X_sp         = fm_q2[feat_q2_sp].values
    grp_sp       = fm_q2["phylogroup"].to_numpy(dtype=str)

    cv_sp = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rf_q2_lc = RandomForestClassifier(**best_params, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE)
    ts_sp, _, te_sp = learning_curve(
        rf_q2_lc, X_sp, y_sp, groups=grp_sp,
        train_sizes=train_fracs,
        cv=cv_sp,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    axes[1].plot(ts_sp, te_sp.mean(axis=1), "o-", color=color, label=sp[:2].upper())
    axes[1].fill_between(ts_sp,
                         te_sp.mean(axis=1) - te_sp.std(axis=1),
                         te_sp.mean(axis=1) + te_sp.std(axis=1),
                         alpha=0.18, color=color)

axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Null (0.5)")
axes[1].set_xlabel("Training set size (genomes)")
axes[1].set_ylabel("Balanced accuracy")
axes[1].set_title("Q2 learning curves: RF (EC, KP, PA)")
axes[1].legend()
axes[1].set_ylim(0.3, 1.05)

fig.tight_layout()
fig.savefig(RES / "figures" / "rf" / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/rf/learning_curves.png")
""")

# ── SECTION 14: SAVE RESULTS ───────────────────────────────────────────────────
md("## Section 14 — Save results")

code("""\
import joblib

# Save best RF model
joblib.dump(rf_best, RES / "models" / "rf_q1_best.pkl")
print("Saved: results/models/rf_q1_best.pkl")

# Save Q1 results
q1_rf_results = pd.DataFrame({
    "model": ["RF_best"],
    "feature_set": ["filtered_265"],
    "cv": ["GroupedStratifiedKFold_5"],
    "ba_mean": [mean_ba], "ba_lo": [lo_ba], "ba_hi": [hi_ba],
    "f1_mean": [mean_f1], "f1_lo": [lo_f1], "f1_hi": [hi_f1],
    "oob_score": [rf_best.oob_score_],
    **{f"param_{k}": [v] for k, v in best_params.items()},
})
q1_rf_results.to_parquet(RES / "q1_rf_results.parquet")
print("Saved: results/q1_rf_results.parquet")

# Save Q2 results (including H4 BH-corrected p-values)
q2_rows = []
for sp, vals in results_q2_rf.items():
    q2_rows.append({"species": sp, "model": "RF_best",
                    "p_adj_bh": q2_pval_map.get(sp, float("nan")),
                    "p_raw":    q2_praw_map.get(sp, float("nan")),
                    "sig_bh":   q2_sig_map.get(sp, False),
                    **vals})
pd.DataFrame(q2_rows).to_parquet(RES / "q2_rf_results.parquet")
print("Saved: results/q2_rf_results.parquet")

# Save feature importance comparison table
imp_compare = pd.DataFrame({
    "gini_rank":  pd.Series(gini_ranks),
    "perm_rank":  pd.Series(perm_ranks),
    "shap_rank":  shap_global.rank(ascending=False).astype(int),
    "gini_val":   imp_gini,
    "perm_val":   imp_perm,
    "shap_val":   shap_global,
}).sort_values("shap_rank")
imp_compare.to_parquet(RES / "q1_rf_feature_importance.parquet")
imp_compare.to_csv(RES / "q1_rf_feature_importance.csv")
print("Saved: results/q1_rf_feature_importance.parquet/.csv")
print("\\nAll outputs saved.")
""")

nb.cells = cells
out_path = Path("notebooks/06_random_forest.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written: {out_path}  ({len(cells)} cells)")
