"""
Build notebooks/07_gradient_boosting.ipynb programmatically.
Run once from the eskape-defence-ml/ root:
    conda run -n eskape-ml python notebooks/build_gb_notebook.py
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
# Phase 9 — Gradient Boosting (XGBoost + LightGBM)

**Research questions addressed:**
- **Q1** (multi-class): Can defence system repertoire classify species?
  How does gradient boosting compare to Random Forest?
- **Q2** (binary, per species): Can defence profile predict high-ARG burden?

**Methodology highlights:**
- All evaluation uses `StratifiedGroupKFold` (5-fold, groups = Mash phylogroups from Phase 6)
- Feature set: identical 265 dp_* features used in Phases 7–8 (specificity filter applied)
- Hyperparameter tuning: GridSearchCV (fixed n_estimators) to find best tree structure,
  then early stopping to set n_estimators in the final model
- Statistical comparison vs RF: McNemar's test on pooled per-genome predictions (n=878)
- Calibration: reliability diagram + Brier score for XGBoost and LightGBM

**Learning thread:**
Concepts introduced: sequential boosting vs parallel bagging, gradient descent in function
space, early stopping and the CV-leakage trap, probability calibration, statistical model
comparison. Each section includes a grounded question before the code runs.
""")

# ── SECTION 1: IMPORTS ─────────────────────────────────────────────────────────
md("## Section 1 — Imports and configuration")

code("""\
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.dummy import DummyClassifier
import joblib

from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

ROOT  = Path("..")
PROC  = ROOT / "data" / "processed"
RES   = ROOT / "results"
FIG   = RES / "figures" / "gb"
FIG.mkdir(parents=True, exist_ok=True)
(RES / "models").mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS     = 5
N_BOOT       = 2000

print(f"XGBoost version: {__import__('xgboost').__version__}")
print(f"LightGBM version: {lgb.__version__}")
print("Imports OK.")
""")

# ── SECTION 2: LOAD DATA + FEATURE SELECTION ───────────────────────────────────
md("""\
## Section 2 — Load data and feature selection

Identical 265-feature set from Phases 7–8. Using the same features across all models
is non-negotiable for valid performance comparison.
""")

code("""\
fm      = pd.read_parquet(PROC / "feature_matrix.parquet")
dp_cols    = sorted([c for c in fm.columns if c.startswith("dp_")])
sp_prev    = fm.groupby("species")[dp_cols].mean()
spec_score = sp_prev.std() / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]

y_q1_str = fm["species"].to_numpy(dtype=str)
groups   = fm["phylogroup"].to_numpy(dtype=str)
X        = fm[FEAT_COLS].to_numpy(dtype=float)

# Integer-encode labels: XGBoost requires integer class labels (or sklearn auto-handles it,
# but explicit encoding avoids edge cases across XGBoost versions)
le = LabelEncoder()
y_q1 = le.fit_transform(y_q1_str)   # 0..5
CLASSES = list(le.classes_)         # alphabetical: abaumannii, ecloaceae, ...

print(f"Feature matrix: {X.shape[0]} genomes x {X.shape[1]} features")
print(f"Classes ({len(CLASSES)}): {CLASSES}")
print(f"Labels encoded: {dict(zip(CLASSES, le.transform(CLASSES)))}")
print(f"Markers removed: {len(markers)}")
""")

# ── SECTION 3: CV SETUP ────────────────────────────────────────────────────────
md("""\
## Section 3 — GroupedStratifiedKFold (same as Phases 7–8)

No new concepts here — this is the same splitter established in Phase 6.
The key constraint: all genomes from one phylogroup land in the same fold.
""")

code("""\
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

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

# Quick fold-size check
fold_sizes = [len(te) for _, te in cv.split(X, y_q1, groups=groups)]
print(f"Fold sizes: {fold_sizes}  (range {min(fold_sizes)}-{max(fold_sizes)})")
""")

# ── SECTION 4: CONCEPT — BOOSTING VS BAGGING ───────────────────────────────────
md("""\
## Section 4 — Concept: boosting vs bagging

**Random Forest (Phase 8) was bagging:**
- Build 100 independent trees *in parallel*, each on a bootstrap sample of genomes.
- Each tree votes; majority wins.
- Error reduction source: **variance reduction** (averaging uncorrelated trees).
- Weak point: each tree is fully grown — high variance but the average is stable.

**Gradient Boosting (this phase) is sequential correction:**
- Build tree 1 on the data. Record the errors (residuals).
- Build tree 2 to predict the *residuals* of tree 1.
- Build tree 3 to predict the *residuals* of tree 1 + tree 2.
- ... and so on for n_estimators trees.
- The final prediction = weighted sum of all trees.

The "gradient" part: fitting residuals is equivalent to gradient descent on the loss
function in function space. Each new tree takes one gradient step toward lower loss.

**Why boosting often beats RF on tabular data:**
Boosting reduces both bias (each step learns the remaining errors) and variance
(shallow trees, regularised with learning_rate). RF reduces variance but not bias
if the trees are already complex. On 265 binary features with ~150 genomes per class,
the bias is non-trivial — boosting has a structural advantage.

**The cost:** Boosting is sensitive to overfitting because errors are amplified forward.
This is why learning_rate and n_estimators must be controlled — which is what early
stopping does.

**Grounded question before running anything:**
If you double the learning_rate (e.g., 0.1 → 0.2), what happens to:
(a) the number of trees needed to converge?
(b) the risk of overfitting?
Write your prediction before looking at Section 5's results.
""")

# ── SECTION 5: XGBOOST QUICKRUN ────────────────────────────────────────────────
md("""\
## Section 5 — XGBoost quickrun (no early stopping, default-like params)

Before tuning, run XGBoost with reasonable defaults and no early stopping.
This gives a reference point: does gradient boosting outperform RF=0.878 at all?

**Parameters:**
- `n_estimators=200`: 200 trees (we'll improve this with early stopping later)
- `learning_rate=0.1`: standard starting point
- `max_depth=6`: moderately deep (RF used max_depth=20; shallower is better for boosting)
- `subsample=0.8`: subsample 80% of genomes per tree (reduces overfitting, adds randomness)
- `colsample_bytree=0.8`: subsample 80% of features per tree

**No class_weight here:** XGBoost handles class imbalance via `sample_weight` in fit(),
not via a constructor parameter. Species distribution is ~150/species so imbalance is
mild; compute sample weights as inverse class frequency.
""")

code("""\
from sklearn.utils.class_weight import compute_sample_weight

xgb_quick = XGBClassifier(
    n_estimators     = 200,
    learning_rate    = 0.1,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "mlogloss",
    verbosity        = 0,
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)

ba_scores = []
yt_quick = np.empty(len(y_q1), dtype=int)
yp_quick = np.empty(len(y_q1), dtype=int)
for tr, te in cv.split(X, y_q1, groups=groups):
    sw = compute_sample_weight("balanced", y_q1[tr])
    xgb_quick.fit(X[tr], y_q1[tr], sample_weight=sw)
    pred = xgb_quick.predict(X[te])
    yt_quick[te] = y_q1[te]
    yp_quick[te] = pred
    ba_scores.append(balanced_accuracy_score(y_q1[te], pred))

mean_ba, lo, hi = bootstrap_ci(yt_quick, yp_quick)
print(f"XGBoost quickrun (grouped CV): BA={mean_ba:.4f} [{lo:.4f}-{hi:.4f}]")
print(f"RF Phase 8 reference:          BA=0.8780 [0.8590-0.8980]")
print(f"Delta XGB_quick vs RF:         {mean_ba - 0.8780:+.4f}")
""")

# ── SECTION 6: EARLY STOPPING MECHANICS ───────────────────────────────────────
md("""\
## Section 6 — Early stopping: the concept and the CV trap

**What early stopping does:**
Train with up to `n_estimators=500` trees, but after each tree evaluate loss on a
*validation set*. If validation loss does not improve for `early_stopping_rounds` trees
in a row, stop. The final model uses the iteration with the best validation loss.

This turns `n_estimators` from a hyperparameter into something that is automatically
determined by the data. The effective n_estimators = `best_iteration_` after fit.

**The CV trap — and why it matters for this project:**
The naive way to add early stopping inside cross-validation is:

```python
# WRONG
for tr, te in cv.split(X, y, groups=groups):
    xgb.fit(X[tr], y[tr], eval_set=[(X[te], y[te])])   # <-- test fold used for early stopping
    pred = xgb.predict(X[te])
```

This is a form of data leakage: the model's training *stops* based on test-fold performance.
The model has seen the test set during training. Validation loss on the test fold will
be optimistically biased.

**The correct approach:**
Split the training fold 80/20. The 20% inner slice is used *only* for early stopping.
The original test fold is never touched until evaluation:

```
Full data
 |
 CV fold split (by phylogroup)
 |—— training fold (80–85% of data) ───|—— 20% inner val ──|
 |                                                           |
 XGBoost trains here        early stopping monitors this
 |
 |—— TEST fold (never seen) ─────────────────────────────────
                    evaluated here after best_iteration_ found
```

**Grounded question:** In the above diagram, can the 20% inner validation slice contain
genomes from the same phylogroups as the test fold? Why does this matter less than it
did for the outer CV split?
""")

code("""\
# Demonstrate early stopping with correct inner-fold split
# Use fold 0 for the demonstration
tr_idx, te_idx = next(cv.split(X, y_q1, groups=groups))

# Split training fold: 80% train, 20% val for early stopping
inner_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_inner_rel, val_inner_rel = next(inner_split.split(X[tr_idx], y_q1[tr_idx]))
tr_inner  = tr_idx[tr_inner_rel]
val_inner = tr_idx[val_inner_rel]

sw_inner = compute_sample_weight("balanced", y_q1[tr_inner])

xgb_es_demo = XGBClassifier(
    n_estimators         = 500,
    early_stopping_rounds = 30,
    learning_rate        = 0.1,
    max_depth            = 6,
    subsample            = 0.8,
    colsample_bytree     = 0.8,
    eval_metric          = "mlogloss",
    verbosity            = 0,
    random_state         = RANDOM_STATE,
    n_jobs               = -1,
)

xgb_es_demo.fit(
    X[tr_inner], y_q1[tr_inner],
    sample_weight = sw_inner,
    eval_set      = [(X[val_inner], y_q1[val_inner])],
    verbose       = False,
)

print(f"Trees trained (n_estimators cap): 500")
print(f"Best iteration (early stopping):  {xgb_es_demo.best_iteration}")
print(f"Validation mlogloss at best iter: {xgb_es_demo.best_score:.5f}")
print()
ba_fold0 = balanced_accuracy_score(y_q1[te_idx], xgb_es_demo.predict(X[te_idx]))
print(f"Fold 0 BA with early stopping ({xgb_es_demo.best_iteration} trees): {ba_fold0:.4f}")
print()
print("Compare: XGBoost fixed 200 trees in Section 5 vs optimal early-stopped trees above.")
print("If best_iteration << 200: we were adding wasteful trees in Section 5.")
print("If best_iteration > 200:  we were underfitting in Section 5.")
""")

# ── SECTION 7: HYPERPARAMETER GRID SEARCH ─────────────────────────────────────
md("""\
## Section 7 — XGBoost hyperparameter grid search

**Strategy:** Use GridSearchCV with a *fixed* n_estimators=200 (no early stopping during
grid search) to identify the best (learning_rate, max_depth, subsample) combination.
Then, in Section 8, take those best params and apply early stopping to set n_estimators.

**Why separate the two steps:**
1. GridSearchCV + early stopping + grouped CV + sample weights is a complex interaction
   that is error-prone to implement correctly.
2. Conceptually: tree structure (max_depth, learning_rate, subsample) and number of
   trees (n_estimators via early stopping) are independent hyperparameters.

**Grid:** 3 × 2 × 2 = 12 combinations × 5 folds = 60 XGBoost fits. At n=878, this
runs in ~1-2 minutes.

**Grounded question before running:**
Why is a smaller learning_rate (e.g., 0.05) generally paired with a larger n_estimators?
""")

code("""\
param_grid_xgb = {
    "learning_rate"   : [0.05, 0.1, 0.3],
    "max_depth"       : [4, 6],
    "subsample"       : [0.8, 1.0],
}

xgb_base = XGBClassifier(
    n_estimators     = 200,          # fixed for grid search
    colsample_bytree = 0.8,
    eval_metric      = "mlogloss",
    verbosity        = 0,
    random_state     = RANDOM_STATE,
    n_jobs           = 1,            # 1 inside; GridSearchCV handles outer parallelism
)

grid_xgb = GridSearchCV(
    estimator  = xgb_base,
    param_grid = param_grid_xgb,
    cv         = cv,
    scoring    = "balanced_accuracy",
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,
)

# M3 fix: sample_weight NOT passed to GridSearchCV — weights computed once on full data
# encode class imbalance of held-out test genomes, a minor form of leakage.
# Hyperparameter tuning uses balanced_accuracy scoring which is already imbalance-aware.
# Per-fold sample weights are correctly computed inside the early-stopping CV loop below.
grid_xgb.fit(X, y_q1, groups=groups)

print("\\nBest hyperparameters (XGBoost, GridSearchCV):")
for k, v in grid_xgb.best_params_.items():
    print(f"  {k:<22} {v}")
print(f"\\nBest CV balanced accuracy: {grid_xgb.best_score_:.4f}")

# Show top 5 combinations
cv_results = pd.DataFrame(grid_xgb.cv_results_)
top5 = (cv_results
        .sort_values("mean_test_score", ascending=False)
        .head(5)[["param_learning_rate", "param_max_depth", "param_subsample",
                  "mean_test_score", "std_test_score"]])
print("\\nTop 5 parameter combinations:")
print(top5.to_string(index=False))
""")

# ── SECTION 8: FINAL XGBOOST WITH EARLY STOPPING ──────────────────────────────
md("""\
## Section 8 — Final XGBoost: best params + early stopping (Q1)

Take the best hyperparameters from Section 7. Now apply early stopping in a proper
grouped CV loop to:
1. Set n_estimators adaptively (no guessing)
2. Get final performance estimate with 95% CI
3. Collect per-genome predictions for McNemar test (Section 10)

**Implementation note — early stopping inside CV:**
A new XGBClassifier instance is created at each fold. This is required: if we reuse the
same instance, `early_stopping_rounds` state from a previous fold's `fit()` persists
and corrupts the iteration count. Recreating the instance resets internal state cleanly.
""")

code("""\
best_params_xgb = grid_xgb.best_params_

ba_scores_xgb = []
f1_scores_xgb = []
best_iters    = []

# For McNemar test: need aligned (genome_idx, true, pred) across all folds
mc_true_xgb  = np.empty(len(y_q1), dtype=int)
mc_pred_xgb  = np.empty(len(y_q1), dtype=int)

for tr, te in cv.split(X, y_q1, groups=groups):
    # Inner validation split for early stopping (from training fold only)
    inner_ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_rel, val_rel = next(inner_ss.split(X[tr], y_q1[tr]))
    tr_inner  = tr[tr_rel]
    val_inner = tr[val_rel]

    sw_inner = compute_sample_weight("balanced", y_q1[tr_inner])

    # Fresh instance per fold — required for early stopping state isolation
    xgb_fold = XGBClassifier(
        **best_params_xgb,
        n_estimators          = 500,
        early_stopping_rounds = 30,
        colsample_bytree      = 0.8,
        eval_metric           = "mlogloss",
        verbosity             = 0,
        random_state          = RANDOM_STATE,
        n_jobs                = -1,
    )

    xgb_fold.fit(
        X[tr_inner], y_q1[tr_inner],
        sample_weight = sw_inner,
        eval_set      = [(X[val_inner], y_q1[val_inner])],
        verbose       = False,
    )

    pred = xgb_fold.predict(X[te])
    ba_scores_xgb.append(balanced_accuracy_score(y_q1[te], pred))
    f1_scores_xgb.append(f1_score(y_q1[te], pred, average="macro"))
    best_iters.append(xgb_fold.best_iteration)
    mc_true_xgb[te] = y_q1[te]
    mc_pred_xgb[te] = pred

mean_ba_xgb, lo_xgb, hi_xgb = bootstrap_ci(mc_true_xgb, mc_pred_xgb)
mean_f1_xgb, lo_f1_xgb, hi_f1_xgb = bootstrap_ci(mc_true_xgb, mc_pred_xgb,
    metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro"))

print(f"=== XGBoost Q1 (best params + early stopping, grouped CV) ===")
print(f"Balanced accuracy: {mean_ba_xgb:.4f} [{lo_xgb:.4f}-{hi_xgb:.4f}]")
print(f"Macro F1:          {mean_f1_xgb:.4f} [{lo_f1_xgb:.4f}-{hi_f1_xgb:.4f}]")
print(f"Best iterations (per fold): {best_iters}  median={np.median(best_iters):.0f}")
print()
print(f"RF Phase 8 reference: BA=0.8780 [0.8590-0.8980]")
print(f"Delta XGB vs RF:      {mean_ba_xgb - 0.8780:+.4f}")
""")

# ── SECTION 9: LIGHTGBM ────────────────────────────────────────────────────────
md("""\
## Section 9 — LightGBM

**LightGBM vs XGBoost — three implementation differences that matter:**

1. **Leaf-wise vs level-wise growth:**
   XGBoost grows trees level by level (all leaves at a given depth simultaneously).
   LightGBM grows the single leaf with the highest gain at each step — it can produce
   deeper, more complex trees for the same number of leaves. `num_leaves` (not `max_depth`)
   is the primary complexity control.

2. **Histogram-based splitting:**
   LightGBM bins continuous features into buckets (histograms), reducing the number of
   split threshold evaluations. On 265 binary features, the advantage is minimal —
   binary features already have only one split threshold each.

3. **Early stopping via callbacks (LightGBM 4.x API):**
   Unlike XGBoost (which takes `early_stopping_rounds` as a constructor parameter),
   LightGBM 4.x uses a callbacks list:
   ```python
   callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
   ```
   Functionally identical; the API difference is a library design choice.

**Parameter choices:**
- `num_leaves=63`: roughly equivalent to `max_depth=6` in XGBoost
  (num_leaves ≈ 2^max_depth / 2 for balanced trees; LightGBM grows asymmetrically)
- `learning_rate`, `subsample`: same as XGBoost best params for fair comparison

**Grounded question before running:**
If LightGBM's leaf-wise growth tends to overfit more aggressively than XGBoost's
level-wise growth, which regularisation parameter would you increase to compensate?
""")

code("""\
ba_scores_lgbm = []
f1_scores_lgbm = []
best_iters_lgbm = []

mc_true_lgbm  = np.empty(len(y_q1), dtype=int)
mc_pred_lgbm  = np.empty(len(y_q1), dtype=int)

lgbm_params = {
    "learning_rate"   : best_params_xgb.get("learning_rate", 0.1),
    "max_depth"       : best_params_xgb.get("max_depth", 6),
    "num_leaves"      : 63,
    "subsample"       : best_params_xgb.get("subsample", 0.8),
    "subsample_freq"  : 1,      # required for subsample to take effect
    "colsample_bytree": 0.8,
    "class_weight"    : "balanced",
    "n_estimators"    : 500,    # ceiling; early stopping will find optimal
    "verbosity"       : -1,
    "random_state"    : RANDOM_STATE,
    "n_jobs"          : -1,
}

for tr, te in cv.split(X, y_q1, groups=groups):
    inner_ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_rel, val_rel = next(inner_ss.split(X[tr], y_q1[tr]))
    tr_inner  = tr[tr_rel]
    val_inner = tr[val_rel]

    lgbm_fold = LGBMClassifier(**lgbm_params)

    lgbm_fold.fit(
        X[tr_inner], y_q1[tr_inner],
        eval_set  = [(X[val_inner], y_q1[val_inner])],
        callbacks = [
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    pred = lgbm_fold.predict(X[te])
    ba_scores_lgbm.append(balanced_accuracy_score(y_q1[te], pred))
    f1_scores_lgbm.append(f1_score(y_q1[te], pred, average="macro"))
    best_iters_lgbm.append(lgbm_fold.best_iteration_)
    mc_true_lgbm[te] = y_q1[te]
    mc_pred_lgbm[te] = pred

mean_ba_lgbm, lo_lgbm, hi_lgbm = bootstrap_ci(mc_true_lgbm, mc_pred_lgbm)
mean_f1_lgbm, lo_f1_lgbm, hi_f1_lgbm = bootstrap_ci(mc_true_lgbm, mc_pred_lgbm,
    metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro"))

print(f"=== LightGBM Q1 (early stopping, grouped CV) ===")
print(f"Balanced accuracy: {mean_ba_lgbm:.4f} [{lo_lgbm:.4f}-{hi_lgbm:.4f}]")
print(f"Macro F1:          {mean_f1_lgbm:.4f} [{lo_f1_lgbm:.4f}-{hi_f1_lgbm:.4f}]")
print(f"Best iterations (per fold): {best_iters_lgbm}  median={np.median(best_iters_lgbm):.0f}")
print()
print("=== Summary so far ===")
lr_ba, lr_lo, lr_hi = 0.8370, 0.8130, 0.8590
rf_ba, rf_lo, rf_hi = 0.8780, 0.8590, 0.8980
print(f"{'Model':<28} {'BA':>6}  {'95% CI':>16}")
print("-" * 52)
print(f"{'LR (Phase 7)':<28} {lr_ba:.4f}  [{lr_lo:.4f}-{lr_hi:.4f}]")
print(f"{'RF (Phase 8)':<28} {rf_ba:.4f}  [{rf_lo:.4f}-{rf_hi:.4f}]")
print(f"{'XGBoost':<28} {mean_ba_xgb:.4f}  [{lo_xgb:.4f}-{hi_xgb:.4f}]")
print(f"{'LightGBM':<28} {mean_ba_lgbm:.4f}  [{lo_lgbm:.4f}-{hi_lgbm:.4f}]")
""")

# ── SECTION 10: STATISTICAL COMPARISON — MCNEMAR ──────────────────────────────
md("""\
## Section 10 — Statistical comparison: McNemar's test

**Why point-estimate comparisons are not enough:**
XGBoost BA=0.882 vs RF BA=0.878 is a difference of 0.004. Is this real or noise?
With 5 fold-level scores, a Wilcoxon test has very low power (n=5). We need more.

**McNemar's test — the correct tool:**
McNemar's test operates on paired binary outcomes: for each genome, was model A
correct and model B wrong, or vice versa? The test statistic uses the *discordant*
pairs — cases where the models disagree.

Contingency table (n=878 genomes):
```
                  Model B correct   Model B wrong
Model A correct        a                 b
Model A wrong          c                 d
```
McNemar statistic: (b - c)² / (b + c) ~ χ²(1)
Only b and c matter — the genomes where the models disagree.
If b ≈ c: the models make errors on the same genomes → no significant difference.
If b >> c (or c >> b): one model is making substantially different errors → real difference.

**What constitutes a meaningful difference here:**
p < 0.05 from McNemar is a statistical threshold. The scientific threshold is whether
the performance gain justifies the added model complexity. For this project:
if delta BA ≤ 0.02 and McNemar p > 0.05, models are equivalent — prefer the simpler one.
""")

code("""\
from scipy.stats import chi2

def mcnemar_test(true, pred_a, pred_b, labels=None):
    \"\"\"McNemar test comparing two classifiers on the same genomes.\"\"\"
    correct_a = (true == pred_a)
    correct_b = (true == pred_b)
    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(correct_b & ~correct_a)   # B right, A wrong
    if b + c == 0:
        return 1.0, b, c, 0.0
    # Continuity-corrected McNemar
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p    = 1 - chi2.cdf(stat, df=1)
    return p, b, c, stat

# Load RF per-genome predictions: re-run RF CV with same splits to get aligned predictions
# (rf_q1_best.pkl was trained on all data for OOB; we need CV predictions)
rf_best = joblib.load(RES / "models" / "rf_q1_best.pkl")
mc_true_rf = np.empty(len(y_q1), dtype=int)
mc_pred_rf = np.empty(len(y_q1), dtype=int)

for tr, te in cv.split(X, y_q1, groups=groups):
    sw = compute_sample_weight("balanced", y_q1[tr])
    rf_fold_params = {k: v for k, v in rf_best.get_params().items()
                      if k in ["n_estimators", "max_depth", "min_samples_leaf",
                               "max_features", "class_weight", "random_state", "n_jobs"]}
    from sklearn.ensemble import RandomForestClassifier
    rf_fold = RandomForestClassifier(**rf_fold_params)
    rf_fold.fit(X[tr], y_q1[tr])
    mc_true_rf[te] = y_q1[te]
    mc_pred_rf[te] = rf_fold.predict(X[te])

# McNemar pairwise tests
print("=== McNemar's test — pairwise model comparison (Q1, n=878) ===")
print()
comparisons = [
    ("XGBoost vs RF",      mc_true_xgb, mc_pred_xgb, mc_true_rf,   mc_pred_rf),
    ("LightGBM vs RF",     mc_true_lgbm, mc_pred_lgbm, mc_true_rf, mc_pred_rf),
    ("XGBoost vs LightGBM", mc_true_xgb, mc_pred_xgb, mc_true_lgbm, mc_pred_lgbm),
]

for label, true_a, pred_a, true_b, pred_b in comparisons:
    # Use model A's true labels (all folds cover same genomes, so any true array is fine)
    p, b, c, stat = mcnemar_test(true_a, pred_a, pred_b)
    acc_a = balanced_accuracy_score(true_a, pred_a)
    acc_b = balanced_accuracy_score(true_b, pred_b)
    sig = "**significant**" if p < 0.05 else "not significant"
    print(f"{label}")
    print(f"  BA: {acc_a:.4f} vs {acc_b:.4f}  delta={acc_a - acc_b:+.4f}")
    print(f"  Discordant pairs: b={b} (A right/B wrong), c={c} (B right/A wrong)")
    print(f"  chi2={stat:.3f}, p={p:.4f}  [{sig}]")
    print()
""")

# ── SECTION 10b: H3 — FAIR COMPARISON (FIXED N_ESTIMATORS) ───────────────────
md("""\
## Section 10b — H3: Fair comparison with fixed n_estimators

**Audit finding H3:** XGBoost and LightGBM use an 80/20 inner split for early
stopping, so they train on only ~64% of the data per fold (80% x 80%).
RF trains on ~80%. The 14pp BA gap (RF=0.878 vs XGB=0.817) may partly reflect
data starvation rather than model quality.

**What this section does:**
Re-run XGB and LGBM with:
- Fixed `n_estimators` from the median of early-stopping best iterations
  (XGB median=143, LGBM median=92)
- No inner validation split -- full training fold used (same effective data as RF)
- Same hyperparameters otherwise

If the gap closes substantially: data starvation was the primary cause.
If the gap persists: RF genuinely outperforms gradient boosting on this dataset.
""")

code("""\
# H3: fixed n_estimators -- use median from early stopping runs
n_est_xgb  = int(np.median(best_iters))
n_est_lgbm = int(np.median(best_iters_lgbm))
print(f"Fixed n_estimators -- XGB: {n_est_xgb}, LGBM: {n_est_lgbm}")

ba_xgb_fixed  = []
ba_lgbm_fixed = []
mc_pred_xgb_fixed  = np.empty(len(y_q1), dtype=int)
mc_pred_lgbm_fixed = np.empty(len(y_q1), dtype=int)
mc_true_fixed      = np.empty(len(y_q1), dtype=int)

for tr, te in cv.split(X, y_q1, groups=groups):
    sw = compute_sample_weight("balanced", y_q1[tr])
    mc_true_fixed[te] = y_q1[te]

    # XGB: fixed n_estimators, full training fold
    xgb_fixed = XGBClassifier(
        **best_params_xgb,
        n_estimators     = n_est_xgb,
        colsample_bytree = 0.8,
        verbosity        = 0,
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    xgb_fixed.fit(X[tr], y_q1[tr], sample_weight=sw)
    pred_xf = xgb_fixed.predict(X[te])
    ba_xgb_fixed.append(balanced_accuracy_score(y_q1[te], pred_xf))
    mc_pred_xgb_fixed[te] = pred_xf

    # LGBM: fixed n_estimators, full training fold
    lgbm_fixed = LGBMClassifier(
        **{k: v for k, v in lgbm_params.items() if k != "n_estimators"},
        n_estimators = n_est_lgbm,
    )
    lgbm_fixed.fit(X[tr], y_q1[tr])
    pred_lf = lgbm_fixed.predict(X[te])
    ba_lgbm_fixed.append(balanced_accuracy_score(y_q1[te], pred_lf))
    mc_pred_lgbm_fixed[te] = pred_lf

ba_xf,  lo_xf,  hi_xf  = bootstrap_ci(mc_true_fixed, mc_pred_xgb_fixed)
ba_lf,  lo_lf,  hi_lf  = bootstrap_ci(mc_true_fixed, mc_pred_lgbm_fixed)

lbl_xf = f"XGB (fixed n={n_est_xgb}, 80% fold)"
lbl_lf = f"LGBM (fixed n={n_est_lgbm}, 80% fold)"

print()
print("=== H3: fixed-n vs early-stopping vs RF (Q1 BA) ===")
print(f"  {'Model':<38} {'BA':>6}  {'95% CI':<18}  {'vs RF delta':>11}")
print("  " + "-" * 78)
ref_rf  = 0.8780
print(f"  {'RF (80% fold, n_estimators free)':<38} {ref_rf:.4f}  [0.8590-0.8980]")
print(f"  {'XGB (early-stop, 64% fold)':<38} {mean_ba_xgb:.4f}  [{lo_xgb:.4f}-{hi_xgb:.4f}]  {mean_ba_xgb - ref_rf:+.4f}")
print(f"  {lbl_xf:<38} {ba_xf:.4f}  [{lo_xf:.4f}-{hi_xf:.4f}]  {ba_xf - ref_rf:+.4f}")
print(f"  {'LGBM (early-stop, 64% fold)':<38} {mean_ba_lgbm:.4f}  [{lo_lgbm:.4f}-{hi_lgbm:.4f}]  {mean_ba_lgbm - ref_rf:+.4f}")
print(f"  {lbl_lf:<38} {ba_lf:.4f}  [{lo_lf:.4f}-{hi_lf:.4f}]  {ba_lf - ref_rf:+.4f}")
print()

# McNemar: fixed-n XGB vs RF
p_xf_rf, b_xf_rf, c_xf_rf, _ = mcnemar_test(mc_true_fixed, mc_pred_xgb_fixed, mc_pred_rf)
sig_xf = "**significant**" if p_xf_rf < 0.05 else "not significant"
print(f"McNemar (XGB fixed-n vs RF): b={b_xf_rf}, c={c_xf_rf}, p={p_xf_rf:.4f}  [{sig_xf}]")
p_lf_rf, b_lf_rf, c_lf_rf, _ = mcnemar_test(mc_true_fixed, mc_pred_lgbm_fixed, mc_pred_rf)
sig_lf = "**significant**" if p_lf_rf < 0.05 else "not significant"
print(f"McNemar (LGBM fixed-n vs RF): b={b_lf_rf}, c={c_lf_rf}, p={p_lf_rf:.4f}  [{sig_lf}]")
""")

# ── SECTION 11: CALIBRATION ────────────────────────────────────────────────────
md("""\
## Section 11 — Probability calibration

**What calibration means:**
A classifier says P(A. baumannii) = 0.80 for a genome. Is it actually *A. baumannii*
80% of the time? If yes: well calibrated. If it is AB only 50% of the time despite
the model saying 0.80: the model is overconfident (typical of tree-based boosted models).

**Why calibration matters for this project:**
- For Q2 (ARG burden risk scoring), the predicted probability is the clinically actionable
  output: "this genome has 80% probability of high ARG burden" drives clinical decisions.
- Overconfident probabilities inflate AUROC but mislead risk assessment.
- Random Forest probabilities are usually well-calibrated (Niculescu-Mizil 2005).
  Boosted classifiers tend to be poorly calibrated — they compress probabilities toward 0 and 1.

**Reliability diagram:**
The x-axis shows the mean predicted probability in 10 bins.
The y-axis shows the actual fraction of correct predictions in each bin.
A perfectly calibrated model falls on the diagonal. Bowing above = underconfident.
Bowing below = overconfident.

**Brier score:** Mean squared error between predicted probability and actual binary outcome.
Lower is better. For reference: a null model predicting 1/6 for all 6 classes has
Brier score = 5/6 × (1/6)² + 1/6 × (5/6)² ≈ 0.139. A perfect model scores 0.

We show calibration for the *one-vs-rest* binary decomposition per class, using the
held-out fold 0 predictions from XGBoost (the cleaner single-fold calibration plot
is easier to read than an aggregated across-fold version).
""")

code("""\
# Use fold 0 for calibration demonstration (same fold as Phase 8 permutation importance)
tr_idx0, te_idx0 = next(cv.split(X, y_q1, groups=groups))

inner_ss0 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_rel0, val_rel0 = next(inner_ss0.split(X[tr_idx0], y_q1[tr_idx0]))
tr_inner0  = tr_idx0[tr_rel0]
val_inner0 = tr_idx0[val_rel0]

# XGBoost fold 0 proba
xgb_cal = XGBClassifier(
    **best_params_xgb,
    n_estimators          = 500,
    early_stopping_rounds = 30,
    colsample_bytree      = 0.8,
    eval_metric           = "mlogloss",
    verbosity             = 0,
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
)
sw0 = compute_sample_weight("balanced", y_q1[tr_inner0])
xgb_cal.fit(X[tr_inner0], y_q1[tr_inner0], sample_weight=sw0,
            eval_set=[(X[val_inner0], y_q1[val_inner0])], verbose=False)
proba_xgb = xgb_cal.predict_proba(X[te_idx0])   # (n_te, 6)

# RF fold 0 proba (for comparison)
from sklearn.ensemble import RandomForestClassifier
rf_cal_params = {k: v for k, v in rf_best.get_params().items()
                 if k in ["n_estimators", "max_depth", "min_samples_leaf",
                          "max_features", "class_weight", "random_state", "n_jobs"]}
rf_cal = RandomForestClassifier(**rf_cal_params)
rf_cal.fit(X[tr_idx0], y_q1[tr_idx0])
proba_rf = rf_cal.predict_proba(X[te_idx0])   # (n_te, 6)

y_te0 = y_q1[te_idx0]

# Plot reliability diagrams: 3 classes (AB worst, SA best, KP middle) for both models
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
classes_to_show = [CLASSES.index(c) for c in ["abaumannii", "kpneumoniae", "saureus"]]

brier_xgb_all, brier_rf_all = [], []

for col, cls_idx in enumerate(classes_to_show):
    y_bin = (y_te0 == cls_idx).astype(int)  # one-vs-rest binary

    for row, (proba, label, color) in enumerate(
        [(proba_xgb, "XGBoost", "#d62728"), (proba_rf, "RF", "#1f77b4")]
    ):
        ax = axes[row][col]
        prob_pos = proba[:, cls_idx]
        frac_pos, mean_pred = calibration_curve(y_bin, prob_pos, n_bins=8, strategy="uniform")
        brier = brier_score_loss(y_bin, prob_pos)
        if row == 0:
            brier_xgb_all.append(brier)
        else:
            brier_rf_all.append(brier)

        ax.plot(mean_pred, frac_pos, "s-", color=color, label=f"{label} (Brier={brier:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Fraction positive")
        ax.set_title(f"{CLASSES[cls_idx]} (1-vs-rest)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle("Reliability diagrams (fold 0): XGBoost vs RF, selected classes", y=1.02)
plt.tight_layout()
fig.savefig(FIG / "q1_calibration_xgb_vs_rf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/gb/q1_calibration_xgb_vs_rf.png")

print(f"\\nMean Brier score (3 classes) — XGBoost: {np.mean(brier_xgb_all):.4f}  "
      f"RF: {np.mean(brier_rf_all):.4f}")
print("Lower = better calibrated (0 = perfect, ~0.139 = null baseline at 1/6)")
""")

# ── SECTION 12: Q2 ARG BURDEN ─────────────────────────────────────────────────
md("""\
## Section 12 — Q2: XGBoost and LightGBM for ARG burden prediction (per species)

**Phase 7–8 Q2 recap:**
LR won 4/6 species; RF won 1 (EF, AUROC 0.900). Conclusion: flexible models overfit
at n~150. The question now: does gradient boosting, with its smoother regularisation via
learning_rate and early stopping, do better than RF on the small-sample species?

**What to look for:**
- EC and KP: LR was dominant (BA 0.752, 0.719). Can XGBoost close the gap?
- EF: RF dominated (BA 0.681). Does boosting match or beat RF here?
- SA and AB: both near chance in Phase 7–8. If boosting improves either, investigate
  leakage — resistance in these species is chromosomal, not defence-linked.
""")

code("""\
results_q2_xgb  = {}
results_q2_lgbm = {}
ba_xgb_dict     = {}  # H4: fold-level XGB BAs per species for BH correction

for sp in sorted(fm["species"].unique()):
    sp_mask  = fm["species"].to_numpy(dtype=str) == sp
    fm_sp    = fm[sp_mask]

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
    print(f"  {sp}: {len(feat_q2_sp)} features after H1 filter (from {len(FEAT_COLS)})")

    cv_sp = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for model_name, results_dict in [("xgb", results_q2_xgb), ("lgbm", results_q2_lgbm)]:
        ba_sp, auc_sp = [], []
        all_yt_sp, all_yp_sp = [], []

        for tr, te in cv_sp.split(X_sp, y_sp, groups=grp_sp):
            if len(set(y_sp[te])) < 2:
                continue

            inner_ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
            tr_rel, val_rel = next(inner_ss.split(X_sp[tr], y_sp[tr]))
            tr_i  = tr[tr_rel]
            val_i = tr[val_rel]
            sw_i  = compute_sample_weight("balanced", y_sp[tr_i])

            if model_name == "xgb":
                m = XGBClassifier(
                    **best_params_xgb,
                    n_estimators=300, early_stopping_rounds=20,
                    colsample_bytree=0.8, eval_metric="logloss",
                    verbosity=0, random_state=RANDOM_STATE, n_jobs=-1,
                )
                m.fit(X_sp[tr_i], y_sp[tr_i], sample_weight=sw_i,
                      eval_set=[(X_sp[val_i], y_sp[val_i])], verbose=False)
            else:
                m = LGBMClassifier(
                    **{k: v for k, v in lgbm_params.items() if k != "n_estimators"},
                    n_estimators=300,
                )
                m.fit(X_sp[tr_i], y_sp[tr_i],
                      eval_set=[(X_sp[val_i], y_sp[val_i])],
                      callbacks=[lgb.early_stopping(20, verbose=False),
                                 lgb.log_evaluation(-1)])

            pred  = m.predict(X_sp[te])
            prob  = m.predict_proba(X_sp[te])[:, 1]
            all_yt_sp.extend(y_sp[te])
            all_yp_sp.extend(pred)
            ba_sp.append(balanced_accuracy_score(y_sp[te], pred))
            auc_sp.append(roc_auc_score(y_sp[te], prob))

        if not ba_sp:
            continue
        if model_name == "xgb":
            ba_xgb_dict[sp] = list(ba_sp)  # H4: store XGB fold BAs for BH correction
        mean_ba_sp, lo_sp, hi_sp = bootstrap_ci(np.array(all_yt_sp), np.array(all_yp_sp))
        results_dict[sp] = {"ba": mean_ba_sp, "lo": lo_sp, "hi": hi_sp,
                            "auroc": np.mean(auc_sp)}

# Print comparison table — load RF results from parquet (C1-corrected)
lr_q2 = {"ecloaceae":0.752,"kpneumoniae":0.719,"paeruginosa":0.645,
          "efaecium":0.512,"saureus":0.470,"abaumannii":0.473}
_rf_df = pd.read_parquet(RES / "q2_rf_results.parquet")
rf_q2  = dict(zip(_rf_df["species"], _rf_df["ba"]))

print(f"\\nQ2 comparison (balanced accuracy):")
print(f"  {'Species':<22} {'LR':>6}  {'RF':>6}  {'XGB':>6}  {'LGBM':>6}  Winner")
print("  " + "-" * 68)
for sp in sorted(results_q2_xgb.keys()):
    lrv = lr_q2.get(sp, float("nan"))
    rfv = rf_q2.get(sp, float("nan"))
    xv  = results_q2_xgb.get(sp, {}).get("ba", float("nan"))
    lv  = results_q2_lgbm.get(sp, {}).get("ba", float("nan"))
    winner = max(["LR", "RF", "XGB", "LGBM"],
                 key=lambda m: {"LR": lrv, "RF": rfv, "XGB": xv, "LGBM": lv}[m])
    print(f"  {sp:<22} {lrv:.3f}  {rfv:.3f}  {xv:.3f}  {lv:.3f}  {winner}")
""")

# ── SECTION 12b: H4 — BH CORRECTION ACROSS Q2 SPECIES ─────────────────────────
md("""\
## Section 12b — H4: BH correction across Q2 species (XGBoost)

**Audit finding H4:** BH correction required across 6 Q2 species tests.
One-sample t-test per species (XGB fold BAs vs BA=0.5 null), then BH correction.
EF, SA, AB may have few usable folds; n_folds reported for transparency.
""")

code("""\
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

null_ba  = 0.5
sp_order = sorted(ba_xgb_dict.keys())
p_raw    = []
for sp in sp_order:
    ba_list = ba_xgb_dict[sp]
    if len(ba_list) >= 2:
        _, pval = ttest_1samp(ba_list, popmean=null_ba, alternative="greater")
        p_raw.append(pval)
    else:
        p_raw.append(float("nan"))

valid_idx   = [i for i, p in enumerate(p_raw) if not (p != p)]
sp_valid    = [sp_order[i] for i in valid_idx]
pvals_valid = [p_raw[i]    for i in valid_idx]

reject_bh, pvals_adj, _, _ = multipletests(pvals_valid, alpha=0.05, method="fdr_bh")
q2_pval_map_xgb = dict(zip(sp_valid, pvals_adj))
q2_praw_map_xgb = dict(zip(sp_valid, pvals_valid))
q2_sig_map_xgb  = dict(zip(sp_valid, reject_bh))

print("H4: Q2 XGB null-baseline significance (one-sample t vs BA=0.5, BH-corrected):")
print(f"  {'Species':<22} {'XGB BA':>7}  {'n_folds':>7}  {'p_raw':>8}  {'p_adj_BH':>10}  {'Sig?':>5}")
print("  " + "-" * 72)
for sp in sp_order:
    ba_v = results_q2_xgb.get(sp, {}).get("ba", float("nan"))
    n_f  = len(ba_xgb_dict.get(sp, []))
    p_r  = q2_praw_map_xgb.get(sp, float("nan"))
    p_a  = q2_pval_map_xgb.get(sp, float("nan"))
    sig  = "YES" if q2_sig_map_xgb.get(sp, False) else "ns"
    print(f"  {sp:<22} {ba_v:.3f}   {n_f:>7}  {p_r:.4f}    {p_a:.4f}      {sig}")
""")

# ── SECTION 13: SAVE RESULTS ───────────────────────────────────────────────────
md("## Section 13 — Save results")

code("""\
# Q1 summary
q1_gb = pd.DataFrame([
    {"model": "XGBoost", "ba_mean": mean_ba_xgb, "ba_lo": lo_xgb, "ba_hi": hi_xgb,
     "f1_mean": mean_f1_xgb, "cv": "GroupedStratifiedKFold_5",
     "early_stopping": True, "median_best_iter": int(np.median(best_iters)),
     **{f"param_{k}": v for k, v in best_params_xgb.items()}},
    {"model": "LightGBM", "ba_mean": mean_ba_lgbm, "ba_lo": lo_lgbm, "ba_hi": hi_lgbm,
     "f1_mean": mean_f1_lgbm, "cv": "GroupedStratifiedKFold_5",
     "early_stopping": True, "median_best_iter": int(np.median(best_iters_lgbm)),
     "param_num_leaves": 63, **{f"param_{k}": v for k, v in lgbm_params.items()
                                if k in ["learning_rate", "max_depth", "subsample"]}},
])
q1_gb.to_parquet(RES / "q1_gb_results.parquet")
print("Saved: results/q1_gb_results.parquet")

# Q2 summary (including H4 BH-corrected p-values for XGB)
q2_rows = []
for sp, vals in results_q2_xgb.items():
    q2_rows.append({"species": sp, "model": "XGBoost",
                    "p_adj_bh": q2_pval_map_xgb.get(sp, float("nan")),
                    "p_raw":    q2_praw_map_xgb.get(sp, float("nan")),
                    "sig_bh":   q2_sig_map_xgb.get(sp, False),
                    **vals})
for sp, vals in results_q2_lgbm.items():
    q2_rows.append({"species": sp, "model": "LightGBM", **vals})
pd.DataFrame(q2_rows).to_parquet(RES / "q2_gb_results.parquet")
print("Saved: results/q2_gb_results.parquet")

# Per-genome prediction arrays for Phase 10 McNemar (if needed)
np.save(RES / "mc_pred_xgb_q1.npy", mc_pred_xgb)
np.save(RES / "mc_pred_lgbm_q1.npy", mc_pred_lgbm)
np.save(RES / "mc_pred_rf_q1.npy",   mc_pred_rf)
np.save(RES / "mc_true_q1.npy",      mc_true_xgb)   # same genome ordering
print("Saved: McNemar prediction arrays (results/mc_pred_*.npy)")

print("\\nAll outputs saved.")
""")

# ── SECTION 14: COMPREHENSION CHECK ────────────────────────────────────────────
md("""\
## Section 14 — Comprehension check

Work through these before the next session. They map directly to what this phase
introduced; if you cannot answer them now, the material hasn't landed yet.

---

**Q1.** Early stopping requires a validation set. In our grouped CV loop, where did that
validation set come from, and why was it a random 20% of the training fold rather than a
phylogenetically-grouped split?

**Q2.** Look at the reliability diagram output from Section 11. Describe what an
overconfident classifier looks like on that plot — which direction does the curve bow,
and what does that mean for how you would use the predicted probability in a clinical
setting?

**Q3.** The McNemar test uses only the *discordant pairs* (b and c). You run the test on
XGBoost vs RF and get b=40, c=35, p=0.67. A colleague says "XGBoost made 40 correct
predictions that RF missed — that's clearly better." What is wrong with this reasoning?

**Q4.** (Grounded in your results.) Look at the Q2 comparison table. If EC shows
LR=0.752 and XGBoost=0.690, what does this tell you about the relationship between
model flexibility and sample size? What would you report in the paper?
""")

# ── WRITE NOTEBOOK ─────────────────────────────────────────────────────────────
nb.cells = cells
out_path = Path("notebooks/07_gradient_boosting.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written: {out_path}  ({len(cells)} cells)")
