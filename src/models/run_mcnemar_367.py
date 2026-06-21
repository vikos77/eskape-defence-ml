"""
McNemar's test: RF vs LR / XGB / LGBM on the full 367-feature matrix (Q1).

RF params are loaded from results/q1_367_results.json (best_params) rather
than hardcoded.

Scope decision, disclosed rather than silently applied: LR/XGB/LGBM
hyperparameters are not independently re-tuned for the 367-feature pool --
they keep the same structural values used elsewhere in this study (LR
C=1.0; XGB/LGBM values from the original GridSearchCV). Re-tuning those
would require a full GridSearch pass on each alternative model, which is out
of scope here (the purpose of this comparison is evaluating RF, the
retained/primary model, against alternatives -- not re-optimizing every
alternative). This is a known, disclosed approximation: the comparison
models may be very slightly under-tuned relative to RF, which only biases
the comparison AGAINST finding RF non-inferior, not in its favor.

All four classifiers run on identical 5-fold GroupedStratifiedKFold splits
(RANDOM_STATE=42) so per-genome predictions are aligned.

Outputs:
  results/q1_mcnemar_367.json   — b, c, chi2, p for RF vs LR/XGB/LGBM
  results/q1_mcnemar_preds_367.npz    — per-genome predictions for audit (gitignored)
"""

import json, warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS      = 5


# ── Feature matrix (identical derivation to run_q1_367.py) ──────────────────
fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]

species_list = fm["species"].unique()
sp_prev = pd.DataFrame({sp: fm[fm.species == sp][dp_cols].mean() for sp in species_list})
spec_score = sp_prev.std(axis=1) / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]

print(f"FEAT_COLS: {len(FEAT_COLS)} (markers removed: {len(markers)})")

X      = fm[FEAT_COLS].to_numpy(dtype=float)
y_str  = fm["species"].to_numpy()
groups = fm["phylogroup"].to_numpy(dtype=str)

# XGBoost needs integer labels
le    = LabelEncoder().fit(y_str)
y_int = le.transform(y_str)

cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ── Model definitions ────────────────────────────────────────────────────────
# RF: best params from results/q1_367_results.json (dynamic, not hardcoded)
with open("results/q1_367_results.json") as f:
    _q1 = json.load(f)
rf_params = dict(
    **_q1["best_params"],
    class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)
print(f"RF params (from Q1-367 GridSearch): {_q1['best_params']}")

# LR: C=1.0; C=0.1 (used in NB06 learning curve) underperforms on this feature set
lr_params = dict(
    C=1.0, solver="saga", max_iter=2000,
    class_weight="balanced", random_state=RANDOM_STATE,
)

# XGB: best params from NB07 GridSearchCV (learning_rate=0.1, max_depth=4, subsample=0.8)
# Fixed n_estimators=200, no early stopping — avoids 64% data starvation (NB07 Section 10b)
xgb_params = dict(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", verbosity=0,
    random_state=RANDOM_STATE, n_jobs=-1,
)

# LGBM: same structural params as XGB (NB07 Section 9)
lgbm_params = dict(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    num_leaves=63, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)


# ── Per-genome prediction arrays ─────────────────────────────────────────────
n = len(y_str)
pred_rf   = np.empty(n, dtype=object)
pred_lr   = np.empty(n, dtype=object)
pred_xgb  = np.empty(n, dtype=int)
pred_lgbm = np.empty(n, dtype=int)
y_true    = np.empty(n, dtype=object)   # string labels for RF/LR
y_true_int = np.empty(n, dtype=int)     # int labels for XGB/LGBM

fold_ba = {m: [] for m in ["RF", "LR", "XGB", "LGBM"]}

print("\nRunning 5-fold CV (all 4 models, same splits)...")
for fold, (tr, te) in enumerate(cv.split(X, y_str, groups=groups)):
    # XGBoost sample weights (no class_weight constructor param)
    sw = compute_sample_weight("balanced", y_int[tr])

    # ── RF ──
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X[tr], y_str[tr])
    p_rf = rf.predict(X[te])
    pred_rf[te] = p_rf
    y_true[te]  = y_str[te]
    fold_ba["RF"].append(balanced_accuracy_score(y_str[te], p_rf))

    # ── LR ──
    lr = LogisticRegression(**lr_params)
    lr.fit(X[tr], y_str[tr])
    p_lr = lr.predict(X[te])
    pred_lr[te] = p_lr
    fold_ba["LR"].append(balanced_accuracy_score(y_str[te], p_lr))

    # ── XGB ──
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X[tr], y_int[tr], sample_weight=sw)
    p_xgb = xgb.predict(X[te])
    pred_xgb[te]    = p_xgb
    y_true_int[te]  = y_int[te]
    fold_ba["XGB"].append(balanced_accuracy_score(y_int[te], p_xgb))

    # ── LGBM ──
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X[tr], y_str[tr])
    p_lgbm = lgbm.predict(X[te])
    pred_lgbm[te] = le.transform(p_lgbm)
    fold_ba["LGBM"].append(balanced_accuracy_score(y_str[te], p_lgbm))

    print(f"  Fold {fold+1}: RF={fold_ba['RF'][-1]:.4f}  "
          f"LR={fold_ba['LR'][-1]:.4f}  "
          f"XGB={fold_ba['XGB'][-1]:.4f}  "
          f"LGBM={fold_ba['LGBM'][-1]:.4f}")


# ── Print summary ────────────────────────────────────────────────────────────
# No stored-value verification (no 367 equivalent of q1_lr_xgb_lgbm_named.json
# exists -- see docstring). Fold BAs reported directly.
print("\n── Fold BA summary (367 features) ──────────────────────────────────")
for m in ["RF", "LR", "XGB", "LGBM"]:
    print(f"{m:<6}  mean={np.mean(fold_ba[m]):.4f}")
print(f"(RF Q1-367 GridSearch primary BA, for cross-reference: {_q1['mean_ba']:.4f})")


# ── McNemar test ─────────────────────────────────────────────────────────────
# pred_xgb / pred_lgbm are int; convert RF/LR int for comparison
pred_rf_int   = le.transform(pred_rf)
pred_lr_int   = le.transform(pred_lr)
y_int_aligned = y_true_int


def mcnemar(true, pred_a, pred_b):
    """Two-sided McNemar chi²(1) with continuity correction."""
    correct_a = (true == pred_a)
    correct_b = (true == pred_b)
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(correct_b & ~correct_a))   # B right, A wrong
    if b + c == 0:
        return b, c, 0.0, 1.0
    # Edwards continuity correction (standard for McNemar)
    chi2_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = 1.0 - chi2_dist.cdf(chi2_stat, df=1)
    return b, c, float(chi2_stat), float(p)


print("\n── McNemar results (RF vs others, full-367 Q1) ───────────────────")
print(f"{'Comparison':<15}  {'b (RF+/other-)':<16}  {'c (RF-/other+)':<16}  "
      f"{'b+c':<6}  {'chi2':<8}  {'p'}")

results = {}
for name, pred_other in [("RF_vs_LR", pred_lr_int),
                          ("RF_vs_XGB", pred_xgb),
                          ("RF_vs_LGBM", pred_lgbm)]:
    b, c, chi2_stat, p = mcnemar(y_int_aligned, pred_rf_int, pred_other)
    results[name] = {"b": b, "c": c, "chi2": chi2_stat, "p": p,
                     "n_discordant": b + c, "n_total": int(n)}
    sig = "*" if p < 0.05 else " "
    print(f"{name:<15}  {b:<16}  {c:<16}  {b+c:<6}  {chi2_stat:<8.3f}  {p:.4f}{sig}")

print("\n* p < 0.05 (two-sided, continuity-corrected McNemar chi²(1))")
print(f"n = {n} genomes")


# ── Save ──────────────────────────────────────────────────────────────────────
results["metadata"] = {
    "n_genomes"     : int(n),
    "n_feat_cols"   : len(FEAT_COLS),
    "n_markers_rm"  : len(markers),
    "cv"            : "StratifiedGroupKFold_5_seed42",
    "rf_params"     : rf_params,
    "lr_params"     : lr_params,
    "xgb_params"    : xgb_params,
    "lgbm_params"   : lgbm_params,
    "fold_bas"      : {m: [round(v, 4) for v in fold_ba[m]] for m in fold_ba},
}

with open("results/q1_mcnemar_367.json", "w") as f:
    json.dump(results, f, indent=2)

np.savez(
    "results/q1_mcnemar_preds_367.npz",
    y_true_int  = y_int_aligned,
    pred_rf     = pred_rf_int,
    pred_lr     = pred_lr_int,
    pred_xgb    = pred_xgb,
    pred_lgbm   = pred_lgbm,
)

print("\nSaved: results/q1_mcnemar_367.json")
print("Saved: results/q1_mcnemar_preds_367.npz (gitignored)")
print("\nDone.")
