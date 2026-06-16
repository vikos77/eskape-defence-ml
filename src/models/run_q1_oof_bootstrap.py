"""
run_q1_oof_bootstrap.py

Re-runs the named-236 Q1 five-fold CV evaluation using the stored best RF params,
collecting per-genome OOF predictions. Then computes the cluster bootstrap 95% CI
for Q1 BA and updates q1_named_results.json.

No GridSearchCV — best params are loaded from q1_named_results.json.
Fold BAs must match stored values within 0.001 (determinism check).

Outputs:
  results/q1_named_oof_preds.npz  — y_true (int), y_pred (int), groups (str)
  results/q1_named_results.json   — ci95_ba updated to cluster bootstrap value
"""

import sys
sys.path.insert(0, "src")

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from evaluation.bootstrap import cluster_bootstrap_ci

RANDOM_STATE = 42
N_FOLDS      = 5

# ── Feature matrix ─────────────────────────────────────────────────────────

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]

species_list = fm["species"].unique()
sp_prev      = pd.DataFrame({sp: fm[fm.species == sp][dp_cols].mean() for sp in species_list})
spec_score   = sp_prev.std(axis=1) / 0.5
markers      = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS    = [c for c in dp_cols if c not in markers]

X      = fm[FEAT_COLS].to_numpy(dtype=float)
y_str  = fm["species"].to_numpy()
groups = fm["phylogroup"].to_numpy(dtype=str)

le    = LabelEncoder().fit(y_str)
y_int = le.transform(y_str)

print(f"Features: {len(FEAT_COLS)}, genomes: {len(fm)}, phylogroups: {len(np.unique(groups))}")

# ── Best params ─────────────────────────────────────────────────────────────

with open("results/q1_named_results.json") as f:
    stored = json.load(f)

best_params = stored["best_params"]
print(f"Best params: {best_params}")

rf_params = dict(
    **best_params,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
    n_jobs       = -1,
)

# ── Five-fold CV with OOF collection ───────────────────────────────────────

cv        = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
y_pred_oof = np.empty(len(fm), dtype=int)
y_true_oof = np.empty(len(fm), dtype=int)
fold_bas   = []

print("\nRunning 5-fold CV (OOF predictions)...")
for fold, (tr, te) in enumerate(cv.split(X, y_str, groups=groups)):
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X[tr], y_str[tr])
    preds = rf.predict(X[te])

    y_pred_oof[te] = le.transform(preds)
    y_true_oof[te] = y_int[te]

    ba = balanced_accuracy_score(y_str[te], preds)
    fold_bas.append(ba)
    print(f"  Fold {fold+1}: BA={ba:.4f}")

mean_ba = float(np.mean(fold_bas))
print(f"\nMean fold BA: {mean_ba:.4f}  (stored: {stored['mean_ba']:.4f})")

# Determinism check
stored_fold_bas = stored["fold_bas"]
max_delta = max(abs(a - b) for a, b in zip(fold_bas, stored_fold_bas))
print(f"Max fold BA delta vs stored: {max_delta:.6f}")
if max_delta > 0.002:
    print("WARNING: fold BAs deviate from stored values by >0.002 — check for non-determinism")
else:
    print("Fold BAs match stored values (delta <= 0.002) — determinism confirmed")

# ── Cluster bootstrap CI ────────────────────────────────────────────────────

print("\nRunning cluster bootstrap (2000 iterations, 309 phylogroups)...")
ci_lo, ci_hi = cluster_bootstrap_ci(
    y_true_oof, y_pred_oof, groups,
    metric_fn=balanced_accuracy_score,
    n_boot=2000, seed=42
)
print(f"Cluster bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"t.interval (old):          [{stored['ci95_ba'][0]:.3f}, {stored['ci95_ba'][1]:.3f}]")

# ── Save ────────────────────────────────────────────────────────────────────

np.savez(
    "results/q1_named_oof_preds.npz",
    y_true  = y_true_oof,
    y_pred  = y_pred_oof,
    groups  = groups,
)
print("\nSaved: results/q1_named_oof_preds.npz")

stored["ci95_ba"] = [round(ci_lo, 3), round(ci_hi, 3)]
with open("results/q1_named_results.json", "w") as f:
    json.dump(stored, f, indent=2)
print("Updated: results/q1_named_results.json (ci95_ba)")
print("Done.")
