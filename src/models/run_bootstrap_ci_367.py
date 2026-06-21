"""
run_bootstrap_ci_367.py

Bootstrap 95% CIs for Q1 BA and Q2 AUROC on the full 367-feature matrix.
Regenerates per-genome OOF predictions (5-fold CV with best_params from
results/q1_367_results.json, determinism-checked against the stored
fold_bas) and computes both CIs in one pass.

Q1 BA  — cluster bootstrap (2000 iterations): resample phylogroups with
          replacement, include all genomes in each drawn group.
Q2 AUROC — fold-level bootstrap (2000 iterations): resample the per-species
           fold AUROC scores (already saved in results/q2_367_results.json
           by run_q2_367.py) with replacement.

Updates results/q1_367_results.json (ci95_ba) and results/q2_367_results.json
(ci95_auroc per species) in place with bootstrap CIs, replacing the
t.interval(df=4)/t.interval(df=n_splits-1) values computed at generation time.

Outputs:
  results/q1_367_oof_preds.npz   — y_true, y_pred, groups (for audit)
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

# ── Q1: regenerate OOF predictions, then cluster bootstrap ──────────────────

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
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

with open("results/q1_367_results.json") as f:
    stored = json.load(f)
best_params = stored["best_params"]
print(f"Best params: {best_params}")

rf_params = dict(**best_params, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)

cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
y_pred_oof = np.empty(len(fm), dtype=int)
y_true_oof = np.empty(len(fm), dtype=int)
fold_bas = []

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

stored_fold_bas = stored["fold_bas"]
max_delta = max(abs(a - b) for a, b in zip(fold_bas, stored_fold_bas))
print(f"Max fold BA delta vs stored: {max_delta:.6f}")
if max_delta > 0.002:
    print("WARNING: fold BAs deviate from stored values by >0.002 -- check for non-determinism")
else:
    print("Fold BAs match stored values (delta <= 0.002) -- determinism confirmed")

print("\nRunning cluster bootstrap (2000 iterations)...")
ci_q1_lo, ci_q1_hi = cluster_bootstrap_ci(
    y_true_oof, y_pred_oof, groups,
    metric_fn=balanced_accuracy_score, n_boot=2000, seed=42,
)
print(f"Q1 cluster bootstrap 95% CI: [{ci_q1_lo:.3f}, {ci_q1_hi:.3f}]")
print(f"t.interval (old):             [{stored['ci95_ba'][0]:.3f}, {stored['ci95_ba'][1]:.3f}]")

np.savez("results/q1_367_oof_preds.npz", y_true=y_true_oof, y_pred=y_pred_oof, groups=groups)
print("Saved: results/q1_367_oof_preds.npz")

stored["ci95_ba"] = [round(ci_q1_lo, 3), round(ci_q1_hi, 3)]
with open("results/q1_367_results.json", "w") as f:
    json.dump(stored, f, indent=2)
print("Updated: results/q1_367_results.json (ci95_ba)")

# ── Q2: fold-level bootstrap ──────────────────────────────────────────────


def fold_bootstrap_ci(fold_scores, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [
        float(np.mean(rng.choice(fold_scores, size=len(fold_scores), replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


with open("results/q2_367_results.json") as f:
    q2 = json.load(f)

print("\nQ2-367 AUROC fold-level bootstrap CIs")
print(f"{'Species':<14}  {'old CI':<20}  {'new CI':<20}")
print("-" * 70)
for sp, res in q2.items():
    fold_aurocs = np.array(res["fold_auroc"])
    lo, hi = fold_bootstrap_ci(fold_aurocs)
    old_lo, old_hi = res["ci95_auroc"]
    res["ci95_auroc"] = [round(lo, 3), round(hi, 3)]
    print(f"{sp:<14}  [{old_lo:.3f}, {old_hi:.3f}]        [{lo:.3f}, {hi:.3f}]")

with open("results/q2_367_results.json", "w") as f:
    json.dump(q2, f, indent=2)

print("\nUpdated results/q1_367_results.json and results/q2_367_results.json")
print("Done.")
