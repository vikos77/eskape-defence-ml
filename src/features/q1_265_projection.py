"""
Q1 265-feature projection: same feature set as the 878-genome study, more data.

Runs LR and RF under GroupedStratifiedKFold-5 using only the 265 features from
the original study (q1_filtered_265_features.csv). Saves results alongside the
352-feature results for a clean "same features, more data" comparison.

Usage:
    conda run -n eskape-ml python3 src/features/q1_265_projection.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RES  = ROOT / "results"

RANDOM_STATE = 42
N_SPLITS     = 5

# ── Load data ────────────────────────────────────────────────────────────────
fm = pd.read_parquet(PROC / "feature_matrix_3335.parquet")
cg = pd.read_parquet(PROC / "cv_groups_3460.parquet")
if "phylogroup" not in fm.columns:
    fm = fm.join(cg, how="left")

# ── Load 265-feature list from original study ────────────────────────────────
feat265_ref = pd.read_csv(PROC / "q1_filtered_265_features.csv", index_col=0)
feat265     = feat265_ref.index.tolist()

# Verify all 265 features exist in the 4x matrix
missing = [f for f in feat265 if f not in fm.columns]
if missing:
    raise ValueError(f"{len(missing)} features from the 265-set are missing: {missing[:5]}")

print(f"265-feature set: {len(feat265)} features, all present in 4× matrix.")

# ── Prepare labels and groups ────────────────────────────────────────────────
le = LabelEncoder()
X  = fm[feat265].values.astype(float)
y  = le.fit_transform(fm["species"])
g  = fm["phylogroup"].values

print(f"Dataset: {X.shape[0]} genomes × {X.shape[1]} features")
print(f"Classes: {le.classes_}")
print(f"Phylogroups: {len(np.unique(g))}")
print()

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def run_model(clf, name):
    ba_scores, f1_scores = [], []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups=g)):
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        ba_scores.append(balanced_accuracy_score(y[te], y_pred))
        f1_scores.append(f1_score(y[te], y_pred, average="macro"))
    ba_arr = np.array(ba_scores)
    f1_arr = np.array(f1_scores)
    # Bootstrap CI over fold scores (same method as nb05/nb06)
    rng = np.random.default_rng(RANDOM_STATE)
    boot = rng.choice(ba_arr, size=(2000, len(ba_arr)), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"{name:<25}  BA={ba_arr.mean():.4f} [{ci_lo:.4f}–{ci_hi:.4f}]  "
          f"F1={f1_arr.mean():.4f}  folds={np.round(ba_arr, 4).tolist()}")
    return {
        "model": name, "feature_set": "projection_265",
        "n_features": len(feat265), "cv": "GroupedStratifiedKFold_5",
        "ba_mean": float(ba_arr.mean()), "ba_lo": float(ci_lo), "ba_hi": float(ci_hi),
        "f1_mean": float(f1_arr.mean()), "fold_scores": ba_arr.tolist(),
    }

print("=== Q1 265-feature projection (GroupedStratifiedKFold-5) ===")
print()

results = []

# LR — same hyperparameters as nb05 filtered-set grouped-CV
lr = LogisticRegression(
    max_iter=2000, class_weight="balanced",
    solver="lbfgs", C=0.1, random_state=RANDOM_STATE,
)
results.append(run_model(lr, "LR (265 features)"))

# RF — same hyperparameters as nb06 tuned model
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)
results.append(run_model(rf, "RF (265 features)"))

# ── Save ─────────────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out.to_parquet(RES / "q1_265projection_results.parquet", index=False)
print()
print(f"Saved → results/q1_265projection_results.parquet")

# ── Comparison table ─────────────────────────────────────────────────────────
print()
print("=== Comparison: 265-feature projection vs 352-feature primary ===")
print()
print(f"{'Model':<30} {'Feature set':<18} {'BA':>6}  {'95% CI':>18}")
print("-" * 75)

# Load 352-feature reference values
lr352 = pd.read_parquet(RES / "q1_lr_results_3460.parquet")
lr352_row = lr352[(lr352["feature_set"] == "filtered_352") &
                  (lr352["cv_strategy"] == "grouped_CV")].iloc[0]
rf352 = pd.read_parquet(RES / "q1_rf_results_3460.parquet")
rf352_row = rf352.iloc[0]

ref_878 = [
    ("LR (878-genome original, 265 feat)", 0.8822, 0.8378, 0.9214),
    ("RF (878-genome original, 265 feat)", 0.8742, 0.8520, 0.8950),
]
for label, ba, lo, hi in ref_878:
    print(f"{label:<30} {'265 (orig)':>18} {ba:>6.4f}  [{lo:.4f}–{hi:.4f}]")

print()
for row in results:
    print(f"{row['model']:<30} {'265 (proj)':>18} {row['ba_mean']:>6.4f}  "
          f"[{row['ba_lo']:.4f}–{row['ba_hi']:.4f}]")

print()
print(f"{'LR (4×, 352 features)':<30} {'352 (primary)':>18} "
      f"{lr352_row['balanced_acc']:>6.4f}  "
      f"[{lr352_row['ci_lo']:.4f}–{lr352_row['ci_hi']:.4f}]")
print(f"{'RF (4×, 352 features)':<30} {'352 (primary)':>18} "
      f"{rf352_row['ba_mean']:>6.4f}  "
      f"[{rf352_row['ba_lo']:.4f}–{rf352_row['ba_hi']:.4f}]")
