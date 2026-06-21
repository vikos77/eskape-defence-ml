"""
Q1: multi-class species classification (RF), full 367-feature defence-system
matrix (every DefenseFinder + PADLOC annotated system, no restriction by
mechanism characterisation).

GridSearchCV tunes hyperparameters from scratch on the 367-feature pool
(StratifiedGroupKFold(5, random_state=42), grouped by phylogroup). A
spec_score>=0.70 taxonomic-marker filter removes features that act as covert
species labels rather than genuine defence signal.

Writes best_params to q1_367_results.json so every downstream script (Q2,
McNemar, C3) reads tuned params dynamically instead of hardcoding them.

Outputs:
  results/models/rf_q1_367.pkl       — best RF model
  results/q1_367_results.json        — fold-level BAs, CI, per-class recall, best_params
  results/q1_367_feat_cols.txt       — FEAT_COLS list
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
N_FOLDS      = 5

# --- Load matrix ---
fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]

# --- Spec_score taxonomic marker filter (threshold = 0.70, registered in PA-2) ---
species_list = fm["species"].unique()
sp_prev = pd.DataFrame({sp: fm[fm.species == sp][dp_cols].mean() for sp in species_list})
spec_score = sp_prev.std(axis=1) / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]

print(f"dp_ features (367 pool): {len(dp_cols)}")
print(f"Markers removed (spec>=0.70): {len(markers)} -> {markers}")
print(f"Q1 FEAT_COLS:   {len(FEAT_COLS)}")

# --- Arrays ---
X      = fm[FEAT_COLS].to_numpy(dtype=float)
y_q1   = fm["species"].to_numpy()
groups = fm["phylogroup"].to_numpy(dtype=str)

cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# --- GridSearchCV ---
param_grid = {
    "n_estimators"    : [100, 300],
    "max_depth"       : [10, 20, None],
    "min_samples_leaf": [1, 3, 5],
    "max_features"    : ["sqrt", 0.3],
}

rf_base = RandomForestClassifier(
    class_weight = "balanced",
    oob_score    = False,
    n_jobs       = 1,
    random_state = RANDOM_STATE,
)

print("\nRunning GridSearchCV...")
grid_search = GridSearchCV(
    estimator  = rf_base,
    param_grid = param_grid,
    cv         = cv,
    scoring    = "balanced_accuracy",
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,
)
grid_search.fit(X, y_q1, groups=groups)
rf_best = grid_search.best_estimator_

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV BA (grid):  {grid_search.best_score_:.4f}")

# --- Fold-level BAs for CI ---
fold_bas = []
fold_f1s = []
per_class_recalls = {sp: [] for sp in species_list}

for fold, (tr, te) in enumerate(cv.split(X, y_q1, groups=groups)):
    rf_fold = RandomForestClassifier(
        **grid_search.best_params_,
        class_weight = "balanced",
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    rf_fold.fit(X[tr], y_q1[tr])
    y_pred = rf_fold.predict(X[te])

    ba = balanced_accuracy_score(y_q1[te], y_pred)
    mf1 = f1_score(y_q1[te], y_pred, average="macro", zero_division=0)
    fold_bas.append(ba)
    fold_f1s.append(mf1)

    recs = recall_score(y_q1[te], y_pred, labels=species_list,
                        average=None, zero_division=0)
    for sp, r in zip(species_list, recs):
        per_class_recalls[sp].append(r)

fold_bas = np.array(fold_bas)
fold_f1s = np.array(fold_f1s)

mean_ba  = fold_bas.mean()
ci95_ba  = stats.t.interval(0.95, df=N_FOLDS-1, loc=mean_ba,
                              scale=stats.sem(fold_bas))
mean_f1  = fold_f1s.mean()
ci95_f1  = stats.t.interval(0.95, df=N_FOLDS-1, loc=mean_f1,
                              scale=stats.sem(fold_f1s))

print(f"\nQ1 RF BA = {mean_ba:.4f} [{ci95_ba[0]:.4f}-{ci95_ba[1]:.4f}]")
print(f"Q1 RF macro-F1 = {mean_f1:.4f} [{ci95_f1[0]:.4f}-{ci95_f1[1]:.4f}]")
print("\nPer-class recall:")
for sp in sorted(species_list):
    r_arr = np.array(per_class_recalls[sp])
    print(f"  {sp}: {r_arr.mean():.4f} [{r_arr.min():.4f}-{r_arr.max():.4f}]")

# --- Sensitivity: spec_score >= 0.50 ---
markers_50 = spec_score[spec_score >= 0.50].index.tolist()
FEAT_50    = [c for c in dp_cols if c not in markers_50]
X_50       = fm[FEAT_50].to_numpy(dtype=float)
print(f"\nSensitivity (spec>=0.50): {len(markers_50)} markers, {len(FEAT_50)} features")

fold_bas_50 = []
for tr, te in cv.split(X_50, y_q1, groups=groups):
    rf_s = RandomForestClassifier(
        **grid_search.best_params_,
        class_weight = "balanced",
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    rf_s.fit(X_50[tr], y_q1[tr])
    fold_bas_50.append(balanced_accuracy_score(y_q1[te], rf_s.predict(X_50[te])))

fold_bas_50 = np.array(fold_bas_50)
mean_ba_50  = fold_bas_50.mean()
ci_50       = stats.t.interval(0.95, df=N_FOLDS-1, loc=mean_ba_50,
                                 scale=stats.sem(fold_bas_50))
print(f"Sensitivity BA = {mean_ba_50:.4f} [{ci_50[0]:.4f}-{ci_50[1]:.4f}]")

# --- Save ---
with open("results/models/rf_q1_367.pkl", "wb") as f:
    pickle.dump(rf_best, f)

with open("results/q1_367_feat_cols.txt", "w") as f:
    f.write("\n".join(FEAT_COLS))

results = {
    "n_features"       : len(FEAT_COLS),
    "n_markers_removed": len(markers),
    "markers"          : markers,
    "best_params"      : grid_search.best_params_,
    "fold_bas"         : fold_bas.tolist(),
    "mean_ba"          : float(mean_ba),
    "ci95_ba"          : [float(ci95_ba[0]), float(ci95_ba[1])],
    "mean_f1"          : float(mean_f1),
    "ci95_f1"          : [float(ci95_f1[0]), float(ci95_f1[1])],
    "per_class_recall" : {sp: np.array(per_class_recalls[sp]).mean().item()
                          for sp in species_list},
    "sensitivity_50"   : {
        "n_markers": len(markers_50),
        "n_features": len(FEAT_50),
        "markers"   : markers_50,
        "mean_ba"   : float(mean_ba_50),
        "ci95_ba"   : [float(ci_50[0]), float(ci_50[1])],
    },
}

with open("results/q1_367_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: results/models/rf_q1_367.pkl")
print("Saved: results/q1_367_results.json")
print("Saved: results/q1_367_feat_cols.txt")
print("\nDone.")
