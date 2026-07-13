"""
Sensitivity analysis: full 367-feature matrix (both DefenseFinder + PADLOC,
PDC/DS-N/All_UG/catch-all all retained), no named-system filter applied.

Companion to run_df_full_sensitivity.py / run_df_only_sensitivity.py, which
covered the DF-only axis (matching the published Acinetobacter paper's tool
choice). This script covers the orthogonal axis: does the named-236 filter
itself (independent of which tool) change Q1/Q2, when both DefenseFinder and
PADLOC are used together as in the actual primary analysis?

Mirrors run_q1_named.py and run_q2_named.py exactly (same RF params, same
StratifiedGroupKFold(5, random_state=42), same spec_score>=0.70 marker filter,
same BH correction) but reads feature_matrix_3335.parquet (367 dp_ features)
instead of feature_matrix_3335_named.parquet (236 dp_ features). RF
hyperparameters are fixed at the named-236 Q1 GridSearch optimum rather than
re-tuned, consistent with how run_df_full_sensitivity.py treated this as a
feature-set sensitivity check, not a re-tuning exercise.

Outputs:
  results/sensitivity_full367_q1.json
  results/sensitivity_full367_q2.json
"""

import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS      = 5
SPEC_THRESH  = 0.70
PREV_THRESH  = 0.05
RF_PARAMS    = {
    "n_estimators"    : 300,
    "max_depth"       : None,
    "min_samples_leaf": 1,
    "max_features"    : "sqrt",
    "class_weight"    : "balanced",
    "random_state"    : RANDOM_STATE,
    "n_jobs"          : -1,
}

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]
print(f"Full (unfiltered) dp_ features: {len(dp_cols)}")

species_list = sorted(fm["species"].unique())

# ── Q1 ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Q1: full 367-feature pool, no named filter")

sp_prev    = pd.DataFrame({sp: fm[fm.species == sp][dp_cols].mean() for sp in species_list})
spec_score = sp_prev.std(axis=1) / 0.5
markers    = spec_score[spec_score >= SPEC_THRESH].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]
print(f"Markers removed (spec>={SPEC_THRESH}): {len(markers)} -> {markers}")
print(f"Q1 FEAT_COLS: {len(FEAT_COLS)}")

X      = fm[FEAT_COLS].to_numpy(dtype=float)
y_q1   = fm["species"].to_numpy()
groups = fm["phylogroup"].to_numpy(dtype=str)
cv     = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_bas, fold_f1s = [], []
per_class_recalls = {sp: [] for sp in species_list}

for tr, te in cv.split(X, y_q1, groups=groups):
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X[tr], y_q1[tr])
    y_pred = rf.predict(X[te])
    fold_bas.append(balanced_accuracy_score(y_q1[te], y_pred))
    fold_f1s.append(f1_score(y_q1[te], y_pred, average="macro", zero_division=0))
    recs = recall_score(y_q1[te], y_pred, labels=species_list, average=None, zero_division=0)
    for sp, r in zip(species_list, recs):
        per_class_recalls[sp].append(r)

fold_bas = np.array(fold_bas)
fold_f1s = np.array(fold_f1s)
mean_ba  = fold_bas.mean()
ci95_ba  = stats.t.interval(0.95, df=N_FOLDS - 1, loc=mean_ba, scale=stats.sem(fold_bas))
mean_f1  = fold_f1s.mean()
ci95_f1  = stats.t.interval(0.95, df=N_FOLDS - 1, loc=mean_f1, scale=stats.sem(fold_f1s))

print(f"\nQ1 RF BA = {mean_ba:.4f} [{ci95_ba[0]:.4f}-{ci95_ba[1]:.4f}]")
print(f"Q1 RF macro-F1 = {mean_f1:.4f} [{ci95_f1[0]:.4f}-{ci95_f1[1]:.4f}]")
print("Per-class recall:")
for sp in sorted(species_list):
    r_arr = np.array(per_class_recalls[sp])
    print(f"  {sp}: {r_arr.mean():.4f} [{r_arr.min():.4f}-{r_arr.max():.4f}]")

q1_results = {
    "n_features_pool"  : len(dp_cols),
    "n_markers_removed": len(markers),
    "markers"          : markers,
    "n_feat_cols"       : len(FEAT_COLS),
    "fold_bas"         : fold_bas.tolist(),
    "mean_ba"          : float(mean_ba),
    "ci95_ba"          : [float(ci95_ba[0]), float(ci95_ba[1])],
    "mean_f1"          : float(mean_f1),
    "ci95_f1"          : [float(ci95_f1[0]), float(ci95_f1[1])],
    "per_class_recall" : {sp: float(np.mean(per_class_recalls[sp])) for sp in species_list},
}

with open("results/sensitivity_full367_q1.json", "w") as f:
    json.dump(q1_results, f, indent=2)
print("Saved: results/sensitivity_full367_q1.json")

# ── Q2 ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Q2: full 367-feature pool, no named filter")

q2_results = {}
p_raws = {}

for sp in species_list:
    sp_df = fm[fm.species == sp].copy()
    q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
    q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

    feat_prev = q2_df[dp_cols].mean()
    sp_feat   = feat_prev[feat_prev >= PREV_THRESH].index.tolist()

    X_sp = q2_df[sp_feat].to_numpy(dtype=float)
    y_sp = q2_df["y_bin"].to_numpy()
    g_sp = q2_df["phylogroup"].to_numpy(dtype=str)

    n_pg     = len(np.unique(g_sp))
    n_splits = min(N_FOLDS, n_pg)
    cv_sp    = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_bas, fold_auroc = [], []
    for tr, te in cv_sp.split(X_sp, y_sp, groups=g_sp):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_sp[tr], y_sp[tr])
        fold_bas.append(balanced_accuracy_score(y_sp[te], rf.predict(X_sp[te])))
        fold_auroc.append(roc_auc_score(y_sp[te], rf.predict_proba(X_sp[te])[:, 1]))

    fold_bas   = np.array(fold_bas)
    fold_auroc = np.array(fold_auroc)
    mean_auroc = fold_auroc.mean()
    ci_auroc   = stats.t.interval(0.95, df=n_splits - 1, loc=mean_auroc, scale=stats.sem(fold_auroc))
    t_stat, p_raw = stats.ttest_1samp(fold_bas, 0.5)

    print(f"  {sp:<15} feat={len(sp_feat):3d}  AUROC={mean_auroc:.3f}  BA={fold_bas.mean():.3f}  p_raw={p_raw:.4f}")

    p_raws[sp] = float(p_raw)
    q2_results[sp] = {
        "n_q2"       : int(len(q2_df)),
        "n_features" : int(len(sp_feat)),
        "mean_ba"    : float(fold_bas.mean()),
        "mean_auroc" : float(mean_auroc),
        "ci95_auroc" : [float(ci_auroc[0]), float(ci_auroc[1])],
        "p_raw"      : float(p_raw),
    }

sp_order = sorted(p_raws.keys())
_, p_adj_v, _, _ = multipletests([p_raws[s] for s in sp_order], method="fdr_bh")
p_adj = {s: float(p_adj_v[i]) for i, s in enumerate(sp_order)}

print("\nQ2 SUMMARY (BH-corrected):")
for sp in sp_order:
    sig = "*" if p_adj[sp] < 0.05 else "ns"
    r = q2_results[sp]
    print(f"  {sp:<15} AUROC={r['mean_auroc']:.3f} [{r['ci95_auroc'][0]:.3f}-{r['ci95_auroc'][1]:.3f}]"
          f"  p_adj={p_adj[sp]:.4f}  {sig}")
    q2_results[sp]["p_adj"] = p_adj[sp]

with open("results/sensitivity_full367_q2.json", "w") as f:
    json.dump(q2_results, f, indent=2)
print("Saved: results/sensitivity_full367_q2.json")
print("Done.")
