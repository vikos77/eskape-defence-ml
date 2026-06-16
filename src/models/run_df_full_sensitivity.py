"""
Sensitivity analysis: Full DefenseFinder feature matrix, no unnamed filter.

Uses ALL DF-detectable features (dp_df_* + dp_* shared) from the original
367-feature matrix, applying only the >=5 genome prevalence filter (already
baked into matrix construction) and the spec_score >=0.70 taxonomic marker
filter for Q1.

This approximates what the published Muthuraman et al. (JAM 2026) analysis
used for DefenseFinder outputs before the uncharacterised-annotation purge.

Outputs:
  results/sensitivity_df_full_q1.json
  results/sensitivity_df_full_q2.json
"""

import json, re, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

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

# Use original matrix (367 dp_ features, includes DS-N / UG / catch-alls)
fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")

# Full DF pool: DF-specific (all, no unnamed filter) + shared; exclude PADLOC-specific
all_dp = [c for c in fm.columns if c.startswith("dp_")]
df_full_pool = [c for c in all_dp if not c.startswith("dp_padloc_")]

print(f"Original dp_ features:        {len(all_dp)}")
print(f"Full DF pool (excl PADLOC):   {len(df_full_pool)}")
print(f"  dp_df_* (DF-specific, all): {sum(1 for c in df_full_pool if c.startswith('dp_df_'))}")
print(f"  dp_* shared:                {sum(1 for c in df_full_pool if not c.startswith('dp_df_'))}")

# DS-N and UG still present in this pool
ds_in_pool = [c for c in df_full_pool if re.search(r'dp_df_DS-', c)]
ug_in_pool = [c for c in df_full_pool if re.search(r'dp_df_UG\d', c)]
print(f"\nDS-N features retained: {len(ds_in_pool)}")
print(f"UG features retained:   {len(ug_in_pool)}")

# ── Q1 ───────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Q1: Full DF (no unnamed filter)")

species_list = sorted(fm["species"].unique())
sp_prev   = pd.DataFrame({sp: fm[fm.species==sp][df_full_pool].mean()
                          for sp in species_list})
spec_score = sp_prev.std(axis=1) / 0.5
markers    = spec_score[spec_score >= SPEC_THRESH].index.tolist()
FEAT_COLS  = [c for c in df_full_pool if c not in markers]

print(f"Taxonomic markers removed (spec>={SPEC_THRESH}): {len(markers)}")
for m in markers:
    print(f"  {m}: spec={spec_score[m]:.3f}")
print(f"Q1 FEAT_COLS: {len(FEAT_COLS)}")

X      = fm[FEAT_COLS].to_numpy(dtype=float)
y_q1   = fm["species"].to_numpy()
groups = fm["phylogroup"].to_numpy(dtype=str)
cv     = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)

fold_bas_q1 = []
for tr, te in cv.split(X, y_q1, groups=groups):
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X[tr], y_q1[tr])
    fold_bas_q1.append(balanced_accuracy_score(y_q1[te], rf.predict(X[te])))

fold_bas_q1 = np.array(fold_bas_q1)
mean_ba_q1  = fold_bas_q1.mean()
ci_q1       = stats.t.interval(0.95, df=N_FOLDS-1, loc=mean_ba_q1,
                                 scale=stats.sem(fold_bas_q1))
print(f"\nQ1 RF BA = {mean_ba_q1:.4f} [{ci_q1[0]:.4f}-{ci_q1[1]:.4f}]")
print(f"Fold BAs: {fold_bas_q1.round(4).tolist()}")

q1_results = {
    "n_features_pool"  : len(df_full_pool),
    "n_markers_removed": len(markers),
    "markers"          : markers,
    "n_feat_cols"      : len(FEAT_COLS),
    "ds_n_count"       : len(ds_in_pool),
    "ug_count"         : len(ug_in_pool),
    "fold_bas"         : fold_bas_q1.tolist(),
    "mean_ba"          : float(mean_ba_q1),
    "ci95_ba"          : [float(ci_q1[0]), float(ci_q1[1])],
}

# ── Q2 ───────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Q2: Full DF (no unnamed filter)")

q2_results = {}
p_raws     = {}

for sp in species_list:
    sp_df  = fm[fm.species == sp].copy()
    q2_df  = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
    q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

    feat_prev = q2_df[df_full_pool].mean()
    sp_feat   = feat_prev[feat_prev >= PREV_THRESH].index.tolist()

    X_sp  = q2_df[sp_feat].to_numpy(dtype=float)
    y_sp  = q2_df["y_bin"].to_numpy()
    g_sp  = q2_df["phylogroup"].to_numpy(dtype=str)

    n_pg     = len(np.unique(g_sp))
    n_splits = min(N_FOLDS, n_pg)
    cv_sp    = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=RANDOM_STATE)

    fold_bas, fold_auroc = [], []
    for tr, te in cv_sp.split(X_sp, y_sp, groups=g_sp):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_sp[tr], y_sp[tr])
        fold_bas.append(balanced_accuracy_score(y_sp[te], rf.predict(X_sp[te])))
        fold_auroc.append(roc_auc_score(y_sp[te],
                                         rf.predict_proba(X_sp[te])[:, 1]))

    fold_bas   = np.array(fold_bas)
    fold_auroc = np.array(fold_auroc)
    mean_auroc = fold_auroc.mean()
    ci_auroc   = stats.t.interval(0.95, df=n_splits-1, loc=mean_auroc,
                                   scale=stats.sem(fold_auroc))
    _, p_raw   = stats.ttest_1samp(fold_bas, 0.5)

    print(f"  {sp:<15} feat={len(sp_feat):3d}  AUROC={mean_auroc:.3f}  "
          f"BA={fold_bas.mean():.3f}  p_raw={p_raw:.4f}")

    p_raws[sp] = float(p_raw)
    q2_results[sp] = {
        "n_q2"       : int(len(q2_df)),
        "n_features" : int(len(sp_feat)),
        "mean_ba"    : float(fold_bas.mean()),
        "mean_auroc" : float(mean_auroc),
        "ci95_auroc" : [float(ci_auroc[0]), float(ci_auroc[1])],
        "p_raw"      : float(p_raw),
    }

sp_order  = sorted(p_raws.keys())
_, p_adj_v, _, _ = multipletests([p_raws[s] for s in sp_order], method="fdr_bh")
p_adj = {s: float(p_adj_v[i]) for i, s in enumerate(sp_order)}

print(f"\nQ2 SUMMARY (BH-corrected):")
for sp in sp_order:
    sig = "★" if p_adj[sp] < 0.05 else "ns"
    r = q2_results[sp]
    print(f"  {sp:<15} AUROC={r['mean_auroc']:.3f} [{r['ci95_auroc'][0]:.3f}-{r['ci95_auroc'][1]:.3f}]"
          f"  p_adj={p_adj[sp]:.4f}  {sig}")
    q2_results[sp]["p_adj"] = p_adj[sp]

with open("results/sensitivity_df_full_q1.json", "w") as f:
    json.dump(q1_results, f, indent=2)
with open("results/sensitivity_df_full_q2.json", "w") as f:
    json.dump(q2_results, f, indent=2)

print("\nSaved: results/sensitivity_df_full_q1.json")
print("Saved: results/sensitivity_df_full_q2.json")
print("Done.")
