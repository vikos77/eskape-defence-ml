"""
run_q2_ad_sensitivity.py

Re-checks decision H7 (2026-05-26, pre-4x-expansion) on the current 3,335-genome
named-236 feature matrix: does adding the 41 ad_* (anti-defence, AntiDefenseFinder)
features to the dp_* pool change the Q2 result? H7's original sensitivity run was
on an earlier dataset version (IC2 % in that notebook output, 266/600=44.3%, does
not match the current authoritative 283/600=47.2%) and was never saved to a results
file, so this reruns it cleanly rather than citing the stale number.

Design: identical to run_q2_named.py (same RF hyperparameters, same
StratifiedGroupKFold(5) on phylogroup, same one-sample t-test + BH correction)
except the per-species feature pool is dp_* + ad_* combined, each independently
passed through the same >=5% prevalence filter on the Q2-eligible subset (H7's
original method: filter dp_ and ad_ separately, then concatenate).

Output:
  results/q2_ad_sensitivity.json
"""

import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS      = 5
PREV_THRESH  = 0.05
RF_PARAMS    = {
    "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
    "max_features": "sqrt", "class_weight": "balanced",
    "random_state": RANDOM_STATE, "n_jobs": -1,
}

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]
ad_cols = [c for c in fm.columns if c.startswith("ad_")]
print(f"dp_ columns: {len(dp_cols)}, ad_ columns: {len(ad_cols)}")

species_list = sorted(fm["species"].unique())
results = {"dp_only": {}, "dp_plus_ad": {}}

for condition, pools in [("dp_only", [dp_cols]), ("dp_plus_ad", [dp_cols, ad_cols])]:
    print(f"\n{'='*60}\n{condition}\n{'='*60}")
    p_raws = {}
    for sp in species_list:
        sp_df = fm[fm.species == sp].copy()
        q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
        q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

        sp_feat = []
        for pool in pools:
            prev = q2_df[pool].mean()
            sp_feat += prev[prev >= PREV_THRESH].index.tolist()

        X = q2_df[sp_feat].to_numpy(dtype=float)
        y = q2_df["y_bin"].to_numpy()
        groups = q2_df["phylogroup"].to_numpy(dtype=str)

        n_pg = len(np.unique(groups))
        n_splits = min(N_FOLDS, n_pg)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        fold_bas, fold_auroc = [], []
        for tr, te in cv.split(X, y, groups=groups):
            rf = RandomForestClassifier(**RF_PARAMS)
            rf.fit(X[tr], y[tr])
            fold_bas.append(balanced_accuracy_score(y[te], rf.predict(X[te])))
            fold_auroc.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))

        fold_bas, fold_auroc = np.array(fold_bas), np.array(fold_auroc)
        mean_ba, mean_auroc = fold_bas.mean(), fold_auroc.mean()
        t_stat, p_raw = stats.ttest_1samp(fold_bas, 0.5)
        n_dp = sum(f.startswith("dp_") for f in sp_feat)
        n_ad = sum(f.startswith("ad_") for f in sp_feat)

        print(f"  {sp:<14} n={len(q2_df):>4}  dp_feat={n_dp:>3}  ad_feat={n_ad:>2}  "
              f"BA={mean_ba:.4f}  AUROC={mean_auroc:.4f}  p_raw={p_raw:.4f}")

        p_raws[sp] = float(p_raw)
        results[condition][sp] = {
            "n_dp_feat": n_dp, "n_ad_feat": n_ad, "mean_ba": float(mean_ba),
            "mean_auroc": float(mean_auroc), "p_raw": float(p_raw),
        }

    sp_order = sorted(p_raws.keys())
    _, p_adj_vec, _, _ = multipletests([p_raws[s] for s in sp_order], method="fdr_bh")
    for i, sp in enumerate(sp_order):
        results[condition][sp]["p_adj"] = float(p_adj_vec[i])
    print("  BH-adjusted: " + ", ".join(f"{sp}={results[condition][sp]['p_adj']:.3f}" for sp in sp_order))

with open("results/q2_ad_sensitivity.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/q2_ad_sensitivity.json")
