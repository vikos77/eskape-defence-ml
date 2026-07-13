"""
run_q2_split_sensitivity.py

Sensitivity check on the Q2 ARG-burden label construction: does the choice of
median (2-way, no exclusion), tertile (current, drop middle third), or quartile
(drop middle half) split change the Q2 outcome?

Same RF hyperparameters, same per-species dp_ prevalence filter (>=5% within the
Q2-eligible subset), same StratifiedGroupKFold(5) on phylogroup, same one-sample
t-test of fold BAs vs 0.5, for every species, repeated three times with only the
label-construction step changed. All three recomputed from arg_count_unique
directly (not from the precomputed arg_burden_tertile column) so the three
conditions are built identically except for the cutoffs.

Output:
  results/q2_split_sensitivity.json
"""

import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS      = 5
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

SPLITS = {
    # q: number of pd.qcut bins. low bin -> 0, high bin -> 1, middle bin(s) excluded.
    # Matches the original label-construction method exactly (NB01: pd.qcut(q=3,
    # duplicates="drop")) so "tertile" here must reproduce the published Q2 n's.
    "median":   2,
    "tertile":  3,
    "quartile": 4,
}

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]

species_list = sorted(fm["species"].unique())
all_results = {split: {} for split in SPLITS}

for split_name, q in SPLITS.items():
    print(f"\n{'#'*60}\nSPLIT TYPE: {split_name} (pd.qcut q={q})\n{'#'*60}")
    p_raws = {}

    for sp in species_list:
        sp_df = fm[fm.species == sp].copy()
        arg = sp_df["arg_count_unique"]

        try:
            bins = pd.qcut(arg, q=q, labels=False, duplicates="drop")
        except ValueError:
            # Same pre-specified floor-effect fallback as NB01: binary split at median.
            med = arg.median()
            bins = pd.Series(np.where(arg < med, 0, np.where(arg > med, q - 1, -1)),
                              index=arg.index)

        n_bins_actual = bins.max() + 1  # duplicates="drop" can yield fewer bins than requested
        low_bin, high_bin = 0, n_bins_actual - 1
        mask = (bins == low_bin) | (bins == high_bin)
        q2_df = sp_df[mask].copy()
        q2_df["y_bin"] = (bins[mask] == high_bin).astype(int).values

        # per-species prevalence filter on this Q2-eligible subset
        feat_prev = q2_df[dp_cols].mean()
        sp_feat = feat_prev[feat_prev >= PREV_THRESH].index.tolist()

        X = q2_df[sp_feat].to_numpy(dtype=float)
        y = q2_df["y_bin"].to_numpy()
        groups = q2_df["phylogroup"].to_numpy(dtype=str)

        n_pg = len(np.unique(groups))
        n_splits = min(N_FOLDS, n_pg)
        if n_splits < 2:
            print(f"  {sp}: WARNING only {n_pg} phylogroups, skipping")
            continue

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_bas, fold_auroc = [], []
        for tr, te in cv.split(X, y, groups=groups):
            rf = RandomForestClassifier(**RF_PARAMS)
            rf.fit(X[tr], y[tr])
            y_pred = rf.predict(X[te])
            y_prob = rf.predict_proba(X[te])[:, 1]
            fold_bas.append(balanced_accuracy_score(y[te], y_pred))
            fold_auroc.append(roc_auc_score(y[te], y_prob))

        fold_bas = np.array(fold_bas)
        fold_auroc = np.array(fold_auroc)
        mean_ba = fold_bas.mean()
        mean_auroc = fold_auroc.mean()
        ci_auroc = stats.t.interval(0.95, df=n_splits - 1, loc=mean_auroc, scale=stats.sem(fold_auroc))
        t_stat, p_raw = stats.ttest_1samp(fold_bas, 0.5)

        n_high = int(y.sum())
        n_low = int(len(y) - n_high)

        print(f"  {sp:<14} n={len(q2_df):>4} (hi={n_high},lo={n_low})  n_pg={n_pg:>3}  "
              f"feat={len(sp_feat):>3}  BA={mean_ba:.4f} (sd={fold_bas.std():.4f})  "
              f"AUROC={mean_auroc:.4f} [{ci_auroc[0]:.3f}-{ci_auroc[1]:.3f}]  "
              f"t={t_stat:+.2f} p_raw={p_raw:.4f}")

        p_raws[sp] = float(p_raw)
        all_results[split_name][sp] = {
            "n_total": int(len(q2_df)), "n_high": n_high, "n_low": n_low,
            "n_phylogroups": int(n_pg), "n_features": int(len(sp_feat)),
            "fold_bas": fold_bas.tolist(), "fold_auroc": fold_auroc.tolist(),
            "mean_ba": float(mean_ba), "ba_sd": float(fold_bas.std()),
            "mean_auroc": float(mean_auroc), "ci95_auroc": [float(ci_auroc[0]), float(ci_auroc[1])],
            "t_stat": float(t_stat), "p_raw": float(p_raw),
        }

    from statsmodels.stats.multitest import multipletests
    sp_order = sorted(p_raws.keys())
    _, p_adj_vec, _, _ = multipletests([p_raws[s] for s in sp_order], method="fdr_bh")
    for i, sp in enumerate(sp_order):
        all_results[split_name][sp]["p_adj"] = float(p_adj_vec[i])
    print(f"\n  BH-adjusted: " + ", ".join(f"{sp}={all_results[split_name][sp]['p_adj']:.3f}" for sp in sp_order))

with open("results/q2_split_sensitivity.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved: results/q2_split_sensitivity.json")
