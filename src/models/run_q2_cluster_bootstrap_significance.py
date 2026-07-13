"""
Q2 primary significance test, corrected for fold non-independence.

The original Q2 significance call (run_q2_367.py) used a one-sample t-test
on 5 CV-fold balanced-accuracy scores against null=0.5. This treats the 5
folds as i.i.d. draws, which they are not -- folds share overlapping
training data, violating the independence assumption (Dietterich 1998;
Bengio & Grandvalet 2004 show no unbiased variance estimator exists for
k-fold CV under this dependence).

Fix: regenerate per-genome out-of-fold (OOF) predictions per species (same
CV split, same RF params, as run_q2_367.py -- determinism-checked against
stored fold_bas), then use a phylogroup-cluster bootstrap (resample
phylogroups with replacement, all genomes in each drawn group) to build the
sampling distribution of BA directly from genome-level OOF predictions. All
six species have >=15 Q2-subset phylogroups (29-79), so cluster_bootstrap_ci
applies uniformly -- no genome-level fallback needed.

Significance: empirical two-sided bootstrap p-value (2 * min of the two
tail proportions relative to null=0.5), BH-corrected across the 6 species --
same correction framework as the original t-test, now applied to a
cluster-correct p-value instead of one that assumes fold independence.

The original t-test p-value/p_adj is retained in the output as a secondary
sensitivity comparison, not deleted.

Output: results/q2_bootstrap_significance.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, "src")
from evaluation.bootstrap import cluster_bootstrap_ci

RANDOM_STATE = 42
N_FOLDS      = 5
PREV_THRESH  = 0.05

q1_res    = json.load(open("results/q1_367_results.json"))
RF_PARAMS = dict(**q1_res["best_params"], class_weight="balanced",
                  random_state=RANDOM_STATE, n_jobs=-1)

q2_stored = json.load(open("results/q2_367_results.json"))
fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]


def boot_p_value(boot_scores, null=0.5):
    boot_scores = np.asarray(boot_scores)
    p_below = float(np.mean(boot_scores <= null))
    p_above = float(np.mean(boot_scores > null))
    return float(2 * min(p_below, p_above))


results = {}
p_boots = {}

for sp in sorted(fm["species"].unique()):
    sp_df = fm[fm.species == sp].copy()
    q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
    q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

    feat_prev = q2_df[dp_cols].mean()
    sp_feat = feat_prev[feat_prev >= PREV_THRESH].index.tolist()

    X = q2_df[sp_feat].to_numpy(dtype=float)
    y = q2_df["y_bin"].to_numpy()
    groups = q2_df["phylogroup"].to_numpy(dtype=str)

    n_pg = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_pg)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_pred_oof = np.empty(len(q2_df), dtype=int)
    fold_bas = []
    for tr, te in cv.split(X, y, groups=groups):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[tr], y[tr])
        preds = rf.predict(X[te])
        y_pred_oof[te] = preds
        fold_bas.append(balanced_accuracy_score(y[te], preds))

    mean_ba_check = float(np.mean(fold_bas))
    stored_mean_ba = q2_stored[sp]["mean_ba"]
    delta = abs(mean_ba_check - stored_mean_ba)
    print(f"{sp}: regenerated mean BA={mean_ba_check:.4f}  stored={stored_mean_ba:.4f}  "
          f"delta={delta:.4f}  n_pg={n_pg}  {'OK' if delta <= 0.002 else 'WARNING: mismatch'}")

    lo, hi = cluster_bootstrap_ci(y, y_pred_oof, groups,
                                    metric_fn=balanced_accuracy_score, n_boot=2000, seed=42)

    rng = np.random.RandomState(42)
    unique_groups = np.unique(groups)
    boot_scores = []
    for _ in range(2000):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        boot_scores.append(balanced_accuracy_score(y[idx], y_pred_oof[idx]))

    p_boot = boot_p_value(boot_scores, null=0.5)
    p_boots[sp] = p_boot

    results[sp] = {
        "n_q2": int(len(q2_df)), "n_phylogroups_q2": int(n_pg),
        "mean_ba_oof": mean_ba_check, "mean_ba_stored_check": stored_mean_ba,
        "ba_delta_vs_stored": delta,
        "ci95_ba_cluster_boot": [round(lo, 4), round(hi, 4)],
        "p_boot_raw": p_boot,
        "p_ttest_original": q2_stored[sp]["p_raw"],
        "p_ttest_adj_original": q2_stored[sp]["p_adj"],
    }
    print(f"  cluster-bootstrap 95% CI BA: [{lo:.4f}, {hi:.4f}]  p_boot={p_boot:.4f}")

sp_order = sorted(p_boots.keys())
p_raw_vec = [p_boots[sp] for sp in sp_order]
_, p_adj_vec, _, _ = multipletests(p_raw_vec, method="fdr_bh")
for i, sp in enumerate(sp_order):
    results[sp]["p_boot_adj"] = float(p_adj_vec[i])
    results[sp]["significant_bootstrap"] = bool(p_adj_vec[i] < 0.05)

print(f"\n{'='*70}\nQ2 SIGNIFICANCE: bootstrap (primary) vs t-test (original) comparison")
print(f"{'Species':<14} {'CI excl 0.5':<12} {'p_boot_adj':<12} {'sig (boot)':<11} "
      f"{'p_ttest_adj':<12} {'sig (t-test)':<12} {'AGREE?'}")
for sp in sp_order:
    r = results[sp]
    ci_excludes = r["ci95_ba_cluster_boot"][0] > 0.5
    sig_boot = r["significant_bootstrap"]
    sig_ttest = r["p_ttest_adj_original"] < 0.05
    agree = "yes" if sig_boot == sig_ttest else "**DISAGREE**"
    print(f"{sp:<14} {str(ci_excludes):<12} {r['p_boot_adj']:<12.4f} {str(sig_boot):<11} "
          f"{r['p_ttest_adj_original']:<12.4f} {str(sig_ttest):<12} {agree}")

with open("results/q2_bootstrap_significance.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/q2_bootstrap_significance.json")
