"""
Cluster-bootstrap significance for the dominant-phylogroup-excluded cohorts
(KP, EF, PA, SA), using the same method as run_q2_cluster_bootstrap_significance.py
(Task 2 fix) rather than the older fold_bootstrap_ci used in
run_subanalysis_dominant_pg.py, for consistency.

Output: results/supplement_dominant_pg_exclusion_bootstrap_sig.json
"""

import sys
sys.path.insert(0, "src")

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from statsmodels.stats.multitest import multipletests
from evaluation.bootstrap import cluster_bootstrap_ci

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
N_FOLDS      = 5
PREV_THRESH  = 0.05

q1_res = json.load(open("results/q1_367_results.json"))
RF_PARAMS = dict(**q1_res["best_params"], class_weight="balanced",
                  random_state=RANDOM_STATE, n_jobs=-1)

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]


def boot_p_value(boot_scores, null=0.5):
    boot_scores = np.asarray(boot_scores)
    p_below = float(np.mean(boot_scores <= null))
    p_above = float(np.mean(boot_scores > null))
    return float(2 * min(p_below, p_above))


SPECIES = ["kpneumoniae", "efaecium", "paeruginosa", "saureus"]
results = {}
p_boots = {}

for sp in SPECIES:
    sp_all = fm[fm["species"] == sp].copy()
    vc = sp_all["phylogroup"].value_counts()
    top_pg = vc.index[0]
    sub = sp_all[sp_all["phylogroup"] != top_pg].copy()

    q33 = sub["arg_count_unique"].quantile(1 / 3)
    q67 = sub["arg_count_unique"].quantile(2 / 3)
    q2_df = sub[(sub["arg_count_unique"] >= q67) | (sub["arg_count_unique"] <= q33)].copy()
    q2_df["y_bin"] = (q2_df["arg_count_unique"] >= q67).astype(int)

    feat_prev = q2_df[dp_cols].mean()
    sp_feat = feat_prev[feat_prev >= PREV_THRESH].index.tolist()

    X = q2_df[sp_feat].to_numpy(dtype=float)
    y = q2_df["y_bin"].to_numpy()
    groups = q2_df["phylogroup"].to_numpy(dtype=str)
    n_pg = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_pg)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_pred_oof = np.empty(len(q2_df), dtype=int)
    for tr, te in cv.split(X, y, groups=groups):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[tr], y[tr])
        y_pred_oof[te] = rf.predict(X[te])

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

    mean_ba = float(balanced_accuracy_score(y, y_pred_oof))
    results[sp] = {
        "top_phylogroup": top_pg, "n_excluded_cohort": int(len(q2_df)),
        "n_phylogroups": int(n_pg), "mean_ba_excluded": mean_ba,
        "ci95_ba_cluster_boot": [round(lo, 4), round(hi, 4)],
        "p_boot_raw": p_boot,
    }
    print(f"{sp:<14} excluded-cohort BA={mean_ba:.3f}  CI=[{lo:.3f},{hi:.3f}]  p_boot={p_boot:.4f}")

sp_order = sorted(p_boots.keys())
_, p_adj_vec, _, _ = multipletests([p_boots[s] for s in sp_order], method="fdr_bh")
for i, sp in enumerate(sp_order):
    results[sp]["p_boot_adj"] = float(p_adj_vec[i])
    results[sp]["significant"] = bool(p_adj_vec[i] < 0.05)

print("\nBH-adjusted:")
for sp in sp_order:
    r = results[sp]
    print(f"  {sp:<14} p_adj={r['p_boot_adj']:.4f}  significant={r['significant']}")

with open("results/supplement_dominant_pg_exclusion_bootstrap_sig.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/supplement_dominant_pg_exclusion_bootstrap_sig.json")
