"""
run_bootstrap_ci.py

Recompute 95% CIs for Q1 BA and Q2 AUROC using the correct bootstrap methods.

Q1 BA  — cluster bootstrap (2000 iterations): resample 309 phylogroups with
          replacement, include all genomes in each drawn group. Consistent with
          the stated effective sample size of 309 phylogroups. Uses stored OOF
          predictions from the McNemar run (mc_pred_rf_q1.npy / mc_true_q1.npy),
          which share the same GroupedStratifiedKFold splits as run_q1_named.py.

Q2 AUROC — fold-level bootstrap (2000 iterations): resample the five per-fold
           AUROC scores with replacement. Correct for the stated Q2 CI estimator
           in methods.md §6.

Replaces the t.interval(df=4) values previously stored in q1_named_results.json
and q2_named_results.json with bootstrap CIs and writes a summary to stdout.
"""

import sys
sys.path.insert(0, "src")

import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from evaluation.bootstrap import cluster_bootstrap_ci

# ── Q1: cluster bootstrap ──────────────────────────────────────────────────

fm      = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
y_true  = np.load("results/mc_true_q1.npy")
y_pred  = np.load("results/mc_pred_rf_q1.npy")
groups  = fm["phylogroup"].values

ci_q1_lo, ci_q1_hi = cluster_bootstrap_ci(
    y_true, y_pred, groups,
    metric_fn=balanced_accuracy_score,
    n_boot=2000, seed=42
)

with open("results/q1_named_results.json") as f:
    q1 = json.load(f)

old_ci_q1 = q1["ci95_ba"]
q1["ci95_ba"] = [round(ci_q1_lo, 3), round(ci_q1_hi, 3)]

with open("results/q1_named_results.json", "w") as f:
    json.dump(q1, f, indent=2)

print("Q1 BA cluster bootstrap CI")
print(f"  old (t.interval df=4): [{old_ci_q1[0]:.3f}, {old_ci_q1[1]:.3f}]")
print(f"  new (cluster bootstrap): [{ci_q1_lo:.3f}, {ci_q1_hi:.3f}]")
print()

# ── Q2: fold-level bootstrap ───────────────────────────────────────────────

def fold_bootstrap_ci(fold_scores, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [
        float(np.mean(rng.choice(fold_scores, size=len(fold_scores), replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


with open("results/q2_named_results.json") as f:
    q2 = json.load(f)

print("Q2 AUROC fold-level bootstrap CIs")
print(f"{'Species':<14}  {'old CI':<20}  {'new CI':<20}  {'fold AUROCs'}")
print("-" * 90)

for sp, res in q2.items():
    fold_aurocs = np.array(res["fold_auroc"])
    lo, hi = fold_bootstrap_ci(fold_aurocs)
    old_lo, old_hi = res["ci95_auroc"]
    res["ci95_auroc"] = [round(lo, 3), round(hi, 3)]
    print(f"{sp:<14}  [{old_lo:.3f}, {old_hi:.3f}]        [{lo:.3f}, {hi:.3f}]        {[round(v,3) for v in fold_aurocs.tolist()]}")

with open("results/q2_named_results.json", "w") as f:
    json.dump(q2, f, indent=2)

print()
print("Updated q1_named_results.json and q2_named_results.json")
print("Done.")
