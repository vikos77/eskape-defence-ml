"""
Q2 sensitivity: does count resolution (dc_) beat presence/absence (dp_) at
separating high/low ARG burden tertiles?

Mirrors run_q2_named.py exactly (same RF params, same CV, same per-species
>=5% filter, same significance test) with one substitution: features are
encoded as dc_ (system-instance copy number, max(DefenseFinder, PADLOC) per
system -- already deduplicated at the instance level, not a raw gene-hit
count and not summed across tools) instead of dp_ (binary presence).

Feature SET is identical to the dp_-based named-236 Q2 result: the >=5%
prevalence filter is computed via dp_ (equivalently, fraction of genomes
with dc_ > 0, since dp_ == (dc_ > 0) by construction in NB01's merge logic)
so the same systems are included/excluded either way. Only the encoding of
each included feature changes from {0,1} to its integer copy number. This
isolates the effect of count-vs-presence cleanly, without also changing
which systems are in the model.

Outputs:
  results/q2_dc_sensitivity_results.json
"""

import json, warnings
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
PREV_THRESH  = 0.05   # >=5% prevalence in Q2 subset (computed via dp_, matches run_q2_named.py)
RF_PARAMS    = {       # identical to run_q2_named.py -- isolates encoding, not hyperparameters
    "n_estimators"    : 300,
    "max_depth"       : None,
    "min_samples_leaf": 1,
    "max_features"    : "sqrt",
    "class_weight"    : "balanced",
    "random_state"    : RANDOM_STATE,
    "n_jobs"          : -1,
}

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]
dc_cols = [c for c in fm.columns if c.startswith("dc_")]
assert len(dp_cols) == len(dc_cols) == 236

# Sanity check the dp_ == (dc_ > 0) identity this script's design depends on.
dc_nonzero = (fm[dc_cols] > 0).to_numpy()
dp_vals    = fm[dp_cols].to_numpy().astype(bool)
assert np.array_equal(dc_nonzero, dp_vals), "dp_ != (dc_ > 0) -- design assumption violated"
print(f"Verified: dp_ == (dc_ > 0) for all {len(dp_cols)} named systems.")

species_list = sorted(fm["species"].unique())
results = {}
p_raws  = {}

for sp in species_list:
    print(f"\n{'='*50}")
    print(f"Species: {sp}")

    sp_df = fm[fm.species == sp].copy()
    q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
    q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

    # Per-species prevalence filter on Q2 subset -- via dp_, so the feature
    # SET matches run_q2_named.py exactly. Filtering via dc_.mean() instead
    # would filter on mean copy number, not prevalence, and silently change
    # which systems are included relative to the dp_ baseline.
    feat_prev = q2_df[dp_cols].mean()
    sp_feat_dp = feat_prev[feat_prev >= PREV_THRESH].index.tolist()
    sp_feat_dc = [c.replace("dp_", "dc_", 1) for c in sp_feat_dp]
    print(f"  Q2 n={len(q2_df)}, features after >=5% filter: {len(sp_feat_dc)}")

    X      = q2_df[sp_feat_dc].to_numpy(dtype=float)
    y      = q2_df["y_bin"].to_numpy()
    groups = q2_df["phylogroup"].to_numpy(dtype=str)

    n_pg = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_pg)
    if n_splits < 2:
        print(f"  WARNING: only {n_pg} phylogroups — skipping")
        continue

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=RANDOM_STATE)

    fold_bas   = []
    fold_auroc = []

    for tr, te in cv.split(X, y, groups=groups):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[tr], y[tr])
        y_pred  = rf.predict(X[te])
        y_prob  = rf.predict_proba(X[te])[:, 1]
        fold_bas.append(balanced_accuracy_score(y[te], y_pred))
        fold_auroc.append(roc_auc_score(y[te], y_prob))

    fold_bas   = np.array(fold_bas)
    fold_auroc = np.array(fold_auroc)

    mean_ba    = fold_bas.mean()
    mean_auroc = fold_auroc.mean()
    ci_auroc   = stats.t.interval(0.95, df=n_splits-1,
                                   loc=mean_auroc, scale=stats.sem(fold_auroc))
    t_stat, p_raw = stats.ttest_1samp(fold_bas, 0.5)

    print(f"  BA={mean_ba:.4f}  AUROC={mean_auroc:.4f} [{ci_auroc[0]:.4f}-{ci_auroc[1]:.4f}]")
    print(f"  t={t_stat:.3f}  p_raw={p_raw:.4f}")

    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X, y)
    gini_imp = pd.Series(rf_full.feature_importances_, index=sp_feat_dc).sort_values(ascending=False)

    # IME removed entirely 2026-06-17 -- see docs/decisions.md "IME removed entirely"
    sp_arg = sp_df["arg_count_unique"].to_numpy()

    spear_arg = {}
    for feat in sp_feat_dc:
        x_vec = sp_df[feat].to_numpy()
        rho_a, p_a = spearmanr(x_vec, sp_arg)
        spear_arg[feat] = (float(rho_a), float(p_a))

    feats_list = list(sp_feat_dc)
    p_arg_raw  = [spear_arg[f][1] for f in feats_list]
    _, p_arg_adj, _, _ = multipletests(p_arg_raw, method="fdr_bh")

    top10 = gini_imp.head(10)
    print("  Top-10 Gini features:")
    for rank, (feat, imp) in enumerate(top10.items(), 1):
        rho_a = spear_arg[feat][0]
        pa_adj = p_arg_adj[feats_list.index(feat)]
        sig_a = "*" if pa_adj < 0.05 else "ns"
        print(f"    {rank:2d}. {feat:<35} imp={imp:.4f}  rho_ARG={rho_a:+.3f}({sig_a})")

    p_raws[sp] = float(p_raw)
    results[sp] = {
        "n_q2"        : int(len(q2_df)),
        "n_features"  : int(len(sp_feat_dc)),
        "fold_bas"    : fold_bas.tolist(),
        "fold_auroc"  : fold_auroc.tolist(),
        "mean_ba"     : float(mean_ba),
        "mean_auroc"  : float(mean_auroc),
        "ci95_auroc"  : [float(ci_auroc[0]), float(ci_auroc[1])],
        "t_stat"      : float(t_stat),
        "p_raw"       : float(p_raw),
        "top10_gini"  : {f: float(v) for f, v in top10.items()},
    }

sp_order = sorted(p_raws.keys())
p_raw_vec = [p_raws[sp] for sp in sp_order]
_, p_adj_vec, _, _ = multipletests(p_raw_vec, method="fdr_bh")
p_adj = {sp: float(p_adj_vec[i]) for i, sp in enumerate(sp_order)}

print(f"\n{'='*50}")
print("Q2 dc_ SENSITIVITY SUMMARY (BH-corrected across 6 species):")
for sp in sp_order:
    r = results[sp]
    sig = "★" if p_adj[sp] < 0.05 else "ns"
    print(f"  {sp:<15} AUROC={r['mean_auroc']:.3f} [{r['ci95_auroc'][0]:.3f}-{r['ci95_auroc'][1]:.3f}]"
          f"  BA={r['mean_ba']:.3f}  p_raw={r['p_raw']:.4f}  p_adj={p_adj[sp]:.4f} {sig}")
    r["p_adj"] = p_adj[sp]

with open("results/q2_dc_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/q2_dc_sensitivity_results.json")
print("Done.")
