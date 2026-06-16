"""
Q2 re-run on the 236-feature named-only matrix.

Feature pool: all 236 named dp_ features.
Spec_score filter NOT applied (Q2 is within-species; between-species
prevalence contrast is irrelevant).
Per-species filter: >=5% prevalence in Q2 tertile subset (top+bottom only).
RF params: best from Q1 GridSearch (n_estimators=300, max_depth=None,
min_samples_leaf=1, max_features=sqrt).
Significance: one-sample t-test of fold BAs vs 0.5, BH-corrected across 6 species.

Outputs:
  results/q2_named_results.json   — AUROC, BA, p-values, top features, Spearman
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
PREV_THRESH  = 0.05   # >=5% prevalence in Q2 subset
RF_PARAMS    = {      # best from Q1 GridSearch
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
print(f"Named dp_ features: {len(dp_cols)}")

species_list = sorted(fm["species"].unique())
results = {}
p_raws  = {}

for sp in species_list:
    print(f"\n{'='*50}")
    print(f"Species: {sp}")

    sp_df = fm[fm.species == sp].copy()
    q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
    q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

    # Per-species prevalence filter on Q2 subset
    feat_prev = q2_df[dp_cols].mean()
    sp_feat   = feat_prev[feat_prev >= PREV_THRESH].index.tolist()
    print(f"  Q2 n={len(q2_df)}, features after >=5% filter: {len(sp_feat)}")

    X      = q2_df[sp_feat].to_numpy(dtype=float)
    y      = q2_df["y_bin"].to_numpy()
    groups = q2_df["phylogroup"].to_numpy(dtype=str)

    # Check enough phylogroups for 5-fold
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

    # Gini importance (train on all Q2 data)
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X, y)
    gini_imp = pd.Series(rf_full.feature_importances_, index=sp_feat).sort_values(ascending=False)

    # Spearman: ARG count and IME count (on all species genomes, BH-corrected)
    sp_arg = sp_df["arg_count_unique"].to_numpy()
    sp_ime = sp_df["ime_count_unique"].to_numpy()

    spear_arg, spear_ime = {}, {}
    for feat in sp_feat:
        x_vec = sp_df[feat].to_numpy()
        rho_a, p_a = spearmanr(x_vec, sp_arg)
        rho_i, p_i = spearmanr(x_vec, sp_ime)
        spear_arg[feat] = (float(rho_a), float(p_a))
        spear_ime[feat] = (float(rho_i), float(p_i))

    # BH correction per species
    feats_list = list(sp_feat)
    p_arg_raw  = [spear_arg[f][1] for f in feats_list]
    p_ime_raw  = [spear_ime[f][1] for f in feats_list]
    _, p_arg_adj, _, _ = multipletests(p_arg_raw, method="fdr_bh")
    _, p_ime_adj, _, _ = multipletests(p_ime_raw, method="fdr_bh")

    # Top 10 features by Gini
    top10 = gini_imp.head(10)
    print("  Top-10 Gini features:")
    for rank, (feat, imp) in enumerate(top10.items(), 1):
        rho_a = spear_arg[feat][0]
        rho_i = spear_ime[feat][0]
        pa_adj = p_arg_adj[feats_list.index(feat)]
        pi_adj = p_ime_adj[feats_list.index(feat)]
        sig_a = "*" if pa_adj < 0.05 else "ns"
        sig_i = "*" if pi_adj < 0.05 else "ns"
        print(f"    {rank:2d}. {feat:<35} imp={imp:.4f}  rho_ARG={rho_a:+.3f}({sig_a})  rho_IME={rho_i:+.3f}({sig_i})")

    p_raws[sp] = float(p_raw)
    results[sp] = {
        "n_q2"        : int(len(q2_df)),
        "n_features"  : int(len(sp_feat)),
        "fold_bas"    : fold_bas.tolist(),
        "fold_auroc"  : fold_auroc.tolist(),
        "mean_ba"     : float(mean_ba),
        "mean_auroc"  : float(mean_auroc),
        "ci95_auroc"  : [float(ci_auroc[0]), float(ci_auroc[1])],
        "t_stat"      : float(t_stat),
        "p_raw"       : float(p_raw),
        "top10_gini"  : {f: float(v) for f, v in top10.items()},
        "spearman_arg": {f: {"rho": spear_arg[f][0], "p_adj": float(p_arg_adj[feats_list.index(f)])}
                         for f in sp_feat},
        "spearman_ime": {f: {"rho": spear_ime[f][0], "p_adj": float(p_ime_adj[feats_list.index(f)])}
                         for f in sp_feat},
    }

# BH correction across 6 species
sp_order = sorted(p_raws.keys())
p_raw_vec = [p_raws[sp] for sp in sp_order]
_, p_adj_vec, _, _ = multipletests(p_raw_vec, method="fdr_bh")
p_adj = {sp: float(p_adj_vec[i]) for i, sp in enumerate(sp_order)}

print(f"\n{'='*50}")
print("Q2 SUMMARY (BH-corrected across 6 species):")
for sp in sp_order:
    r = results[sp]
    sig = "★" if p_adj[sp] < 0.05 else "ns"
    print(f"  {sp:<15} AUROC={r['mean_auroc']:.3f} [{r['ci95_auroc'][0]:.3f}-{r['ci95_auroc'][1]:.3f}]"
          f"  BA={r['mean_ba']:.3f}  p_raw={r['p_raw']:.4f}  p_adj={p_adj[sp]:.4f} {sig}")
    r["p_adj"] = p_adj[sp]

with open("results/q2_named_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/q2_named_results.json")
print("Done.")
