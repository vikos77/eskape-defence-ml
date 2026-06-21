"""
Q2: ARG-burden tertile classification (RF, per species), full 367-feature
defence-system matrix. The phylogroup-robustness gate is applied UNIFORMLY to
named and unnamed top drivers alike.

Reads feature_matrix_3335.parquet (367 dp_ features, no restriction by
mechanism characterisation) so PDC/DS-N/All_UG/catch-all features are
eligible top drivers, not excluded before analysis. RF params are read from
results/q1_367_results.json ("best_params") rather than hardcoded.

The within-phylogroup robustness check (demean feature and target within
phylogroup, singleton phylogroups excluded, recompute Spearman) is applied to
every top driver per species, named or unnamed alike, using one criterion:
p_within_pg < 0.05 AND sign preserved (rho_within_pg same sign as rho_raw).
This is not a magnitude threshold (|rho| is not gated) -- the criterion
matches how every named feature is adjudicated throughout the manuscript
(e.g. a feature with p=0.054 is rejected regardless of effect size; a feature
with p<0.05 but a sign flip is rejected; a feature with p<0.05 and sign held
is accepted). A "top driver" = appears in the per-species top-10 Gini OR is
raw species-level BH-significant (p_adj<0.05) for Spearman vs ARG count.

Output:
  results/q2_367_results.json   -- AUROC, BA, p-values, top drivers tagged
                                    named/unnamed with robustness-survival flag
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
PREV_THRESH  = 0.05    # >=5% prevalence in Q2 subset
TARGET       = "arg_count_unique"
ROBUST_P_THRESH   = 0.05   # survival criterion: p<0.05 AND sign preserved (see docstring)

q1_res    = json.load(open("results/q1_367_results.json"))
RF_PARAMS = dict(
    **q1_res["best_params"],
    class_weight = "balanced",
    random_state = RANDOM_STATE,
    n_jobs       = -1,
)
print(f"RF params (from Q1-367 GridSearch): {q1_res['best_params']}")

PDC_RE       = re.compile(r"PDC-?[SM]\d")
DS_N_RE      = re.compile(r"_DS-\d")
ALL_UG_RE    = re.compile(r"_UG\d")
CATCH_ALL_RE = re.compile(r"(_other|_unknown|_unsubtyped|_merge)$")


def categorize(col: str) -> str:
    if PDC_RE.search(col):
        return "PDC"
    if DS_N_RE.search(col):
        return "DS-N"
    if ALL_UG_RE.search(col):
        return "All_UG"
    if CATCH_ALL_RE.search(col):
        return "catch-all"
    return "named"


def within_pg_spearman(sp_df: pd.DataFrame, feat: str, target: str) -> dict:
    x_raw = sp_df[feat].to_numpy(dtype=float)
    y_raw = sp_df[target].to_numpy(dtype=float)
    pg    = sp_df["phylogroup"].to_numpy(dtype=str)

    pg_sizes  = pd.Series(pg).value_counts()
    valid_pgs = set(pg_sizes[pg_sizes >= 2].index)
    mask      = np.array([g in valid_pgs for g in pg])
    n_valid   = int(mask.sum())
    n_sgl     = int(len(sp_df) - n_valid)

    x_use, y_use, pg_use = x_raw[mask].copy(), y_raw[mask].copy(), pg[mask]
    for g in valid_pgs:
        idx = (pg_use == g)
        x_use[idx] -= x_use[idx].mean()
        y_use[idx] -= y_use[idx].mean()

    rho_pg, p_pg = spearmanr(x_use, y_use)
    return {
        "rho_within_pg": float(rho_pg), "p_within_pg": float(p_pg),
        "n_within_pg": n_valid, "n_singletons": n_sgl,
        "n_phylogroups": len(valid_pgs),
    }


fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]
print(f"Full (unfiltered) dp_ features: {len(dp_cols)}")

species_list = sorted(fm["species"].unique())
results = {}
p_raws  = {}

for sp in species_list:
    print(f"\n{'='*60}\nSpecies: {sp}")

    sp_df = fm[fm.species == sp].copy()
    q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
    q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

    feat_prev = q2_df[dp_cols].mean()
    sp_feat   = feat_prev[feat_prev >= PREV_THRESH].index.tolist()
    n_unnamed = sum(1 for f in sp_feat if categorize(f) != "named")
    print(f"  Q2 n={len(q2_df)}, features after >=5% filter: {len(sp_feat)} "
          f"({n_unnamed} unnamed)")

    X      = q2_df[sp_feat].to_numpy(dtype=float)
    y      = q2_df["y_bin"].to_numpy()
    groups = q2_df["phylogroup"].to_numpy(dtype=str)

    n_pg     = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_pg)
    if n_splits < 2:
        print(f"  WARNING: only {n_pg} phylogroups -- skipping")
        continue

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_bas, fold_auroc = [], []
    for tr, te in cv.split(X, y, groups=groups):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[tr], y[tr])
        fold_bas.append(balanced_accuracy_score(y[te], rf.predict(X[te])))
        fold_auroc.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))

    fold_bas   = np.array(fold_bas)
    fold_auroc = np.array(fold_auroc)
    mean_ba    = fold_bas.mean()
    mean_auroc = fold_auroc.mean()
    ci_auroc   = stats.t.interval(0.95, df=n_splits-1, loc=mean_auroc, scale=stats.sem(fold_auroc))
    t_stat, p_raw = stats.ttest_1samp(fold_bas, 0.5)

    print(f"  BA={mean_ba:.4f}  AUROC={mean_auroc:.4f} [{ci_auroc[0]:.4f}-{ci_auroc[1]:.4f}]  p_raw={p_raw:.4f}")

    # Gini importance (train on all Q2 data)
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X, y)
    gini_imp = pd.Series(rf_full.feature_importances_, index=sp_feat).sort_values(ascending=False)
    top10_gini = set(gini_imp.head(10).index)

    # Species-level Spearman vs ARG count, all Q2-eligible features, BH-corrected
    arg = sp_df[TARGET].to_numpy(dtype=float)
    raw = {}
    for f in sp_feat:
        x = sp_df[f].to_numpy(dtype=float)
        rho, p = spearmanr(x, arg)
        raw[f] = (float(rho), float(p))

    p_raw_vec = [raw[f][1] for f in sp_feat]
    _, p_adj_vec, _, _ = multipletests(p_raw_vec, method="fdr_bh")
    p_adj_map = {f: float(p_adj_vec[i]) for i, f in enumerate(sp_feat)}
    bh_sig = {f for f in sp_feat if p_adj_map[f] < 0.05}

    # Top drivers = top-10 Gini UNION BH-significant Spearman -- tested for robustness
    top_drivers = sorted(top10_gini | bh_sig)
    print(f"  Top drivers to test for robustness: {len(top_drivers)} "
          f"(top10 Gini: {len(top10_gini)}, BH-sig Spearman: {len(bh_sig)}, "
          f"overlap: {len(top10_gini & bh_sig)})")

    driver_rows = []
    for f in top_drivers:
        rho_raw, p_arg_raw = raw[f]
        pg_res = within_pg_spearman(sp_df, f, TARGET)
        sign_preserved = (rho_raw > 0) == (pg_res["rho_within_pg"] > 0)
        survives = (pg_res["p_within_pg"] < ROBUST_P_THRESH) and sign_preserved
        cat = categorize(f)
        row = {
            "feature": f, "category": cat, "is_named": cat == "named",
            "in_top10_gini": f in top10_gini, "bh_significant_raw": f in bh_sig,
            "gini_importance": float(gini_imp[f]),
            "rho_raw": rho_raw, "p_adj_raw": p_adj_map[f],
            **pg_res,
            "sign_preserved": bool(sign_preserved),
            "survives_robustness": bool(survives),
        }
        driver_rows.append(row)
        sig_raw = "*" if p_adj_map[f] < 0.05 else " "
        print(f"    {f:<32} [{cat:<9}] gini={gini_imp[f]:.4f}  "
              f"rho_raw={rho_raw:+.3f}{sig_raw}  rho_pg={pg_res['rho_within_pg']:+.3f}  "
              f"p_pg={pg_res['p_within_pg']:.3f}  {'SURVIVES' if survives else 'collapses'}")

    n_named_drivers   = sum(1 for r in driver_rows if r["is_named"])
    n_unnamed_drivers = len(driver_rows) - n_named_drivers
    n_named_survive   = sum(1 for r in driver_rows if r["is_named"] and r["survives_robustness"])
    n_unnamed_survive = sum(1 for r in driver_rows if not r["is_named"] and r["survives_robustness"])
    print(f"  Robustness: named {n_named_survive}/{n_named_drivers} survive, "
          f"unnamed {n_unnamed_survive}/{n_unnamed_drivers} survive")

    p_raws[sp] = float(p_raw)
    results[sp] = {
        "n_q2"       : int(len(q2_df)),
        "n_features" : int(len(sp_feat)),
        "n_unnamed_features": int(n_unnamed),
        "fold_bas"   : fold_bas.tolist(),
        "fold_auroc" : fold_auroc.tolist(),
        "mean_ba"    : float(mean_ba),
        "mean_auroc" : float(mean_auroc),
        "ci95_auroc" : [float(ci_auroc[0]), float(ci_auroc[1])],
        "t_stat"     : float(t_stat),
        "p_raw"      : float(p_raw),
        "top_drivers": driver_rows,
        "robustness_summary": {
            "n_named_drivers": n_named_drivers, "n_named_survive": n_named_survive,
            "n_unnamed_drivers": n_unnamed_drivers, "n_unnamed_survive": n_unnamed_survive,
        },
    }

# BH correction across 6 species (classifier significance)
sp_order = sorted(p_raws.keys())
p_raw_vec = [p_raws[sp] for sp in sp_order]
_, p_adj_vec, _, _ = multipletests(p_raw_vec, method="fdr_bh")
p_adj = {sp: float(p_adj_vec[i]) for i, sp in enumerate(sp_order)}

print(f"\n{'='*60}")
print("Q2-367 SUMMARY (BH-corrected across 6 species):")
for sp in sp_order:
    r = results[sp]
    sig = "*" if p_adj[sp] < 0.05 else "ns"
    rs = r["robustness_summary"]
    print(f"  {sp:<15} AUROC={r['mean_auroc']:.3f} [{r['ci95_auroc'][0]:.3f}-{r['ci95_auroc'][1]:.3f}]"
          f"  p_adj={p_adj[sp]:.4f} {sig}  "
          f"| robust drivers: named {rs['n_named_survive']}/{rs['n_named_drivers']}, "
          f"unnamed {rs['n_unnamed_survive']}/{rs['n_unnamed_drivers']}")
    r["p_adj"] = p_adj[sp]

with open("results/q2_367_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/q2_367_results.json")
print("Done.")
