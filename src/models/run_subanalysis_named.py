"""
run_subanalysis_named.py

Two pre-registered subanalyses on the named-236 feature matrix (Supplementary
Tables S7 and S8):

  S7 — AB IC2 exclusion Q2:
       Exclude AB genomes where dp_SspBCDE=1 (IC2 proxy). Rerun Q2 on the
       remaining non-IC2 AB genomes to test whether RM restriction of ARG
       burden is detectable once IC2 clonal compression is removed.

  S8 — EC E. hormaechei subanalysis Q2:
       Filter EC to E. hormaechei sensu stricto (MLST scheme ecoli_achtman_4
       or enterobacter_oxford; filter by species_annotation field if available,
       otherwise by most common MLST complex). Rerun Q2 on E. hormaechei only.

Both follow the same Q2 logic as run_q2_named.py:
  - Per-subset tertile split of arg_count_unique
  - Per-subset prevalence filter (dp_* >= 5%)
  - GroupedStratifiedKFold-5 RF Q2
  - AUROC + fold-bootstrap CI (2000 iters)
  - Spearman rho for top Gini features

Outputs:
  results/supplement_ab_ic2_q2.csv       — S7
  results/supplement_ec_complex_q2.csv   — S8
"""

import sys
sys.path.insert(0, "src")

import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
N_FOLDS      = 5
PREV_THRESH  = 0.05
RF_PARAMS    = dict(
    n_estimators=300, max_depth=None, max_features="sqrt",
    min_samples_leaf=1, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)


def fold_bootstrap_ci(fold_scores, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [
        float(np.mean(rng.choice(fold_scores, size=len(fold_scores), replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_q2_subanalysis(subset, label, feat_cols_pool):
    """
    Run Q2 RF on a genome subset.

    Parameters
    ----------
    subset       : pd.DataFrame  — rows are genomes, must include phylogroup,
                                   arg_count_unique, and all dp_* cols
    label        : str           — name for printing
    feat_cols_pool : list        — named dp_* columns to filter from

    Returns
    -------
    dict of results
    """
    # Tertile labels within subset
    q33 = subset["arg_count_unique"].quantile(1 / 3)
    q67 = subset["arg_count_unique"].quantile(2 / 3)
    high = subset[subset["arg_count_unique"] >= q67].copy()
    low  = subset[subset["arg_count_unique"] <= q33].copy()
    q2   = pd.concat([high, low]).copy()
    q2["label"] = (q2["arg_count_unique"] >= q67).astype(int)

    n_high, n_low, n_q2 = len(high), len(low), len(q2)
    n_pgs = q2["phylogroup"].nunique()
    print(f"\n{label}: n_total={len(subset)}, Q2 high={n_high}, low={n_low}, total={n_q2}, phylogroups={n_pgs}")

    # Prevalence filter on Q2-eligible genomes
    prev = q2[feat_cols_pool].mean()
    feat_cols = [c for c in feat_cols_pool if prev[c] >= PREV_THRESH]
    print(f"  Features after prevalence filter: {len(feat_cols)}")

    if n_pgs < N_FOLDS:
        print(f"  WARNING: fewer phylogroups ({n_pgs}) than folds ({N_FOLDS}) — skipping")
        return None

    X      = q2[feat_cols].to_numpy(dtype=float)
    y      = q2["label"].to_numpy()
    groups = q2["phylogroup"].to_numpy(dtype=str)

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_auroc = []
    fold_ba    = []
    gini_sum   = np.zeros(len(feat_cols))

    for tr, te in cv.split(X, y, groups=groups):
        if len(np.unique(y[te])) < 2:
            continue
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[tr], y[tr])
        proba = rf.predict_proba(X[te])[:, 1]
        fold_auroc.append(roc_auc_score(y[te], proba))
        fold_ba.append(balanced_accuracy_score(y[te], rf.predict(X[te])))
        gini_sum += rf.feature_importances_

    mean_auroc = float(np.mean(fold_auroc))
    ci_lo, ci_hi = fold_bootstrap_ci(fold_auroc)
    mean_ba = float(np.mean(fold_ba))

    # Top Gini features + Spearman (species-level, full subset)
    gini_avg = gini_sum / N_FOLDS
    top_idx  = np.argsort(gini_avg)[::-1][:10]
    top_feats = [(feat_cols[i], float(gini_avg[i])) for i in top_idx]

    spearman_rows = []
    for feat, gini in top_feats[:5]:
        rho, p = spearmanr(subset[feat], subset["arg_count_unique"])
        spearman_rows.append({"feature": feat, "gini": round(gini, 5),
                               "rho_arg": round(rho, 4), "p_arg": round(p, 6)})

    print(f"  AUROC={mean_auroc:.3f} [{ci_lo:.3f}-{ci_hi:.3f}]  BA={mean_ba:.3f}")
    print(f"  Top 5 Gini features:")
    for r in spearman_rows:
        print(f"    {r['feature']:<25}  gini={r['gini']:.4f}  rho_ARG={r['rho_arg']:+.3f}  p={r['p_arg']:.4f}")

    return {
        "label":       label,
        "n_total":     len(subset),
        "n_q2":        n_q2,
        "n_features":  len(feat_cols),
        "n_phylogroups": n_pgs,
        "mean_auroc":  mean_auroc,
        "ci95_lo":     ci_lo,
        "ci95_hi":     ci_hi,
        "mean_ba":     mean_ba,
        "fold_auroc":  fold_auroc,
        "top5_spearman": spearman_rows,
    }


# ── Load feature matrix ───────────────────────────────────────────────────

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]

# Named-236 pool (same derivation as run_q2_named.py)
species_list = fm["species"].unique()
sp_prev = pd.DataFrame({sp: fm[fm.species == sp][dp_cols].mean() for sp in species_list})
spec_score = sp_prev.std(axis=1) / 0.5
# Do NOT apply spec_score filter for Q2 — all 236 dp_* are the starting pool
feat_cols_pool = dp_cols  # prevalence filter applied per-subset below

print(f"Named dp_* pool: {len(feat_cols_pool)} features")

# ── S7: AB IC2 exclusion ──────────────────────────────────────────────────

print("\n" + "="*60)
print("S7 — AB IC2 exclusion (dp_SspBCDE=0)")
print("="*60)

ab_all   = fm[fm["species"] == "abaumannii"].copy()
ab_non_ic2 = ab_all[ab_all["dp_SspBCDE"] == 0].copy()

print(f"AB total: {len(ab_all)}, IC2 excluded (SspBCDE=1): {ab_all['dp_SspBCDE'].sum()}, "
      f"non-IC2 remaining: {len(ab_non_ic2)} ({len(ab_non_ic2)/len(ab_all)*100:.1f}%)")

res_s7 = run_q2_subanalysis(ab_non_ic2, "AB non-IC2", feat_cols_pool)

# ── S8: EC E. hormaechei subanalysis ──────────────────────────────────────

print("\n" + "="*60)
print("S8 — EC E. hormaechei subanalysis")
print("="*60)

ec_all = fm[fm["species"] == "ecloaceae"].copy()

# Identify E. hormaechei using MLST species_annotation if available,
# otherwise use the 'mlst_species' or 'subspecies' column.
ec_cols = ec_all.columns.tolist()
anno_cols = [c for c in ec_cols if "species" in c.lower() or "mlst" in c.lower() or "subspecies" in c.lower()]
print(f"Annotation columns available: {anno_cols}")

if "mlst_species" in ec_all.columns:
    print(f"mlst_species values:\n{ec_all['mlst_species'].value_counts().head(10)}")
    hormaechei = ec_all[ec_all["mlst_species"].str.contains("hormaechei", case=False, na=False)].copy()
elif "subspecies" in ec_all.columns:
    print(f"subspecies values:\n{ec_all['subspecies'].value_counts().head(10)}")
    hormaechei = ec_all[ec_all["subspecies"].str.contains("hormaechei", case=False, na=False)].copy()
elif "complex_member" in ec_all.columns:
    print(f"complex_member values:\n{ec_all['complex_member'].value_counts()}")
    hormaechei = ec_all[ec_all["complex_member"] == "E. hormaechei"].copy()
else:
    print("No species annotation column found — EC hormaechei subanalysis requires MLST species field.")
    print("Available columns:", [c for c in ec_all.columns if not c.startswith("dp_") and not c.startswith("dc_") and not c.startswith("ac_")])
    hormaechei = None

if hormaechei is not None and len(hormaechei) >= 30:
    res_s8 = run_q2_subanalysis(hormaechei, "EC E. hormaechei", feat_cols_pool)
else:
    if hormaechei is not None:
        print(f"Too few E. hormaechei genomes ({len(hormaechei)}) for Q2 — skipping")
    res_s8 = None

# ── Save outputs ──────────────────────────────────────────────────────────

rows = []
for res, fname in [(res_s7, "results/supplement_ab_ic2_q2.csv"),
                   (res_s8, "results/supplement_ec_complex_q2.csv")]:
    if res is None:
        continue
    out_rows = []
    for r in res["top5_spearman"]:
        out_rows.append({
            "subanalysis":   res["label"],
            "n_total":       res["n_total"],
            "n_q2":          res["n_q2"],
            "n_features":    res["n_features"],
            "n_phylogroups": res["n_phylogroups"],
            "mean_auroc":    res["mean_auroc"],
            "ci95_lo":       res["ci95_lo"],
            "ci95_hi":       res["ci95_hi"],
            "mean_ba":       res["mean_ba"],
            **r,
        })
    pd.DataFrame(out_rows).to_csv(fname, index=False)
    print(f"\nSaved: {fname}")

print("\nDone.")
