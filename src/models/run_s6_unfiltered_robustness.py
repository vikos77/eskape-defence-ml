"""
run_s6_unfiltered_robustness.py

Systematic within-phylogroup robustness sweep for the unnamed (PDC/DS-N/All_UG/
catch-all) features driving Q2 BH-significance in the full-367 unfiltered
sensitivity check (run_q1q2_full367_sensitivity.py), for the four species that
crossed BH significance there but are null under the named-236 primary
analysis: ecloaceae, efaecium, kpneumoniae, paeruginosa.

Extends the same methodology as run_s6_spearman_named.py (species-level
Spearman vs. within-phylogroup-demeaned Spearman, singleton phylogroups
excluded) from a single hand-picked feature per species to every unnamed
feature in that species' Q2-eligible pool (>=5% prevalence in the Q2
high/low-ARG subset) that is BH-significant at the raw species level.

Purpose: determine whether the apparent Q2 significance gain from including
unnamed candidate systems reflects features that are robust to clonal/
phylogroup confounding (like the named facilitative markers RM_Type_II,
Gabija already reported) or features whose correlation collapses under the
same correction already applied to every other Q2 claim in this manuscript
(like the named RM_Type_I/RM_Type_IV signals that did NOT survive S6).

Output:
  results/s6_unfiltered_robustness.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

PREV_THRESH = 0.05
TARGET = "arg_count_unique"
SPECIES = ["ecloaceae", "efaecium", "kpneumoniae", "paeruginosa"]

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]


def categorize(col: str) -> str:
    import re
    if re.search(r"PDC-?[SM]\d", col):
        return "PDC"
    if re.search(r"_DS-\d", col):
        return "DS-N"
    if re.search(r"_UG\d", col):
        return "All_UG"
    if re.search(r"(_other|_unknown|_unsubtyped|_merge)$", col):
        return "catch-all"
    return "named"


def within_pg_spearman(sp_df: pd.DataFrame, feat: str, target: str) -> tuple[float, float, int, int]:
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
    return float(rho_pg), float(p_pg), n_valid, n_sgl


rows = []

for sp in SPECIES:
    sp_df = fm[fm.species == sp].copy()
    q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()

    feat_prev = q2_df[dp_cols].mean()
    sp_feat   = feat_prev[feat_prev >= PREV_THRESH].index.tolist()
    unnamed_feat = [f for f in sp_feat if categorize(f) != "named"]

    print(f"\n{'='*70}\n{sp}: {len(sp_feat)} Q2-eligible features, {len(unnamed_feat)} unnamed")

    # Raw species-level Spearman for ALL eligible features (named+unnamed),
    # BH-corrected across the full eligible pool -- matches run_q2_named.py convention.
    arg = sp_df[TARGET].to_numpy(dtype=float)
    raw_results = {}
    for f in sp_feat:
        x = sp_df[f].to_numpy(dtype=float)
        rho, p = spearmanr(x, arg)
        raw_results[f] = (rho, p)

    p_raw_vec = [raw_results[f][1] for f in sp_feat]
    _, p_adj_vec, _, _ = multipletests(p_raw_vec, method="fdr_bh")
    p_adj_map = {f: p_adj_vec[i] for i, f in enumerate(sp_feat)}

    sig_unnamed = [f for f in unnamed_feat if p_adj_map[f] < 0.05]
    print(f"  BH-significant unnamed features (raw, species-level): {len(sig_unnamed)}/{len(unnamed_feat)}")

    for f in sig_unnamed:
        rho_raw, p_raw = raw_results[f]
        rho_pg, p_pg, n_valid, n_sgl = within_pg_spearman(sp_df, f, TARGET)
        survives = (p_pg < 0.05) and (np.sign(rho_pg) == np.sign(rho_raw))
        cat = categorize(f)
        prev = q2_df[f].mean()
        print(f"    {f:<28} [{cat:<9}] prev={prev:.3f}  rho_raw={rho_raw:+.3f}(p_adj={p_adj_map[f]:.2e})  "
              f"rho_pg={rho_pg:+.3f}(p={p_pg:.3f})  {'SURVIVES' if survives else 'COLLAPSES'}")
        rows.append({
            "species": sp, "feature": f, "category": cat, "prevalence": prev,
            "rho_raw": rho_raw, "p_adj_raw": p_adj_map[f],
            "rho_within_pg": rho_pg, "p_within_pg": p_pg,
            "n_within_pg": n_valid, "n_singletons": n_sgl,
            "survives": survives,
        })

out = pd.DataFrame(rows)
out.to_csv("results/s6_unfiltered_robustness.csv", index=False)

print(f"\n{'='*70}")
print(f"TOTAL unnamed features tested: {len(out)}")
if len(out):
    print(f"Survive within-phylogroup correction: {out['survives'].sum()}/{len(out)}")
    print("\nBy species:")
    for sp in SPECIES:
        sub = out[out.species == sp]
        if len(sub):
            print(f"  {sp:<15} {sub['survives'].sum()}/{len(sub)} survive")
print(f"\nSaved: results/s6_unfiltered_robustness.csv")
