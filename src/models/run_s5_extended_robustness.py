"""
run_s5_extended_robustness.py

Extends the within-phylogroup partial-Spearman robustness check (run_s6_spearman_named.py)
beyond the single top-ranked Gini feature per species, addressing reviewer items S5
(Methodology) and M1 (Devil's Advocate): "the within-phylogroup partial Spearman robustness
check is applied... for the 'top facilitative predictor per species'... but not for
alternative top-2/top-3 features per species."

Scope decision: extend to the full RM-family (all dp_RM_Type_* / dp_padloc_RM_type_* /
dp_df_RM_Type_*_* features reaching within-species BH significance, per species), rather
than an unbounded "top-10 Gini features regardless of identity" sweep. RM is the
literature-anchored system class the RESTRICT/FACILITATE dichotomy is built around (ref 20);
testing the whole family is a pre-statable, theory-motivated extension rather than a new
forking-paths exercise. For *E. faecium*, where RM is not the dominant Gini driver, the
named facilitative drivers already in the manuscript text (Gabija, AbiE, AbiJ, AbiH) are
also tested, since the manuscript's EF facilitative claim rests on this specific quartet,
not just Gabija alone.

Significant RM-type features per species (source: results/q2_named_results.json,
spearman_arg, p_adj < 0.05, identified 2026-06-18 while building this check):
  AB: RM_Type_I (-0.287), RM_Type_II (-0.096), RM_Type_IV (-0.334)   -- all restrictive
  EC: RM_Type_II (+0.298), RM_Type_IV (+0.115)                       -- both facilitative
  KP: RM_Type_I (+0.232), RM_Type_II (+0.387)                        -- both facilitative
  PA: RM_Type_I (+0.333), RM_Type_II (+0.178), RM_Type_IIG (+0.105),
      RM_Type_III (+0.152), df_RM_Type_IIG_2 (+0.130),
      df_RM_Type_IV_1 (+0.179)                                       -- all facilitative
  EF: RM_Type_I (+0.141)                                             -- facilitative (weak)
  SA: RM_Type_IIG (-0.117)                                           -- restrictive (weak);
      NOT discussed anywhere in the current manuscript text -- found while building this
      check. AB is not the only species with a significant negative RM correlation.

Output:
  results/s5_extended_robustness.csv
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
q2 = json.load(open("results/q2_named_results.json"))

# (species, feature) pairs: full RM family per species (all p_adj<0.05 RM-type hits)
# plus EF's named non-RM facilitative quartet (Gabija already in S6; AbiE/AbiJ/AbiH added).
PAIRS = [
    ("abaumannii",  "dp_RM_Type_I"),
    ("abaumannii",  "dp_RM_Type_II"),
    ("abaumannii",  "dp_RM_Type_IV"),
    ("ecloaceae",   "dp_RM_Type_II"),
    ("ecloaceae",   "dp_RM_Type_IV"),
    ("kpneumoniae", "dp_RM_Type_I"),
    ("kpneumoniae", "dp_RM_Type_II"),
    ("paeruginosa", "dp_RM_Type_I"),
    ("paeruginosa", "dp_RM_Type_II"),
    ("paeruginosa", "dp_RM_Type_IIG"),
    ("paeruginosa", "dp_RM_Type_III"),
    ("paeruginosa", "dp_df_RM_Type_IIG_2"),
    ("paeruginosa", "dp_df_RM_Type_IV_1"),
    ("efaecium",    "dp_RM_Type_I"),
    ("efaecium",    "dp_Gabija"),
    ("efaecium",    "dp_AbiE"),
    ("efaecium",    "dp_df_AbiJ"),
    ("efaecium",    "dp_df_AbiH"),
    ("saureus",     "dp_RM_Type_IIG"),
]
TARGET = "arg_count_unique"


def within_pg_spearman(species: str, feat: str, target: str) -> dict:
    sp = fm[fm["species"] == species].copy()
    x_raw = sp[feat].to_numpy(dtype=float)
    y_raw = sp[target].to_numpy(dtype=float)
    pg = sp["phylogroup"].to_numpy(dtype=str)
    n_total = len(sp)

    rho_sp, p_sp = spearmanr(x_raw, y_raw)

    pg_sizes = pd.Series(pg).value_counts()
    valid_pgs = set(pg_sizes[pg_sizes >= 2].index)
    mask = np.array([g in valid_pgs for g in pg])
    n_valid = int(mask.sum())

    x_use = x_raw[mask].copy()
    y_use = y_raw[mask].copy()
    pg_use = pg[mask]
    for g in valid_pgs:
        idx = pg_use == g
        x_use[idx] -= x_use[idx].mean()
        y_use[idx] -= y_use[idx].mean()

    rho_pg, p_pg = spearmanr(x_use, y_use)

    # species-level p_adj from the already-BH-corrected Q2 run, for cross-reference
    spear_rec = q2.get(species, {}).get("spearman_arg", {}).get(feat, {})
    p_adj_species_level = spear_rec.get("p_adj", float("nan"))

    return {
        "species": species,
        "feature": feat,
        "n_total": n_total,
        "n_within_pg": n_valid,
        "n_phylogroups": len(valid_pgs),
        "rho_species": float(rho_sp),
        "p_species": float(p_sp),
        "p_adj_species_level_q2run": p_adj_species_level,
        "rho_within_pg": float(rho_pg),
        "p_within_pg": float(p_pg),
        "survives_robustness": bool(abs(rho_pg) > 0.10 and p_pg < 0.05),
    }


rows = [within_pg_spearman(sp, feat, TARGET) for sp, feat in PAIRS]
out = pd.DataFrame(rows)

print(f"{'Species':<14}{'Feature':<24}{'rho_sp':>8}{'p_sp':>11}{'rho_pg':>8}{'p_pg':>11}  survives")
print("-" * 90)
for r in rows:
    sig_sp = "*" if r["p_species"] < 0.05 else " "
    sig_pg = "*" if r["p_within_pg"] < 0.05 else " "
    print(f"{r['species']:<14}{r['feature']:<24}{r['rho_species']:>+7.3f}{sig_sp}"
          f"{r['p_species']:>11.2e}{r['rho_within_pg']:>+7.3f}{sig_pg}{r['p_within_pg']:>11.2e}"
          f"  {'YES' if r['survives_robustness'] else 'no'}")

out.to_csv("results/s5_extended_robustness.csv", index=False)
print(f"\nSaved: results/s5_extended_robustness.csv ({len(out)} rows)")
