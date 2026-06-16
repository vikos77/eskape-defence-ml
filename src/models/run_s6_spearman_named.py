"""
run_s6_spearman_named.py

Within-phylogroup partial Spearman for the named-236 analysis (Supplementary Table S6).

For each (species, feature, target) triplet:
  1. Compute species-level Spearman (full species, all genomes).
  2. Demean both feature and target within each phylogroup (subtract group mean).
     Singleton phylogroups (size=1) are excluded — nothing to demean against.
  3. Compute Spearman on the demeaned residuals.

Rationale: tests whether the Q2 facilitative direction (positive rho for RM types in
EC/KP/PA and Gabija in EF) persists after controlling for clonal lineage structure.
If a high-ARG clone happens to carry a defence system, the species-level rho could be
driven by that clone rather than a within-lineage association.

Features tested:
  EC  → dp_RM_Type_II   (Q2 Gini rank 1)
  KP  → dp_RM_Type_II   (Q2 Gini rank 1)
  PA  → dp_RM_Type_I    (Q2 Gini rank 1)
  EF  → dp_Gabija       (Q2 Gini rank 1)
  AB  → dp_RM_Type_IV   (published restrictive marker)
  SA  → dp_RM_Type_I    (high prevalence, non-significant Q2)

Targets: arg_count, ime_count (total ICE/IME per genome).

Output:
  results/supplement_within_phylogroup_spearman.csv  (overwrites old file)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")

PAIRS = [
    ("ecloaceae",    "dp_RM_Type_II",  ),
    ("kpneumoniae",  "dp_RM_Type_II",  ),
    ("paeruginosa",  "dp_RM_Type_I",   ),
    ("efaecium",     "dp_Gabija",      ),
    ("abaumannii",   "dp_RM_Type_IV",  ),
    ("saureus",      "dp_RM_Type_I",   ),
]
TARGETS = ["arg_count_unique", "ime_count_total"]

# ime_count column name — check what's in the matrix
print("Columns with 'ime' or 'ice':", [c for c in fm.columns if "ime" in c.lower() or "ice" in c.lower()])
print("Columns with 'arg':", [c for c in fm.columns if "arg" in c.lower()][:5])
print()


def within_pg_spearman(species: str, feat: str, target: str) -> dict:
    sp = fm[fm["species"] == species].copy()

    # Resolve target column name
    if target not in sp.columns:
        raise ValueError(f"Column '{target}' not found. Available: {sp.columns.tolist()}")

    x_raw = sp[feat].to_numpy(dtype=float)
    y_raw = sp[target].to_numpy(dtype=float)
    pg    = sp["phylogroup"].to_numpy(dtype=str)
    n_total = len(sp)

    # Species-level Spearman
    rho_sp, p_sp = spearmanr(x_raw, y_raw)

    # Within-phylogroup demeaning
    # Exclude singletons: keep only genomes in phylogroups of size >= 2
    pg_sizes = pd.Series(pg).value_counts()
    valid_pgs = set(pg_sizes[pg_sizes >= 2].index)
    mask = np.array([g in valid_pgs for g in pg])
    n_valid = mask.sum()

    x_use = x_raw[mask].copy()
    y_use = y_raw[mask].copy()
    pg_use = pg[mask]

    # Demean within each phylogroup
    for g in valid_pgs:
        idx = (pg_use == g)
        x_use[idx] -= x_use[idx].mean()
        y_use[idx] -= y_use[idx].mean()

    rho_pg, p_pg = spearmanr(x_use, y_use)

    return {
        "species":        species,
        "feature":        feat,
        "target":         target,
        "n_total":        n_total,
        "n_within_pg":    int(n_valid),
        "n_singletons":   int(n_total - n_valid),
        "n_phylogroups":  int(len(valid_pgs)),
        "rho_species":    float(rho_sp),
        "p_species":      float(p_sp),
        "rho_within_pg":  float(rho_pg),
        "p_within_pg":    float(p_pg),
    }


rows = []
print(f"{'Species':<14}  {'Feature':<18}  {'Target':<10}  "
      f"{'rho_sp':>7}  {'p_sp':>9}  {'rho_pg':>7}  {'p_pg':>9}  {'n_pg':>5}  {'n_sgl':>5}")
print("-" * 100)

for species, feat in PAIRS:
    if feat not in fm.columns:
        print(f"  WARNING: {feat} not in feature matrix — skipping {species}")
        continue
    for target in TARGETS:
        r = within_pg_spearman(species, feat, target)
        rows.append(r)
        sig_sp = "*" if r["p_species"]  < 0.05 else " "
        sig_pg = "*" if r["p_within_pg"] < 0.05 else " "
        print(f"{species:<14}  {feat:<18}  {target:<10}  "
              f"{r['rho_species']:>+7.3f}{sig_sp} "
              f"{r['p_species']:>9.2e}  "
              f"{r['rho_within_pg']:>+7.3f}{sig_pg} "
              f"{r['p_within_pg']:>9.2e}  "
              f"{r['n_within_pg']:>5}  {r['n_singletons']:>5}")

print("\n* p < 0.05")

out = pd.DataFrame(rows)
out.to_csv("results/supplement_within_phylogroup_spearman.csv", index=False)
print(f"\nSaved: results/supplement_within_phylogroup_spearman.csv ({len(out)} rows)")
print("Done.")
