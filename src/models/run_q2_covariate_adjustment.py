"""
Q2 covariate-adjustment test: does the phylogroup-robust facilitative signal
survive controlling for total mobilome/defence burden (the generic-HGT
confound raised independently by the Methodology and Devil's Advocate
reviewers)?

For every driver that already survives the within-phylogroup robustness gate
(run_q2_367.py) in the four Q2-significant species, this adds a second gate:
within phylogroup, residualize both the feature and arg_count_unique on two
burden covariates --
  z1 = defence_system_count - feature itself (total OTHER defence systems,
       excluding the driver under test to avoid tautological self-control)
  z2 = is_count_total (total IS elements)
-- then Spearman-correlate the residuals. A driver survives only if
p_partial < 0.05 AND sign matches rho_within_pg.

This does not retest drivers that already failed the phylogroup gate --
those are already excluded from the manuscript's reported driver lists.

Output: results/q2_covariate_adjustment.json
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

TARGET = "arg_count_unique"

SPECIES_MAP = {
    "ecloaceae": "E. cloacae",
    "kpneumoniae": "K. pneumoniae",
    "efaecium": "E. faecium",
    "paeruginosa": "P. aeruginosa",
}

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
q2 = json.load(open("results/q2_367_results.json"))


def within_pg_partial_spearman(sp_df, feat, target, covariates):
    pg = sp_df["phylogroup"].to_numpy(dtype=str)
    pg_sizes = pd.Series(pg).value_counts()
    valid_pgs = set(pg_sizes[pg_sizes >= 2].index)
    mask = np.array([g in valid_pgs for g in pg])

    x = sp_df[feat].to_numpy(dtype=float)[mask].copy()
    y = sp_df[target].to_numpy(dtype=float)[mask].copy()
    Z = np.column_stack([sp_df[c].to_numpy(dtype=float)[mask].copy() for c in covariates])
    pg_use = pg[mask]

    for g in valid_pgs:
        idx = pg_use == g
        x[idx] -= x[idx].mean()
        y[idx] -= y[idx].mean()
        Z[idx] -= Z[idx].mean(axis=0)

    resid_x = x - LinearRegression().fit(Z, x).predict(Z)
    resid_y = y - LinearRegression().fit(Z, y).predict(Z)
    rho, p = spearmanr(resid_x, resid_y)
    return {
        "rho_partial": float(rho), "p_partial": float(p),
        "n_used": int(mask.sum()), "n_phylogroups": len(valid_pgs),
    }


out = {}
for sp_key, sp_label in SPECIES_MAP.items():
    sp_df = fm[fm.species == sp_key].copy()
    sp_df["defence_other"] = sp_df["defence_system_count"]  # per-feature subtract below

    drivers = [r for r in q2[sp_key]["top_drivers"] if r["survives_robustness"]]
    print(f"\n{'='*70}\n{sp_label} ({sp_key}): {len(drivers)} phylogroup-robust drivers to test")

    rows = []
    for r in drivers:
        feat = r["feature"]
        sp_df["_z1"] = sp_df["defence_system_count"] - sp_df[feat]
        res = within_pg_partial_spearman(sp_df, feat, TARGET, ["_z1", "is_count_total"])
        sign_preserved = (r["rho_within_pg"] > 0) == (res["rho_partial"] > 0)
        survives = (res["p_partial"] < 0.05) and sign_preserved
        row = {
            "feature": feat, "category": r["category"],
            "rho_within_pg": r["rho_within_pg"], "p_within_pg": r["p_within_pg"],
            **res, "sign_preserved": bool(sign_preserved),
            "survives_covariate_adjustment": bool(survives),
        }
        rows.append(row)
        flag = "SURVIVES" if survives else "collapses"
        print(f"  {feat:<28} rho_pg={r['rho_within_pg']:+.3f}  "
              f"rho_partial={res['rho_partial']:+.3f}  p_partial={res['p_partial']:.4f}  {flag}")

    n_total = len(rows)
    n_survive = sum(1 for r in rows if r["survives_covariate_adjustment"])
    print(f"  -> {n_survive}/{n_total} survive covariate adjustment")
    out[sp_key] = {"species_label": sp_label, "n_drivers_tested": n_total,
                    "n_survive_covariate": n_survive, "drivers": rows}

print(f"\n{'='*70}\nSUMMARY (drivers surviving BOTH phylogroup gate AND covariate adjustment):")
for sp_key, res in out.items():
    print(f"  {res['species_label']:<16} {res['n_survive_covariate']}/{res['n_drivers_tested']}")

with open("results/q2_covariate_adjustment.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/q2_covariate_adjustment.json")
