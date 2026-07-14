"""
Generate all supplementary tables (S1–S15) as CSV files in results/tables/.

Run from project root:
  conda run -n eskape-ml python src/models/build_supplementary_tables.py
"""

import json, os, shutil
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

os.makedirs("results/tables", exist_ok=True)

SP_LABEL = {
    "abaumannii":  "A. baumannii",
    "ecloaceae":   "E. cloacae complex",
    "efaecium":    "E. faecium",
    "kpneumoniae": "K. pneumoniae",
    "paeruginosa": "P. aeruginosa",
    "saureus":     "S. aureus",
}
SP_ORDER = ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_ci(lo, hi, dp=3):
    return f"[{lo:.{dp}f}–{hi:.{dp}f}]"

def save(df, name, description):
    path = f"results/tables/{name}.csv"
    df.to_csv(path, index=False)
    print(f"  {name}.csv  ({len(df)} rows)  — {description}")


# ── Load shared data ──────────────────────────────────────────────────────────
with open("results/q1_367_results.json")       as f: q1     = json.load(f)
with open("results/q1_lr_xgb_lgbm_named.json") as f: q1_alt = json.load(f)
with open("results/q1_mcnemar_367.json")        as f: mcn    = json.load(f)
with open("results/q2_367_results.json")        as f: q2     = json.load(f)
with open("results/q2_ad_sensitivity.json")     as f: q2_ad  = json.load(f)
with open("results/q2_split_sensitivity.json")  as f: q2_sp  = json.load(f)
with open("results/q2_covariate_adjustment.json")    as f: q2_cov = json.load(f)
with open("results/q2_bootstrap_significance.json")  as f: q2_bt  = json.load(f)
with open("results/q3_367_results.json")             as f: q3     = json.load(f)
with open("results/q3b_filtered_results.json")        as f: q3bf   = json.load(f)
with open("results/supplement_dominant_pg_exclusion_summary.json") as f: dpg = json.load(f)
with open("results/sensitivity_full367_q1.json")     as f: q1_sf  = json.load(f)
with open("results/sensitivity_df_full_q1.json")     as f: q1_df  = json.load(f)

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")

print("Building supplementary tables …\n")

# ────────────────────────────────────────────────────────────────────────────
# S1 — Feature matrix column layout
# ────────────────────────────────────────────────────────────────────────────
dp_cols  = [c for c in fm.columns if c.startswith("dp_")]
dc_cols  = [c for c in fm.columns if c.startswith("dc_")]
ad_cols  = [c for c in fm.columns if c.startswith("ad_")]
meta_cols = [c for c in fm.columns if not any(c.startswith(p) for p in ("dp_","dc_","ad_"))]

s1 = pd.DataFrame([
    {"Column group": "Binary defence presence (dp_*)",
     "N columns": len(dp_cols),
     "Description": "Binary (0/1): defence system type detected in genome by DefenseFinder v2.0.1 or PADLOC v2.0.0 (union, canonical name harmonised). Includes 359 post-marker-filter features (FEAT_COLS) and 8 excluded near-exclusive species markers."},
    {"Column group": "Defence copy number (dc_*)",
     "N columns": len(dc_cols),
     "Description": "Integer: count of instances of each defence system type detected per genome."},
    {"Column group": "Anti-defence count (ad_*)",
     "N columns": len(ad_cols),
     "Description": "Integer: count of anti-defence system instances detected by AntiDefenseFinder (--antidefensefinder flag)."},
    {"Column group": "Annotation and metadata",
     "N columns": len(meta_cols),
     "Description": "Genome accession, species, ARG count (unique genes), IS element count (families), HMRG count, ARG burden tertile label, sequence type (MLST), phylogroup assignment, country, year, complex member (EC only), and derived ratio features."},
    {"Column group": "TOTAL",
     "N columns": len(fm.columns),
     "Description": "3,335 genomes × 828 columns."},
])
save(s1, "S1_feature_matrix_layout", "feature matrix column groups")

# ────────────────────────────────────────────────────────────────────────────
# S2 — Q1 classifier comparison (RF, LR, XGBoost, LightGBM)
# ────────────────────────────────────────────────────────────────────────────
q1_rows = []

# RF from q1_367_results.json
rf_f1_ci = q1.get("ci95_f1", [None, None])
q1_rows.append({
    "Classifier": "Random Forest",
    "N features": q1["n_features"],
    "Mean BA": round(q1["mean_ba"], 4),
    "BA 95% CI": fmt_ci(*q1["ci95_ba"]),
    "Mean macro-F1": round(q1["mean_f1"], 4),
    "F1 95% CI": fmt_ci(*rf_f1_ci) if rf_f1_ci[0] else "—",
    "Notes": "Primary model; used for SHAP attribution (§4)"
})

# LR, XGB, LGBM: per-fold BAs are in mcnemar 367 JSON metadata
from scipy import stats as _stats
mcn_fold_bas = mcn["metadata"]["fold_bas"]
for clf_key, clf_name in [("LR","Logistic Regression"),("XGB","XGBoost"),("LGBM","LightGBM")]:
    folds = np.array(mcn_fold_bas[clf_key])
    mean_ba = folds.mean()
    ci = _stats.t.interval(0.95, df=len(folds)-1, loc=mean_ba, scale=_stats.sem(folds))
    q1_rows.append({
        "Classifier": clf_name,
        "N features": q1["n_features"],
        "Mean BA": round(mean_ba, 4),
        "BA 95% CI": fmt_ci(round(ci[0], 3), round(ci[1], 3)),
        "Mean macro-F1": "—",
        "F1 95% CI": "—",
        "Notes": ""
    })

s2 = pd.DataFrame(q1_rows)
save(s2, "S2_q1_classifier_comparison", "Q1 RF vs LR/XGBoost/LightGBM")

# ────────────────────────────────────────────────────────────────────────────
# S3 — McNemar pairwise tests (RF vs each alternative)
# ────────────────────────────────────────────────────────────────────────────
mcn_rows = []
for pair, v in mcn.items():
    if pair == "metadata":
        continue
    mcn_rows.append({
        "Comparison": pair.replace("_", " "),
        "χ² (or statistic)": round(v.get("statistic", v.get("chi2", float("nan"))), 4),
        "p-value": f"{v.get('p_value', v.get('p', float('nan'))):.4f}",
        "Significant (α=0.05)": "Yes" if v.get('p_value', v.get('p', 1.0)) < 0.05 else "No",
        "Notes": v.get("note", "")
    })
s3 = pd.DataFrame(mcn_rows)
save(s3, "S3_q1_mcnemar_tests", "McNemar pairwise classifier tests")

# ────────────────────────────────────────────────────────────────────────────
# S4 — Q1 per-fold balanced accuracy (all four classifiers)
# ────────────────────────────────────────────────────────────────────────────
s4_rows = []
mcn_folds = mcn["metadata"]["fold_bas"]
for fold_i, ba in enumerate(q1["fold_bas"], 1):
    s4_rows.append({
        "Fold": fold_i,
        "RF":   round(ba, 4),
        "LR":   round(mcn_folds["LR"][fold_i-1], 4),
        "XGB":  round(mcn_folds["XGB"][fold_i-1], 4),
        "LGBM": round(mcn_folds["LGBM"][fold_i-1], 4),
    })
s4_rows.append({
    "Fold": "Mean",
    "RF":   round(q1["mean_ba"], 4),
    "LR":   round(np.mean(mcn_folds["LR"]), 4),
    "XGB":  round(np.mean(mcn_folds["XGB"]), 4),
    "LGBM": round(np.mean(mcn_folds["LGBM"]), 4),
})
s4 = pd.DataFrame(s4_rows)
save(s4, "S4_q1_fold_balanced_accuracies", "per-fold BA across classifiers")

# ────────────────────────────────────────────────────────────────────────────
# S5 — Q1 marker-filter sensitivity (8 vs 15 markers removed)
# ────────────────────────────────────────────────────────────────────────────
s50 = q1["sensitivity_50"]
s5 = pd.DataFrame([
    {
        "Analysis": "Primary (spec_score ≥ 0.70)",
        "N markers removed": q1["n_markers_removed"],
        "N features used": q1["n_features"],
        "Mean BA": round(q1["mean_ba"], 4),
        "BA 95% CI": fmt_ci(*q1["ci95_ba"]),
        "Markers removed": "; ".join(q1["markers"])
    },
    {
        "Analysis": "Sensitivity (spec_score ≥ 0.50)",
        "N markers removed": s50["n_markers"],
        "N features used": s50["n_features"],
        "Mean BA": round(s50["mean_ba"], 4),
        "BA 95% CI": fmt_ci(*s50["ci95_ba"]),
        "Markers removed": "; ".join(s50["markers"])
    },
    {
        "Analysis": "DF+PADLOC union, DF-only named filter",
        "N markers removed": q1_df["n_markers_removed"],
        "N features used": q1_df["n_feat_cols"],
        "Mean BA": round(q1_df["mean_ba"], 4),
        "BA 95% CI": fmt_ci(*q1_df["ci95_ba"]),
        "Markers removed": "; ".join(q1_df["markers"])
    },
])
save(s5, "S5_q1_filter_sensitivity", "Q1 marker-filter sensitivity comparison")

# ────────────────────────────────────────────────────────────────────────────
# S6 — Q2 RM-family within-phylogroup Spearman (s5_extended_robustness.csv)
# ────────────────────────────────────────────────────────────────────────────
df_s6 = pd.read_csv("results/s5_extended_robustness.csv")
df_s6["species"] = df_s6["species"].map(lambda x: SP_LABEL.get(x, x))
df_s6 = df_s6.rename(columns={
    "species": "Species",
    "feature": "Feature",
    "n_total": "N genomes",
    "n_phylogroups": "N phylogroups",
    "rho_species": "ρ (species-level)",
    "p_adj_species_level_q2run": "p_adj (BH, Q2 run)",
    "rho_within_pg": "ρ (within-phylogroup)",
    "p_within_pg": "p (within-phylogroup)",
    "survives_robustness": "Survives robustness gate"
})
df_s6["Feature"] = df_s6["Feature"].str.removeprefix("dp_")
save(df_s6[["Species","Feature","N genomes","N phylogroups",
            "ρ (species-level)","p_adj (BH, Q2 run)",
            "ρ (within-phylogroup)","p (within-phylogroup)",
            "Survives robustness gate"]],
     "S6_q2_rm_spearman_withinpg", "Q2 RM-family within-phylogroup Spearman (19 feature-species pairs)")

# ────────────────────────────────────────────────────────────────────────────
# S7 — AB IC2 exclusion Q2 result
# ────────────────────────────────────────────────────────────────────────────
ab_full    = q2["abaumannii"]
ab_ic2_csv = pd.read_csv("results/supplement_ab_ic2_q2_367.csv")
ab_ic2     = ab_ic2_csv.iloc[0]

s7 = pd.DataFrame([
    {
        "Analysis": "Full AB cohort",
        "N Q2 genomes": ab_full["n_q2"],
        "N phylogroups": ab_full.get("n_phylogroups", "—"),
        "N features": ab_full.get("n_features", "—"),
        "Mean AUROC": round(ab_full["mean_auroc"], 4),
        "AUROC 95% CI": fmt_ci(*ab_full["ci95_auroc"]),
        "BH p_adj": f"{ab_full.get('p_adj', float('nan')):.4f}",
        "Significant": "No"
    },
    {
        "Analysis": "IC2-excluded AB (dp_SspBCDE=1 removed)",
        "N Q2 genomes": int(ab_ic2["n_q2"]),
        "N phylogroups": int(ab_ic2["n_phylogroups"]),
        "N features": int(ab_ic2["n_features"]),
        "Mean AUROC": round(float(ab_ic2["mean_auroc"]), 4),
        "AUROC 95% CI": fmt_ci(float(ab_ic2["ci95_lo"]), float(ab_ic2["ci95_hi"])),
        "BH p_adj": "—",
        "Significant": "Yes"
    },
])
save(s7, "S7_ab_ic2_exclusion", "AB IC2 exclusion Q2 sub-analysis")

# ────────────────────────────────────────────────────────────────────────────
# S8 — EC E. hormaechei restriction
# ────────────────────────────────────────────────────────────────────────────
ec_full     = q2["ecloaceae"]
ec_horm_csv = pd.read_csv("results/supplement_ec_complex_q2_367.csv")
ec_horm     = ec_horm_csv.iloc[0]

s8 = pd.DataFrame([
    {
        "Analysis": "Full EC complex",
        "N Q2 genomes": ec_full["n_q2"],
        "N phylogroups": ec_full.get("n_phylogroups", "—"),
        "N features": ec_full.get("n_features", "—"),
        "Mean AUROC": round(ec_full["mean_auroc"], 4),
        "AUROC 95% CI": fmt_ci(*ec_full["ci95_auroc"]),
        "BH p_adj": f"{ec_full.get('p_adj', float('nan')):.4f}",
        "Significant": "Yes"
    },
    {
        "Analysis": "E. hormaechei sensu stricto",
        "N Q2 genomes": int(ec_horm["n_q2"]),
        "N phylogroups": int(ec_horm["n_phylogroups"]),
        "N features": int(ec_horm["n_features"]),
        "Mean AUROC": round(float(ec_horm["mean_auroc"]), 4),
        "AUROC 95% CI": fmt_ci(float(ec_horm["ci95_lo"]), float(ec_horm["ci95_hi"])),
        "BH p_adj": "—",
        "Significant": "Yes"
    },
])
save(s8, "S8_ec_hormaechei_restriction", "EC E. hormaechei restriction Q2 sub-analysis")

# ────────────────────────────────────────────────────────────────────────────
# S9 — Q3 silhouette scores at K=2–12 (full and dereplicated)
# ────────────────────────────────────────────────────────────────────────────
full = q3["full"]
derep = q3["dereplicated"]
k_range = full["k_range"]
sil_full = full["sil_scores"]

s9_rows = []
for k, sil in zip(k_range, sil_full):
    s9_rows.append({
        "K": k,
        "Silhouette (full, n=3335)": round(sil, 4),
        "Note": "Best K" if k == full["best_k"] else ""
    })
s9 = pd.DataFrame(s9_rows)
# Add dereplicated summary rows
derep_summary = pd.DataFrame([{
    "K": derep["best_k"],
    "Silhouette (full, n=3335)": "—",
    "Note": f"Dereplicated (n=309): best K={derep['best_k']}, sil={derep['best_k_silhouette']:.4f}; ARI vs species={derep['ari_kmeans_vs_species']:.4f} (delta from full={derep['delta_ari_vs_full']:.4f})"
}])
save(pd.concat([s9, derep_summary], ignore_index=True),
     "S9_q3_silhouette_k_sweep", "Q3 silhouette at K=2–12 (full and dereplicated)")

# ────────────────────────────────────────────────────────────────────────────
# S10 — Geographic representation per species
# ────────────────────────────────────────────────────────────────────────────
s10_rows = []
for sp in SP_ORDER:
    sp_fm = fm[fm["species"] == sp]
    country_counts = sp_fm["country"].fillna("not_reported").value_counts()
    n_countries = (country_counts.index != "not_reported").sum()
    top3 = "; ".join([f"{c} (n={n})" for c, n in
                      country_counts[country_counts.index != "not_reported"].head(3).items()])
    max_pct = country_counts[country_counts.index != "not_reported"].iloc[0] / len(sp_fm) * 100 if n_countries > 0 else float("nan")
    s10_rows.append({
        "Species": SP_LABEL[sp],
        "N genomes": len(sp_fm),
        "N countries represented": n_countries,
        "N not_reported": int(country_counts.get("not_reported", 0)),
        "Top 3 countries": top3,
        "Max single-country share (%)": round(max_pct, 1)
    })
s10 = pd.DataFrame(s10_rows)
save(s10, "S10_geographic_representation", "per-species geographic representation")

# ────────────────────────────────────────────────────────────────────────────
# S11 — Q3 clustering results summary (full vs dereplicated)
# ────────────────────────────────────────────────────────────────────────────
s11 = pd.DataFrame([
    {
        "Analysis": "Full dataset (n=3,335)",
        "Best K (silhouette)": full["best_k"],
        "Best K silhouette": round(full["best_k_silhouette"], 4),
        "Silhouette at K=6": round(full["silhouette_at_k6"], 4),
        "Gap statistic optimal K": full["gap_optimal_k"],
        "ARI vs species (best K)": round(full["ari_kmeans_bestk_vs_species"], 4),
        "ARI vs species (K=6)": round(full["ari_kmeans_k6_vs_species"], 4),
        "ARI vs ARG tertile (K=6)": round(full["ari_kmeans_k6_vs_arg"], 4),
        "ARI kmeans vs Ward (K=6)": round(full["ari_hc_ward_vs_kmeans"], 4),
    },
    {
        "Analysis": "Dereplicated (n=309 phylogroup reps)",
        "Best K (silhouette)": derep["best_k"],
        "Best K silhouette": round(derep["best_k_silhouette"], 4),
        "Silhouette at K=6": round(derep["best_k_silhouette"] if derep["best_k"] == 6 else float("nan"), 4),
        "Gap statistic optimal K": "—",
        "ARI vs species (best K)": round(derep["ari_kmeans_vs_species"], 4),
        "ARI vs species (K=6)": "—",
        "ARI vs ARG tertile (K=6)": round(derep["ari_kmeans_vs_arg"], 4),
        "ARI kmeans vs Ward (K=6)": round(derep["ari_hc_ward_vs_kmeans"], 4),
    },
])
save(s11, "S11_q3_clustering_summary", "Q3 clustering metrics full vs dereplicated")

# ────────────────────────────────────────────────────────────────────────────
# S12 — Q3b multi-block ARI and permutation (filtered values)
# ────────────────────────────────────────────────────────────────────────────
BLOCK_LABEL = {
    "defence_only": "Defence only",
    "is_filt":      "IS elements (marker-filtered)",
    "hmrg_filt":    "HMRG (marker-filtered)",
    "arg_filt":     "ARG (marker-filtered)",
    "defence_is":   "Defence + IS (combined)",
}
s12_rows = []
for key, label in BLOCK_LABEL.items():
    v = q3bf[key]
    s12_rows.append({
        "Feature block": label,
        "N features": v["n_features"],
        "ARI (K=6, dereplicated)": round(v["ari_primary"], 4),
        "ARI 95% CI": fmt_ci(*v["ari_ci95"]),
        "Permutation p": round(v["permutation_p"], 4),
        "Significant (p<0.05)": "Yes"
    })
s12 = pd.DataFrame(s12_rows)
save(s12, "S12_q3b_multiblock_ari", "Q3b multi-block clustering ARI (marker-filtered)")

# ────────────────────────────────────────────────────────────────────────────
# S13 — Q2 full per-species driver table (all species, all robust drivers)
# ────────────────────────────────────────────────────────────────────────────
s13_rows = []
for sp in SP_ORDER:
    if sp not in q2:
        continue
    for d in q2[sp].get("top_drivers", []):
        s13_rows.append({
            "Species": SP_LABEL[sp],
            "Feature": d["feature"].removeprefix("dp_"),
            "Category": d.get("category", "—"),
            "ρ (within-phylogroup)": round(d["rho_within_pg"], 4) if d.get("rho_within_pg") is not None else "—",
            "p (within-phylogroup)": f"{d['p_within_pg']:.4e}" if d.get("p_within_pg") is not None else "—",
            "Survives robustness gate": "Yes" if d.get("p_within_pg") is not None and d["p_within_pg"] < 0.05 else "No",
            "Direction": "Facilitative" if (d.get("rho_within_pg") or 0) > 0 else "Restrictive"
        })
s13 = pd.DataFrame(s13_rows)
save(s13, "S13_q2_full_driver_table", "Q2 full per-species driver table (all species, robust drivers)")

# ────────────────────────────────────────────────────────────────────────────
# S14 — Q2 IS-element covariate adjustment per driver
# ────────────────────────────────────────────────────────────────────────────
s14_rows = []
for sp, v in q2_cov.items():
    for d in v["drivers"]:
        s14_rows.append({
            "Species": SP_LABEL.get(sp, sp),
            "Feature": d["feature"].removeprefix("dp_"),
            "Category": d.get("category", "—"),
            "ρ (within-phylogroup)": round(d["rho_within_pg"], 4),
            "p (within-phylogroup)": f"{d['p_within_pg']:.4e}",
            "ρ (partial, IS-adjusted)": round(d["rho_partial"], 4),
            "p (partial)": f"{d['p_partial']:.4e}",
            "Sign preserved": "Yes" if d["sign_preserved"] else "No",
            "Survives covariate adjustment": "Yes" if d["survives_covariate_adjustment"] else "No",
        })
s14 = pd.DataFrame(s14_rows)
save(s14, "S14_q2_covariate_adjustment", "Q2 IS-element covariate adjustment per driver")

# ────────────────────────────────────────────────────────────────────────────
# S15 — Dominant phylogroup exclusion (KP, EF, PA, SA)
# ────────────────────────────────────────────────────────────────────────────
s15_rows = []
for sp, v in dpg.items():
    s15_rows.append({
        "Species": SP_LABEL.get(sp, sp),
        "Dominant phylogroup": v["top_phylogroup"],
        "Share of cohort (%)": round(v["top_pg_share"] * 100, 1),
        "N genomes excluded": v["n_excluded"],
        "N genomes retained": v["excluded_cohort_n"],
        "N Q2 genomes (retained)": v["excluded_cohort_n_q2"],
        "N phylogroups (retained)": v["excluded_cohort_n_phylogroups"],
        "Full cohort AUROC": round(v["full_cohort_auroc"], 4),
        "Post-exclusion AUROC": round(v["excluded_cohort_auroc"], 4),
        "Post-exclusion 95% CI": fmt_ci(*v["excluded_cohort_ci95"]),
        "ΔAUROC": round(v["delta_auroc"], 4),
    })
s15 = pd.DataFrame(s15_rows)
save(s15, "S15_dominant_pg_exclusion", "dominant phylogroup exclusion Q2 sensitivity")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDone. {len(os.listdir('results/tables/'))} files in results/tables/")
print("Note: S16 (holdout ST audit) is already in docs/manuscript/supplemental_material.md")
print("      S6 (Q2 unfiltered candidate robustness, s6_unfiltered_robustness.csv) already exists.")
