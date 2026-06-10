"""
Q2 per-species feature importance analysis.
Fits per-species RF Q2 models and reports:
  1. Gini importance ranking for all features (top 20 per species)
  2. Rank of each RM-related feature within each species Q2 model
  3. Spearman correlation: RM presence vs raw ARG count (direction test)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES  = ROOT / "results"

RANDOM_STATE = 42
N_SPLITS     = 5

RM_FEATURES = [
    "dp_RM_Type_I", "dp_RM_Type_II", "dp_RM_Type_IIG",
    "dp_RM_Type_III", "dp_RM_Type_IV",
    "dp_df_RM_Type_IIG_2", "dp_df_RM_Type_IV_1",
    "dp_padloc_RM_type_HNH",
    "dp_DISARM_I", "dp_DISARM_II",
]

# Best Q1 RF hyperparameters (same used for Q2 per pre-registered plan)
BEST_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=1,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)


def main():
    fm = pd.read_parquet(PROC / "feature_matrix_3460.parquet")
    dp_cols = sorted([c for c in fm.columns if c.startswith("dp_")])

    # Specificity filter identical to Q1 (remove 8 taxonomic markers)
    sp_prev    = fm.groupby("species")[dp_cols].mean()
    spec_score = sp_prev.std() / 0.5
    markers    = spec_score[spec_score >= 0.70].index.tolist()
    base_feats = [c for c in dp_cols if c not in markers]  # 352 features

    print("=" * 70)
    print("Q2 PER-SPECIES RF FEATURE IMPORTANCE")
    print("=" * 70)

    gini_records   = []
    spearman_rows  = []

    for sp in sorted(fm["species"].unique()):
        fm_sp   = fm[fm["species"] == sp]
        mask_q2 = fm_sp["arg_burden_tertile"].isin(["low_ARG", "high_ARG"])
        fm_q2   = fm_sp[mask_q2].copy()

        y_sp  = (fm_q2["arg_burden_tertile"] == "high_ARG").astype(int).values
        grp   = fm_q2["phylogroup"].to_numpy(dtype=str)

        # H1: per-species sparsity filter (>5% prevalence in Q2 genomes)
        prev_sp = fm_q2[base_feats].mean()
        feat_sp = [f for f in base_feats if prev_sp[f] >= 0.05]

        X = fm_q2[feat_sp].to_numpy(dtype=float)

        print(f"\n{'─'*60}")
        print(f"  {sp.upper()}  |  Q2 n={len(fm_q2)}  |  features={len(feat_sp)}")
        print(f"  high_ARG={y_sp.sum()}  low_ARG={(1-y_sp).sum()}")

        # Fit RF on full Q2 set for Gini importance
        rf = RandomForestClassifier(**BEST_PARAMS)
        rf.fit(X, y_sp)

        imp = pd.Series(rf.feature_importances_, index=feat_sp).sort_values(ascending=False)
        imp_df = imp.reset_index()
        imp_df.columns = ["feature", "gini_importance"]
        imp_df["rank"]    = range(1, len(imp_df) + 1)
        imp_df["species"] = sp
        gini_records.append(imp_df)

        # Report top 20
        print(f"\n  Top 20 features by Gini importance:")
        print(f"  {'Rank':>4}  {'Feature':<35}  {'Gini':>8}")
        print(f"  {'─'*4}  {'─'*35}  {'─'*8}")
        for _, row in imp_df.head(20).iterrows():
            marker = "  *** RM/DISARM ***" if row["feature"] in RM_FEATURES else ""
            print(f"  {int(row['rank']):>4}  {row['feature']:<35}  {row['gini_importance']:>8.5f}{marker}")

        # RM feature ranks
        rm_present = [f for f in RM_FEATURES if f in feat_sp]
        rm_absent  = [f for f in RM_FEATURES if f not in feat_sp]
        if rm_present:
            print(f"\n  RM/DISARM feature ranks within {sp}:")
            for f in rm_present:
                if f in imp.index:
                    rank = imp.index.get_loc(f) + 1
                    val  = imp[f]
                    print(f"    {f:<35}  rank={rank:>3}/{len(feat_sp)}  gini={val:.5f}")
        if rm_absent:
            print(f"\n  RM/DISARM features below 5% prevalence (excluded from model):")
            for f in rm_absent:
                prev = fm_q2[f].mean() if f in fm_q2.columns else 0
                print(f"    {f:<35}  prevalence={prev:.3f}")

        # Spearman: RM presence vs total ARG count (direction test)
        # Use raw total_arg_count on Q2 + mid genomes (full species set for direction)
        print(f"\n  Spearman: RM feature presence vs total_arg_count (full species, n={len(fm_sp)}):")
        arg_counts = fm_sp["arg_count_total"].values
        for f in RM_FEATURES:
            if f not in fm_sp.columns:
                continue
            prev_full = fm_sp[f].mean()
            if prev_full < 0.02:  # skip near-zero prevalence
                continue
            rho, pval = spearmanr(fm_sp[f].values, arg_counts)
            spearman_rows.append({
                "species": sp, "feature": f, "rho": rho,
                "p_raw": pval, "prevalence": prev_full
            })
            direction = "RESTRICT (negative)" if rho < 0 else "FACILITATE (positive)"
            print(f"    {f:<35}  rho={rho:+.3f}  p={pval:.4f}  {direction}")

    # BH correction on all Spearman tests
    sp_df = pd.DataFrame(spearman_rows)
    if len(sp_df) > 0:
        _, p_adj, _, _ = multipletests(sp_df["p_raw"].values, method="fdr_bh")
        sp_df["p_adj_bh"] = p_adj
        sp_df["sig"] = sp_df["p_adj_bh"] < 0.05

        print("\n\n" + "=" * 70)
        print("SPEARMAN SUMMARY (BH-corrected): RM/DISARM vs ARG count")
        print("=" * 70)
        print(f"{'Species':<14} {'Feature':<35} {'rho':>6} {'p_adj':>8} {'Sig':>5}")
        print("─" * 70)
        for _, row in sp_df.sort_values(["species", "rho"]).iterrows():
            sig_marker = "YES" if row["sig"] else "ns"
            print(f"{row['species']:<14} {row['feature']:<35} {row['rho']:>+6.3f} "
                  f"{row['p_adj_bh']:>8.4f} {sig_marker:>5}")

    # Save Gini importance table
    all_gini = pd.concat(gini_records, ignore_index=True)
    out_path = RES / "q2_rf_feature_importance_3460.csv"
    all_gini.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Save Spearman summary
    sp_out = RES / "q2_rm_spearman_3460.csv"
    sp_df.to_csv(sp_out, index=False)
    print(f"Saved: {sp_out}")


if __name__ == "__main__":
    main()
