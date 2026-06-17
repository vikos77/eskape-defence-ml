"""
Q2 seed-stability sweep: how robust is the dp_ vs dc_ Q2 result to CV fold-split
randomness?

Motivation: a single-seed run (random_state=42) showed E. faecium crossing the
p<0.05 (BH-corrected) threshold under dc_ (count) encoding but not dp_
(presence) encoding. A 5-seed spot check showed both encodings' EF p_raw
swinging from 0.004 to 0.19 depending on the seed -- i.e. the apparent
"counts rescue EF" result may just be fold-split lottery. This script runs
N_SEEDS independent CV splits per species per encoding to characterize that
properly: what fraction of seeds give p_raw < 0.05, and how much does AUROC
vary, for dp_ vs dc_ side by side.

Feature SET is identical between dp_ and dc_ runs (same >=5% prevalence
filter via dp_, see run_q2_dc_sensitivity.py) -- only the encoding differs.

Outputs:
  results/q2_seed_stability.json
"""

import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

N_SEEDS     = 20
SEEDS       = list(range(N_SEEDS))
N_FOLDS     = 5
PREV_THRESH = 0.05
RF_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : None,
    "min_samples_leaf": 1,
    "max_features"    : "sqrt",
    "class_weight"    : "balanced",
    "n_jobs"          : -1,
}

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]

species_list = sorted(fm["species"].unique())
all_results = {}

for sp in species_list:
    sp_df = fm[fm.species == sp].copy()
    q2_df = sp_df[sp_df.arg_burden_tertile.isin(["high_ARG", "low_ARG"])].copy()
    q2_df["y_bin"] = (q2_df.arg_burden_tertile == "high_ARG").astype(int)

    feat_prev  = q2_df[dp_cols].mean()
    sp_feat_dp = feat_prev[feat_prev >= PREV_THRESH].index.tolist()
    sp_feat_dc = [c.replace("dp_", "dc_", 1) for c in sp_feat_dp]

    y      = q2_df["y_bin"].to_numpy()
    groups = q2_df["phylogroup"].to_numpy(dtype=str)
    n_pg   = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_pg)
    if n_splits < 2:
        print(f"{sp}: only {n_pg} phylogroups, skipping")
        continue

    print(f"\n{sp}: n_q2={len(q2_df)}, n_features={len(sp_feat_dp)}, n_phylogroups={n_pg}")
    all_results[sp] = {}

    for encoding, feats in [("dp", sp_feat_dp), ("dc", sp_feat_dc)]:
        X = q2_df[feats].to_numpy(dtype=float)
        seed_p_raw  = []
        seed_auroc  = []
        seed_ba     = []

        for seed in SEEDS:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_bas, fold_auroc = [], []
            for tr, te in cv.split(X, y, groups=groups):
                rf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
                rf.fit(X[tr], y[tr])
                y_pred = rf.predict(X[te])
                y_prob = rf.predict_proba(X[te])[:, 1]
                fold_bas.append(balanced_accuracy_score(y[te], y_pred))
                fold_auroc.append(roc_auc_score(y[te], y_prob))
            fold_bas = np.array(fold_bas)
            _, p_raw = stats.ttest_1samp(fold_bas, 0.5)
            seed_p_raw.append(float(p_raw))
            seed_auroc.append(float(np.mean(fold_auroc)))
            seed_ba.append(float(np.mean(fold_bas)))

        seed_p_raw = np.array(seed_p_raw)
        seed_auroc = np.array(seed_auroc)
        seed_ba    = np.array(seed_ba)
        frac_sig   = float((seed_p_raw < 0.05).mean())

        print(f"  {encoding}: mean_AUROC={seed_auroc.mean():.3f} (sd={seed_auroc.std():.3f})  "
              f"mean_BA={seed_ba.mean():.3f} (sd={seed_ba.std():.3f})  "
              f"frac_seeds_p<0.05={frac_sig:.2f}  "
              f"p_raw range=[{seed_p_raw.min():.3f}, {seed_p_raw.max():.3f}]")

        all_results[sp][encoding] = {
            "n_features"     : len(feats),
            "seed_p_raw"     : seed_p_raw.tolist(),
            "seed_auroc"     : seed_auroc.tolist(),
            "seed_ba"        : seed_ba.tolist(),
            "mean_auroc"     : float(seed_auroc.mean()),
            "sd_auroc"       : float(seed_auroc.std()),
            "mean_ba"        : float(seed_ba.mean()),
            "sd_ba"          : float(seed_ba.std()),
            "frac_seeds_sig" : frac_sig,
        }

with open("results/q2_seed_stability.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*70}")
print("SUMMARY: fraction of 20 seeds with raw p<0.05, dp_ vs dc_")
print(f"{'species':<14}{'frac_sig(dp)':>14}{'frac_sig(dc)':>14}{'AUROC(dp)':>12}{'AUROC(dc)':>12}")
for sp in species_list:
    if sp not in all_results:
        continue
    r = all_results[sp]
    print(f"{sp:<14}{r['dp']['frac_seeds_sig']:>14.2f}{r['dc']['frac_seeds_sig']:>14.2f}"
          f"{r['dp']['mean_auroc']:>12.3f}{r['dc']['mean_auroc']:>12.3f}")

print("\nSaved: results/q2_seed_stability.json")
