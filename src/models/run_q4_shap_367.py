"""
Q4 SHAP interpretation — full 367-feature defence-system matrix (359
FEAT_COLS). RF params read dynamically from q1_367_results.json.

Every feature in the global/per-class top-20 output is tagged named/unnamed
(PDC/DS-N/All_UG/catch-all categories: PADLOC cluster IDs, DefensePredictor ML
candidates, DefenseFinder uncharacterised groups, and tool catch-alls, none
with an assigned mechanism class). Per the citability-hedging requirement: an
unnamed top feature can be reported as a statistically real SHAP driver, but
no mechanistic claim may be attached to it (e.g. "PDC-S04, an uncharacterised
PADLOC candidate cluster, ranks Nth -- mechanism unknown" vs "RM_Type_IV, a
published restriction-modification system, ranks 1st").

Fold-averaged SHAP: for each of 5 GroupedStratifiedKFold folds, train a fresh
RF (367 best params) on the training partition, compute SHAP on the held-out
test partition, reassemble into (3335, 359, 6). Every genome's SHAP values come
from a model that never saw it in training.

Outputs:
  results/q4_shap_367_global.csv     — feature, global_shap, global_rank, category, is_named
  results/q4_shap_367_perclass.csv   — feature, species, mean_abs_shap, rank, category, is_named
  results/q4_shap_367_array.npy      — raw (3335, 359, 6) SHAP array

Prints:
  - Global top-20 features (named/unnamed tagged)
  - Per-class top-10 for all 6 species (named/unnamed tagged)
  - Alignment table: published Fisher's exact systems vs 367 SHAP ranks
  - Ranks for RM, SspBCDE, Gao_Qat specifically
"""

import json
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
import shap

warnings.filterwarnings("ignore")

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

ROOT         = Path(__file__).resolve().parents[2]
PROC         = ROOT / "data" / "processed"
RES          = ROOT / "results"
RANDOM_STATE = 42
N_SPLITS     = 5

SPECIES_ORDER = [
    "abaumannii", "ecloaceae", "efaecium",
    "kpneumoniae", "paeruginosa", "saureus",
]

# ── Load matrix and derive FEAT_COLS ─────────────────────────────────────────
fm      = pd.read_parquet(PROC / "feature_matrix_3335.parquet")
dp_cols = sorted([c for c in fm.columns if c.startswith("dp_")])

sp_prev    = fm.groupby("species")[dp_cols].mean()
spec_score = sp_prev.std() / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]

X      = fm[FEAT_COLS].to_numpy(dtype=float)
y      = fm["species"].to_numpy(dtype=str)
groups = fm["phylogroup"].to_numpy(dtype=str)

# Load best params from prior Q1 run
with open(RES / "q1_367_results.json") as f:
    _res = json.load(f)
BEST_PARAMS = _res["best_params"]

# Derive sorted class list to match sklearn's label ordering
CLASSES = sorted(fm["species"].unique())   # alphabetical: same as sklearn

print(f"Feature matrix: {X.shape[0]} genomes x {X.shape[1]} FEAT_COLS")
print(f"Markers removed: {markers}")
print(f"Classes: {CLASSES}")
print(f"Best params: {BEST_PARAMS}")
print()

# ── Fold-averaged SHAP ────────────────────────────────────────────────────────
n_genomes, n_feats = X.shape
n_classes          = len(CLASSES)

shap_3d = np.full((n_genomes, n_feats, n_classes), np.nan)

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups=groups)):
    print(f"Fold {fold_i + 1}/{N_SPLITS}: train={len(tr_idx)}, test={len(te_idx)}", end="  ")

    rf_fold = RandomForestClassifier(
        **BEST_PARAMS,
        class_weight = "balanced",
        n_jobs       = -1,
        random_state = RANDOM_STATE + fold_i,
    )
    rf_fold.fit(X[tr_idx], y[tr_idx])

    explainer = shap.TreeExplainer(rf_fold)
    sv = explainer.shap_values(X[te_idx], check_additivity=False)

    if isinstance(sv, list):          # older SHAP: list of (n, p) per class
        sv_3d = np.stack(sv, axis=2)
    else:                             # newer SHAP >= 0.46: (n, p, c)
        sv_3d = sv

    shap_3d[te_idx] = sv_3d
    print("done")

assert not np.isnan(shap_3d).any(), "Some genomes are missing SHAP values"

# ── Save raw array ────────────────────────────────────────────────────────────
np.save(RES / "q4_shap_367_array.npy", shap_3d)
print(f"\nSaved: {RES / 'q4_shap_367_array.npy'}  shape={shap_3d.shape}")

# ── Global SHAP: mean |SHAP| over samples and classes ─────────────────────────
mean_abs_global = np.abs(shap_3d).mean(axis=(0, 2))   # (n_feats,)
global_shap     = pd.Series(mean_abs_global, index=FEAT_COLS).sort_values(ascending=False)
global_df       = global_shap.reset_index()
global_df.columns = ["feature", "global_shap"]
global_df["global_rank"] = range(1, len(global_df) + 1)
global_df["category"] = global_df["feature"].apply(categorize)
global_df["is_named"] = global_df["category"] == "named"

global_df.to_csv(RES / "q4_shap_367_global.csv", index=False)
print(f"Saved: {RES / 'q4_shap_367_global.csv'}")

# ── Per-class SHAP ────────────────────────────────────────────────────────────
perclass_rows = []
for cls_idx, sp in enumerate(CLASSES):
    shap_cls   = np.abs(shap_3d[:, :, cls_idx]).mean(axis=0)   # (n_feats,)
    ranked     = pd.Series(shap_cls, index=FEAT_COLS).sort_values(ascending=False)
    for rank_i, (feat, val) in enumerate(ranked.items(), start=1):
        cat = categorize(feat)
        perclass_rows.append({
            "species":       sp,
            "feature":       feat,
            "mean_abs_shap": val,
            "rank":          rank_i,
            "category":      cat,
            "is_named":      cat == "named",
        })

perclass_df = pd.DataFrame(perclass_rows)
perclass_df.to_csv(RES / "q4_shap_367_perclass.csv", index=False)
print(f"Saved: {RES / 'q4_shap_367_perclass.csv'}")

# ── Print global top-20 (named/unnamed tagged) ─────────────────────────────────
print()
print("=" * 80)
print("GLOBAL TOP-20 FEATURES  (mean |SHAP| over 3335 genomes x 6 classes)")
print(f"{'Rank':>4}  {'Feature':<35}  {'Mean |SHAP|':>11}  {'Category':<10}")
print("-" * 80)
for _, row in global_df.head(20).iterrows():
    print(f"  {int(row['global_rank']):>2}  {row['feature']:<35}  {row['global_shap']:.5f}      {row['category']:<10}")
n_unnamed_top20 = (global_df.head(20)["category"] != "named").sum()
print(f"\nUnnamed (PDC/DS-N/All_UG/catch-all) features in global top-20: {n_unnamed_top20}/20")
print("Citability note: unnamed features above are statistically real SHAP drivers")
print("but carry no mechanistic claim -- report as 'uncharacterised candidate, mechanism")
print("unknown', not as a named system would be reported.")

# ── Per-class top-10 (named/unnamed tagged) ────────────────────────────────────
print()
print("=" * 80)
print("PER-CLASS TOP-10 FEATURES")
print("=" * 80)
for sp in SPECIES_ORDER:
    sub = perclass_df[perclass_df["species"] == sp].head(10)
    print(f"\n  {sp.upper()}")
    print(f"  {'Rank':>4}  {'Feature':<35}  {'Mean |SHAP|':>11}  {'Category':<10}")
    print("  " + "-" * 65)
    for _, row in sub.iterrows():
        print(f"  {int(row['rank']):>4}  {row['feature']:<35}  {row['mean_abs_shap']:.5f}      {row['category']:<10}")

# ── Alignment table: published Fisher's exact vs full-367 SHAP ───────────────
PUBLISHED_TO_DP = {
    "RM":          [c for c in FEAT_COLS if "RM_Type" in c or c == "dp_RM"],
    "SspBCDE":     ["dp_SspBCDE"],
    "Gao_Qat":     ["dp_Gao_Qat"],
    "CBASS":       [c for c in FEAT_COLS if "CBASS" in c],
    "Cas":         [c for c in FEAT_COLS if "CAS_" in c or "cas_" in c],
    "Gabija":      ["dp_Gabija"]    if "dp_Gabija"    in FEAT_COLS else [],
    "DRT":         [c for c in FEAT_COLS if "DRT" in c],
    "Paris":       [c for c in FEAT_COLS if "PARIS" in c],
    "Retron":      [c for c in FEAT_COLS if "Retron" in c],
    "RosmerTA":    ["dp_RosmerTA"]  if "dp_RosmerTA"  in FEAT_COLS else [],
    "Septu":       ["dp_Septu"]     if "dp_Septu"     in FEAT_COLS else [],
    "Lamassu-Fam": ["dp_Lamassu"]   if "dp_Lamassu"   in FEAT_COLS else [],
    "RloC":        [c for c in FEAT_COLS if "RloC" in c],
    "Mokosh":      [c for c in FEAT_COLS if "Mokosh" in c],
    "AbiD":        [c for c in FEAT_COLS if "AbiD" in c],
    "AbiH":        [c for c in FEAT_COLS if "AbiH" in c],
    "AbiJ":        [c for c in FEAT_COLS if "AbiJ" in c],
    "PD-T4-5":     [c for c in FEAT_COLS if "PD-T4-5" in c],
    "PD-T7-2":     [c for c in FEAT_COLS if "PD-T7-2" in c],
    "PD-T7-5":     [c for c in FEAT_COLS if "PD-T7-5" in c],
}

# AB-class SHAP ranks for fair comparison to published AB-only Fisher's exact
ab_idx     = CLASSES.index("abaumannii")
shap_ab    = np.abs(shap_3d[:, :, ab_idx]).mean(axis=0)
shap_ab_sr = pd.Series(shap_ab, index=FEAT_COLS).sort_values(ascending=False)
ab_rank    = {feat: int(rank) for rank, feat in enumerate(shap_ab_sr.index, start=1)}
global_rank = {row["feature"]: int(row["global_rank"]) for _, row in global_df.iterrows()}

print()
print("=" * 75)
print("ALIGNMENT TABLE  — published Fisher's exact systems vs full-367 SHAP")
print(f"{'System':<16}  {'Best dp_ feature':<35}  {'Global rank':>11}  {'AB rank':>7}")
print("-" * 75)

for pub_sys, dp_feats in PUBLISHED_TO_DP.items():
    present = [f for f in dp_feats if f in FEAT_COLS]
    if not present:
        print(f"  {pub_sys:<14}  {'(not in 367 FEAT_COLS)':<35}  {'—':>11}  {'—':>7}")
        continue
    best_global = min(present, key=lambda f: global_rank.get(f, 9999))
    best_ab     = min(present, key=lambda f: ab_rank.get(f, 9999))
    gr = global_rank.get(best_global, 9999)
    ar = ab_rank.get(best_ab, 9999)
    feat_label = best_global if best_global == best_ab else f"{best_global} / {best_ab}"
    print(f"  {pub_sys:<14}  {best_global:<35}  {gr:>11}  {ar:>7}")

# ── Key systems summary for discussion.md ────────────────────────────────────
print()
print("=" * 68)
print("KEY SYSTEMS FOR DISCUSSION.MD")
print("=" * 68)
key_systems = {
    "RM":      [c for c in FEAT_COLS if "RM_Type" in c],
    "SspBCDE": ["dp_SspBCDE"],
    "Gao_Qat": ["dp_Gao_Qat"],
}
for pub_sys, dp_feats in key_systems.items():
    present = [f for f in dp_feats if f in FEAT_COLS]
    print(f"\n  {pub_sys}:")
    for feat in present:
        gr = global_rank.get(feat, 9999)
        ar = ab_rank.get(feat, 9999)
        print(f"    {feat:<35}  global_rank={gr:>3}  AB_rank={ar:>3}  "
              f"global_shap={global_shap.get(feat, 0):.5f}")

print()
print("Done.")
