#!/usr/bin/env python
"""
Build notebooks/10_phase12_sensitivity.ipynb for the 3335-genome cohort.
Run from eskape-defence-ml project root:
    conda run -n eskape-ml python src/models/build_nb10.py

NOTE (2026-06-16): Tests A and B were dropped from the final manuscript.
This notebook is retained as a historical record and for potential future work.
The Q2 baselines in the TITLE cell below are from the 367-feature analysis
(pre-named-236 filter); the named-236 Q2 result is 0/6 species significant.

Key differences from the 878-genome original:
- feature_matrix_3335.parquet  (3335 × 791, 367 dp_ features before named-236 filter)
- cv_groups_3460.parquet joined for phylogroup  (309 groups)
- q2_gb_results_3460.parquet and q2_rf_results_3460.parquet
- Test A baseline uses dp-RF (same function, same model) not XGBoost parquet
- ResFinder rf_long intersected with fm.index (guard against 31 leaked Ab dirs)
- SHAP scope expanded to all significant species
"""
import nbformat

def md(src):  return nbformat.v4.new_markdown_cell(src)
def code(src): return nbformat.v4.new_code_cell(src)


TITLE = """\
# NB10 — Phase 12: Mechanism-Level ARG Burden & RM Sensitivity Analysis (3,335-genome cohort)

**Pre-registered:** 2026-05-27. Full specification locked before touching NB10.

**NOTE (2026-06-16):** Tests A and B were dropped from the final manuscript after the
feature matrix was restricted to 236 named systems (named-236 filter). On the named-236
matrix, Q2 produces 0/6 significant species (all p_adj > 0.10), making Test A's
differential comparison non-applicable. This notebook remains for historical record.

Two tests, both independent of IS element data (ISEScan not required):

**Test A:** Does replacing binary RM presence (dp_RM_*) with copy-count (dc_RM_*) in the
Q2 feature set improve ARG burden prediction? Tests whether RM restriction is a dose effect.

**Test B:** Can defence system profile predict mechanism-class-specific ARG burden
(β-lactam, aminoglycoside, etc.) rather than total ARG burden? Tests whether RM
restriction operates selectively on plasmid-borne classes.

Dataset: 3,335 genomes × 367 dp_ features (original matrix, pre-named-236 filter).
Q2 XGBoost baselines (367-feature matrix): Ab=0.767*(p=0.049), EC=0.762*,
Ef=0.701*, KP=0.793*, PA=0.790*, SA=0.630 ns. [Named-236 Q2: 0/6 significant.]
"""

IMPORTS = """\
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from scipy import stats as sp_stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import shap

warnings.filterwarnings("ignore")

# ── PATH GUARD ────────────────────────────────────────────────────────────────
ROOT = Path("..").resolve()
assert ROOT.name == "eskape-defence-ml", (
    f"Wrong project root: {ROOT}\\n"
    "Run this notebook from eskape-defence-ml/notebooks/"
)
PROC = ROOT / "data" / "processed"
INTER = ROOT / "data" / "interim"
CONF = ROOT / "config"
RES  = ROOT / "results"
FIG  = RES / "figures" / "phase12"
FIG.mkdir(parents=True, exist_ok=True)

assert (PROC / "feature_matrix_3335.parquet").exists(), "FM not found"
assert (PROC / "cv_groups_3460.parquet").exists(), "cv_groups not found"
assert (RES / "q2_gb_results_3460.parquet").exists(), "Q2 GB parquet not found"
assert (RES / "q2_rf_results_3460.parquet").exists(), "Q2 RF parquet not found"
assert (CONF / "arg_class_mapping.yaml").exists(), "arg_class_mapping.yaml not found"

RANDOM_STATE = 42
N_SPLITS     = 5
N_BOOT       = 2000

# Phase 7 best RF hyperparameters (frozen — same model spec for all NB10 runs)
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=20,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
print("Imports OK.")
print(f"Project root : {ROOT}")
"""

LOAD_FM = """\
## Section 2 — Load data and reproduce Phase 7 feature selection

Recomputing spec_score from scratch (not caching) to guarantee FEAT_COLS is identical
to what was used in NB06-NB08. The 0.70 threshold removes 8 taxonomic markers.
"""

LOAD_FM_CODE = """\
fm = pd.read_parquet(PROC / "feature_matrix_3335.parquet")
fm["accession"] = fm.index   # expose index as column for downstream joins

# Ensure phylogroup column is present (may already be in FM or needs joining)
if "phylogroup" not in fm.columns:
    pg = pd.read_parquet(PROC / "cv_groups_3460.parquet")
    fm = fm.join(pg.iloc[:, 0].rename("phylogroup"))
assert fm["phylogroup"].notna().all(), "Some genomes missing phylogroup assignment"

groups = fm["phylogroup"].to_numpy(dtype=str)   # 309 phylogroups

# ── Reproduce Phase 7 specificity filter ─────────────────────────────────────
dp_cols    = sorted([c for c in fm.columns if c.startswith("dp_")])
sp_prev    = fm.groupby("species")[dp_cols].mean()
spec_score = sp_prev.std() / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]

print(f"Feature matrix: {fm.shape[0]} genomes x {len(FEAT_COLS)} filtered dp_ features")
print(f"Markers removed ({len(markers)}): {markers}")
print(f"Phylogroups: {len(np.unique(groups))}")
print()
print("Species distribution:")
for sp, n in fm["species"].value_counts().items():
    print(f"  {sp:<22} {n:>4}")
"""

RM_CHECK_HEADING = """\
## Section 3 — Pre-check: are RM count (dc_) and presence (dp_) equivalent?

For Test A to be meaningful, at least one RM subtype must vary in copy number across
genomes (dc ≠ dp in ≥10% of genomes). If all RM subtypes are effectively binary
(one copy or zero), substituting dc for dp adds no information and Test A is moot.

*Moot = the test can be run but its result is predetermined: ΔAUROC ≈ 0.*
*Live = dc genuinely differs from dp; count-based prediction is possible.*
"""

RM_CHECK_CODE = """\
dc_rm_cols = [c for c in fm.columns if c.startswith("dc_RM")]
dp_rm_cols = [c.replace("dc_", "dp_") for c in dc_rm_cols]

rows = []
for dc_col, dp_col in zip(dc_rm_cols, dp_rm_cols):
    dc = fm[dc_col]
    dp = fm[dp_col]
    n_diff    = (dc != dp).sum()
    pct_diff  = n_diff / len(fm) * 100
    max_count = dc.max()
    count_dist = dc.value_counts().sort_index().to_dict()
    rows.append({
        "RM subtype":             dc_col.replace("dc_", ""),
        "Genomes where dc≠dp":    n_diff,
        "% differ":               round(pct_diff, 1),
        "Max dc value":           int(max_count),
        "Count distribution (dc)": str(count_dist),
    })

precheck = pd.DataFrame(rows)
print(precheck.to_string(index=False))
print()

moot_cols = [r["RM subtype"] for _, r in precheck.iterrows() if r["% differ"] < 10]
live_cols = [r["RM subtype"] for _, r in precheck.iterrows() if r["% differ"] >= 10]
print("Test A assessment:")
print(f"  Live (dc≠dp ≥10%): {live_cols}")
print(f"  Moot (dc≈dp <10%): {moot_cols}")
if len(live_cols) == 0:
    print("  VERDICT: Test A is MOOT — all RM subtypes are effectively binary. Report as finding.")
else:
    print("  VERDICT: Test A is LIVE — multi-copy RM variation warrants count-feature test.")
"""

RM_VIZ_CODE = """\
# Visualise dc distribution for live RM subtype(s)
if live_cols:
    fig, axes = plt.subplots(1, len(live_cols), figsize=(4 * len(live_cols), 4))
    if len(live_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, live_cols):
        vc = fm[f"dc_{col}"].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color="steelblue", alpha=0.8)
        ax.set_title(f"dc_{col} distribution")
        ax.set_xlabel("Copy number"); ax.set_ylabel("Genome count")
    plt.tight_layout()
    plt.savefig(FIG / "testa_rm_count_dist.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: testa_rm_count_dist.png")
else:
    print("No live RM subtypes — skipping distribution visualisation.")
"""

TESTA_HEADING = """\
## Section 4 — Test A: replace dp_RM_* with dc_RM_* in Q2 feature set

Only *live* RM subtypes (dc≠dp in ≥10% of genomes) are swapped. Moot subtypes stay as
dp_* to keep feature-count constant and avoid confounding the comparison.

**Scope (367-feature analysis, historical):** All species where Q2 XGBoost AUROC was
BH-significant (q<0.05) on the 367-feature matrix: Ab (0.767, p=0.049), EC (0.762,
p=0.007), Ef (0.701, p=0.024), KP (0.793, p=0.015), PA (0.790, p=0.008). SA (0.630,
p=0.215) was out of scope. NOTE: on named-236, 0/6 species reach significance; Test A
is not applied in the final analysis.
"""

TESTA_FEATCOLS = """\
# Build Test A feature list — only swap live RM subtypes
live_rm_dp = {f"dp_{c}" for c in live_cols}
FEAT_COLS_A = []
for c in FEAT_COLS:
    if c in live_rm_dp:
        FEAT_COLS_A.append(c.replace("dp_", "dc_"))
    else:
        FEAT_COLS_A.append(c)

assert len(FEAT_COLS_A) == len(FEAT_COLS), "Feature count must not change in Test A"
swapped = [a for a, b in zip(FEAT_COLS_A, FEAT_COLS) if a != b]
print(f"Test A feature count: {len(FEAT_COLS_A)} (same as dp_ set)")
print(f"Swapped ({len(swapped)}): {swapped}")
print(f"Unchanged: {len(FEAT_COLS_A) - len(swapped)}")
"""

TESTA_RUN_HEADING = """\
## Section 5 — Test A: Q2 Random Forest with dp_ baseline and dc_RM features

**Advisor fix:** Both the dp_ baseline and Test A (dc_RM) use the SAME function,
hyperparameters, GroupKFold, and H1 sparsity filter. This isolates the feature swap
(dp→dc) as the only variable. Comparing dc-RF to the XGBoost parquet would confound
model-type with feature-type.

The XGBoost Q2 headline numbers (NB07) are loaded for context only, not as Test A reference.
"""

TESTA_RUN_CODE = """\
# ── XGBoost Q2 baselines (context only — not Test A reference) ───────────────
_gb = pd.read_parquet(RES / "q2_gb_results_3460.parquet")
_rf = pd.read_parquet(RES / "q2_rf_results_3460.parquet")
def _auroc(df, sp, model):
    row = df[(df["species"] == sp) & (df["model"] == model)]
    return float(row["auroc"].values[0]) if len(row) else None

xgb_q2_auroc = {sp: _auroc(_gb, sp, "XGBoost") for sp in fm["species"].unique()}
print("Q2 XGBoost headline (NB07, context only):")
for sp, v in sorted(xgb_q2_auroc.items()):
    sig = "*" if _gb.loc[_gb.species==sp, "p_adj_bh"].values[0] < 0.05 else " "
    print(f"  {sp:<22} {f'{v:.3f}' if v else '  -  '} {sig}")

# ── Core function: dp- or dc-RF Q2 per species ────────────────────────────────
def bootstrap_ci(scores, n_boot=N_BOOT, alpha=0.05, rng=42):
    rng = np.random.default_rng(rng)
    boot = [np.mean(rng.choice(scores, size=len(scores), replace=True))
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(scores)), lo, hi

def q2_rf_per_species(feat_cols, label_col="arg_burden_tertile"):
    # Run Q2 RF for all 6 species; return dict of AUROC + CI.
    results = {}
    cv = GroupKFold(n_splits=N_SPLITS)
    for sp in fm["species"].unique():
        sp_mask = fm["species"] == sp
        sp_fm   = fm[sp_mask].copy()
        sp_grp  = groups[sp_mask]
        q2_mask = sp_fm[label_col].isin(["high_ARG", "low_ARG"])
        if q2_mask.sum() < 60:
            results[sp] = {"auroc": None, "ci": (None, None),
                           "n": int(q2_mask.sum()), "note": "n<60"}
            continue
        sp_q2  = sp_fm[q2_mask].copy()
        grp_q2 = sp_grp[q2_mask.to_numpy()]
        y_q2   = (sp_q2[label_col] == "high_ARG").astype(int).to_numpy()
        feat_keep = [c for c in feat_cols
                     if c in sp_q2.columns and sp_q2[c].mean() >= 0.05]
        if len(feat_keep) == 0:
            results[sp] = {"auroc": None, "ci": (None, None),
                           "n": int(q2_mask.sum()), "note": "no features after H1"}
            continue
        X_q2 = sp_q2[feat_keep].to_numpy(dtype=float)
        rf   = RandomForestClassifier(**RF_PARAMS)
        fold_aurocs = []
        for train_idx, test_idx in cv.split(X_q2, y_q2, groups=grp_q2):
            if len(np.unique(y_q2[test_idx])) < 2:
                continue
            rf.fit(X_q2[train_idx], y_q2[train_idx])
            proba = rf.predict_proba(X_q2[test_idx])[:, 1]
            fold_aurocs.append(roc_auc_score(y_q2[test_idx], proba))
        if len(fold_aurocs) < 3:
            results[sp] = {"auroc": None, "ci": (None, None),
                           "n": int(q2_mask.sum()), "note": "insufficient folds"}
            continue
        mean_a, lo, hi = bootstrap_ci(np.array(fold_aurocs))
        results[sp] = {
            "auroc":  round(mean_a, 3),
            "ci":     (round(lo, 3), round(hi, 3)),
            "n_feat": len(feat_keep),
            "n":      int(q2_mask.sum()),
            "note":   "ok",
        }
    return results

print()
print("Running dp-RF baseline (Test A reference)... (~2 min)")
dp_baseline = q2_rf_per_species(FEAT_COLS)
print("Running Test A dc-RF ... (~2 min)")
testa_results = q2_rf_per_species(FEAT_COLS_A)
print("Done.")
"""

TESTA_DISPLAY = """\
# Display Test A results vs dp-RF baseline (fair RF-vs-RF comparison)
print("\\nTest A: dc_RM vs dp_RM  (both RF, same hyperparams, same CV)")
print()
header = (f"{'Species':<22} {'dp AUROC':>10} {'dc AUROC':>10} "
          f"{'95% CI':>16} {'Δ AUROC':>9} {'N feat':>7}")
print(header)
print("-" * len(header))
for sp in sorted(testa_results.keys()):
    r    = testa_results[sp]
    dp_r = dp_baseline.get(sp, {})
    dp_auroc = dp_r.get("auroc")
    if r["auroc"] is None:
        print(f"{sp:<22} {'   -  ':>10} {'  skipped':>10}")
        continue
    ci_str    = f"[{r['ci'][0]:.3f}–{r['ci'][1]:.3f}]"
    delta     = r["auroc"] - dp_auroc if dp_auroc else None
    delta_str = f"{delta:+.3f}" if delta is not None else "   -  "
    dp_str    = f"{dp_auroc:.3f}" if dp_auroc else "  -  "
    print(f"{sp:<22} {dp_str:>10} {r['auroc']:>10.3f} {ci_str:>16} {delta_str:>9} {r['n_feat']:>7}")
"""

TESTA_SHAP_HEADING = """\
## Section 6 — Test A: SHAP for RM count features

For species where Test A was live, does dc_RM rank higher in SHAP than dp_RM would?
If RM count is more informative than RM presence, its SHAP rank should increase.
SHAP is computed in-sample (full Q2 data) for explanatory ranking only — not for accuracy.
"""

TESTA_SHAP_CODE = """\
shap_results_a = {}
cv_shap = GroupKFold(n_splits=N_SPLITS)

# Run SHAP for all significant species (Ab/EC/Ef/KP/PA) — scope expanded vs 878 study
sig_species = [sp for sp in fm["species"].unique()
               if testa_results.get(sp, {}).get("auroc") is not None]

for sp in sig_species:
    sp_mask = fm["species"] == sp
    sp_fm   = fm[sp_mask].copy()
    sp_grp  = groups[sp_mask]
    q2_mask = sp_fm["arg_burden_tertile"].isin(["high_ARG", "low_ARG"])
    sp_q2   = sp_fm[q2_mask].copy()
    grp_q2  = sp_grp[q2_mask.to_numpy()]
    y_q2    = (sp_q2["arg_burden_tertile"] == "high_ARG").astype(int).to_numpy()

    feat_keep = [c for c in FEAT_COLS_A
                 if c in sp_q2.columns and sp_q2[c].mean() >= 0.05]
    X_q2 = sp_q2[feat_keep].to_numpy(dtype=float)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_q2, y_q2)

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_q2)
    if isinstance(sv, list):
        sv = sv[1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs = pd.DataFrame(np.abs(sv), columns=feat_keep).mean().sort_values(ascending=False)
    rm_feats = [c for c in feat_keep if "RM" in c]
    shap_results_a[sp] = {"mean_abs_shap": mean_abs, "feat_keep": feat_keep, "rm_in_set": rm_feats}

    rm_ranks = {f: int(mean_abs.rank(ascending=False)[f]) for f in rm_feats}
    print(f"{sp}: RM features in Test A SHAP")
    for f, rk in sorted(rm_ranks.items(), key=lambda x: x[1]):
        print(f"  {f:<32} rank={rk:>4}/{len(feat_keep)}  mean|SHAP|={mean_abs[f]:.4f}")
    print()
"""

TESTA_VERDICT_HEADING = """\
## Section 7 — Test A summary and interpretation

Three pre-specified outcomes:
- **MOOT:** All RM subtypes are binary (dc ≈ dp). ΔAUROC ≈ 0. Report that RM copy number
  adds no predictive signal beyond presence/absence in ESKAPE.
- **POSITIVE:** ΔAUROC > 0 for ≥1 species. Multi-copy RM systems impose stronger
  restriction — count matters, not just presence.
- **NEGATIVE (active):** ΔAUROC < 0 for ≥1 species. Counting RM copies hurts prediction,
  suggesting noisy copy-count annotation. Defend the dp_ choice in the Methods.
"""

TESTA_VERDICT_CODE = """\
print("Test A narrative verdict:")
print("─" * 50)

live_results = {sp: r for sp, r in testa_results.items()
                if r.get("auroc") is not None}

if not live_cols:
    print("MOOT: all RM subtypes are effectively binary (dc ≈ dp).")
    print("ΔAUROC is not computed. Reported as finding: RM restriction is an")
    print("on/off switch, not a dose effect, at 3,335-genome scale.")
else:
    if live_results:
        deltas = []
        for sp, r in live_results.items():
            dp_a = dp_baseline.get(sp, {}).get("auroc")
            if dp_a:
                deltas.append((sp, r["auroc"] - dp_a))
        max_delta = max(d for _, d in deltas) if deltas else 0
        print(f"Max ΔAUROC (dc-RF minus dp-RF): {max_delta:+.3f}")
        if max_delta > 0.01:
            print("  → dc_RM improves Q2 prediction. Multi-copy RM signal is real.")
            print("  → Substitute dc_RM in the main analysis and note in Methods.")
        elif max_delta < -0.01:
            print("  → dc_RM hurts Q2 prediction. dp_* choice validated.")
        else:
            print("  → ΔAUROC ≈ 0. RM is effectively binary despite live classification.")
            print("  → dp_* choice validated. Report as supplementary finding.")
    else:
        print("No species had sufficient data for Test A.")
"""

TESTB_HEADING = """\
## Section 8 — Test B: ARG mechanism-class counting from ResFinder

**Why class-specific, not total ARG burden?**
Total ARG count conflates plasmid-borne genes (β-lactam, aminoglycoside — gated by RM
horizontal-transfer restriction) with chromosomal mutations (quinolone — intrinsic, not
plasmid-mediated). If RM restricts ARG uptake via horizontal gene transfer, the signal
should be strongest for plasmid-borne classes and absent for chromosomal classes.

Pre-specified predictions:
- **Negative SHAP for dp_RM_* in β-lactam and aminoglycoside cells**: RM restricts
  transfer of the most common plasmid-borne resistance genes.
- **Near-zero SHAP for dp_RM_* in quinolone cells**: chromosomal mutations, no HGT gating.
- **Exploratory (not pre-specified):** tetracycline and macrolide in Gram-positive species —
  plasmid-borne but with GT mediated by IS elements rather than conjugation.
"""

RESFINDER_CODE = """\
# Load drug→class mapping
with open(CONF / "arg_class_mapping.yaml") as f:
    raw_map = yaml.safe_load(f)

drug_to_class = {}
for cls, drugs in raw_map.items():
    if cls == "other":
        continue
    for d in (drugs or []):
        d = d.strip()
        drug_to_class.setdefault(d, [])
        drug_to_class[d].append(cls)

# Parse ResFinder outputs
print("Parsing ResFinder outputs for mechanism-class ARG counts...")
records = []
for sp in ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]:
    rf_files = glob.glob(str(INTER / sp / "resfinder" / "*" / "ResFinder_results_tab.txt"))
    for fp in rf_files:
        raw_acc = Path(fp).parent.name
        acc = re.sub(r"_(\\d+)$", r".\\1", raw_acc)
        try:
            df = pd.read_csv(fp, sep="\\t")
        except Exception:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            gene     = str(row.get("Resistance gene", "")).strip()
            phenostr = str(row.get("Phenotype", "")).strip()
            drugs    = [d.strip() for d in re.split(r",\\s*", phenostr) if d.strip()]
            for drug in drugs:
                for cls in drug_to_class.get(drug, []):
                    records.append({"accession": acc, "species": sp,
                                    "gene": gene, "arg_class": cls})

rf_long = pd.DataFrame(records)

# ── LEAK GUARD: intersect with FM index to exclude 31 leaked Ab dirs ─────────
before = rf_long["accession"].nunique()
rf_long = rf_long[rf_long["accession"].isin(fm.index)]
after  = rf_long["accession"].nunique()
print(f"  Genomes before FM intersection: {before}")
print(f"  Genomes after  FM intersection: {after}  ({before-after} excluded — not in clean FM)")
print(f"  Total (genome × class × gene) rows: {len(rf_long)}")
"""

ARGCLASS_WIDE = """\
arg_class_counts = (
    rf_long.groupby(["accession", "species", "arg_class"])["gene"]
    .nunique()
    .reset_index(name="n_genes_class")
)
arg_class_wide = (
    arg_class_counts.pivot_table(
        index=["accession", "species"], columns="arg_class",
        values="n_genes_class", aggfunc="sum", fill_value=0
    )
    .reset_index()
)
arg_class_wide.columns.name = None

# Add rows for genomes with 0 ARGs in all classes
all_genomes   = fm[["accession", "species"]].copy()
arg_class_wide = all_genomes.merge(arg_class_wide, on=["accession", "species"], how="left").fillna(0)

class_cols = [c for c in arg_class_wide.columns if c not in ("accession", "species")]
print(f"ARG classes found: {class_cols}")
print(f"\\nGenomes with ≥1 ARG per class:")
for cls in sorted(class_cols):
    n = (arg_class_wide[cls] > 0).sum()
    print(f"  {cls:<22} {n:>4} / {len(arg_class_wide)}")
"""

TESTB_LABELS_HEADING = """\
## Section 9 — Test B: label construction and 30/30 floor check

Each (species × ARG class) cell needs ≥30 high-burden and ≥30 low-burden genomes to
run a classifier. The 30/30 floor filters out sparse combinations (e.g. glycopeptide
in PA) where class-imbalance would make GroupKFold AUROC unreliable.
"""

TESTB_LABELS_CODE = """\
def make_class_labels(sp_df, class_col):
    # Tertile labels for one (species x ARG class). Returns (Series, method_str).
    counts = sp_df[class_col]
    try:
        labels = pd.qcut(counts, q=3, labels=["low_class", "mid_class", "high_class"],
                         duplicates="drop")
        if labels.isna().sum() > 0 or labels.nunique() < 3:
            raise ValueError("degenerate tertile")
        return labels, "tertile"
    except (ValueError, KeyError):
        med = counts.median()
        labels = pd.Series("mid_class", index=sp_df.index, dtype="object")
        labels[counts < med] = "low_class"
        labels[counts > med] = "high_class"
        return labels, "median_fallback"

floor_report = []
label_store  = {}

for sp in fm["species"].unique():
    sp_acc = fm[fm["species"] == sp]["accession"].tolist()
    sp_arg = arg_class_wide[arg_class_wide["accession"].isin(sp_acc)].set_index("accession").copy()
    for cls in class_cols:
        labels, method = make_class_labels(sp_arg, cls)
        n_high = (labels == "high_class").sum()
        n_low  = (labels == "low_class").sum()
        passes = (n_high >= 30) and (n_low >= 30)
        floor_report.append({
            "species": sp, "arg_class": cls,
            "n_high": int(n_high), "n_low": int(n_low),
            "label_method": method, "passes_30_30": passes,
        })
        if passes:
            label_store[(sp, cls)] = labels

floor_df = pd.DataFrame(floor_report)
print("30/30 floor check — (species × class) cells:")
print(floor_df.pivot_table(
    index="species", columns="arg_class", values="passes_30_30", aggfunc="first"
).to_string())
print()
passing = floor_df[floor_df["passes_30_30"]]
print(f"Cells passing 30/30 floor: {len(passing)}")
for _, row in passing.iterrows():
    print(f"  {row['species']:<22} {row['arg_class']:<22} n_H={row['n_high']:>3}, n_L={row['n_low']:>3}  ({row['label_method']})")
"""

TESTB_RF_HEADING = """\
## Section 10 — Test B: RF models per (species × ARG class)

One classifier per passing (species × class) cell, same GroupKFold-5 setup as main Q2.
Per-species sparsity filter (H1: feature present in ≥5% of Q2 genomes) applied.
BH correction across all passing cells (q=0.05).
"""

TESTB_RF_CODE = """\
def run_q2_cell(sp, cls, feat_cols):
    # Run Q2 RF for one (species x class) cell. Returns dict or None if skipped.
    if (sp, cls) not in label_store:
        return None
    labels = label_store[(sp, cls)]
    q2_acc = labels[labels.isin(["high_class", "low_class"])].index
    sp_q2  = fm[fm["accession"].isin(q2_acc)].set_index("accession")
    sp_q2  = sp_q2.loc[q2_acc]
    y_q2   = (labels.loc[q2_acc] == "high_class").astype(int).to_numpy()
    grp_q2 = sp_q2["phylogroup"].to_numpy(dtype=str)
    feat_keep = [c for c in feat_cols
                 if c in sp_q2.columns and sp_q2[c].mean() >= 0.05]
    if len(feat_keep) < 5:
        return {"note": f"n_feat={len(feat_keep)} < 5 after H1"}
    X_q2 = sp_q2[feat_keep].to_numpy(dtype=float)
    rf   = RandomForestClassifier(**RF_PARAMS)
    cv   = GroupKFold(n_splits=N_SPLITS)
    fold_aurocs = []
    for tr, te in cv.split(X_q2, y_q2, groups=grp_q2):
        if len(np.unique(y_q2[te])) < 2:
            continue
        rf.fit(X_q2[tr], y_q2[tr])
        fold_aurocs.append(roc_auc_score(y_q2[te], rf.predict_proba(X_q2[te])[:, 1]))
    if len(fold_aurocs) < 3:
        return {"note": "insufficient valid folds"}
    mean_a, lo, hi = bootstrap_ci(np.array(fold_aurocs))
    t_stat, p_val  = ttest_1samp(fold_aurocs, 0.5, alternative="greater")
    return {
        "auroc": round(mean_a, 3),
        "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
        "p_raw": float(p_val),
        "n_feat": len(feat_keep), "n_folds": len(fold_aurocs),
        "note": "ok",
    }

print("Running Test B RF models (may take 5-10 min)...")
testb_raw = {}
for sp in fm["species"].unique():
    for cls in class_cols:
        key = (sp, cls)
        if key not in label_store:
            continue
        r = run_q2_cell(sp, cls, FEAT_COLS)
        if r is not None:
            testb_raw[key] = r
            status = (f"AUROC={r.get('auroc',' - ')} p={r.get('p_raw',0):.3f}"
                      if r.get("auroc") else r.get("note", "?"))
            print(f"  {sp:<22} {cls:<22} {status}")
print(f"\\nCompleted {len(testb_raw)} cells.")
"""

TESTB_BH_CODE = """\
passing_keys = [(sp, cls) for (sp, cls), r in testb_raw.items() if r.get("note") == "ok"]
p_raws = [testb_raw[k]["p_raw"] for k in passing_keys]
if p_raws:
    reject, p_adj, _, _ = multipletests(p_raws, method="fdr_bh", alpha=0.05)
    for i, key in enumerate(passing_keys):
        testb_raw[key]["p_adj"] = float(p_adj[i])
        testb_raw[key]["significant"] = bool(reject[i])

rows = []
for (sp, cls), r in testb_raw.items():
    fl = floor_df[(floor_df["species"]==sp) & (floor_df["arg_class"]==cls)]
    n_high = int(fl["n_high"].values[0]) if len(fl) else None
    n_low  = int(fl["n_low"].values[0])  if len(fl) else None
    rows.append({
        "species": sp, "arg_class": cls,
        "AUROC": r.get("auroc"), "CI_lo": r.get("ci_lo"), "CI_hi": r.get("ci_hi"),
        "p_raw": r.get("p_raw"), "p_adj": r.get("p_adj"),
        "significant": r.get("significant", False),
        "n_high": n_high, "n_low": n_low, "n_feat": r.get("n_feat"),
        "note": r.get("note"),
    })
testb_df = pd.DataFrame(rows).sort_values(["arg_class","AUROC"], ascending=[True, False])
testb_df.to_parquet(RES / "testb_results_3460.parquet", index=False)

print("Test B results (BH-corrected, q=0.05):")
print(f"{'Species':<22} {'Class':<22} {'AUROC':>6} {'95% CI':>14} {'p_adj':>8} {'Sig':>4}")
print("-" * 85)
for _, row in testb_df.iterrows():
    if row["note"] != "ok":
        continue
    sig = "★" if row.get("significant") else " "
    ci  = f"[{row['CI_lo']:.3f}–{row['CI_hi']:.3f}]"
    print(f"{row['species']:<22} {row['arg_class']:<22} {row['AUROC']:>6.3f} {ci:>14} "
          f"{row['p_adj']:>8.4f} {sig:>4}")
"""

TESTB_SHAP_HEADING = """\
## Section 11 — Test B: SHAP directionality for RM features

The key biological prediction: if RM systems restrict plasmid-mediated ARG transfer,
signed SHAP for dp_RM_* features should be **negative** (predicts low ARG burden)
in β-lactam and aminoglycoside cells, and **near-zero** in quinolone (chromosomal).

Signed SHAP (not |SHAP|): positive = feature pushes prediction toward high_class (more ARG).
"""

TESTB_SHAP_CODE = """\
shap_sign_results = {}
sig_cells = [(sp, cls) for (sp, cls) in testb_raw
             if testb_raw[(sp, cls)].get("significant", False)]

print(f"Running SHAP for {len(sig_cells)} significant cells...")
for sp, cls in sig_cells:
    labels = label_store[(sp, cls)]
    q2_acc = labels[labels.isin(["high_class", "low_class"])].index
    sp_q2  = fm[fm["accession"].isin(q2_acc)].set_index("accession").loc[q2_acc]
    y_q2   = (labels.loc[q2_acc] == "high_class").astype(int).to_numpy()
    feat_keep = [c for c in FEAT_COLS
                 if c in sp_q2.columns and sp_q2[c].mean() >= 0.05]
    X_q2 = sp_q2[feat_keep].to_numpy(dtype=float)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_q2, y_q2)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_q2)
    if isinstance(sv, list):
        sv = sv[1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]
    shap_df  = pd.DataFrame(sv, columns=feat_keep)
    rm_feats = [c for c in feat_keep if "RM" in c]
    shap_sign_results[(sp, cls)] = {f: float(shap_df[f].mean()) for f in rm_feats}
    print(f"  {sp} / {cls}:")
    for feat, val in sorted(shap_sign_results[(sp, cls)].items(), key=lambda x: x[1]):
        direction = "← RESTRICTS" if val < -0.001 else ("→ FACILITATES" if val > 0.001 else "≈ 0")
        print(f"    {feat:<32} signed_SHAP={val:+.4f}  {direction}")
    print()
"""

TESTB_SHAP_SUMMARY = """\
predictions = {
    "beta_lactam":    "negative",
    "aminoglycoside": "negative",
    "sulfonamide":    "negative",
    "trimethoprim":   "negative",
    "quinolone":      "near_zero",
    "tetracycline":   "negative_exploratory",
    "glycopeptide":   "negative",
    "macrolide_mlsb": "negative_exploratory",
    "phenicol":       "ambiguous",
}

print("SHAP direction match vs pre-specified prediction:")
print("─" * 85)
for (sp, cls), rm_vals in shap_sign_results.items():
    predicted  = predictions.get(cls, "unspecified")
    dp_rm_vals = {k: v for k, v in rm_vals.items() if k.startswith("dp_RM")}
    if not dp_rm_vals:
        print(f"{sp:<22} {cls:<20} pred={predicted:<22} obs=no dp_RM_* in feature set")
        continue
    best_feat = min(dp_rm_vals, key=dp_rm_vals.get)
    best_val  = dp_rm_vals[best_feat]
    if best_val < -0.002:
        obs_dir = f"negative [{best_feat} = {best_val:+.4f}]"
    elif max(dp_rm_vals.values()) > 0.002:
        pos_feat = max(dp_rm_vals, key=dp_rm_vals.get)
        obs_dir  = f"positive [{pos_feat} = {dp_rm_vals[pos_feat]:+.4f}]"
    else:
        obs_dir = f"near zero [max|SHAP|={max(abs(v) for v in dp_rm_vals.values()):.4f}]"
    if predicted in ("negative", "negative_exploratory"):
        match = "MATCH ✓" if best_val < -0.002 else "MISMATCH ✗"
    elif predicted == "near_zero":
        match = "MATCH ✓" if max(abs(v) for v in dp_rm_vals.values()) <= 0.002 else "MISMATCH ✗"
    else:
        match = "(not pre-specified)"
    print(f"{sp:<22} {cls:<20} pred={predicted:<22} obs={obs_dir:<40} {match}")
"""

VIZ_HEADING = """\
## Section 12 — Visualisation: AUROC heatmap and SHAP sign plot
"""

VIZ_CODE = """\
# Figure 1: Test B AUROC heatmap
ok_df = testb_df[testb_df["note"] == "ok"].copy()
if not ok_df.empty:
    pivot_auroc = ok_df.pivot_table(
        index="species", columns="arg_class", values="AUROC", aggfunc="first"
    )
    pivot_sig = ok_df.pivot_table(
        index="species", columns="arg_class", values="significant", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_auroc.columns)*1.2), 5))
    sns.heatmap(
        pivot_auroc, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.4, vmax=0.9, linewidths=0.5, ax=ax,
        mask=pivot_auroc.isna()
    )
    # Add ★ for significant cells
    for i, sp in enumerate(pivot_auroc.index):
        for j, cls in enumerate(pivot_auroc.columns):
            try:
                if pivot_sig.loc[sp, cls]:
                    ax.text(j+0.85, i+0.2, "★", color="black", fontsize=9)
            except (KeyError, TypeError):
                pass
    ax.set_title("Test B AUROC by (species × ARG class)  |  ★ = BH-significant (q<0.05)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(FIG / "testb_auroc_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: testb_auroc_heatmap.png")
else:
    print("No Test B results to plot.")

# Figure 2: RM SHAP sign heatmap
if shap_sign_results:
    sign_rows = []
    for (sp, cls), rm_vals in shap_sign_results.items():
        dp_vals = {k: v for k, v in rm_vals.items() if k.startswith("dp_RM")}
        if dp_vals:
            best_feat = min(dp_vals, key=dp_vals.get)
            sign_rows.append({"species": sp, "arg_class": cls,
                               "min_rm_shap": dp_vals[best_feat],
                               "feature": best_feat})
    if sign_rows:
        sign_df = pd.DataFrame(sign_rows)
        pivot_sign = sign_df.pivot_table(
            index="species", columns="arg_class", values="min_rm_shap", aggfunc="first"
        )
        fig, ax = plt.subplots(figsize=(max(10, len(pivot_sign.columns)*1.2), 5))
        sns.heatmap(
            pivot_sign, annot=True, fmt=".3f", cmap="RdBu",
            center=0, vmin=-0.05, vmax=0.05, linewidths=0.5, ax=ax,
            mask=pivot_sign.isna()
        )
        ax.set_title("Most-negative dp_RM_* signed SHAP per cell\\n"
                     "Blue = RM restricts ARGs; Red = RM facilitates ARGs")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(FIG / "testb_rm_shap_sign.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: testb_rm_shap_sign.png")
"""

SYNTHESIS_HEADING = """\
## Section 13 — Synthesis: what Tests A and B tell us together

Test A and Test B answer complementary questions:
- Test A: *Is RM restriction a binary switch or a dose effect?*
- Test B: *Is RM restriction class-selective (plasmid-borne ARGs only)?*

Together, they determine whether the defence-ARG co-exclusion pattern seen in Q2
represents genuine RM-mediated plasmid restriction or a generic defence-depleted
genome confound.
"""

SYNTHESIS_CODE = """\
print("=" * 65)
print("PHASE 12 SYNTHESIS  (3,335-genome cohort)")
print("=" * 65)

# Test A
if not live_cols:
    print("\\nTest A: MOOT — all RM subtypes binary (dc ≈ dp).")
    print("  RM restriction is an on/off switch, not a dose effect.")
    print("  dp_* feature choice validated.")
else:
    live_res = {sp: testa_results[sp] for sp in testa_results
                if testa_results[sp].get("auroc") is not None}
    deltas = [(sp, r["auroc"] - dp_baseline.get(sp, {}).get("auroc", r["auroc"]))
              for sp, r in live_res.items()
              if dp_baseline.get(sp, {}).get("auroc") is not None]
    if deltas:
        max_d_sp, max_d = max(deltas, key=lambda x: x[1])
        print(f"\\nTest A: max ΔAUROC = {max_d:+.3f} ({max_d_sp})")
        if max_d > 0.01:
            print("  → dc_RM improves prediction. Multi-copy RM signal is real.")
        elif max_d < -0.01:
            print("  → dc_RM hurts prediction. dp_* choice validated.")
        else:
            print("  → |ΔAUROC| ≤ 0.01 across all species. RM effectively binary.")

# Test B
sig_cells = [(sp, cls) for (sp, cls) in testb_raw
             if testb_raw[(sp, cls)].get("significant", False)]
print(f"\\nTest B: {len(sig_cells)} significant (species × class) cells (BH q<0.05):")
for sp, cls in sorted(sig_cells):
    r = testb_raw[(sp, cls)]
    print(f"  {sp:<22} {cls:<22} AUROC={r['auroc']:.3f}  p_adj={r['p_adj']:.4f}")

# SHAP direction
plasmid_classes = ("beta_lactam", "aminoglycoside", "sulfonamide", "trimethoprim")
def _min_rm(sp, cls):
    dp_vals = {k: v for k, v in shap_sign_results.get((sp,cls),{}).items()
               if k.startswith("dp_RM")}
    if not dp_vals: return None, 0.0
    feat = min(dp_vals, key=dp_vals.get)
    return feat, dp_vals[feat]

matched    = [(sp, cls) for (sp, cls) in shap_sign_results
              if cls in plasmid_classes and _min_rm(sp, cls)[1] < -0.002]
mismatch   = [(sp, cls) for (sp, cls) in shap_sign_results
              if cls in plasmid_classes and _min_rm(sp, cls)[1] >= -0.002]
quinolone_neutral = [(sp, cls) for (sp, cls) in shap_sign_results
                     if cls == "quinolone" and
                     all(abs(v) <= 0.002 for v in shap_sign_results[(sp,cls)].values()
                         if True)]
print(f"\\nSHAP direction (plasmid classes):")
print(f"  RM restricts (SHAP < -0.002):  {matched}")
print(f"  RM near-zero/positive (mismatch): {mismatch}")
print(f"  Quinolone ≈ 0 (expected negative control): {quinolone_neutral}")
print()
print("Phase 12 complete. Figures in results/figures/phase12/")
print("Test B results: results/testb_results_3460.parquet")
"""

COMPREHENSION = """\
## Section 14 — Comprehension check

Answer these before ending the session. Do not read ahead in the Results.

**Q1.** Test A observes ΔAUROC ≈ 0 for all species (dc_RM minus dp_RF baseline).
A colleague says "this means RM systems have no effect on ARG burden." What is the
correct interpretation? What would the data need to show for RM to be truly
uninformative — as distinct from binary?

**Q2.** Test B shows β-lactam AUROC significantly above chance for EC and KP, but
quinolone AUROC is not significant in either. A reviewer says: "this just reflects that
β-lactam ARG counts are more variable than quinolone ARG counts, not any mechanism."
How do you respond — and which feature of your pre-registration directly addresses this?

**Q3.** SHAP for dp_RM_Type_I in a significant β-lactam cell is +0.005 (small positive).
You predicted negative. Name TWO explanations that do not require abandoning the
RESTRICT hypothesis.

**Q4.** If Test B shows aminoglycoside and β-lactam are both significant for EC but
quinolone is not, write one sentence for the Results section. It must:
(a) state the finding, (b) name the correct metric, (c) use no causal language.
"""


# ── Assemble notebook ────────────────────────────────────────────────────────
cells = [
    md(TITLE),
    md("## Section 1 — Imports and configuration"),
    code(IMPORTS),
    md(LOAD_FM),
    code(LOAD_FM_CODE),
    md("## Section 3 — Pre-check: are RM count (dc_) and presence (dp_) equivalent?"),
    code(RM_CHECK_CODE),
    code(RM_VIZ_CODE),
    md(TESTA_HEADING),
    code(TESTA_FEATCOLS),
    md(TESTA_RUN_HEADING),
    code(TESTA_RUN_CODE),
    code(TESTA_DISPLAY),
    md(TESTA_SHAP_HEADING),
    code(TESTA_SHAP_CODE),
    md(TESTA_VERDICT_HEADING),
    code(TESTA_VERDICT_CODE),
    md(TESTB_HEADING),
    code(RESFINDER_CODE),
    code(ARGCLASS_WIDE),
    md(TESTB_LABELS_HEADING),
    code(TESTB_LABELS_CODE),
    md(TESTB_RF_HEADING),
    code(TESTB_RF_CODE),
    code(TESTB_BH_CODE),
    md(TESTB_SHAP_HEADING),
    code(TESTB_SHAP_CODE),
    code(TESTB_SHAP_SUMMARY),
    md(VIZ_HEADING),
    code(VIZ_CODE),
    md(SYNTHESIS_HEADING),
    code(SYNTHESIS_CODE),
    md(COMPREHENSION),
]

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (eskape-ml)",
        "language": "python",
        "name": "eskape-ml"
    },
    "language_info": {"name": "python", "version": "3.11.0"}
}

out = "notebooks/10_phase12_sensitivity.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)
print(f"Written: {out}")
print(f"Cells: {len(nb.cells)}")
