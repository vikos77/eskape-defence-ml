"""
Build notebooks/10_phase12_sensitivity.ipynb programmatically.
Run once from the eskape-defence-ml/ root:
    conda run -n eskape-ml python notebooks/build_phase12_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (eskape-ml)",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ── TITLE ──────────────────────────────────────────────────────────────────────
md("""\
# Phase 12 — Mechanism-level ARG burden and RM count sensitivity analysis

**Pre-registered:** 2026-05-27. Full specification in `docs/pre_analysis_plan.md §10`.

**Two independent tests:**

- **Test A** — Replace `dp_RM_*` (binary presence) with `dc_RM_*` (count) in Q2 RF.
  All other features remain `dp_*`. Same total-ARG-burden target as Phase 8/9 Q2.
  Question: does knowing *how many* RM systems a genome carries improve ARG-burden prediction
  beyond knowing whether the genome has at least one RM system?

- **Test B** — Replace total-ARG-burden target with mechanism-class ARG burden.
  One binary RF per (species × ARG class). Original `dp_*` features unchanged.
  Question: is the RESTRICT/FACILITATE principle class-specific — stronger for
  plasmid-mediated ARG classes (β-lactam, aminoglycoside) than for chromosomal classes
  (quinolone)?

**Ordering:** Test A must be completed before Test B. If Test A shows RM is already
binary (dc ≈ dp), that is informative context for interpreting RM SHAP values in Test B.
""")

# ── SECTION 1: IMPORTS ─────────────────────────────────────────────────────────
md("## Section 1 — Imports and configuration")

code("""\
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.utils import resample
import shap

warnings.filterwarnings("ignore")

ROOT   = Path("..")
PROC   = ROOT / "data" / "processed"
INTER  = ROOT / "data" / "interim"
CONF   = ROOT / "config"
RES    = ROOT / "results"
FIG    = RES / "figures" / "phase12"
FIG.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS     = 5
N_BOOT       = 2000

# Phase 8 best RF hyperparameters (frozen — same model spec for all Phase 12 runs)
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=20,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Imports OK.")
""")

# ── SECTION 2: LOAD DATA ───────────────────────────────────────────────────────
md("""\
## Section 2 — Load data and reproduce Phase 8 feature selection

**Why reproduce, not cache?**
Recomputing spec_score from the same feature matrix guarantees that Phase 12
uses the identical 265-feature filtered set as Phase 8. Caching a list of feature
names would be fine for a stable pipeline, but here we want the code to be
self-verifying: if the feature matrix ever changes upstream, the spec_score
recalculation would catch any drift.

**Reminder on the specificity filter:**
For each `dp_*` feature, compute the standard deviation of per-species prevalence
across 6 species, normalised by 0.5. Features with score ≥ 0.70 are taxonomic
markers (near-universal in one species) and are excluded from Q2 features.
9 features removed → 265 retained (`FEAT_COLS`).
""")

code("""\
fm     = pd.read_parquet(PROC / "feature_matrix.parquet")
fm["accession"] = fm.index   # index holds GCF accessions; expose as column for downstream joins
groups = fm["phylogroup"].to_numpy(dtype=str)   # 95 phylogroups (Phase 6, frozen)

# ── Reproduce Phase 8 specificity filter ──
dp_cols    = sorted([c for c in fm.columns if c.startswith("dp_")])
sp_prev    = fm.groupby("species")[dp_cols].mean()
spec_score = sp_prev.std() / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in dp_cols if c not in markers]   # 265 dp_* features

print(f"Feature matrix: {fm.shape[0]} genomes × {len(FEAT_COLS)} filtered dp_* features")
print(f"Markers removed ({len(markers)}): {markers}")
print(f"Species distribution:")
for sp, n in fm['species'].value_counts().items():
    print(f"  {sp:<20} {n:>4}")
""")

# ── SECTION 3: PRE-CHECK — dc vs dp RM ────────────────────────────────────────
md("""\
## Section 3 — Pre-check: are RM count (dc) and presence (dp) equivalent?

**Why this check comes before any modelling:**

The pre-analysis plan (§10, Test A) requires this pre-check. The logic is:

- `dp_RM_Type_I` = 1 if a genome has ≥1 RM Type I system; 0 otherwise.
- `dc_RM_Type_I` = the actual count (0, 1, 2, 3…).

If dc == dp for all genomes (i.e., no genome ever has more than 1 RM system of any subtype),
then binary presence already captures all the information in the count. Test A would be
testing noise. Reporting *that* finding is itself scientifically informative: RM systems
in ESKAPE are always present in single copy per type.

If dc ≠ dp for a substantial fraction of genomes, multi-copy RM systems exist and the count
carries extra information that binary presence discards.

**Falsification criterion (pre-specified):**
If dc == dp for ≥90% of genomes across ALL RM subtypes → Test A is moot.
Report dc/dp equivalence as a finding in the Methods; do not run the classifier.
""")

code("""\
dc_rm_cols = [c for c in fm.columns if c.startswith("dc_RM")]
dp_rm_cols = [c.replace("dc_", "dp_") for c in dc_rm_cols]

rows = []
for dc_col, dp_col in zip(dc_rm_cols, dp_rm_cols):
    dc = fm[dc_col]
    dp = fm[dp_col]
    n_diff     = (dc != dp).sum()
    pct_diff   = n_diff / len(fm) * 100
    max_count  = dc.max()
    count_dist = dc.value_counts().sort_index().to_dict()
    rows.append({
        "RM subtype": dc_col.replace("dc_", ""),
        "Genomes where dc≠dp": n_diff,
        "% differ": round(pct_diff, 1),
        "Max dc value": int(max_count),
        "Count distribution (dc)": str(count_dist),
    })

precheck = pd.DataFrame(rows)
print(precheck.to_string(index=False))
print()

# Is Test A moot? Check 90% threshold per column
moot_cols = [r["RM subtype"] for _, r in precheck.iterrows() if r["% differ"] < 10]
live_cols = [r["RM subtype"] for _, r in precheck.iterrows() if r["% differ"] >= 10]
print(f"Test A assessment:")
print(f"  Live (dc≠dp ≥10% of genomes): {live_cols}")
print(f"  Moot (dc≈dp, <10% differ): {moot_cols}")
if len(live_cols) == 0:
    print("  VERDICT: Test A is MOOT — all RM subtypes are effectively binary. Report as finding.")
else:
    print("  VERDICT: Test A is LIVE — multi-copy RM variation warrants the count-feature test.")
""")

code("""\
# Visualise dc distribution for the live RM subtype(s)
fig, axes = plt.subplots(1, len(dc_rm_cols), figsize=(4 * len(dc_rm_cols), 3))
if len(dc_rm_cols) == 1:
    axes = [axes]
for ax, dc_col in zip(axes, dc_rm_cols):
    vals = fm[dc_col].value_counts().sort_index()
    ax.bar(vals.index.astype(str), vals.values, color="#4C72B0", edgecolor="white")
    ax.set_title(dc_col.replace("dc_", ""), fontsize=9)
    ax.set_xlabel("Count (dc value)")
    ax.set_ylabel("# Genomes")
plt.suptitle("RM count distributions across 878 genomes", y=1.02)
plt.tight_layout()
plt.savefig(FIG / "01_rm_count_distributions.pdf", bbox_inches="tight")
plt.show()
print("Figure saved: 01_rm_count_distributions.pdf")
""")

# ── SECTION 4: TEST A — FEATURE ENGINEERING ───────────────────────────────────
md("""\
## Section 4 — Test A: replace dp_RM_* with dc_RM_* in Q2 feature set

**What changes exactly:**

Phase 8 Q2 used `FEAT_COLS` (265 dp_* features). For Test A, we build
`FEAT_COLS_A`: the same 265 features but with the 5 `dp_RM_*` columns swapped
for their `dc_RM_*` counterparts.

Example:
- Phase 8: `dp_RM_Type_I` ∈ {0, 1}
- Test A: `dc_RM_Type_I` ∈ {0, 1, 2, …, max}

All other features remain binary dp_*.

**Why not include BOTH dp and dc?**
The pre-registration says "replace", not "add". Including both would create
near-collinear features (dc > 0 ⟺ dp = 1 for all genomes where dc > 0), which
inflates permutation importance variance and makes SHAP interpretation ambiguous.
If Test A improves performance, the combined exploratory run is permitted in the
supplementary section.
""")

code("""\
# Build Test A feature list
dp_rm_set  = set(dp_rm_cols)
FEAT_COLS_A = []
for c in FEAT_COLS:
    if c in dp_rm_set:
        FEAT_COLS_A.append(c.replace("dp_", "dc_"))  # swap presence → count
    else:
        FEAT_COLS_A.append(c)

assert len(FEAT_COLS_A) == len(FEAT_COLS), "Feature count must not change in Test A"
swapped = [a for a, b in zip(FEAT_COLS_A, FEAT_COLS) if a != b]
print(f"Test A feature count: {len(FEAT_COLS_A)} (same as Phase 8)")
print(f"Swapped features ({len(swapped)}): {swapped}")
print(f"All other features unchanged: {len(FEAT_COLS_A) - len(swapped)} dp_* features")
""")

# ── SECTION 5: TEST A — Q2 RF PER SPECIES ─────────────────────────────────────
md("""\
## Section 5 — Test A: Q2 Random Forest with dc_RM features

**Setup identical to Phase 8 Q2 (frozen spec):**
- Same hyperparameters: max_depth=20, max_features=sqrt, min_samples_leaf=1, n_estimators=100
- Same CV: GroupKFold(5), groups = 95 phylogroups (Phase 6, frozen)
- Same Q2 labels: `arg_burden_tertile` (top vs bottom tertile; PA uses binary median fallback per PA-1)
- Same H1 sparsity filter applied per species on FEAT_COLS_A
- Primary metric: AUROC (same as Phase 8 Q2 reporting)

**What we're measuring:**
ΔAUROC = AUROC_TestA − AUROC_Phase8, per species.
If ΔType_I_AUROC > 0 for species where RM Type I is multi-copy (AB, PA): count signal improves prediction.
If Δ ≈ 0: binary presence captures the full RM signal already in this dataset.

**Null hypothesis:** RM count features do not improve Q2 AUROC vs Phase 8 binary-presence baseline.
""")

code("""\
# Phase 8 Q2 AUROC baseline (from decisions.md / 06_random_forest.ipynb)
# XGB primary for EC/KP; RF primary for PA. Both reported here.
phase8_q2_auroc = {
    "ecloaceae":   0.824,   # XGB (primary)
    "kpneumoniae": 0.789,   # XGB (primary)
    "paeruginosa": 0.677,   # RF  (primary — XGB 0.568)
    "efaecium":    None,    # chance
    "saureus":     None,    # marginal
    "abaumannii":  None,    # chance
}

def bootstrap_ci(scores, n_boot=N_BOOT, alpha=0.05, rng=42):
    rng = np.random.default_rng(rng)
    boot = [np.mean(rng.choice(scores, size=len(scores), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(scores)), lo, hi

def q2_rf_per_species(feat_cols, label_col="arg_burden_tertile"):
    \"\"\"Run Q2 RF for all 6 species; return dict of AUROC + CI.\"\"\"
    results = {}
    cv = GroupKFold(n_splits=N_SPLITS)

    for sp in fm["species"].unique():
        sp_mask = fm["species"] == sp
        sp_fm   = fm[sp_mask].copy()
        sp_grp  = groups[sp_mask]

        # Q2: top vs bottom tertile only (exclude mid_ARG)
        q2_mask = sp_fm[label_col].isin(["high_ARG", "low_ARG"])
        if q2_mask.sum() < 60:
            results[sp] = {"auroc": None, "ci": (None, None), "n": int(q2_mask.sum()), "note": "n<60"}
            continue

        sp_q2  = sp_fm[q2_mask].copy()
        grp_q2 = sp_grp[q2_mask.to_numpy()]
        y_q2   = (sp_q2[label_col] == "high_ARG").astype(int).to_numpy()

        # H1 per-species sparsity filter: keep features present in ≥5% of Q2 genomes
        feat_keep = [c for c in feat_cols
                     if c in sp_q2.columns and sp_q2[c].mean() >= 0.05]
        if len(feat_keep) == 0:
            results[sp] = {"auroc": None, "ci": (None, None), "n": int(q2_mask.sum()), "note": "no features after H1"}
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
            results[sp] = {"auroc": None, "ci": (None, None), "n": int(q2_mask.sum()), "note": "insufficient folds"}
            continue

        mean_a, lo, hi = bootstrap_ci(np.array(fold_aurocs))
        results[sp] = {
            "auroc": round(mean_a, 3),
            "ci": (round(lo, 3), round(hi, 3)),
            "n_feat": len(feat_keep),
            "n": int(q2_mask.sum()),
            "note": "ok",
        }
    return results

print("Running Test A Q2 RF (dc_RM features)... this may take 2-3 minutes")
testa_results = q2_rf_per_species(FEAT_COLS_A)
print("Done.")
""")

code("""\
# Display Test A results vs Phase 8 baseline
print("\\nTest A results — Q2 AUROC with dc_RM features vs Phase 8 dp_RM baseline\\n")
header = f"{'Species':<20} {'Ph8 AUROC':>10} {'TestA AUROC':>12} {'95% CI':>18} {'Δ AUROC':>9} {'N feat':>7} {'N genomes':>10}"
print(header)
print("-" * len(header))
for sp in sorted(testa_results.keys()):
    r = testa_results[sp]
    ph8 = phase8_q2_auroc.get(sp)
    if r["auroc"] is None:
        auroc_str = f"  {'chance':>10}"
        ci_str    = "        —"
        delta_str = "       —"
    else:
        auroc_str = f"  {r['auroc']:>10.3f}"
        ci_str    = f"[{r['ci'][0]:.3f}–{r['ci'][1]:.3f}]"
        delta     = r["auroc"] - ph8 if ph8 is not None else None
        delta_str = f"  {delta:>+.3f}" if delta is not None else "       —"
    ph8_str = f"{ph8:.3f}" if ph8 is not None else "  —"
    n_feat  = r.get("n_feat", "—")
    print(f"{sp:<20} {ph8_str:>10} {auroc_str:>12} {ci_str:>18} {delta_str:>9} {str(n_feat):>7} {r['n']:>10}")
""")

# ── SECTION 6: TEST A — SHAP FOR RM FEATURES ──────────────────────────────────
md("""\
## Section 6 — Test A: SHAP for RM count features

**What we're looking for:**

In Phase 8, `dp_RM_Type_I` (binary presence) had relatively low SHAP values — the binary
feature did not strongly separate genomes with many RM systems from those with one.
If `dc_RM_Type_I` (count) has higher absolute SHAP values, it means the **number** of RM
systems matters, not just their presence.

This is biologically meaningful: a genome with 8 RM Type I systems imposes a much stronger
restriction barrier against foreign DNA (including plasmid-encoded ARGs) than one with
just 1. The published Muthuraman et al. (2026) Spearman correlations operated on count
data for this reason.

**We run SHAP on the three species with Q2 signal (EC, KP, PA) using the full Q2
training data for a single-model SHAP explanation (not fold-by-fold).**
""")

code("""\
shap_results_a = {}
cv = GroupKFold(n_splits=N_SPLITS)

for sp in ["ecloaceae", "kpneumoniae", "paeruginosa"]:
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

    # Train on full Q2 data for SHAP (in-sample explanation; not for accuracy)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_q2, y_q2)

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_q2)
    if isinstance(sv, list):
        sv = sv[1]           # old shap: list[class0, class1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]     # new shap >=0.44: (samples, features, classes)

    rm_feats_in_set = [c for c in feat_keep if "RM" in c]
    shap_df = pd.DataFrame(np.abs(sv), columns=feat_keep)
    mean_abs = shap_df.mean().sort_values(ascending=False)

    shap_results_a[sp] = {
        "mean_abs_shap": mean_abs,
        "feat_keep": feat_keep,
        "rm_in_set": rm_feats_in_set,
    }

    rm_ranks = {f: int(mean_abs.rank(ascending=False)[f]) for f in rm_feats_in_set}
    print(f"{sp}: RM features in Test A SHAP")
    for f, rk in sorted(rm_ranks.items(), key=lambda x: x[1]):
        shap_val = mean_abs[f]
        print(f"  {f:<30} rank={rk:>4}/{len(feat_keep)}  mean|SHAP|={shap_val:.4f}")
    print()
""")

# ── SECTION 7: TEST A SUMMARY ─────────────────────────────────────────────────
md("""\
## Section 7 — Test A summary and interpretation

**Interpreting the results:**

Three possible outcomes are pre-specified:

1. **ΔAUROC > 0 for EC and/or KP (significant species):** RM count adds information
   beyond binary presence. The published Spearman correlation (count data) is more
   informative than a binary presence test in the ML context. Manuscript claim: "Replacing
   binary RM presence with RM count improves Q2 prediction in [species], consistent with
   a dose-response relationship between RM system burden and ARG restriction."

2. **ΔAUROC ≈ 0 across all species:** RM is effectively binary in this dataset (no genome
   gains much from having >1 RM system per type over having 1). The binary encoding was
   adequate. Manuscript claim: "RM count did not improve Q2 prediction over binary presence,
   suggesting that the presence/absence of RM systems captures the biologically relevant
   threshold effect at n=150 per species."

3. **ΔAUROC < 0 (unexpected):** dc_RM features somehow degrade performance. Most likely
   explanation: the larger numeric range of dc features interacts with sqrt max_features
   in RF subsampling — dc features occupy a wider space and are selected proportionally less
   often. This is a nuisance effect, not a biological finding. Report and note.

**Regardless of AUROC direction:** the SHAP analysis tells you whether the RM *count*
value drives the SHAP score (SHAP increasing monotonically with dc value) or whether SHAP
essentially treats 1 and 2+ identically (equivalent to binary presence effect).
""")

code("""\
# Print narrative verdict
print("Test A narrative verdict:")
print("─" * 50)
any_live = any(r["auroc"] is not None for r in testa_results.values())
if any_live:
    live_sps  = {sp: r for sp, r in testa_results.items() if r["auroc"] is not None}
    pos_delta = {sp: r["auroc"] - phase8_q2_auroc[sp]
                 for sp, r in live_sps.items()
                 if phase8_q2_auroc.get(sp) is not None}

    improving = {sp: d for sp, d in pos_delta.items() if d > 0.005}
    degrading  = {sp: d for sp, d in pos_delta.items() if d < -0.005}
    neutral    = {sp: d for sp, d in pos_delta.items() if abs(d) <= 0.005}

    if improving:
        print(f"  RM count IMPROVES AUROC in: {list(improving.keys())}")
        print(f"  Max improvement: +{max(improving.values()):.3f}")
    if degrading:
        print(f"  RM count DEGRADES AUROC in: {list(degrading.keys())} (nuisance effect likely)")
    if neutral:
        print(f"  No meaningful AUROC change in: {list(neutral.keys())} (RM effectively binary)")

    if not improving:
        print("  Overall verdict: RM is effectively binary in this dataset.")
        print("  Binary presence encoding was adequate. Report as positive methodological finding.")
else:
    print("  No Q2-significant species — cannot assess Test A AUROC effect.")
print()
print("This section is pre-registered as Test A. Results above are confirmatory.")
""")

# ── SECTION 8: TEST B — ARG CLASS COUNTING ────────────────────────────────────
md("""\
## Section 8 — Test B: ARG mechanism-class counting from ResFinder

**Why mechanism-class ARG burden, not total ARG burden?**

Total ARG burden mixes signals from fundamentally different resistance mechanisms:

| ARG class | Typical acquisition route | RM gating predicted? |
|---|---|---|
| β-lactam (bla genes) | Plasmid conjugation | Yes — RM restricts plasmid entry |
| Aminoglycoside (aac/aph) | Plasmid / integron | Yes |
| Sulfonamide (sul genes) | Class 1 integrons on plasmids | Yes |
| Trimethoprim (dfr genes) | Class 1 integrons on plasmids | Yes |
| Quinolone | **Chromosomal point mutations** (GyrA/ParC) | **No** — no plasmid, no RM gate |
| Tetracycline | Mixed (plasmid tet genes + some chromosomal) | Partial |

A classifier predicting total ARG burden sees a noisy mixture of these signals. Test B
separates them: if the RESTRICT/FACILITATE principle is truly about plasmid-mediated ARG
acquisition via RM gatekeeping, we expect:

- **β-lactam AUROC ≥ total-ARG AUROC** for EC/KP (high plasmid burden)
- **Quinolone AUROC ≈ 0.5** (chromosomal — no RM gating signal)
- **RM SHAP for β-lactam: negative** (high RM → low β-lactam ARG burden)
- **RM SHAP for quinolone: ≈ 0**

These are directional predictions written before any Test B modelling.

**ARG class mapping:** `config/arg_class_mapping.yaml` (versioned, written 2026-05-27).
Source: ResFinder `Phenotype` field (comma-separated drug names → antibiotic class).
Parsing: `re.split(r',\\s*', phenotype_string)` to handle spacing inconsistency.
Counting unit: unique resistance gene names per (genome × class), matching `arg_count_unique`.
""")

code("""\
# Load drug→class mapping
with open(CONF / "arg_class_mapping.yaml") as f:
    raw_map = yaml.safe_load(f)

# Build drug→class lookup (one drug can map to multiple classes if polyspecific)
drug_to_class = {}
for cls, drugs in raw_map.items():
    if cls == "other":
        continue   # "other" class not modelled in Test B
    for d in (drugs or []):
        d = d.strip()
        drug_to_class.setdefault(d, [])
        drug_to_class[d].append(cls)

# Parse all ResFinder outputs
print("Parsing ResFinder outputs for mechanism-class ARG counts...")
records = []
for sp in ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]:
    rf_files = glob.glob(str(INTER / sp / "resfinder" / "*" / "ResFinder_results_tab.txt"))
    for fp in rf_files:
        raw_acc = Path(fp).parent.name
        # ResFinder dirs use _ as version separator (GCF_000746645_1),
        # but fm.index uses dots (GCF_000746645.1) — normalise here.
        acc = re.sub(r"_(\d+)$", r".\\1", raw_acc)
        try:
            df = pd.read_csv(fp, sep="\\t")
        except Exception:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            gene    = str(row.get("Resistance gene", "")).strip()
            phenostr = str(row.get("Phenotype", "")).strip()
            drugs   = [d.strip() for d in re.split(r",\\s*", phenostr) if d.strip()]
            for drug in drugs:
                for cls in drug_to_class.get(drug, []):
                    records.append({"accession": acc, "species": sp, "gene": gene, "arg_class": cls})

rf_long = pd.DataFrame(records)
print(f"  Total (genome × class × gene) rows parsed: {len(rf_long)}")
print(f"  Genomes with ≥1 ARG in any class: {rf_long['accession'].nunique()}")
""")

code("""\
# Per-genome unique gene count per class
arg_class_counts = (
    rf_long.groupby(["accession", "species", "arg_class"])["gene"]
    .nunique()
    .reset_index(name="n_genes_class")
)

# Pivot to wide: one row per genome, one column per class (fill 0 for missing)
arg_class_wide = (
    arg_class_counts.pivot_table(
        index=["accession", "species"], columns="arg_class",
        values="n_genes_class", aggfunc="sum", fill_value=0
    )
    .reset_index()
)
arg_class_wide.columns.name = None

# Add rows for genomes with 0 ARGs in all classes
all_genomes = fm[["accession", "species"]].copy()
arg_class_wide = all_genomes.merge(arg_class_wide, on=["accession", "species"], how="left").fillna(0)

class_cols = [c for c in arg_class_wide.columns if c not in ("accession", "species")]
print(f"ARG classes found: {class_cols}")
print(f"\\nGenomes with ≥1 ARG per class:")
for cls in sorted(class_cols):
    n = (arg_class_wide[cls] > 0).sum()
    print(f"  {cls:<20} {n:>4} / {len(arg_class_wide)}")
""")

# ── SECTION 9: TEST B — LABEL CONSTRUCTION ────────────────────────────────────
md("""\
## Section 9 — Test B: label construction and 30/30 floor check

**Label construction (same protocol as Phase 8 Q2 primary labels):**

For each (species × ARG class) combination:
1. Compute per-genome count of unique ARGs in that class.
2. Attempt tertile split within species (top 33% = high_class; bottom 33% = low_class; middle excluded).
3. If tertile fails (PA-1: 0th percentile == 33rd percentile), fall back to binary median split.
4. Check 30/30 floor: ≥30 high_class AND ≥30 low_class genomes required.
5. If floor fails → excluded. Document count in results.

**30/30 floor justification (from pre_analysis_plan.md §10):**
At n=30 per class under 5-fold GroupKFold, each test fold sees ~6 positives.
Bootstrap CI on AUROC = ±0.11 — borderline but can distinguish AUROC=0.70 from 0.50.
Below n=30, CI exceeds ±0.15 and is uninformative.

**Directional predictions (written before any modelling):**
- β-lactam: high AUROC in EC/KP; RM SHAP < 0 (restricts plasmid entry)
- Aminoglycoside: high AUROC in EC/KP/PA; RM SHAP < 0
- Sulfonamide: moderate AUROC; RM SHAP < 0
- Trimethoprim: moderate AUROC; RM SHAP < 0
- Quinolone: AUROC ≈ 0.5 (negative control — chromosomal mutations)
- Tetracycline: mixed prediction; depends on proportion of plasmid-borne tet genes per species
""")

code("""\
def make_class_labels(sp_df, class_col):
    \"\"\"Tertile labels for one (species × ARG class). Returns Series with index of sp_df.\n    Labels: high_class, mid_class, low_class (or high_class / low_class for PA-1 fallback).\"\"\"
    counts = sp_df[class_col]
    try:
        labels = pd.qcut(counts, q=3, labels=["low_class", "mid_class", "high_class"],
                         duplicates="drop")
        if labels.isna().sum() > 0 or labels.nunique() < 3:
            raise ValueError("degenerate tertile")
        return labels, "tertile"
    except (ValueError, KeyError):
        # PA-1 fallback: binary median split
        med = counts.median()
        labels = pd.Series("mid_class", index=sp_df.index, dtype="object")
        labels[counts < med] = "low_class"
        labels[counts > med] = "high_class"
        return labels, "median_fallback"

# Build label table for all (species × class) combinations
floor_report = []
label_store  = {}   # key: (species, arg_class) → Series of labels

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
print("30/30 floor check — all (species × class) combinations:")
print(floor_df.pivot_table(
    index="species", columns="arg_class",
    values="passes_30_30", aggfunc="first"
).to_string())
print()
passing = floor_df[floor_df["passes_30_30"]]
print(f"Cells passing 30/30 floor: {len(passing)}")
for _, row in passing.iterrows():
    print(f"  {row['species']:<20} {row['arg_class']:<20} n_high={row['n_high']:>3}, n_low={row['n_low']:>3}  ({row['label_method']})")
""")

# ── SECTION 10: TEST B — RF PER (SPECIES × CLASS) ─────────────────────────────
md("""\
## Section 10 — Test B: RF models per (species × ARG class)

**One classifier per passing (species × class) cell.**

Each classifier:
- Features: `FEAT_COLS` (265 dp_* features, same as Phase 8 Q2 — unchanged)
- H1 filter applied per-cell (≥5% prevalence within Q2-eligible genomes for that cell)
- GroupKFold(5) on 95 phylogroups (frozen from Phase 6)
- RF with Phase 8 best hyperparameters
- Primary metric: AUROC (with bootstrap CI)

**BH correction:** Applied across all passing cells within Test B (not across phases).
Threshold: q = 0.05 (adjusted p-value).

**Note on interpretation:** Chance AUROC = 0.5. An AUROC significantly > 0.5 means
defence features predict class-specific ARG burden. The absolute AUROC is less important
than whether it is (a) above chance and (b) consistent with the directional SHAP prediction.
""")

code("""\
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

def run_q2_cell(sp, cls, feat_cols):
    \"\"\"Run Q2 RF for one (species × class) cell. Returns dict or None if skipped.\"\"\"
    if (sp, cls) not in label_store:
        return None

    labels = label_store[(sp, cls)]
    # labels has accession as index (from set_index in Section 9 label construction)
    q2_acc = labels[labels.isin(["high_class", "low_class"])].index

    # Use fm (with accession column) to get features and phylogroup for q2_acc genomes
    sp_q2  = fm[fm["accession"].isin(q2_acc)].set_index("accession")
    sp_q2  = sp_q2.loc[q2_acc]         # reorder to match label order
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
        "ci_lo": round(lo, 3),
        "ci_hi": round(hi, 3),
        "p_raw": float(p_val),
        "n_feat": len(feat_keep),
        "n_folds": len(fold_aurocs),
        "note": "ok",
    }

print("Running Test B RF models (may take 5-10 minutes)...")
testb_raw = {}
for sp in fm["species"].unique():
    for cls in class_cols:
        key = (sp, cls)
        if key not in label_store:
            continue
        r = run_q2_cell(sp, cls, FEAT_COLS)
        if r is not None:
            testb_raw[key] = r
            status = f"AUROC={r.get('auroc','—')} p={r.get('p_raw','—'):.3f}" if r.get("auroc") else r.get("note","?")
            print(f"  {sp:<20} {cls:<20} {status}")
print(f"\\nCompleted {len(testb_raw)} cells.")
""")

code("""\
# BH correction across all passing cells
passing_keys = [(sp, cls) for (sp, cls), r in testb_raw.items() if r.get("note") == "ok"]
p_raws = [testb_raw[k]["p_raw"] for k in passing_keys]

if p_raws:
    reject, p_adj, _, _ = multipletests(p_raws, method="fdr_bh", alpha=0.05)
    for i, key in enumerate(passing_keys):
        testb_raw[key]["p_adj"] = float(p_adj[i])
        testb_raw[key]["significant"] = bool(reject[i])

# Summary table
rows = []
for (sp, cls), r in testb_raw.items():
    fl = floor_df[(floor_df["species"]==sp) & (floor_df["arg_class"]==cls)]
    n_high = int(fl["n_high"].values[0]) if len(fl) else "—"
    n_low  = int(fl["n_low"].values[0])  if len(fl) else "—"
    rows.append({
        "species": sp, "arg_class": cls,
        "AUROC": r.get("auroc"), "CI_lo": r.get("ci_lo"), "CI_hi": r.get("ci_hi"),
        "p_raw": r.get("p_raw"), "p_adj": r.get("p_adj"),
        "significant": r.get("significant", False),
        "n_high": n_high, "n_low": n_low, "n_feat": r.get("n_feat"),
        "note": r.get("note"),
    })

testb_df = pd.DataFrame(rows).sort_values(["arg_class","AUROC"], ascending=[True, False])
testb_df.to_parquet(RES / "testb_results.parquet", index=False)

print("Test B results (BH-corrected, q=0.05):")
print(f"{'Species':<20} {'Class':<20} {'AUROC':>6} {'95% CI':>14} {'p_adj':>8} {'Sig':>5} {'n_H':>5} {'n_L':>5}")
print("-" * 90)
for _, row in testb_df.iterrows():
    if row["note"] != "ok":
        continue
    sig_flag = "★" if row.get("significant") else " "
    ci_str   = f"[{row['CI_lo']:.3f}–{row['CI_hi']:.3f}]"
    print(f"{row['species']:<20} {row['arg_class']:<20} {row['AUROC']:>6.3f} {ci_str:>14} "
          f"{row['p_adj']:>8.4f} {sig_flag:>5} {row['n_high']:>5} {row['n_low']:>5}")
""")

# ── SECTION 11: TEST B — SHAP DIRECTIONALITY ──────────────────────────────────
md("""\
## Section 11 — Test B: SHAP directionality for RM features

**The key biological prediction:**

- **β-lactam and aminoglycoside:** RM systems are gatekeepers against plasmid entry.
  A genome with RM systems is less likely to acquire new plasmid-borne β-lactam ARGs.
  Therefore: `SHAP(dp_RM_Type_I | high_beta_lactam_burden) < 0` — having RM pushes
  the prediction *away* from high β-lactam burden.

- **Quinolone:** GyrA/ParC point mutations are chromosomal — no plasmid is required.
  RM systems have no gate to operate. Therefore: `SHAP(dp_RM | high_quinolone_burden) ≈ 0`

If SHAP directions match these predictions: this is the strongest mechanistic confirmation
of RESTRICT available in this dataset — it shows the principle is class-specific, not
just correlated with generic genome-wide ARG load.

If SHAP does NOT match: the Phase 8 RESTRICT signal may reflect genome-wide correlates
(e.g., IC2 clonal genomes are both RM-poor AND ARG-rich for confounded reasons), not
direct plasmid-restriction biology.

**Technical note:** SHAP is computed in-sample on the full Q2-eligible set for each
significant (species × class) cell. The sign of the SHAP value for `dp_RM_*` features
is the key output — not the magnitude.
""")

code("""\
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

    rm_feats = [c for c in feat_keep if "RM" in c]
    shap_df  = pd.DataFrame(sv, columns=feat_keep)

    rm_shap_means = {}
    for rf_feat in rm_feats:
        mean_signed = shap_df[rf_feat].mean()   # signed mean (not abs)
        rm_shap_means[rf_feat] = float(mean_signed)

    shap_sign_results[(sp, cls)] = rm_shap_means
    print(f"  {sp} / {cls}:")
    for feat, val in sorted(rm_shap_means.items(), key=lambda x: x[1]):
        direction = "← RESTRICTS" if val < -0.001 else ("→ FACILITATES" if val > 0.001 else "≈ 0")
        print(f"    {feat:<30} signed_SHAP={val:+.4f}  {direction}")
    print()
""")

code("""\
# Summary: does SHAP direction match the pre-specified prediction?
print("SHAP direction match vs pre-specified prediction:")
print("─" * 65)
predictions = {
    "beta_lactam":   "negative",   # RM restricts plasmid-mediated
    "aminoglycoside":"negative",
    "sulfonamide":   "negative",
    "trimethoprim":  "negative",
    "quinolone":     "near_zero",  # chromosomal — no RM gating
    "tetracycline":  "ambiguous",
    "glycopeptide":  "negative",
    "macrolide_mlsb":"ambiguous",
    "phenicol":      "ambiguous",
}

for (sp, cls), rm_vals in shap_sign_results.items():
    predicted = predictions.get(cls, "unspecified")
    rm_type_i = rm_vals.get("dp_RM_Type_I", None)
    if rm_type_i is None:
        obs_dir = "RM_I not in feature set"
        match   = "—"
    else:
        if rm_type_i < -0.002:
            obs_dir = "negative (RM restricts)"
        elif rm_type_i > 0.002:
            obs_dir = "positive (unexpected)"
        else:
            obs_dir = "near zero"

        if predicted == "negative":
            match = "MATCH ✓" if rm_type_i < -0.002 else "MISMATCH ✗"
        elif predicted == "near_zero":
            match = "MATCH ✓" if abs(rm_type_i) <= 0.002 else "MISMATCH ✗"
        else:
            match = "(not pre-specified)"

    print(f"{sp:<20} {cls:<18} pred={predicted:<10} obs={obs_dir:<25} {match}")
""")

# ── SECTION 12: VISUALISATION ─────────────────────────────────────────────────
md("""\
## Section 12 — Visualisation: AUROC heatmap and SHAP sign plot

Two figures:

1. **Heatmap of Test B AUROC** by (species × class) — highlights where defence features
   predict mechanism-class ARG burden and where they don't.

2. **RM SHAP sign heatmap** for significant cells — shows direction of RM effect per class.
   Green = negative (RM restricts), red = positive (unexpected), grey = near zero.
""")

code("""\
# Figure 1: Test B AUROC heatmap
pivot_auroc = testb_df[testb_df["note"]=="ok"].pivot_table(
    index="species", columns="arg_class", values="AUROC", aggfunc="first"
)

fig, ax = plt.subplots(figsize=(max(8, len(pivot_auroc.columns)*1.2), 4))
mask_na = pivot_auroc.isna()

sns.heatmap(
    pivot_auroc.fillna(0.5),
    annot=True, fmt=".2f", cmap="RdYlGn",
    vmin=0.45, vmax=0.85, center=0.5,
    linewidths=0.5, ax=ax,
    mask=mask_na,
)
ax.set_title("Test B — Q2 AUROC per (species × ARG class)", fontsize=11)
ax.set_xlabel("ARG mechanism class")
ax.set_ylabel("")

# Mark significant cells with ★
for i, sp in enumerate(pivot_auroc.index):
    for j, cls in enumerate(pivot_auroc.columns):
        key = (sp, cls)
        if testb_raw.get(key, {}).get("significant", False):
            ax.text(j+0.85, i+0.15, "★", color="black", fontsize=8, ha="center", va="center")

plt.tight_layout()
plt.savefig(FIG / "02_testb_auroc_heatmap.pdf", bbox_inches="tight")
plt.show()
print("Figure saved: 02_testb_auroc_heatmap.pdf")
""")

code("""\
# Figure 2: RM SHAP sign heatmap for significant cells
if shap_sign_results:
    sign_rows = []
    for (sp, cls), rm_vals in shap_sign_results.items():
        for feat, val in rm_vals.items():
            sign_rows.append({"species": sp, "arg_class": cls, "rm_feature": feat, "signed_shap": val})
    sign_df = pd.DataFrame(sign_rows)

    if not sign_df.empty:
        pivot_shap = sign_df[sign_df["rm_feature"]=="dp_RM_Type_I"].pivot_table(
            index="species", columns="arg_class", values="signed_shap", aggfunc="first"
        )
        if not pivot_shap.empty:
            fig, ax = plt.subplots(figsize=(max(6, len(pivot_shap.columns)*1.2), 3))
            sns.heatmap(
                pivot_shap, annot=True, fmt="+.3f", cmap="coolwarm_r",
                vmin=-0.02, vmax=0.02, center=0,
                linewidths=0.5, ax=ax,
            )
            ax.set_title("RM Type I signed mean SHAP per (species × ARG class)", fontsize=10)
            ax.set_xlabel("ARG mechanism class")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(FIG / "03_rm_shap_sign_heatmap.pdf", bbox_inches="tight")
            plt.show()
            print("Figure saved: 03_rm_shap_sign_heatmap.pdf")
        else:
            print("No dp_RM_Type_I entries in significant cells — check feature set.")
    else:
        print("No SHAP sign data collected (no significant cells, or SHAP empty).")
else:
    print("No significant cells reached SHAP stage.")
""")

# ── SECTION 13: SYNTHESIS ─────────────────────────────────────────────────────
md("""\
## Section 13 — Synthesis: what Tests A and B tell us together

**Interpreting Test A and Test B together:**

Test A and Test B are *orthogonal* sensitivity analyses — they modify different
parts of the Q2 pipeline without interacting. Their results combine as follows:

| Scenario | Interpretation |
|---|---|
| Test A: ΔAUROC > 0 AND Test B: β-lactam AUROC > quinolone AUROC | Strongest result: RM count matters AND the restriction is class-specific |
| Test A: ΔAUROC ≈ 0 AND Test B: β-lactam AUROC > quinolone | Binary RM adequately captured the signal; class specificity is the new finding |
| Test A: ΔAUROC > 0 AND Test B: no class specificity | Multi-copy RM helps Q2 but the mechanism is non-specific (may reflect genome complexity) |
| Test A: ≈ 0 AND Test B: no class specificity | Phase 8 results are robust; the sensitivity analyses confirm no gains beyond the primary design |

**What this analysis does NOT claim:**
- Causation. SHAP direction shows statistical association, not mechanism.
- Experimental validation. The plasmid-RM gating hypothesis requires conjugation assays.
- Generalisation beyond this 878-genome ESKAPE dataset.

**Manuscript framing:**
Results labelled "confirmatory" (pre-specified directional predictions that are met).
Results labelled "exploratory" (unplanned classes, combined runs, etc.).
""")

code("""\
# Automated synthesis narrative
print("=" * 60)
print("PHASE 12 SYNTHESIS")
print("=" * 60)

# Test A verdict
testa_live = {sp: r for sp, r in testa_results.items()
              if r.get("auroc") is not None and phase8_q2_auroc.get(sp) is not None}
if testa_live:
    deltas = {sp: r["auroc"] - phase8_q2_auroc[sp] for sp, r in testa_live.items()}
    max_gain = max(deltas.values())
    print(f"\\nTest A: Max ΔAUROC from dc_RM substitution = {max_gain:+.3f}")
    if max_gain > 0.01:
        print("  → dc_RM improves Q2 prediction. Multi-copy RM signal is real.")
    else:
        print("  → RM is effectively binary in this dataset. dc ≈ dp finding reported.")

# Test B verdict
sig_cells = [(sp, cls) for (sp, cls) in testb_raw
             if testb_raw[(sp, cls)].get("significant", False)]
print(f"\\nTest B: {len(sig_cells)} significant (species × class) cells (BH q=0.05):")
for sp, cls in sig_cells:
    r = testb_raw[(sp, cls)]
    print(f"  {sp:<20} {cls:<20} AUROC={r['auroc']:.3f} p_adj={r['p_adj']:.4f}")

# SHAP direction check
matched   = [(sp, cls) for (sp, cls) in shap_sign_results
             if cls in ("beta_lactam", "aminoglycoside", "sulfonamide", "trimethoprim")
             and shap_sign_results[(sp, cls)].get("dp_RM_Type_I", 0) < -0.002]
mismatched = [(sp, cls) for (sp, cls) in shap_sign_results
              if cls in ("beta_lactam", "aminoglycoside", "sulfonamide", "trimethoprim")
              and shap_sign_results[(sp, cls)].get("dp_RM_Type_I", 0) > 0.002]
quinolone_cells = [(sp, cls) for (sp, cls) in shap_sign_results
                   if cls == "quinolone"
                   and abs(shap_sign_results[(sp, cls)].get("dp_RM_Type_I", 1)) <= 0.002]

print(f"\\nSHAP direction (plasmid classes — RM negative predicted):")
print(f"  Matches (RM → negative SHAP): {matched}")
print(f"  Mismatches (RM → positive): {mismatched}")
print(f"  Quinolone ≈ 0 (expected): {quinolone_cells}")
print()
print("Phase 12 complete. Figures in results/figures/phase12/")
print("Results table: results/testb_results.parquet")
""")

# ── SECTION 14: COMPREHENSION CHECK ───────────────────────────────────────────
md("""\
## Section 14 — Comprehension check

Answer these before ending the session. Do not look at the results before answering.

**Q1.** Test A replaces dp_RM_* with dc_RM_* and observes ΔAUROC ≈ 0 for all species.
A colleague says "this means RM systems have no effect on ARG burden." What is the
correct interpretation, and what would the data need to show for RM to be truly
uninformative for ARG burden prediction?

**Q2.** Test B shows β-lactam AUROC = 0.81 (BH-significant) in KP and quinolone
AUROC = 0.52 (not significant) in the same species. A reviewer asks "couldn't this just
reflect that β-lactam ARG counts are more variable than quinolone ARG counts?" How do
you respond — and what feature in your pre-registration addresses this concern?

**Q3.** SHAP for dp_RM_Type_I in the β-lactam KP cell is +0.003 (small positive).
You predicted negative. Name TWO explanations for this mismatch that do not require
abandoning the RESTRICT hypothesis.

**Q4.** If Test B shows that aminoglycoside and β-lactam AUROC are both significantly
above chance for EC but quinolone is not, write a single sentence you would put in the
Results section. It must: (a) state the finding, (b) name the correct metric, (c) not
use causal language.
""")

# ── FINALIZE ───────────────────────────────────────────────────────────────────
nb.cells = cells
out_path = Path("notebooks/10_phase12_sensitivity.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {out_path}")
