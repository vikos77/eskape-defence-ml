"""Append Section 3 (RESTRICT/FACILITATE Spearman check) to notebooks/01_eda.ipynb."""
import json

def md(text, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": [text]}

def code(text, cell_id):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": [text]}

nb = json.load(open("notebooks/01_eda.ipynb"))
cells = nb["cells"]

# ── Section 3 header ─────────────────────────────────────────────────────────
cells.append(md("""\
## Section 3 — RESTRICT/FACILITATE check

### What we are testing and why

The published *JAM* paper's central biological claim is a **dichotomy**:
- **RESTRICT systems** (primarily RM types): negatively correlated with ARG, IME,
  and HMRG counts. These systems block MGE entry, preventing ARG acquisition.
- **FACILITATE systems** (SspBCDE, Gao_Qat): *positively* correlated with ARG
  and IME counts. These systems live on MGEs — they arrived in the genome via
  horizontal transfer alongside the ARGs they co-occur with.

If this dichotomy is a general principle of defence biology rather than an
*Acinetobacter*-specific phenomenon, we expect:
1. RM systems negatively correlated with ARG count **in all six ESKAPE species**
2. SspBCDE positively correlated with ARG in AB (replication), but the
   correlation is not testable in species where SspBCDE prevalence ≈ 0%
3. The per-species correlations should all be negative for RM; if any species
   shows a consistently positive RM–ARG correlation, that species is an exception
   requiring biological explanation

**What we do NOT do:** We do not pool all 878 genomes and compute one correlation.
Section 1 Q3 established why: the pooled signal is dominated by between-species
ecological differences, not within-species biology. KP has both high defence AND
high ARG, so the pooled correlation for defence vs ARG would be positive — the
opposite of what the RESTRICT hypothesis predicts. Per-species correlations are
the only valid test.

**Statistical note:** We use Spearman's ρ throughout, consistent with the published
paper. Defence counts are right-skewed integers; Spearman ranks, so it is
distribution-free. We apply BH correction across all tests per species (one species
at a time, correcting within the tests run for that species).""", "eda-s3-md"))

# ── Core Spearman correlations ────────────────────────────────────────────────
cells.append(code("""\
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ── Define which variables to test against ARG ────────────────────────────────
# dc_* (count) for RM types so we capture dose-dependence
# dp_* (presence) for SspBCDE and Gao_Qat because they are near-absent in most species
# and count would be mostly 0 with rare 1/2
RESTRICT_VARS = ["dc_RM_Type_I", "dc_RM_Type_II", "dc_RM_Type_III", "dc_RM_Type_IV"]
FACILITATE_VARS = ["dp_SspBCDE", "dp_Gao_Qat"]
SUMMARY_VARS_CORR = ["defence_system_count", "adef_system_count",
                     "ime_count_unique", "is_count_total"]
TARGET = "arg_count_unique"

ALL_VARS = RESTRICT_VARS + FACILITATE_VARS + SUMMARY_VARS_CORR

# ── Filter: only include vars that exist in the feature matrix ────────────────
ALL_VARS = [v for v in ALL_VARS if v in fm.columns]

print(f"Testing {len(ALL_VARS)} variables against {TARGET} per species")
print(f"Restrict: {RESTRICT_VARS}")
print(f"Facilitate: {FACILITATE_VARS}")
print()

# ── Per-species Spearman + BH correction ─────────────────────────────────────
results = []
for sp in SPECIES_ORDER:
    sub = fm[fm["species"] == sp].copy()
    rhos, pvals, var_names = [], [], []
    for var in ALL_VARS:
        if var not in sub.columns:
            continue
        # Drop rows where either variable is NaN
        pair = sub[[var, TARGET]].dropna()
        if len(pair) < 10 or pair[var].nunique() < 2:
            # Skip if too few observations or zero variance
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(pair[var], pair[TARGET])
        rhos.append(rho)
        pvals.append(p)
        var_names.append(var)

    # BH correction within this species
    valid = [(i, p) for i, p in enumerate(pvals) if not np.isnan(p)]
    padj = [np.nan] * len(pvals)
    if valid:
        idx_v, pv = zip(*valid)
        _, p_adj, _, _ = multipletests(list(pv), method="fdr_bh")
        for i, pa in zip(idx_v, p_adj):
            padj[i] = pa

    for var, rho, p, pa in zip(var_names, rhos, pvals, padj):
        results.append({
            "species": SPECIES_LABELS[sp],
            "variable": var,
            "rho": rho,
            "p_raw": p,
            "p_adj_BH": pa,
            "sig": "*" if pa < 0.05 else ("†" if pa < 0.10 else ""),
        })

corr_df = pd.DataFrame(results)

# ── Print per-species results for RM types ────────────────────────────────────
print("Spearman ρ: RM systems vs ARG count (per species, BH-corrected)")
print("* = FDR < 0.05    † = FDR < 0.10")
print("=" * 85)
for sp_label in [SPECIES_LABELS[s] for s in SPECIES_ORDER]:
    sp_data = corr_df[
        (corr_df["species"] == sp_label) &
        (corr_df["variable"].isin(RESTRICT_VARS + FACILITATE_VARS))
    ]
    print(f"\\n{sp_label}:")
    for _, row in sp_data.iterrows():
        rho_str = f"{row['rho']:+.3f}" if not np.isnan(row["rho"]) else "  n/a "
        p_str   = f"{row['p_adj_BH']:.3f}" if not np.isnan(row["p_adj_BH"]) else "  n/a"
        print(f"  {row['variable']:<25} ρ={rho_str}  p_adj={p_str}  {row['sig']}")""",
"eda-s3-spearman"))

# ── Summary heatmap of correlations ──────────────────────────────────────────
cells.append(code("""\
# ── Correlation heatmap: all tested variables × all species ──────────────────
pivot_rho = corr_df.pivot(index="variable", columns="species", values="rho")
pivot_sig = corr_df.pivot(index="variable", columns="species", values="sig")

# Reorder columns to SPECIES_ORDER
sp_labels_ordered = [SPECIES_LABELS[s] for s in SPECIES_ORDER]
pivot_rho = pivot_rho[sp_labels_ordered]
pivot_sig = pivot_sig[sp_labels_ordered]

# Row order: RM restrict → facilitate → summary
var_order = [v for v in RESTRICT_VARS + FACILITATE_VARS + SUMMARY_VARS_CORR
             if v in pivot_rho.index]
pivot_rho = pivot_rho.loc[var_order]
pivot_sig = pivot_sig.loc[var_order]

# Clean up row labels
row_labels = {
    "dc_RM_Type_I":        "RM Type I (count)",
    "dc_RM_Type_II":       "RM Type II (count)",
    "dc_RM_Type_III":      "RM Type III (count)",
    "dc_RM_Type_IV":       "RM Type IV (count)",
    "dp_SspBCDE":          "SspBCDE (P/A)  [FACILITATE]",
    "dp_Gao_Qat":          "Gao_Qat (P/A)  [FACILITATE]",
    "defence_system_count":"Total defence count",
    "adef_system_count":   "Anti-defence count",
    "ime_count_unique":    "IME count",
    "is_count_total":      "IS element count",
}
pivot_rho.index = [row_labels.get(v, v) for v in pivot_rho.index]
pivot_sig.index = pivot_rho.index

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    pivot_rho,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-0.6, vmax=0.6,
    annot=pivot_rho.round(2),
    fmt=".2f",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Spearman ρ", "shrink": 0.6},
)

# Overlay significance stars
for row_i, row_name in enumerate(pivot_rho.index):
    for col_i, col_name in enumerate(pivot_rho.columns):
        sig = pivot_sig.loc[row_name, col_name]
        if sig:
            ax.text(col_i + 0.85, row_i + 0.25, sig,
                    ha="center", va="center", fontsize=11, color="black", weight="bold")

ax.set_title(
    "Spearman ρ: defence/MGE features vs ARG count, per species\\n"
    "* FDR < 0.05   † FDR < 0.10   (BH correction within each species)",
    fontsize=10, pad=10,
)
ax.set_xlabel("Species", fontsize=10)
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_restrict_facilitate_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/eda/05_restrict_facilitate_heatmap.png")""",
"eda-s3-heatmap"))

# ── Species-level scatter: defence_system_count vs arg_count_unique ──────────
cells.append(code("""\
# ── Scatter: defence_system_count vs arg_count_unique, faceted by species ─────
# This is the most direct visual for the RESTRICT/FACILITATE pattern:
# negative slope = more defence → fewer ARGs (RESTRICT dominates)
# positive slope = more defence → more ARGs (facilitative systems dominate)
# flat = no relationship within this species

fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
axes = axes.flatten()
fig.suptitle(
    "Defence system count vs ARG count per species\\n"
    "Line = Spearman regression-equivalent LOWESS smoother",
    fontsize=11, y=1.01
)

from scipy.stats import spearmanr
import statsmodels.api as sm

for ax, sp in zip(axes, SPECIES_ORDER):
    sub = fm[fm["species"] == sp]
    x = sub["defence_system_count"].values
    y = sub["arg_count_unique"].values

    rho, p = spearmanr(x, y)
    sig_label = "*" if p < 0.05 else ""

    ax.scatter(x, y, alpha=0.35, s=18, color="#4878CF", edgecolors="none")

    # LOWESS smoother
    lowess = sm.nonparametric.lowess(y, x, frac=0.5)
    ax.plot(lowess[:, 0], lowess[:, 1], color="#D65F5F", lw=2)

    ax.set_title(
        f"{SPECIES_LABELS[sp]}\\nρ={rho:+.2f}{sig_label}  (n={len(sub)})",
        fontsize=9
    )
    ax.set_xlabel("Defence system count", fontsize=8)
    ax.set_ylabel("ARG count (unique)", fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(FIG_DIR / "06_defence_arg_scatter_facet.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/eda/06_defence_arg_scatter_facet.png")""",
"eda-s3-scatter"))

# ── Interpretive note ─────────────────────────────────────────────────────────
cells.append(md("""\
### How to read the RESTRICT/FACILITATE heatmap

**Negative ρ (blue cells) for RM rows = RESTRICT signal present in this species.**
The more RM systems a genome carries, the fewer ARGs it has. This is the published
*Acinetobacter* finding — if it generalises, all RM rows should be blue across all
six species.

**Positive ρ (red cells) for SspBCDE / Gao_Qat = FACILITATE signal.**
These are MGE-borne; genomes that acquired them via HGT also acquired ARGs on the
same mobile element. Only testable in AB where both systems have non-trivial
prevalence.

**IME count row:** Should be positive — more IMEs = more horizontal transfer
vehicles = more opportunity for ARG acquisition. If IME ρ is positive across
all species, this confirms IMEs as ARG vectors regardless of species.

**IS element row:** Also expected positive, but for a different reason — IS elements
amplify and rearrange existing ARG genes rather than importing new ones. The
relationship may be weaker than IME, or may be driven by specific IS families
(IS26 in KP being a well-known ARG-capture element).

**Exceptions:** A species with a positive RM–ARG correlation is a genuine anomaly.
It could mean RM systems in that species are on plasmids that also carry ARGs
(i.e., RM itself is the cargo of a resistance plasmid, not the gatekeeper).
Investigate before modelling.""", "eda-s3-note"))

# ── Comprehension check ───────────────────────────────────────────────────────
cells.append(md("""\
### Section 3 comprehension check

**Q1.** You find that RM_Type_I has ρ = −0.30 in K. pneumoniae (FDR = 0.001) and
ρ = +0.05 in S. aureus (FDR = 0.70). What is the correct biological interpretation
of the SA result — and does the positive sign in SA falsify the RESTRICT/FACILITATE
hypothesis?

**Q2.** The IME count vs ARG count correlation is strongly positive in KP (ρ = +0.55)
but near-zero in S. aureus (ρ = +0.03). Give a mechanistic explanation for why
ICE/IME-mediated ARG acquisition would be strong in a gram-negative (KP) but weak
in a gram-positive (SA).

**Q3.** You want to include this RESTRICT/FACILITATE analysis in the paper's Results
section. A reviewer demands a multivariate model (e.g., multiple regression of ARG
count on all RM types simultaneously). Why might the reviewer's demand be justified —
and what would the multivariate model tell you that the pairwise Spearman correlations
cannot?""", "eda-s3-cc"))

# ── Write updated notebook ────────────────────────────────────────────────────
nb["cells"] = cells
with open("notebooks/01_eda.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

nb2 = json.load(open("notebooks/01_eda.ipynb"))
print(f"Updated notebooks/01_eda.ipynb: {len(nb2['cells'])} cells total")
