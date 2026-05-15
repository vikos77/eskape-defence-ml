"""Append Section 2 (defence system prevalence heatmap) to notebooks/01_eda.ipynb."""
import json

def md(text, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": [text]}

def code(text, cell_id):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": [text]}

nb = json.load(open("notebooks/01_eda.ipynb"))
cells = nb["cells"]

# ── Section 2 header ─────────────────────────────────────────────────────────
cells.append(md("""\
## Section 2 — Defence system prevalence

### What we are looking for and why

The summary table in Section 1 told you *how many* defence systems each species
carries on average. This section reveals *which* systems they are.

**The core question:** Are defence systems species-specific (each species has its own
repertoire) or generalist (the same systems appear across all six species)?

If defence systems are species-specific, Q1 classification (predict species from
defence repertoire) will work easily. If they are generalist, the classifier must
find subtle *abundance* differences rather than presence/absence differences.

**What we plot:** A heatmap where:
- Rows = individual defence system types (274 total from DefenseFinder)
- Columns = the six ESKAPE species
- Cell colour = fraction of genomes in that species carrying that system (0–1)
- Rows are hierarchically clustered so systems with similar prevalence patterns
  cluster together — the clustering itself is biologically informative

**Three things to read from this plot:**
1. Near-universal systems (hot across all columns): present in >80% of genomes
   of every species — uninformative for Q1 but may be important for Q2 if they
   vary in count
2. Species-exclusive systems (hot in one column, cold in all others): these are
   Q1's strongest features. If a system appears in 90% of PA genomes and 0% of
   AB genomes, it nearly classifies the species on its own
3. The A. baumannii column: should be notably sparser than others, consistent
   with the published paper's finding of depauperate AB defence repertoires""",
"eda-s2-md"))

# ── Prevalence matrix computation ─────────────────────────────────────────────
cells.append(code("""\
# ── Compute per-species prevalence for each defence system type ───────────────
# dp_* columns are binary (0/1). Mean per species = fraction of genomes carrying
# each system. Result: 274 systems × 6 species matrix.

dp_cols = [c for c in fm.columns if c.startswith("dp_")]

prev = pd.DataFrame(
    {sp: fm[fm["species"] == sp][dp_cols].mean() for sp in SPECIES_ORDER},
    index=dp_cols,
)
prev.index = [c.replace("dp_", "") for c in prev.index]  # strip prefix for legibility
prev.columns = [SPECIES_LABELS[s] for s in SPECIES_ORDER]

print(f"Prevalence matrix shape: {prev.shape}  (defence types × species)")
print()

# ── How many systems are near-universal? (>80% in all 6 species) ─────────────
near_universal = prev[(prev > 0.80).all(axis=1)]
print(f"Near-universal systems (>80% in ALL 6 species): {len(near_universal)}")
if len(near_universal):
    print(near_universal.to_string())
print()

# ── How many are completely absent from AB? (prevalence = 0) ─────────────────
ab_absent = prev[prev["A. baumannii"] == 0]
print(f"Systems absent from ALL A. baumannii genomes: {len(ab_absent)}")
print()

# ── Top 10 most prevalent systems per species ─────────────────────────────────
print("Top 5 most prevalent systems per species:")
for sp_label in [SPECIES_LABELS[s] for s in SPECIES_ORDER]:
    top5 = prev[sp_label].nlargest(5)
    print(f"  {sp_label}:")
    for sys_name, val in top5.items():
        print(f"    {sys_name:<35} {val:.2f}")
    print()""", "eda-s2-prev"))

# ── Heatmap — full 274 × 6 ────────────────────────────────────────────────────
cells.append(code("""\
# ── Full prevalence heatmap (hierarchically clustered rows) ──────────────────
# seaborn clustermap clusters both rows and columns by default.
# col_cluster=False keeps species in our fixed order.
# Row clustering groups systems with similar cross-species patterns.

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Filter: keep only systems present in ≥1 genome (prevalence > 0 in any species)
prev_nonzero = prev[(prev > 0).any(axis=1)]
print(f"Systems with any presence: {len(prev_nonzero)} of {len(prev)}")

g = sns.clustermap(
    prev_nonzero,
    col_cluster=False,         # keep species columns in SPECIES_ORDER
    row_cluster=True,          # cluster defence types by co-occurrence pattern
    cmap="YlOrRd",
    figsize=(8, 20),
    xticklabels=True,
    yticklabels=False,         # 274 labels illegible at this size
    linewidths=0,
    cbar_pos=(1.02, 0.3, 0.03, 0.4),
    cbar_kws={"label": "Prevalence (fraction of genomes)"},
)
g.ax_heatmap.set_xlabel("Species", fontsize=11)
g.ax_heatmap.set_ylabel(f"Defence system types (n={len(prev_nonzero)})", fontsize=11)
g.fig.suptitle(
    "Defence system prevalence across ESKAPE species\\n"
    "(rows clustered by co-occurrence pattern; columns fixed)",
    y=1.01, fontsize=12,
)
plt.savefig(FIG_DIR / "03_defence_prevalence_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/eda/03_defence_prevalence_heatmap.png")""", "eda-s2-heatmap"))

# ── Zoom: top-50 most variable systems ────────────────────────────────────────
cells.append(code("""\
# ── Zoom heatmap: top 50 systems by cross-species variance ────────────────────
# The full heatmap has 274 rows — most of them near-zero.
# Selecting systems with high cross-species variance shows the discriminating ones.

row_variance = prev_nonzero.var(axis=1)
top50 = prev_nonzero.loc[row_variance.nlargest(50).index]
top50 = top50.sort_values(by=list(top50.columns), ascending=False)

fig, ax = plt.subplots(figsize=(8, 14))
sns.heatmap(
    top50,
    ax=ax,
    cmap="YlOrRd",
    vmin=0, vmax=1,
    linewidths=0.4,
    linecolor="white",
    xticklabels=True,
    yticklabels=True,
    cbar_kws={"label": "Prevalence (fraction of genomes)", "shrink": 0.4},
)
ax.set_title(
    "Top 50 defence systems by cross-species prevalence variance",
    fontsize=12, pad=12,
)
ax.set_xlabel("Species", fontsize=10)
ax.set_ylabel("Defence system", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_defence_prevalence_top50.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/eda/04_defence_prevalence_top50.png")""", "eda-s2-zoom"))

# ── AB-specific check: published systems ─────────────────────────────────────
cells.append(code("""\
# ── Check prevalence of the published key systems in all 6 species ────────────
# The JAM paper identified RM systems, SspBCDE, and Gao_Qat as the main
# discriminating systems in Acinetobacter. Check if they appear in the other
# five species — if yes, cross-species generalisation is plausible.

KEY_SYSTEMS = [
    "RM_Type_I",
    "RM_Type_II",
    "RM_Type_III",
    "RM_Type_IV",
    "SspBCDE",
    "Gao_Qat",
    "Gabija",
    "Druantia",
    "Lamassu",
    "Thoeris",
    "Zorya",
    "CRISPR_Cas",
]

found_systems = [s for s in KEY_SYSTEMS if s in prev.index]
missing_systems = [s for s in KEY_SYSTEMS if s not in prev.index]

print("Published key systems — prevalence across species:")
print("=" * 90)
print(prev.loc[found_systems].to_string(float_format=lambda x: f"{x:.2f}"))
print()
if missing_systems:
    print(f"Not found in feature matrix (may be absent from all genomes): {missing_systems}")""",
"eda-s2-keysystems"))

# ── Interpretive note ─────────────────────────────────────────────────────────
cells.append(md("""\
### What to read from these plots

**Full heatmap (03):** The vertical stripes of colour tell you each species'
overall defence density. A. baumannii should appear notably paler than KP.
The horizontal bands of colour are system *co-occurrence clusters* — groups of
systems that tend to appear together. These clusters often reflect shared MGE
carriers: if five systems are always on the same ICE, they will always be
co-present and will form a tight cluster.

**Top-50 zoom (04):** These are the systems with the highest cross-species
*variance* in prevalence — meaning they discriminate species most effectively.
A system that is 90% prevalent in PA but 5% in SA has high variance. These
50 systems are the ones a Q1 classifier will lean on hardest.

**Key systems table:** The published paper found RM_Type_I, SspBCDE, and
Gao_Qat as the main Acinetobacter discriminators. Their prevalence in the
other five ESKAPE species determines whether Q4 SHAP results can replicate
the published ranking. If SspBCDE appears only in AB, the SHAP comparison is
trivial. If it appears across species at variable rates, the comparison is
meaningful.

**The A. baumannii column test:** If AB is consistently the palest column in
both heatmaps, the published finding holds in the new data. If AB is not the
palest, something unexpected is in the RefSeq genome selection — investigate
before modelling.""", "eda-s2-note"))

# ── Comprehension check ───────────────────────────────────────────────────────
cells.append(md("""\
### Section 2 comprehension check

**Q1.** The clustermap clusters rows (defence systems) but not columns (species).
Why is it correct to keep columns in a fixed order here rather than letting the
algorithm cluster them too?

**Q2.** You find that RM_Type_II has prevalence 0.88 in K. pneumoniae and 0.05
in A. baumannii. You also find that RM_Type_II is the top-ranked feature in the
Q1 Random Forest (Phase 7). A colleague says: "This just means the RF learned
that KP has RM and AB doesn't — that's phylogenetics, not defence biology. The
result is trivial." Is this critique valid? How would you respond?

**Q3.** You see a tight cluster of 8 systems that are all 0 in S. aureus and
E. faecium but present at 60–80% in the four gram-negative species. Before
looking up the system names, predict the most likely biological explanation for
this gram-stain-stratified pattern.""", "eda-s2-cc"))

# ── Write updated notebook ────────────────────────────────────────────────────
nb["cells"] = cells
with open("notebooks/01_eda.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

nb2 = json.load(open("notebooks/01_eda.ipynb"))
print(f"Updated notebooks/01_eda.ipynb: {len(nb2['cells'])} cells total")
