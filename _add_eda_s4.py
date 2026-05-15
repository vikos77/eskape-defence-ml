"""Append Section 4 (variance decomposition) to notebooks/01_eda.ipynb."""
import json

def md(text, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": [text]}

def code(text, cell_id):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": [text]}

nb = json.load(open("notebooks/01_eda.ipynb"))
cells = nb["cells"]

# ── Section 4 header ─────────────────────────────────────────────────────────
cells.append(md("""\
## Section 4 — Variance decomposition: between-species vs within-species

### What we are computing and why

Every feature column has a total variance across all 878 genomes. That variance
has two sources:
1. **Between-species variance** — the KP mean differs from the AB mean; the
   differences between species means drives this component.
2. **Within-species variance** — even within KP, not all 132 KP genomes are
   identical; genome-to-genome variation within species drives this component.

The ratio of between-species variance to total variance is **η² (eta-squared)**
from a one-way ANOVA of feature ~ species:

```
η² = SS_between / SS_total
```

- η² ≈ 1.0: almost all variance is between species → this feature is a
  **species marker**. Useful for Q1 (species classification); risky for Q2
  (predicts ARG by detecting species, not within-species biology).
- η² ≈ 0.0: almost all variance is within species → this feature varies
  primarily *within* each species. This is the signal Q2 needs — it reflects
  individual-genome biology, not species identity.
- η² ≈ 0.3–0.7: mixed → informative for both, but using in Q2 requires care.

**Why this matters for modelling:** If you include high-η² features in a Q2
(ARG burden) model, the model will partly classify species (which has a
predictable ARG baseline) rather than learning the within-species
defence→ARG relationship. This is a subtle form of confounding — not
technically leakage (test data still unseen), but biologically misleading.
The Phase 6+ models will use this decomposition to decide which features
are appropriate for Q1 vs Q2.""", "eda-s4-md"))

# ── Compute η² for all feature columns ───────────────────────────────────────
cells.append(code("""\
from scipy import stats as scipy_stats

feat_cols = [c for c in fm.columns if c not in
             ["species", "arg_burden_tertile", "country", "year_bin",
              "complex_member", "sequence_type", "mlst_scheme"]]

print(f"Computing η² for {len(feat_cols)} feature columns ...")

eta2_vals = {}
for col in feat_cols:
    groups = [fm[fm["species"] == sp][col].dropna().values for sp in SPECIES_ORDER]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        eta2_vals[col] = np.nan
        continue
    # One-way ANOVA
    grand_mean = fm[col].mean()
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2 for g in groups
    )
    ss_total = sum(((v - grand_mean) ** 2) for g in groups for v in g)
    eta2_vals[col] = ss_between / ss_total if ss_total > 0 else np.nan

eta2 = pd.Series(eta2_vals, name="eta2").dropna().sort_values(ascending=False)

print(f"η² computed for {len(eta2)} features")
print()
print("Distribution of η² across all features:")
for cut, label in [(0.8, ">0.80 (strong species marker)"),
                   (0.5, "0.50–0.80 (moderate)"),
                   (0.20, "0.20–0.50 (mixed)"),
                   (0.0,  "<0.20 (mainly within-species)")]:
    n = (eta2 > cut).sum() if cut > 0 else len(eta2)
    if cut == 0.8:
        n_band = (eta2 > 0.8).sum()
    elif cut == 0.5:
        n_band = ((eta2 > 0.5) & (eta2 <= 0.8)).sum()
    elif cut == 0.20:
        n_band = ((eta2 > 0.20) & (eta2 <= 0.5)).sum()
    else:
        n_band = (eta2 <= 0.20).sum()
    print(f"  {label}: {n_band} features  ({100*n_band/len(eta2):.1f}%)")

print()
print("Top 20 features by η² (strongest species markers):")
print(eta2.head(20).to_string())
print()
print("Bottom 20 features by η² (most within-species variation):")
print(eta2.tail(20).to_string())""", "eda-s4-eta2"))

# ── Visualise distribution ────────────────────────────────────────────────────
cells.append(code("""\
# ── Distribution of η² + highlight key features ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Variance decomposition: between-species (η²) across all 624 features",
             fontsize=12, y=1.01)

# Left: histogram of all η² values
ax = axes[0]
ax.hist(eta2.values, bins=40, color="#4878CF", alpha=0.75, edgecolor="none")
ax.axvline(0.5, color="red", lw=1.5, ls="--", label="η²=0.50")
ax.axvline(0.2, color="orange", lw=1.5, ls="--", label="η²=0.20")
ax.set_xlabel("η² (between-species fraction of total variance)", fontsize=10)
ax.set_ylabel("Number of features", fontsize=10)
ax.set_title("Distribution across all features", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# Right: scatter — Q1 vs Q2 candidates among the dp_* and dc_* columns
# x-axis = η²; y-axis = within-species mean absolute deviation (MAD)
# Features in top-right: high between AND within variation — useful for both Q1 and Q2
ax2 = axes[1]

dp_eta2 = eta2[[c for c in eta2.index if c.startswith("dp_")]]
within_mad = {}
for col in dp_eta2.index:
    mads = []
    for sp in SPECIES_ORDER:
        s = fm[fm["species"] == sp][col]
        mads.append((s - s.mean()).abs().mean())
    within_mad[col] = np.mean(mads)

within_mad_s = pd.Series(within_mad)
ax2.scatter(dp_eta2.values, within_mad_s[dp_eta2.index].values,
            alpha=0.35, s=12, color="#6ACC65", edgecolors="none")

# Annotate the key published systems
key_systems = {
    "dp_SspBCDE": "SspBCDE",
    "dp_Gao_Qat": "Gao_Qat",
    "dp_RM_Type_I": "RM_I",
    "dp_RM_Type_IV": "RM_IV",
    "dp_Gabija": "Gabija",
}
for col, label in key_systems.items():
    if col in dp_eta2.index:
        ax2.annotate(
            label,
            xy=(dp_eta2[col], within_mad_s[col]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=8, color="darkred",
        )
        ax2.scatter([dp_eta2[col]], [within_mad_s[col]],
                    s=50, color="darkred", zorder=5)

ax2.axvline(0.5, color="red", lw=1, ls="--", alpha=0.5, label="η²=0.50")
ax2.axvline(0.2, color="orange", lw=1, ls="--", alpha=0.5, label="η²=0.20")
ax2.set_xlabel("η² (between-species variance fraction)", fontsize=10)
ax2.set_ylabel("Mean within-species MAD", fontsize=10)
ax2.set_title("dp_* features: Q1 vs Q2 signal\\n(right = species marker; top = within-species variable)",
              fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(FIG_DIR / "07_variance_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/eda/07_variance_decomposition.png")""", "eda-s4-plot"))

# ── Key features: where do the published systems sit? ────────────────────────
cells.append(code("""\
# ── Where do the published key systems sit in η² space? ─────────────────────
key_cols = {
    "dp_RM_Type_I":   "RM_Type_I (P/A)",
    "dp_RM_Type_II":  "RM_Type_II (P/A)",
    "dp_RM_Type_III": "RM_Type_III (P/A)",
    "dp_RM_Type_IV":  "RM_Type_IV (P/A)",
    "dp_SspBCDE":     "SspBCDE (P/A)",
    "dp_Gao_Qat":     "Gao_Qat (P/A)",
    "dp_Gabija":      "Gabija (P/A)",
    "dc_RM_Type_I":   "RM_Type_I (count)",
    "dc_RM_Type_IV":  "RM_Type_IV (count)",
    "dc_SspBCDE":     "SspBCDE (count)" if "dc_SspBCDE" in eta2.index else None,
    "defence_system_count": "Total defence count",
    "arg_count_unique":     "ARG count (unique)  [TARGET]",
    "ime_count_unique":     "IME count (unique)",
    "is_count_total":       "IS count (total)",
}
key_cols = {k: v for k, v in key_cols.items() if v is not None and k in eta2.index}

print("η² for published key systems and summary features:")
print("=" * 65)
print(f"{'Feature':<35} {'η²':>8}   Interpretation")
print("-" * 65)
for col, label in key_cols.items():
    e = eta2[col]
    if e >= 0.8:
        interp = "strong species marker (Q1)"
    elif e >= 0.5:
        interp = "moderate species signal"
    elif e >= 0.2:
        interp = "mixed — both Q1 and Q2"
    else:
        interp = "mainly within-species (Q2)"
    print(f"{label:<35} {e:>8.3f}   {interp}")""", "eda-s4-keycols"))

# ── Interpretive note ─────────────────────────────────────────────────────────
cells.append(md("""\
### How to use this for modelling decisions

**High η² features (>0.5):** These will dominate Q1 classification easily.
They are risky for Q2 — if included, the Q2 model will partly classify species
(high-ARG baseline = KP; low-ARG baseline = SA) rather than learning within-species
biology. Phase 6 will investigate whether removing high-η² features changes Q2
performance.

**Low η² features (<0.2):** These are the Q2 candidates. They vary within each
species, meaning they might discriminate high-ARG from low-ARG genomes of the
same species. These are the features the RESTRICT/FACILITATE biological story
is built on.

**The target variable (arg_count_unique) η²:** If the ARG count has high η², it
means ARG burden is largely predictable from species identity alone. If it has
low η², most of the ARG variation is within-species. This number determines how
hard Q2 actually is: if arg_count_unique η² is high, Q2 is partly a species
classification problem in disguise.""", "eda-s4-note"))

# ── Comprehension check ───────────────────────────────────────────────────────
cells.append(md("""\
### Section 4 comprehension check

**Q1.** A feature has η² = 0.85. You include it in a Q2 (ARG burden prediction)
model trained on all 878 genomes with species as a covariate (i.e., species is
also a feature). Does the η² = 0.85 feature still cause a problem? Or does
including species as a covariate fix it?

**Q2.** arg_count_unique (the Q2 target variable) has some η². If η² were 0.80
for the target, what would this imply about the difficulty of Q2 — and what
would you expect the null baseline accuracy to look like?

**Q3.** You find that SspBCDE has η² = 0.45. This means it has both between-species
AND within-species variance. For Q1 it might help classify AB (high prevalence)
from other species. For Q2 in AB specifically, it should correlate with high ARG
burden (per Section 3). But when you run Q2 across all species pooled, what
confound does η² = 0.45 on SspBCDE introduce — and how do you avoid it?""",
"eda-s4-cc"))

# ── Write updated notebook ────────────────────────────────────────────────────
nb["cells"] = cells
with open("notebooks/01_eda.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

nb2 = json.load(open("notebooks/01_eda.ipynb"))
print(f"Updated notebooks/01_eda.ipynb: {len(nb2['cells'])} cells total")
