"""
Supplementary Figure — Q3b multi-block clustering comparison (marker-filtered).

Data source: results/q3b_filtered_results.json
             results/q3b_367_results.json   (unfiltered values for annotation only)
Output:      results/figures/q3b/sfig_q3b_block_comparison.png  (300 dpi)
             results/figures/q3b/sfig_q3b_block_comparison.pdf

Design:
- 5 bars: Defence (baseline), IS filtered, HMRG filtered, ARG filtered, Defence+IS combined.
- Near-exclusive species markers (spec_score >= 0.70) removed from all blocks before clustering.
- ARG and HMRG show pre-filter values as ghost bars to quantify taxonomic inflation.
- All conditions significant by permutation test (p=0.002, 500 permutations).
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT_DIR = "results/figures/q3b"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open("results/q3b_filtered_results.json") as f:
    filt = json.load(f)

with open("results/q3b_367_results.json") as f:
    unfilt = json.load(f)

# Unfiltered reference values for ghost annotation
UNFILT = {
    "is_filt":   unfilt["is_only"]["ari_primary"],     # 0.304
    "arg_filt":  unfilt["arg_only"]["ari_primary"],    # 0.557
    "hmrg_filt": unfilt["hmrg_only"]["ari_primary"],   # 0.380
}

# ── Conditions — ordered left to right ───────────────────────────────────────
CONDITIONS = [
    ("defence_only", "Defence\n(359 feat)",        "#0072B2", False),  # Okabe-Ito blue
    ("is_filt",      "IS elements†\n(19 feat)",      "#009E73", False),  # Okabe-Ito bluish green
    ("hmrg_filt",    "HMRG†\n(9 feat)",             "#E69F00", False),  # Okabe-Ito orange
    ("arg_filt",     "ARG†\n(134 feat)",             "#D55E00", False),  # Okabe-Ito vermilion
    ("defence_is",   "Defence + IS\n(378 feat)",    "#CC79A7", True),   # Okabe-Ito reddish purple
]

keys     = [c[0] for c in CONDITIONS]
labels   = [c[1] for c in CONDITIONS]
colors   = [c[2] for c in CONDITIONS]

ari     = [filt[k]["ari_primary"]  for k in keys]
ci_lo   = [filt[k]["ari_ci95"][0] for k in keys]
ci_hi   = [filt[k]["ari_ci95"][1] for k in keys]
yerr_lo = [ari[i] - ci_lo[i]     for i in range(len(keys))]
yerr_hi = [ci_hi[i] - ari[i]     for i in range(len(keys))]

DEFENCE_ARI = ari[0]   # 0.219 — reference line

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))

# extra gap before the combined bar
x = np.array([0, 1, 2, 3, 4.4])

# background shading
ax.axvspan(-0.55, 3.55, color="#F0F4F8", alpha=0.55, zorder=0)
ax.axvspan(3.90,  5.00, color="#E8F5E9", alpha=0.55, zorder=0)

# section labels
ax.text(1.5, 0.625, "Single blocks (marker-filtered)", ha="center", va="bottom",
        fontsize=8, color="#555555", style="italic")
ax.text(4.4, 0.625, "Combined", ha="center", va="bottom",
        fontsize=8, color="#CC79A7", style="italic", fontweight="semibold")

# defence baseline reference line
ax.axhline(DEFENCE_ARI, color="#0072B2", linewidth=1.2,
           linestyle="--", alpha=0.75, zorder=1)

# ── Ghost bars for ARG and HMRG (unfiltered heights) ────────────────────────
# Visualise the pre-filter values to show taxonomic marker inflation
ghost_info = {
    "is_filt":   (x[1], UNFILT["is_filt"],   "#009E73"),
    "hmrg_filt": (x[2], UNFILT["hmrg_filt"], "#E69F00"),
    "arg_filt":  (x[3], UNFILT["arg_filt"],  "#D55E00"),
}
for key, (xi, uari, col) in ghost_info.items():
    ax.bar(xi, uari, width=0.72, color=col, alpha=0.12,
           zorder=1, edgecolor=col, linewidth=0.8, linestyle="--")
    ax.text(xi, uari + 0.012, f"pre-filter: {uari:.3f}",
            ha="center", va="bottom", fontsize=6.8, color=col, alpha=0.7,
            style="italic")

# ── Main bars (filtered values) ──────────────────────────────────────────────
ax.bar(x, ari, width=0.72, color=colors, alpha=0.88,
       zorder=2, edgecolor="white", linewidth=0.8)

# CI error bars
ax.errorbar(x, ari,
            yerr=[yerr_lo, yerr_hi],
            fmt="none", color="#333333", linewidth=1.4,
            capsize=4, capthick=1.4, zorder=3)

# ARI value labels above filtered bars
for i, (xi, ai, hi) in enumerate(zip(x, ari, ci_hi)):
    ax.text(xi, hi + 0.018, f"{ai:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=colors[i])

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlim(-0.55, 5.00)
ax.set_ylim(0, 0.66)
ax.set_ylabel("ARI vs species identity (K = 6, dereplicated genomes)", fontsize=9)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.tick_params(axis="y", labelsize=8)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ───────────────────────────────────────────────────────────────────
baseline_line = Line2D([0], [0], color="#0072B2", linewidth=1.4,
                       linestyle="--", alpha=0.8,
                       label=f"Defence baseline (ARI = {DEFENCE_ARI:.3f})")
ghost_patch = mpatches.Patch(facecolor="#AAAAAA", alpha=0.25, edgecolor="#888888",
                              linestyle="--", linewidth=0.8,
                              label="Pre-filter value (near-exclusive markers included)")
ax.legend(handles=[baseline_line, ghost_patch], fontsize=7.5,
          loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=2, framealpha=0.9, edgecolor="#CCCCCC",
          bbox_transform=ax.transAxes)

# ── Footnote ─────────────────────────────────────────────────────────────────
ax.text(0.5, -0.14,
        "† Near-exclusive species markers (spec_score ≥ 0.70) removed from IS, ARG, and HMRG blocks before clustering.\n"
        "  Ghost bars show pre-filter ARI (IS: 0.304; HMRG: 0.380; ARG: 0.557). "
        "All conditions: permutation p = 0.002 (500 permutations, K = 6).",
        transform=ax.transAxes, fontsize=7, color="#555555",
        ha="center", va="top")

# ── Title ────────────────────────────────────────────────────────────────────
ax.set_title("Multi-block clustering: species-discriminating power after marker filtering",
             fontsize=8.5, loc="left", pad=42)

fig.tight_layout()
fig.subplots_adjust(bottom=0.20, top=0.84)

FINAL_DIR = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/results/figures/final"
for ext in ("png", "pdf"):
    for d in (OUT_DIR, FINAL_DIR):
        path = os.path.join(d, f"fig3_q3b_block_comparison.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

plt.close(fig)
