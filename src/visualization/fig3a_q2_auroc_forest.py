"""
Fig 3A — Q2 per-species AUROC forest plot.

Data source: results/q2_367_results.json (authoritative).
Output:      results/figures/q2/fig2a_q2_auroc_forest.png  (300 dpi)
             results/figures/q2/fig2a_q2_auroc_forest.pdf

Design:
- Horizontal forest plot, species ordered by AUROC descending.
- 4 significant species (BH-corrected p_adj < 0.05) in steel blue.
- 2 non-significant species in mid-grey.
- 95% CI bars from fold-level bootstrap (stored in JSON).
- Null reference line at AUROC = 0.5.
- Annotations: p_adj (right side) + mean fold BA (left side).
- Sized at 5.5 × 4 inches to pair with Fig 3B heatmap in a two-panel layout.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── data ────────────────────────────────────────────────────────────────────
RESULTS = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/results/q2_367_results.json"
OUT_DIR = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/results/figures/q2"
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS) as f:
    d = json.load(f)

# Order by AUROC descending; ties (KP=PA=0.791) broken alphabetically
rows = [
    ("efaecium",    "E. faecium"),
    ("kpneumoniae", "K. pneumoniae"),
    ("paeruginosa", "P. aeruginosa"),
    ("ecloaceae",   "E. cloacae complex"),
    ("abaumannii",  "A. baumannii"),
    ("saureus",     "S. aureus"),
]

auroc   = [d[k]["mean_auroc"]       for k, _ in rows]
ci_lo   = [d[k]["ci95_auroc"][0]    for k, _ in rows]
ci_hi   = [d[k]["ci95_auroc"][1]    for k, _ in rows]
mean_ba = [d[k]["mean_ba"]          for k, _ in rows]
p_adj   = [d[k]["p_adj"]            for k, _ in rows]
labels  = [lab                      for _, lab in rows]
sig     = [p < 0.05                 for p in p_adj]

# ── colours ─────────────────────────────────────────────────────────────────
COL_SIG = "#0072B2"   # Okabe-Ito blue — significant
COL_NS  = "#969696"   # mid-grey       — not significant
colors  = [COL_SIG if s else COL_NS for s in sig]

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 4.5))

y_pos = np.arange(len(rows))[::-1]   # top = highest AUROC

# null line only — no shading
ax.axvline(0.5, color="#BDBDBD", linewidth=1.2, linestyle="--", zorder=1)

for i, y in enumerate(y_pos):
    col = colors[i]
    # CI bar
    ax.plot([ci_lo[i], ci_hi[i]], [y, y],
            color=col, linewidth=2.0, solid_capstyle="round", zorder=2)
    # point estimate
    ax.scatter(auroc[i], y, color=col, s=60, zorder=3,
               marker="D" if sig[i] else "o")

    # right annotation: AUROC value, then p_adj / ns on second line
    if sig[i]:
        if p_adj[i] < 0.001:
            p_str = "p_adj < 0.001"
        elif p_adj[i] < 0.01:
            p_str = f"p_adj = {p_adj[i]:.4f}"   # e.g. 0.0012, 0.0082
        else:
            p_str = f"p_adj = {p_adj[i]:.3f}"   # e.g. 0.038
    else:
        p_str = "ns"
    right_x = ci_hi[i] + 0.015
    ax.text(right_x, y + 0.18, f"AUROC {auroc[i]:.3f}  BA {mean_ba[i]:.3f}",
            ha="left", va="center", fontsize=7.0, color=col)
    ax.text(right_x, y - 0.18, p_str,
            ha="left", va="center", fontsize=7.0, color=col,
            style="italic")

# ── y-axis: italic species names ─────────────────────────────────────────────
# Genus + epithet italic; "complex" not italic
ITALIC_LABELS = [
    r"$\it{E.\ faecium}$",
    r"$\it{K.\ pneumoniae}$",
    r"$\it{P.\ aeruginosa}$",
    r"$\it{E.\ cloacae}$ complex",
    r"$\it{A.\ baumannii}$",
    r"$\it{S.\ aureus}$",
]
ax.set_yticks(y_pos)
ax.set_yticklabels(ITALIC_LABELS, fontsize=8)

# ── x-axis ──────────────────────────────────────────────────────────────────
ax.set_xlim(0.40, 1.02)
ax.set_xlabel("AUROC (high vs low ARG burden)", fontsize=9)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.tick_params(axis="x", labelsize=8)

# ── legend ───────────────────────────────────────────────────────────────────
sig_patch = mpatches.Patch(color=COL_SIG, label="Significant (BH q < 0.05)")
ns_patch  = mpatches.Patch(color=COL_NS,  label="Not significant")
ax.legend(handles=[sig_patch, ns_patch], fontsize=8,
          loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")

# ── grid + spines ────────────────────────────────────────────────────────────
ax.grid(axis="x", color="#E0E0E0", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(0.01, 0.995, "(A)", fontsize=10, fontweight="bold", va="top")

fig.tight_layout()

FINAL_DIR = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/results/figures/final"
for ext in ("png", "pdf"):
    for d in (OUT_DIR, FINAL_DIR):
        path = os.path.join(d, f"fig2a_q2_auroc_forest.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

plt.close(fig)
