"""
Fig 2 — UMAP of ESKAPE defence system repertoires (publication version).

Two panels:
  (A) Coloured by species — centroid italic labels, no legend.
  (B) Coloured by ARG burden tertile, shape = Gram stain.

UMAP parameters: n_neighbors=15, min_dist=0.1, metric=jaccard, random_state=42.
Embedding cached to results/umap_embedding_3335.npy on first run (~3 min);
reloaded on subsequent runs (<5 s).

Output:
  results/figures/dimred/fig2_umap.png  (300 dpi)
  results/figures/dimred/fig2_umap.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import umap
import warnings
warnings.filterwarnings("ignore")

OUT_DIR   = "results/figures/dimred"
EMBED_NPY = "results/umap_embedding_3335.npy"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load feature matrix ───────────────────────────────────────────────────────
fm      = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]
X       = fm[dp_cols].values.astype(float)

y_species = fm["species"].values
y_arg     = fm["arg_burden_tertile"].values

# ── UMAP: compute or reload ───────────────────────────────────────────────────
if os.path.exists(EMBED_NPY):
    X_umap = np.load(EMBED_NPY)
    print(f"Loaded embedding from {EMBED_NPY}")
else:
    print("Computing UMAP (n_neighbors=15, metric=jaccard) …")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        metric="jaccard", random_state=42, verbose=False)
    X_umap = reducer.fit_transform(X)
    np.save(EMBED_NPY, X_umap)
    print(f"Saved embedding to {EMBED_NPY}")

# ── Colour / style maps ───────────────────────────────────────────────────────
SPECIES_ORDER = ["kpneumoniae", "ecloaceae", "abaumannii",
                 "efaecium", "paeruginosa", "saureus"]

SP_COLORS = {
    "kpneumoniae": "#4878CF",
    "ecloaceae":   "#6ACC65",
    "abaumannii":  "#D65F5F",
    "efaecium":    "#B47CC7",
    "paeruginosa": "#C4AD66",
    "saureus":     "#77BEDB",
}

# Italic centroid labels (LaTeX math-italic for species names)
SP_LABELS = {
    "kpneumoniae": r"$\it{K.\ pneumoniae}$",
    "ecloaceae":   r"$\it{E.\ cloacae}$ cx",
    "abaumannii":  r"$\it{A.\ baumannii}$",
    "efaecium":    r"$\it{E.\ faecium}$",
    "paeruginosa": r"$\it{P.\ aeruginosa}$",
    "saureus":     r"$\it{S.\ aureus}$",
}

ARG_COLORS = {
    "low_ARG":  "#4878CF",
    "high_ARG": "#D65F5F",
    "mid_ARG":  "#BBBBBB",
}

GRAM_NEG = {"kpneumoniae", "ecloaceae", "abaumannii", "paeruginosa"}

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# ── Panel A: by species ───────────────────────────────────────────────────────
ax = axes[0]

for sp in SPECIES_ORDER:
    mask = y_species == sp
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               c=SP_COLORS[sp], alpha=0.60, s=18,
               edgecolors="none", rasterized=True)

# Centroid italic labels
for sp in SPECIES_ORDER:
    mask = y_species == sp
    cx = np.median(X_umap[mask, 0])
    cy = np.median(X_umap[mask, 1])
    ax.text(cx, cy, SP_LABELS[sp],
            fontsize=8.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      alpha=0.75, ec="none"))

ax.set_xlabel("UMAP 1", fontsize=9, labelpad=4)
ax.set_ylabel("UMAP 2", fontsize=9, labelpad=4)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)
ax.set_title("(A)  Coloured by species", fontsize=9, loc="left", pad=6)

# ── Panel B: by ARG burden ────────────────────────────────────────────────────
ax2 = axes[1]

# Draw mid ARG first (background), then low, then high on top
for arg_tier in ["mid_ARG", "low_ARG", "high_ARG"]:
    col   = ARG_COLORS[arg_tier]
    alpha = 0.35 if arg_tier == "mid_ARG" else 0.65
    size  = 14  if arg_tier == "mid_ARG" else 20
    for i, (sp, arg) in enumerate(zip(y_species, y_arg)):
        if arg != arg_tier:
            continue
        marker = "o" if sp in GRAM_NEG else "^"
        ax2.scatter(X_umap[i, 0], X_umap[i, 1],
                    c=col, alpha=alpha, s=size,
                    marker=marker, edgecolors="none", rasterized=True)

# Legend — lower left (dense clusters are upper centre/right)
legend_handles = [
    mpatches.Patch(color=ARG_COLORS["low_ARG"],  label="Low ARG burden"),
    mpatches.Patch(color=ARG_COLORS["high_ARG"], label="High ARG burden"),
    mpatches.Patch(color=ARG_COLORS["mid_ARG"],  label="Mid ARG (Q2 excluded)"),
    mlines.Line2D([], [], marker="o", color="#555555", lw=0,
                  markersize=6, label="Gram-negative"),
    mlines.Line2D([], [], marker="^", color="#555555", lw=0,
                  markersize=6, label="Gram-positive"),
]
ax2.legend(handles=legend_handles, fontsize=7.5,
           loc="lower left", framealpha=0.9, edgecolor="#CCCCCC")

ax2.set_xlabel("UMAP 1", fontsize=9, labelpad=4)
ax2.set_ylabel("UMAP 2", fontsize=9, labelpad=4)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_linewidth(0.6)
ax2.spines["bottom"].set_linewidth(0.6)
ax2.set_title("(B)  Coloured by ARG burden tertile  (shape = Gram stain)",
              fontsize=9, loc="left", pad=6)

fig.tight_layout(pad=1.5)

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"fig2_umap.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
