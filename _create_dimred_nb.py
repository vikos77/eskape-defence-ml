"""Create notebooks/03_dimensionality_reduction.ipynb — Phase 5."""
import json

def md(text, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": [text]}

def code(text, cell_id):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": [text]}

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md("""\
# 03 — Dimensionality Reduction

**Goal:** Project 625 defence-system features into 2D to answer:
1. Do ESKAPE species separate by defence-system repertoire?
2. Does ARG burden create visible sub-structure within species clusters?
3. Where does A. baumannii sit relative to the other five species?

**Methods:** PCA (linear), UMAP (non-linear), t-SNE (non-linear, local-focused)

**What this phase does NOT do:** It does not train classifiers or compute
accuracy. Visualisation only. Patterns seen here generate hypotheses that
Phases 6–8 will test quantitatively.""", "dr-title"))

# ── Imports + load ───────────────────────────────────────────────────────────
cells.append(md("## Imports and data load", "dr-imports-md"))

cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path("..")
PROC    = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures" / "dimred"
FIG_DIR.mkdir(parents=True, exist_ok=True)

fm = pd.read_parquet(PROC / "feature_matrix.parquet")

LABEL_COLS = ["species", "arg_burden_tertile", "country", "year_bin",
              "complex_member", "sequence_type", "mlst_scheme"]
feat_cols  = [c for c in fm.columns if c not in LABEL_COLS]

# Use only dp_* (binary presence/absence) as features for dimensionality reduction.
# Why not dc_* as well? Because dp_* and dc_* represent the same biology in two
# formats. Including both would double-count defence systems in the distance metric,
# inflating their apparent importance relative to count-only features (IS, ARG counts).
# For visualisation the binary representation is sufficient and cleaner.
dp_cols = [c for c in fm.columns if c.startswith("dp_")]

X = fm[dp_cols].values.astype(float)  # 878 × 274
y_species = fm["species"].values
y_arg     = fm["arg_burden_tertile"].values

SPECIES_ORDER = ["kpneumoniae", "ecloaceae", "abaumannii",
                 "efaecium", "paeruginosa", "saureus"]
SPECIES_LABELS = {
    "kpneumoniae": "K. pneumoniae",
    "ecloaceae":   "E. cloacae",
    "abaumannii":  "A. baumannii",
    "efaecium":    "E. faecium",
    "paeruginosa": "P. aeruginosa",
    "saureus":     "S. aureus",
}
SP_COLORS = {
    "kpneumoniae": "#4878CF",
    "ecloaceae":   "#6ACC65",
    "abaumannii":  "#D65F5F",
    "efaecium":    "#B47CC7",
    "paeruginosa": "#C4AD66",
    "saureus":     "#77BEDB",
}
ARG_COLORS = {"low_ARG": "#4878CF", "mid_ARG": "#AAAAAA", "high_ARG": "#D65F5F"}

print(f"Feature matrix: {fm.shape[0]} genomes × {len(dp_cols)} dp_* features")
print(f"Species: { {sp: (y_species==sp).sum() for sp in SPECIES_ORDER} }")
print(f"ARG tertile: { pd.Series(y_arg).value_counts().to_dict() }")""",
"dr-load"))

# ── Section 1: PCA ───────────────────────────────────────────────────────────
cells.append(md("""\
## Section 1 — PCA

PCA finds linear combinations of the 274 dp_* features that explain the most
variance. We run it on **scaled** features (StandardScaler: mean=0, std=1 per
feature). Scaling is necessary because binary features with very different
prevalences (0.01 vs 0.99) would otherwise have very different variances, and
PCA would be dominated by the high-prevalence features regardless of their
biological importance.""", "dr-pca-md"))

cells.append(code("""\
# ── Scale and run PCA ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ── Scree plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
indvar = pca.explained_variance_ratio_ * 100

axes[0].bar(range(1, 21), indvar[:20], color="#4878CF", alpha=0.8)
axes[0].set_xlabel("Principal component", fontsize=10)
axes[0].set_ylabel("Variance explained (%)", fontsize=10)
axes[0].set_title("Scree plot — top 20 PCs", fontsize=10)
axes[0].grid(axis="y", alpha=0.3)

axes[1].plot(range(1, 51), cumvar, color="#D65F5F", lw=2)
axes[1].axhline(50, color="grey", ls="--", lw=1, label="50%")
axes[1].axhline(80, color="grey", ls=":", lw=1, label="80%")
axes[1].set_xlabel("Number of PCs", fontsize=10)
axes[1].set_ylabel("Cumulative variance explained (%)", fontsize=10)
axes[1].set_title("Cumulative variance — 50 PCs", fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "01_pca_scree.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"PC1 explains: {indvar[0]:.1f}%  |  PC2: {indvar[1]:.1f}%")
print(f"PCs to reach 50% variance: {np.searchsorted(cumvar, 50)+1}")
print(f"PCs to reach 80% variance: {np.searchsorted(cumvar, 80)+1}")
print("Saved: results/figures/dimred/01_pca_scree.png")""", "dr-pca-scree"))

cells.append(code("""\
# ── PCA scatter — coloured by species ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, color_by, title_suffix in zip(
    axes,
    ["species", "arg"],
    ["coloured by species", "coloured by ARG burden tertile"]
):
    for sp in SPECIES_ORDER:
        mask = y_species == sp
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=SP_COLORS[sp], alpha=0.5, s=18, edgecolors="none",
            label=SPECIES_LABELS[sp],
        )
    ax.set_xlabel(f"PC1 ({indvar[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({indvar[1]:.1f}% variance)", fontsize=10)
    ax.set_title(f"PCA — {title_suffix}", fontsize=10)
    ax.grid(alpha=0.2)

axes[0].legend(fontsize=8, markerscale=1.5,
               loc="upper right", framealpha=0.8)

# Second panel: colour by ARG tertile, shape by gram stain
gram_neg = {"kpneumoniae", "ecloaceae", "abaumannii", "paeruginosa"}
for i in range(len(y_species)):
    sp = y_species[i]
    arg = y_arg[i]
    if arg not in ARG_COLORS:
        continue
    marker = "o" if sp in gram_neg else "^"
    axes[1].scatter(
        X_pca[i, 0], X_pca[i, 1],
        c=ARG_COLORS[arg], alpha=0.45, s=20,
        marker=marker, edgecolors="none",
    )

legend_patches = [
    mpatches.Patch(color=ARG_COLORS["low_ARG"], label="low ARG"),
    mpatches.Patch(color=ARG_COLORS["high_ARG"], label="high ARG"),
    plt.scatter([], [], marker="o", color="grey", label="gram-negative"),
    plt.scatter([], [], marker="^", color="grey", label="gram-positive"),
]
axes[1].legend(handles=legend_patches, fontsize=8, framealpha=0.8)

plt.tight_layout()
plt.savefig(FIG_DIR / "02_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/dimred/02_pca_scatter.png")""", "dr-pca-scatter"))

cells.append(code("""\
# ── PCA loadings: what drives PC1 and PC2? ────────────────────────────────────
loadings = pd.DataFrame(
    pca.components_[:2].T,
    index=[c.replace("dp_", "") for c in dp_cols],
    columns=["PC1", "PC2"],
)

print("Top 10 features driving PC1 (positive = pushes genomes to the right):")
print(loadings["PC1"].abs().nlargest(10).index.tolist())
print()
print("PC1 — top 5 positive (push right):")
print(loadings["PC1"].nlargest(5).to_string())
print()
print("PC1 — top 5 negative (push left):")
print(loadings["PC1"].nsmallest(5).to_string())
print()
print("Top 5 features driving PC2:")
print(loadings["PC2"].abs().nlargest(5).index.tolist())""", "dr-pca-loadings"))

# ── Section 2: UMAP ──────────────────────────────────────────────────────────
cells.append(md("""\
## Section 2 — UMAP

UMAP preserves neighbourhood relationships (similar genomes stay close in 2D).
Unlike PCA, it can reveal curved or nested cluster structure.

**Two parameters matter most:**
- `n_neighbors`: how many neighbours each point considers when building the
  graph. Low value (~5) = focus on very local structure (tight clusters, may
  fragment). High value (~50) = focus on global structure (spread-out, smoother).
- `min_dist`: how tightly points are packed within a cluster. Low = tight;
  high = spread.

We run with `random_state=42` for reproducibility. Then we do a sensitivity
check to confirm the main structure is not an artefact of parameter choice.""",
"dr-umap-md"))

cells.append(code("""\
# ── UMAP default run ──────────────────────────────────────────────────────────
# Run UMAP on the scaled binary feature matrix
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                    metric="jaccard",   # appropriate for binary P/A data
                    random_state=42, verbose=False)
X_umap = reducer.fit_transform(X)

print("UMAP embedding computed.")
print(f"Embedding shape: {X_umap.shape}")""", "dr-umap-run"))

cells.append(code("""\
# ── UMAP scatter — species and ARG burden ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: colour by species
for sp in SPECIES_ORDER:
    mask = y_species == sp
    axes[0].scatter(
        X_umap[mask, 0], X_umap[mask, 1],
        c=SP_COLORS[sp], alpha=0.55, s=18, edgecolors="none",
        label=SPECIES_LABELS[sp],
    )
axes[0].set_title("UMAP (Jaccard, n_neighbors=15) — by species", fontsize=10)
axes[0].set_xlabel("UMAP 1", fontsize=10)
axes[0].set_ylabel("UMAP 2", fontsize=10)
axes[0].legend(fontsize=8, markerscale=1.5, framealpha=0.8)
axes[0].grid(alpha=0.2)

# Right: colour by ARG burden, shape by gram stain
for i in range(len(y_species)):
    sp = y_species[i]
    arg = y_arg[i]
    if arg not in ARG_COLORS:
        continue
    marker = "o" if sp in gram_neg else "^"
    axes[1].scatter(
        X_umap[i, 0], X_umap[i, 1],
        c=ARG_COLORS[arg], alpha=0.45, s=20,
        marker=marker, edgecolors="none",
    )
axes[1].set_title("UMAP — by ARG burden (circle=gram-neg, triangle=gram-pos)",
                  fontsize=10)
axes[1].set_xlabel("UMAP 1", fontsize=10)
axes[1].set_ylabel("UMAP 2", fontsize=10)
axes[1].legend(handles=legend_patches, fontsize=8, framealpha=0.8)
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(FIG_DIR / "03_umap_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/dimred/03_umap_scatter.png")""", "dr-umap-scatter"))

cells.append(code("""\
# ── UMAP parameter sensitivity: n_neighbors sweep ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("UMAP sensitivity: n_neighbors (min_dist=0.1, metric=jaccard)",
             fontsize=11, y=1.01)

for ax, nn in zip(axes, [5, 15, 30, 50]):
    emb = umap.UMAP(n_neighbors=nn, min_dist=0.1, n_components=2,
                    metric="jaccard", random_state=42, verbose=False
                   ).fit_transform(X)
    for sp in SPECIES_ORDER:
        mask = y_species == sp
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=SP_COLORS[sp], alpha=0.5, s=8, edgecolors="none",
                   label=SPECIES_LABELS[sp])
    ax.set_title(f"n_neighbors={nn}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

axes[0].legend(fontsize=6, markerscale=1.5, loc="lower left")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_umap_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/dimred/04_umap_sensitivity.png")""", "dr-umap-sens"))

# ── Section 3: t-SNE ─────────────────────────────────────────────────────────
cells.append(md("""\
## Section 3 — t-SNE

t-SNE is run for comparison only. Key differences from UMAP:
- More aggressive at compressing within-cluster distances
- Distances between clusters are not interpretable at all
- Slower; more sensitive to `perplexity` (analogous to n_neighbors)

We run with perplexity=30 (standard default) and compare to UMAP.""",
"dr-tsne-md"))

cells.append(code("""\
# ── t-SNE run ─────────────────────────────────────────────────────────────────
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            learning_rate="auto", init="pca", n_jobs=-1)
X_tsne = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
for sp in SPECIES_ORDER:
    mask = y_species == sp
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=SP_COLORS[sp], alpha=0.55, s=18, edgecolors="none",
               label=SPECIES_LABELS[sp])
ax.set_title("t-SNE (perplexity=30) — by species", fontsize=11)
ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.legend(fontsize=9, markerscale=1.5, framealpha=0.8)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_tsne_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/dimred/05_tsne_scatter.png")""", "dr-tsne"))

# ── Comprehension check ───────────────────────────────────────────────────────
cells.append(md("""\
## Comprehension check — Phase 5

**Q1.** In the PCA scree plot, PC1 explains a certain percentage of variance and
PC2 explains less. If you needed 20 PCs to explain 80% of the variance, what
would that tell you about the structure of the defence-system feature space?

**Q2.** UMAP uses `metric="jaccard"` for the binary presence/absence features.
From what you learned in Phase 3 about distance metrics on sparse binary data,
why is Jaccard more appropriate here than Euclidean distance?

**Q3.** You notice that KP and EC partially overlap in both PCA and UMAP.
Give two reasons — one biological, one technical — why this overlap is expected
and does not mean the Q1 classifier will fail to separate them.""",
"dr-cc"))

# ── Write notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "eskape-ml", "language": "python",
                       "name": "eskape-ml"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open("notebooks/03_dimensionality_reduction.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

nb2 = json.load(open("notebooks/03_dimensionality_reduction.ipynb"))
print(f"Created notebooks/03_dimensionality_reduction.ipynb with {len(nb2['cells'])} cells")
