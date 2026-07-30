"""
Fig 4B — Compact per-species SHAP heatmap.

Shows mean signed SHAP value for each (feature, species-class) pair for the
top 20 globally important features.

- Red  = positive SHAP  → feature presence increases probability of that species.
- Blue = negative SHAP  → feature presence decreases probability of that species.
- Mean taken over all 3,335 genomes (not just the true-class genomes).

Data sources:
  results/q4_shap_367_array.npy   — raw SHAP 3D array (n, p, c)
  results/q4_shap_367_global.csv  — global feature ranking
  results/q1_367_feat_cols.txt    — FEAT_COLS in sorted order
Output:
  results/figures/interpretation/fig4b_shap_heatmap.png  (300 dpi)
  results/figures/interpretation/fig4b_shap_heatmap.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUT_DIR = "results/figures/interpretation"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
shap_3d   = np.load("results/q4_shap_367_array.npy")          # (3335, 359, 6)
global_df = pd.read_csv("results/q4_shap_367_global.csv")
pc_df     = pd.read_csv("results/q4_shap_367_perclass.csv")

with open("results/q1_367_feat_cols.txt") as f:
    feat_cols = [l.strip() for l in f if l.strip()]           # sorted alphabetically

# Load feature matrix to identify presence rows (dp_* = 1)
fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")

# class order from script: sorted alphabetically
CLASSES = ["abaumannii", "ecloaceae", "efaecium",
           "kpneumoniae", "paeruginosa", "saureus"]
COL_LABELS = ["AB", "EC", "EF", "KP", "PA", "SA"]

# ── Top 20 features ──────────────────────────────────────────────────────────
N_FEATS = 20
top20     = global_df.head(N_FEATS)
top20_idx = [feat_cols.index(f) for f in top20["feature"]]   # position in FEAT_COLS

# ── Heatmap: mean |SHAP| per class (magnitude) with sign from feature-present rows
# For each (feature, class): take mean SHAP over genomes where feature==1
# This answers "when this system is present, does it push toward/away from each species?"
X_all    = fm[feat_cols].to_numpy(dtype=float)   # (3335, 359)
heatmap  = np.zeros((N_FEATS, len(CLASSES)))
for row_i, feat_i in enumerate(top20_idx):
    present_mask = X_all[:, feat_i] > 0          # genomes where system detected
    if present_mask.sum() < 5:
        continue
    # mean signed SHAP among presence genomes, per class
    heatmap[row_i, :] = shap_3d[present_mask, feat_i, :].mean(axis=0)

# ── Feature labels ───────────────────────────────────────────────────────────
def clean_label(feat, category):
    """Strip dp_/df_/padloc_ prefixes; mark uncharacterised with dagger."""
    name = feat.removeprefix("dp_").removeprefix("df_").removeprefix("padloc_")
    if category != "named":
        name = name + "†"
    return name

row_labels = [
    clean_label(row["feature"], row["category"])
    for _, row in top20.iterrows()
]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 7.0))

# symmetric colour scale around 0
abs_max = np.abs(heatmap).max()
vmax    = abs_max * 1.05

CB_DIV = mcolors.LinearSegmentedColormap.from_list(
    "cb_div", ["#0072B2", "#FFFFFF", "#D55E00"]  # Okabe-Ito blue → white → vermilion
)
im = ax.imshow(heatmap, cmap=CB_DIV, vmin=-vmax, vmax=vmax,
               aspect="auto", interpolation="nearest")

# cell text (values)
for i in range(N_FEATS):
    for j in range(len(CLASSES)):
        val   = heatmap[i, j]
        color = "white" if abs(val) > 0.55 * vmax else "#222222"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=6.5, color=color)

# axes
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(COL_LABELS, fontsize=9)
ax.set_yticks(range(N_FEATS))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_xlabel("Species", fontsize=9, labelpad=6)
ax.set_ylabel("Feature (global SHAP rank, top 20)", fontsize=9, labelpad=6)

# y-axis rank numbers on the right (on a twinx, no label — rank obvious from position)
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(N_FEATS))
ax2.set_yticklabels([f"#{i + 1}" for i in range(N_FEATS)],
                     fontsize=7, color="#999999")
ax2.tick_params(right=False)

# colour bar on its own axes to avoid overlap with twinx
cbar_ax = fig.add_axes([0.93, 0.12, 0.025, 0.75])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label("Mean SHAP (present genomes)\norange = toward species, blue = away",
             fontsize=8, labelpad=6)
cb.ax.tick_params(labelsize=7)

# title
ax.set_title("(B)  Per-species mean SHAP — top 20 global features (Q1 RF)\n"
             "† uncharacterised candidate system",
             fontsize=8.5, loc="left", pad=8)

# light grid lines between cells
for i in range(N_FEATS + 1):
    ax.axhline(i - 0.5, color="white", linewidth=0.5)
for j in range(len(CLASSES) + 1):
    ax.axvline(j - 0.5, color="white", linewidth=0.5)

fig.subplots_adjust(right=0.88)

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"fig4b_shap_heatmap.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
