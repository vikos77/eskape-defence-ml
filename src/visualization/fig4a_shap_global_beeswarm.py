"""
Fig 4A — Global SHAP beeswarm, top 20 features (Q1 RF, all species).

Collapses the 3D SHAP array (n × features × classes) to 2D by taking
mean |SHAP| over the six class dimensions, then plots the standard SHAP
summary (beeswarm) for the 20 highest-ranked features.

Dot colour = feature value (feature matrix; 0=binary absent, 1=present).
x-axis = mean |SHAP| per genome.

Data sources:
  results/q4_shap_367_array.npy     — raw SHAP 3D array (n, 359, 6)
  results/q4_shap_367_global.csv    — global feature ranking (pre-computed)
  results/q1_367_feat_cols.txt      — FEAT_COLS, 359 features in sorted order
  data/processed/feature_matrix_3335.parquet  — original binary feature matrix

Output:
  results/figures/interpretation/fig4a_shap_global_beeswarm.png  (300 dpi)
  results/figures/interpretation/fig4a_shap_global_beeswarm.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

OUT_DIR = "results/figures/interpretation"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load cached SHAP array and feature metadata ───────────────────────────────
shap_3d = np.load("results/q4_shap_367_array.npy")          # (3335, 359, 6)

with open("results/q1_367_feat_cols.txt") as f:
    feat_cols = [line.strip() for line in f if line.strip()]   # 359 names

global_df = pd.read_csv("results/q4_shap_367_global.csv")    # 'feature', 'mean_abs_shap', ...

# ── Collapse to 2D: mean |SHAP| over 6 classes ────────────────────────────────
# Avoids sign cancellation from opposing class effects.
shap_2d = np.abs(shap_3d).mean(axis=2)                        # (3335, 359)

# ── Select top 20 by global mean |SHAP| ──────────────────────────────────────
top20_feats = global_df.head(20)["feature"].tolist()
top20_mask  = [feat_cols.index(f) for f in top20_feats]
shap_2d_top20 = shap_2d[:, top20_mask]                        # (3335, 20)

# ── Load feature values (for beeswarm dot colour) ────────────────────────────
fm    = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
X_top = fm[top20_feats].to_numpy(dtype=float)                 # (3335, 20)

# ── Feature labels: strip all tool-name prefixes ─────────────────────────────
def clean_name(feat):
    return feat.removeprefix("dp_").removeprefix("df_").removeprefix("padloc_")

feature_names = [clean_name(f) for f in top20_feats]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

shap.summary_plot(
    shap_2d_top20,
    X_top,
    feature_names=feature_names,
    show=False,
    max_display=20,
    plot_size=None,
)

# Force canvas draw so SHAP's tick labels are fully populated, then
# re-set them explicitly — this also resets SHAP's italic style to normal.
fig.canvas.draw()
main_ax = fig.axes[0]   # axes[0] is beeswarm; axes[1] is SHAP's colorbar

ylabels = [t.get_text() for t in main_ax.get_yticklabels()]
main_ax.set_yticklabels(ylabels, fontsize=8)
main_ax.tick_params(axis="x", labelsize=8)
main_ax.set_xlabel(main_ax.get_xlabel(), fontsize=9)

main_ax.set_title(
    "(A)  Global SHAP beeswarm — top 20 features\n"
    "mean |SHAP| over 6 classes, Q1 RF (n = 3,335 genomes)",
    fontsize=10,
    loc="left",
    pad=8,
)

plt.tight_layout()

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"fig4a_shap_global_beeswarm.{ext}")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
