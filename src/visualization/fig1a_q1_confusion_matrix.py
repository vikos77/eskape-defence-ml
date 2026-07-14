"""
Fig 1A — Q1 species classification confusion matrix.

Data source: results/q1_367_oof_preds.npz (integer-encoded OOF predictions,
             LabelEncoder alphabetical order: AB EC EF KP PA SA)
             results/q1_367_results.json (BA CI, fold BAs)
Output:      results/figures/rf/fig1a_q1_confusion_matrix.png  (300 dpi)
             results/figures/rf/fig1a_q1_confusion_matrix.pdf

Design:
- 6×6 row-normalised confusion matrix (diagonal = per-class recall).
- Blues colormap.
- Short species codes: AB, EC, EF, KP, PA, SA.
- Subtitle: mean fold BA and cluster bootstrap 95% CI.
- Sized at 5.0 × 4.5 inches for a single-column or half-page panel.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ── data ────────────────────────────────────────────────────────────────────
OOF_FILE    = "results/q1_367_oof_preds.npz"
RESULTS     = "results/q1_367_results.json"
OUT_DIR     = "results/figures/rf"
os.makedirs(OUT_DIR, exist_ok=True)

data    = np.load(OOF_FILE)
y_true  = data["y_true"]
y_pred  = data["y_pred"]

with open(RESULTS) as f:
    res = json.load(f)

# LabelEncoder alphabetical order
CLASS_NAMES  = ["AB", "EC", "EF", "KP", "PA", "SA"]
n_classes    = len(CLASS_NAMES)

# Row-normalised confusion matrix
cm     = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Fold-mean BA and CI from JSON (more reliable than pooled-OOF BA)
ci_lo, ci_hi = res["ci95_ba"]
mean_ba = float(np.mean(res["fold_bas"]))

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 4.5))

im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

# cell annotations
for i in range(n_classes):
    for j in range(n_classes):
        val   = cm_norm[i, j]
        color = "white" if val > 0.55 else "#222222"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8.0, color=color, fontweight="bold" if i == j else "normal")

# axes
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(CLASS_NAMES, fontsize=9)
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted species", fontsize=9, labelpad=6)
ax.set_ylabel("True species", fontsize=9, labelpad=6)

# colour bar
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Recall (row-normalised)", fontsize=8)
cb.ax.tick_params(labelsize=7)

# title
ax.set_title(
    f"(A)   BA = {mean_ba:.3f}  [{ci_lo:.3f}–{ci_hi:.3f}]  "
    f"(5-fold grouped CV, n=3,335)",
    fontsize=8.5, loc="left", pad=8
)

fig.tight_layout()

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"fig1a_q1_confusion_matrix.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
