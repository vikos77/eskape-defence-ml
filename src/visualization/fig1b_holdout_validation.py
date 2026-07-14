"""
Fig 1B — C3 external holdout validation.

Data sources:
  data/processed/holdout_feature_matrix.parquet  — 180-genome holdout
  results/models/rf_q1_367.pkl                   — trained RF Q1 model
  results/q1_367_feat_cols.txt                   — FEAT_COLS (359 features)
  results/q1_367_results.json                    — CV BA and cluster bootstrap CI

Output:
  results/figures/rf/fig1b_holdout_validation.png  (300 dpi)
  results/figures/rf/fig1b_holdout_validation.pdf
"""

import json, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample

warnings.filterwarnings("ignore")

OUT_DIR = "results/figures/rf"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
hfm = pd.read_parquet("data/processed/holdout_feature_matrix.parquet")
with open("results/q1_367_feat_cols.txt") as f:
    feat_cols = [l.strip() for l in f if l.strip()]
with open("results/models/rf_q1_367.pkl", "rb") as f:
    rf = pickle.load(f)
with open("results/q1_367_results.json") as f:
    q1_res = json.load(f)

X_hold = hfm[feat_cols].values
y_str  = hfm["species"].values

# ── Holdout predictions ──────────────────────────────────────────────────────
y_pred     = rf.predict(X_hold)
ba_holdout = balanced_accuracy_score(y_str, y_pred)

rng = np.random.default_rng(42)
ba_boot = []
for _ in range(2000):
    idx = resample(np.arange(len(y_str)), stratify=y_str,
                   random_state=rng.integers(1_000_000))
    ba_boot.append(balanced_accuracy_score(y_str[idx], y_pred[idx]))
ci_lo, ci_hi = np.percentile(ba_boot, [2.5, 97.5])

# ── CV reference ─────────────────────────────────────────────────────────────
cv_ba  = float(np.mean(q1_res["fold_bas"]))
cv_lo  = q1_res["ci95_ba"][0]
cv_hi  = q1_res["ci95_ba"][1]

# ── Per-species recall ───────────────────────────────────────────────────────
SPECIES_ORDER = [
    "abaumannii", "ecloaceae", "efaecium",
    "kpneumoniae", "paeruginosa", "saureus",
]
SPECIES_LABELS = {
    "abaumannii":  "A. baumannii",
    "ecloaceae":   "E. cloacae",
    "efaecium":    "E. faecium",
    "kpneumoniae": "K. pneumoniae",
    "paeruginosa": "P. aeruginosa",
    "saureus":     "S. aureus",
}
SPECIES_COLORS = {
    "abaumannii":  "#E63946",
    "ecloaceae":   "#F4A261",
    "efaecium":    "#2A9D8F",
    "kpneumoniae": "#457B9D",
    "paeruginosa": "#8338EC",
    "saureus":     "#FB5607",
}

recalls = []
for sp in SPECIES_ORDER:
    mask = y_str == sp
    recalls.append((y_pred[mask] == sp).sum() / mask.sum())

# ── Figure (two panels, wide) ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                         gridspec_kw={"width_ratios": [1.1, 1]})

# ── Left: per-species recall bar chart ───────────────────────────────────────
ax = axes[0]
bars = ax.bar(
    [SPECIES_LABELS[sp] for sp in SPECIES_ORDER],
    recalls,
    color=[SPECIES_COLORS[sp] for sp in SPECIES_ORDER],
    edgecolor="white", linewidth=0.5, width=0.65,
)
ax.axhline(1.0, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.7)
for bar, val in zip(bars, recalls):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0, 1.14)
ax.set_ylabel("Recall (per species, n=30)", fontsize=9)
ax.set_title("(B)  C3 holdout: per-species recall\n"
             "(30 novel-ST-prioritised genomes per species)",
             fontsize=8.5, loc="left")
ax.tick_params(axis="x", rotation=35, labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# ── Right: holdout BA vs CV estimate ─────────────────────────────────────────
ax2 = axes[1]
y_pos = [1, 0]    # CV on top, Holdout below
row_labels = ["CV (GroupedKFold)", "Holdout (C3, n=180)"]

# CI bars
ax2.barh(y_pos[0], cv_hi - cv_lo, left=cv_lo, height=0.35,
         color="#AAAAAA", alpha=0.65, label="CV CI [2.5–97.5%]")
ax2.barh(y_pos[1], ci_hi - ci_lo, left=ci_lo, height=0.35,
         color="#457B9D", alpha=0.65, label="Holdout CI [2.5–97.5%]")

# Point estimates
ax2.plot(cv_ba,      y_pos[0], "o", color="#333333", markersize=8,
         label=f"CV BA = {cv_ba:.3f}")
ax2.plot(ba_holdout, y_pos[1], "o", color="#457B9D", markersize=8,
         label=f"Holdout BA = {ba_holdout:.3f}")

# Value annotations
ax2.text(cv_ba + 0.003,      y_pos[0] + 0.22, f"{cv_ba:.3f}  [{cv_lo:.3f}–{cv_hi:.3f}]",
         fontsize=8, va="bottom", color="#333333")
ax2.text(ba_holdout + 0.003, y_pos[1] + 0.22, f"{ba_holdout:.3f}  [{ci_lo:.3f}–{ci_hi:.3f}]",
         fontsize=8, va="bottom", color="#457B9D")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(row_labels, fontsize=9)
ax2.set_xlim(0.79, 1.04)
ax2.set_ylim(-0.5, 1.75)
ax2.set_xlabel("Balanced accuracy", fontsize=9)
ax2.axvline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
ax2.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(axis="x", labelsize=8)
ax2.set_title("Holdout BA vs CV estimate\n"
              f"Consistent: holdout {ba_holdout:.3f} [{ci_lo:.3f}–{ci_hi:.3f}], "
              f"CV {cv_ba:.3f} [{cv_lo:.3f}–{cv_hi:.3f}]",
              fontsize=8.5)

fig.tight_layout(pad=1.5)

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"fig1b_holdout_validation.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
