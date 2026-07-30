"""Regenerate Section 2.4, Supplement S-A, and Section 2.5 figures using cached SHAP array.

Uses results/q4_shap_367_array.npy to avoid recomputing SHAP from scratch.
Run from project root:
    conda run -n eskape-ml python src/models/regen_shap_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

ROOT = Path(".")
RES  = ROOT / "results"
FIGS_INTERP = RES / "figures" / "interpretation"
FIGS_INTERP.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size":  11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SPECIES_ORDER  = ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]
SPECIES_LABELS = {
    "abaumannii":  "A. baumannii",
    "ecloaceae":   "E. cloacae complex",
    "efaecium":    "E. faecium",
    "kpneumoniae": "K. pneumoniae",
    "paeruginosa": "P. aeruginosa",
    "saureus":     "S. aureus",
}
SPECIES_PALETTE = {
    "abaumannii":  "#D55E00",  # Okabe-Ito vermilion
    "ecloaceae":   "#009E73",  # Okabe-Ito bluish green
    "efaecium":    "#CC79A7",  # Okabe-Ito reddish purple
    "kpneumoniae": "#0072B2",  # Okabe-Ito blue
    "paeruginosa": "#E69F00",  # Okabe-Ito orange
    "saureus":     "#56B4E9",  # Okabe-Ito sky blue
}

# ── Uniform colors (same in every panel) ─────────────────────────────────────
POS_COLOR = "#0072B2"   # Okabe-Ito blue    — positive-direction / feature-present
NEG_COLOR = "#AAAAAA"   # grey              — negative-direction / feature-absent

# ── Load data ─────────────────────────────────────────────────────────────────
fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
FEAT_COLS = open("results/q1_367_feat_cols.txt").read().splitlines()

shap_3d = np.load("results/q4_shap_367_array.npy")   # (3335, 359, 6)
assert shap_3d.shape == (len(fm), len(FEAT_COLS), len(SPECIES_ORDER)), "SHAP array shape mismatch"

y_full = fm["species"].to_numpy(dtype=str)
X_full = fm[FEAT_COLS].to_numpy()

print(f"Loaded: {shap_3d.shape} SHAP array, {len(fm)} genomes, {len(FEAT_COLS)} features")

# ── Section 2.4: Per-class SHAP bar plots ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()

for cls_idx, (species, ax) in enumerate(zip(SPECIES_ORDER, axes_flat)):
    shap_cls = shap_3d[:, :, cls_idx]
    mean_abs = np.abs(shap_cls).mean(axis=0)
    top15    = pd.Series(mean_abs, index=FEAT_COLS).sort_values(ascending=False).head(15)

    colors_bar = [
        POS_COLOR if v >= 0 else NEG_COLOR
        for v in shap_cls.mean(axis=0)[
            [FEAT_COLS.index(f) for f in top15.index]
        ]
    ]
    ax.barh(
        y=[f.replace("dp_", "").replace("df_", "").replace("padloc_", "") for f in top15.index[::-1]],
        width=top15.values[::-1],
        color=colors_bar[::-1],
        edgecolor="white", linewidth=0.4,
    )
    ax.set_title(SPECIES_LABELS[species], fontsize=11, color=SPECIES_PALETTE[species])
    ax.set_xlabel("Mean |SHAP|", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    legend_patches = [
        mpatches.Patch(color=POS_COLOR, label="Pushes toward this species"),
        mpatches.Patch(color=NEG_COLOR, label="Pushes away from this species"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right",
              framealpha=0.7, edgecolor="none")

plt.suptitle("Per-class top-15 features by mean |SHAP|  -  Q1 RF", fontsize=12, y=1.01)
plt.tight_layout()
out = FIGS_INTERP / "q1_shap_per_class.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Supplement S-A: Within-species violin plots ───────────────────────────────
N_TOP = 6

fig, axes = plt.subplots(2, 3, figsize=(18, 13))
axes_flat = axes.flatten()

for cls_idx, (species, ax) in enumerate(zip(SPECIES_ORDER, axes_flat)):
    species_mask = (y_full == species)

    mean_abs_all = np.abs(shap_3d[:, :, cls_idx]).mean(axis=0)
    top_feats = (
        pd.Series(mean_abs_all, index=FEAT_COLS)
        .sort_values(ascending=False)
        .head(N_TOP)
        .index.tolist()
    )

    rows = []
    for feat_col in top_feats:
        fi        = FEAT_COLS.index(feat_col)
        shap_vals = shap_3d[species_mask, fi, cls_idx]
        feat_vals = X_full[species_mask, fi].astype(int)
        label     = feat_col.replace("dp_", "").replace("df_", "").replace("padloc_", "")
        for sv, fv in zip(shap_vals, feat_vals):
            rows.append({"Feature": label, "SHAP": sv,
                         "Presence": "Present" if fv == 1 else "Absent"})

    df_plot = pd.DataFrame(rows)

    valid_feats = []
    for feat_col in top_feats:
        label = feat_col.replace("dp_", "").replace("df_", "").replace("padloc_", "")
        sub   = df_plot[df_plot["Feature"] == label]
        if (sub["Presence"] == "Present").sum() >= 5 and (sub["Presence"] == "Absent").sum() >= 5:
            valid_feats.append(label)

    df_valid = df_plot[df_plot["Feature"].isin(valid_feats)]
    order    = [f.replace("dp_", "").replace("df_", "").replace("padloc_", "") for f in top_feats if f.replace("dp_", "").replace("df_", "").replace("padloc_", "") in valid_feats]

    n_excluded = len(top_feats) - len(valid_feats)
    if df_valid.empty:
        ax.text(0.5, 0.5, "All features near-universal\nin this species",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    else:
        sns.violinplot(
            data=df_valid, x="SHAP", y="Feature",
            hue="Presence", hue_order=["Present", "Absent"],
            palette={"Present": POS_COLOR, "Absent": NEG_COLOR},
            order=order, orient="h", inner="quart",
            cut=0, linewidth=0.5, ax=ax,
        )
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.legend(title="Feature value", fontsize=7, title_fontsize=7,
                  loc="lower right", framealpha=0.7, edgecolor="none")

    n_sp = species_mask.sum()
    title_str = f"{SPECIES_LABELS[species]}  (n={n_sp} genomes)"
    if n_excluded > 0:
        title_str += (f"\n{n_excluded} of top {N_TOP} features absent in this species"
                      " (shown in grey)")
    ax.set_title(title_str, fontsize=9, color=SPECIES_PALETTE[species])
    ax.set_xlabel(f"SHAP value for {SPECIES_LABELS[species]} class", fontsize=8)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=8)

plt.suptitle(
    "Within-species SHAP distribution: top 6 features per class\n"
    "Blue = genomes carrying the feature; Grey = genomes lacking it.  "
    "Dashed line = 0 (no effect).\n"
    "Features filtered to groups with ≥5 genomes; features universal in this species are excluded.",
    fontsize=9, y=1.02,
)
plt.tight_layout()
for ext in ("png", "pdf"):
    out = FIGS_INTERP / f"sfig_sa_shap_within_species.{ext}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()

# ── Section 2.5: Top-6 dependence plots (2×3 grid) ───────────────────────────
global_shap = pd.read_csv(RES / "q4_shap_367_global.csv", index_col="feature")["global_shap"]
global_shap = global_shap.reindex(FEAT_COLS).dropna()
top6_features = global_shap.sort_values(ascending=False).head(6).index.tolist()
KEY_FEATURES  = {f: f.replace("dp_", "").replace("df_", "").replace("padloc_", "").replace("_", " ") for f in top6_features}

print("Top 6 features (dependence plots):")
for f, label in KEY_FEATURES.items():
    print(f"  {f:40s}  SHAP = {global_shap[f]:.5f}")

species_labels_arr = fm["species"].to_numpy(dtype=str)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 10))
axes_flat = axes.flatten()

rng = np.random.default_rng(42)
for ax, (feat_col, feat_label) in zip(axes_flat, KEY_FEATURES.items()):
    feat_idx  = FEAT_COLS.index(feat_col)
    shap_feat = shap_3d[:, feat_idx, :].mean(axis=1)
    feat_vals = X_full[:, feat_idx]

    plot_df = pd.DataFrame({
        "species":  species_labels_arr,
        "shap_val": shap_feat,
        "feat_val": feat_vals.astype(int),
    })

    for species in SPECIES_ORDER:
        sub = plot_df[plot_df["species"] == species]
        for val, marker, alpha in [(0, "x", 0.4), (1, "o", 0.7)]:
            pts = sub[sub["feat_val"] == val]
            ax.scatter(
                val + rng.uniform(-0.1, 0.1, len(pts)),
                pts["shap_val"],
                color=SPECIES_PALETTE[species], alpha=alpha,
                marker=marker, s=18, linewidths=0.5,
                label=SPECIES_LABELS[species] if val == 1 else "",
            )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Absent", "Present"])
    ax.set_title(feat_label, fontsize=10)
    ax.set_ylabel("Mean SHAP value (over 6 classes)", fontsize=9)
    ax.set_xlabel("Feature presence", fontsize=9)

handles = [
    mpatches.Patch(color=SPECIES_PALETTE[s], label=SPECIES_LABELS[s])
    for s in SPECIES_ORDER
]
axes_flat[2].legend(handles=handles, fontsize=8, title="Species",
                    bbox_to_anchor=(1.02, 1), loc="upper left")

plt.suptitle("SHAP dependence plots  -  top 6 features by global mean |SHAP|\n"
             "(x: feature presence, y: mean SHAP effect over 6 classes)", fontsize=11)
plt.tight_layout()
out = FIGS_INTERP / "q1_shap_top6_dependence.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
