"""
Fig 3B — Q2 within-phylogroup Spearman ρ for robust ARG-burden drivers.

Four panels (2×2), one per Q2-significant species. Each panel shows
features that survived the within-phylogroup robustness check (p_within_pg
< 0.05, sign preserved), ordered by rho_within_pg descending. Dots are
coloured by mechanism category (named vs uncharacterised candidate).

Data source: results/q2_367_results.json (top_drivers per species)
Output:      results/figures/q2/fig3b_q2_driver_dotplot.png  (300 dpi)
             results/figures/q2/fig3b_q2_driver_dotplot.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "results/figures/q2"
os.makedirs(OUT_DIR, exist_ok=True)

with open("results/q2_367_results.json") as f:
    q2 = json.load(f)

# ── Species to show — ordered by AUROC descending (matches forest plot order)
PANELS = [
    ("efaecium",    r"$\it{E.\ faecium}$",   "#2CA25F"),
    ("kpneumoniae", r"$\it{K.\ pneumoniae}$", "#457B9D"),
    ("paeruginosa", r"$\it{P.\ aeruginosa}$", "#8338EC"),
    ("ecloaceae",   r"$\it{E.\ cloacae}$ cx", "#F4A261"),
]

COL_NAMED = "#2171B5"    # blue — named systems
COL_UNC   = "#E07B54"    # terracotta — uncharacterised candidates

MAX_FEATS = 8   # cap per panel for readability

NAMED_CATS = {"named"}

def is_named(cat):
    return cat in NAMED_CATS

def short_name(feat):
    name = feat.removeprefix("dp_")
    # tidy catch-all labels for display
    if name == "cbass_other":
        name = "CBASS_other"
    return name

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=False)
axes_flat = axes.flatten()

for ax_i, (sp, sp_label, sp_color) in enumerate(PANELS):
    ax = axes_flat[ax_i]
    drivers = q2[sp]["top_drivers"]

    # Filter: within-PG significant, sign known
    robust_all = [
        d for d in drivers
        if d.get("rho_within_pg") is not None
        and d.get("p_within_pg", 1) < 0.05
    ]
    n_survive = len(robust_all)
    # Sort by rho_within_pg descending
    robust_all = sorted(robust_all, key=lambda d: d["rho_within_pg"], reverse=True)
    # Cap for readability: top positives + most negative
    if n_survive > MAX_FEATS:
        n_pos = sum(1 for d in robust_all if d["rho_within_pg"] > 0)
        n_neg = n_survive - n_pos
        pos_keep = min(MAX_FEATS - min(2, n_neg), n_pos)
        neg_keep = MAX_FEATS - pos_keep
        robust = robust_all[:pos_keep] + robust_all[n_survive - neg_keep:]
    else:
        robust = robust_all

    labels  = [short_name(d["feature"]) for d in robust]
    rho_vals = [d["rho_within_pg"] for d in robust]
    colors   = [COL_NAMED if is_named(d.get("category","")) else COL_UNC for d in robust]

    y_pos = np.arange(len(labels))[::-1]   # top = highest rho

    # lollipop sticks
    for i, (y, rho) in enumerate(zip(y_pos, rho_vals)):
        ax.plot([0, rho], [y, y], color=colors[i], linewidth=1.4, alpha=0.7)

    # dots
    ax.scatter(rho_vals, y_pos, color=colors, s=60, zorder=3, linewidths=0.5,
               edgecolors="white")

    # value labels
    for i, (y, rho) in enumerate(zip(y_pos, rho_vals)):
        offset = 0.012 if rho >= 0 else -0.012
        ha = "left" if rho >= 0 else "right"
        ax.text(rho + offset, y, f"{rho:+.3f}", ha=ha, va="center",
                fontsize=7.0, color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(-0.35, 0.55)
    ax.set_xlabel("Within-phylogroup Spearman ρ", fontsize=8)
    n_shown = len(robust)
    shown_str = f"showing top {n_shown}" if n_survive > n_shown else f"n = {n_shown}"
    ax.set_title(f"{sp_label}\nAUROC = {q2[sp]['mean_auroc']:.3f}  "
                 f"({n_survive} robust drivers; {shown_str})",
                 fontsize=8.5, color=sp_color, loc="left")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.5)

# ── Legend ───────────────────────────────────────────────────────────────────
named_patch = mpatches.Patch(color=COL_NAMED, label="Named system")
unc_patch   = mpatches.Patch(color=COL_UNC,   label="Uncharacterised candidate")
fig.legend(handles=[named_patch, unc_patch], fontsize=8,
           loc="lower center", ncol=2, framealpha=0.9,
           edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.01))

fig.suptitle("(B)  Top within-phylogroup-robust ARG-burden drivers, Q2-significant species\n"
             "Spearman ρ corrected for phylogroup structure; p_within_pg < 0.05",
             fontsize=9, y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.98])

for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"fig3b_q2_driver_dotplot.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
