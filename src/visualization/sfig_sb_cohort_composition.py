"""
sfig_sb_cohort_composition.py — Supplementary Figure S-B
Two-panel figure: genome year-bin distribution (Panel A) and retrospective
isolation-source classification per ESKAPE species (Panel B).

Outputs:
  results/figures/sfig_sb_cohort_composition.png
  results/figures/sfig_sb_cohort_composition.pdf
"""

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

SPECIES_KEYS = ["abaumannii", "ecloaceae", "efaecium",
                "kpneumoniae", "paeruginosa", "saureus"]
SPECIES_LABELS = [
    "A. baumannii",
    "E. cloacae complex",
    "E. faecium",
    "K. pneumoniae",
    "P. aeruginosa",
    "S. aureus",
]

# ── Year-bin settings ─────────────────────────────────────────────────────────
YEAR_BINS  = ["pre-2010", "2010–2015", "2016–2020", "2021–present"]
YEAR_BINS_JSON = ["pre-2010", "2010-2015", "2016-2020", "2021-present"]  # JSON keys
YEAR_MAP   = dict(zip(YEAR_BINS_JSON, YEAR_BINS))
YEAR_COLORS = ["#d4e6f1", "#7fb3d3", "#2980b9", "#1a5276"]   # light→dark blue

# ── Isolation-source settings ─────────────────────────────────────────────────
ISO_CAT_MAP = {
    "clinical_human":      "Clinical (human)",
    "clinical_animal":     "Clinical (animal)",
    "environmental_water": "Environmental",
    "environmental_soil":  "Environmental",
    "food":                "Food",
    "other":               "Other/unspecified",
    "unknown":             "Not provided",
    "not_fetched":         "Not provided",
}
ISO_CATS   = ["Clinical (human)", "Clinical (animal)", "Environmental",
              "Food", "Other/unspecified", "Not provided"]
ISO_COLORS = ["#c0392b", "#e67e22", "#27ae60", "#f1c40f", "#95a5a6", "#d5d8dc"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_year_data() -> pd.DataFrame:
    rows = []
    for sp_key, sp_label in zip(SPECIES_KEYS, SPECIES_LABELS):
        meta = json.load(open(f"data/interim/{sp_key}_accession_metadata.json"))
        counts = Counter(YEAR_MAP.get(r.get("year_bin", "unknown"), "Other")
                         for r in meta)
        total = sum(counts.values())
        rows.append({
            "species": sp_label,
            **{yb: counts.get(yb, 0) for yb in YEAR_BINS},
            "total": total,
        })
    df = pd.DataFrame(rows).set_index("species")
    return df


def load_iso_data() -> pd.DataFrame:
    iso_df = pd.read_csv("data/interim/isolation_source_all.csv")
    iso_df["iso_group"] = iso_df["iso_class"].map(ISO_CAT_MAP).fillna("Not provided")

    rows = []
    for sp_key, sp_label in zip(SPECIES_KEYS, SPECIES_LABELS):
        subset = iso_df[iso_df["species"] == next(
            lab for k, lab in [
                ("abaumannii", "Acinetobacter baumannii"),
                ("ecloaceae",  "Enterobacter cloacae complex"),
                ("efaecium",   "Enterococcus faecium"),
                ("kpneumoniae","Klebsiella pneumoniae"),
                ("paeruginosa","Pseudomonas aeruginosa"),
                ("saureus",    "Staphylococcus aureus"),
            ] if k == sp_key
        )]
        counts = Counter(subset["iso_group"])
        total = len(subset)
        rows.append({
            "species": sp_label,
            **{cat: counts.get(cat, 0) for cat in ISO_CATS},
            "total": total,
        })
    df = pd.DataFrame(rows).set_index("species")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _stacked_hbar(ax, df: pd.DataFrame, categories: list, colors: list,
                  xlabel: str, show_n: bool = True):
    """Draw horizontal stacked percentage bars (species on y-axis)."""
    species_list = list(df.index)
    y = np.arange(len(species_list))
    totals = df["total"].values

    lefts = np.zeros(len(species_list))
    for cat, col in zip(categories, colors):
        vals = df[cat].values
        pcts = 100 * vals / totals
        bars = ax.barh(y, pcts, left=lefts, color=col, height=0.62,
                       edgecolor="white", linewidth=0.5)
        # Label segments ≥ 8%
        for j, (pct, left) in enumerate(zip(pcts, lefts)):
            if pct >= 8:
                ax.text(left + pct / 2, y[j], f"{pct:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if col not in ("#f1c40f", "#d5d8dc", "#d4e6f1") else "#333",
                        fontweight="bold")
        lefts += pcts

    ax.set_yticks(y)
    ax.set_yticklabels([f"$\\it{{{s.split()[0]}}}$ {' '.join(s.split()[1:])}"
                        if len(s.split()) > 1 else f"$\\it{{{s}}}$"
                        for s in species_list], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_n:
        for j, (sp, n) in enumerate(zip(species_list, totals)):
            ax.text(101, y[j], f"n={n}", va="center", fontsize=7.5, color="#555")


def make_figure():
    year_df = load_year_data()
    iso_df  = load_iso_data()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13, 4.2),
        gridspec_kw={"wspace": 0.55}
    )

    # ── Panel A: Year-bin ─────────────────────────────────────────────────────
    _stacked_hbar(ax_a, year_df, YEAR_BINS, YEAR_COLORS,
                  "Percentage of genomes", show_n=True)
    ax_a.set_title("A.  Genome release-year distribution", loc="left",
                   fontsize=10, fontweight="bold", pad=8)

    year_handles = [
        mpatches.Patch(color=c, label=yb)
        for c, yb in zip(YEAR_COLORS, YEAR_BINS)
    ]
    ax_a.legend(handles=year_handles, loc="lower right", fontsize=8,
                framealpha=0.8, edgecolor="none", title="Year bin",
                title_fontsize=8)

    # ── Panel B: Isolation source ─────────────────────────────────────────────
    _stacked_hbar(ax_b, iso_df, ISO_CATS, ISO_COLORS,
                  "Percentage of genomes", show_n=False)
    ax_b.set_title("B.  Retrospective isolation-source classification",
                   loc="left", fontsize=10, fontweight="bold", pad=8)

    iso_handles = [
        mpatches.Patch(color=c, label=cat)
        for c, cat in zip(ISO_COLORS, ISO_CATS)
    ]
    ax_b.legend(handles=iso_handles, loc="lower right", fontsize=8,
                framealpha=0.8, edgecolor="none",
                title="Isolation source", title_fontsize=8)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.12, wspace=0.55)

    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sfig_sb_cohort_composition.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "sfig_sb_cohort_composition.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("Saved:")
    print(f"  {out_dir / 'sfig_sb_cohort_composition.png'}")
    print(f"  {out_dir / 'sfig_sb_cohort_composition.pdf'}")


if __name__ == "__main__":
    make_figure()
