"""
Combine individual figure panels into submission-ready multi-panel figures.

Run from project root:
    conda run -n eskape-ml python src/visualization/combine_panels.py

Outputs (all 300 DPI PNG + PDF stub):
  results/figures/final/fig1_q1_classification.{png,pdf}
  results/figures/final/fig2_q2_arg_prediction.{png,pdf}
  results/figures/final/fig3_q3b_block_comparison.{png,pdf}   (copy)
  results/figures/final/fig4_shap_interpretation.{png,pdf}
"""

import os, shutil
from PIL import Image

ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL   = os.path.join(ROOT, "results", "figures", "final")
os.makedirs(FINAL, exist_ok=True)

DPI = 300
GAP = 30   # px gap between stacked or side-by-side panels
BG  = (255, 255, 255)   # white background


def p(rel):
    return os.path.join(ROOT, rel)


def load(rel):
    return Image.open(p(rel)).convert("RGBA")


def stack_vertical(imgs, gap=GAP):
    """Stack images vertically, centred horizontally on white background."""
    W = max(i.width for i in imgs)
    H = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    y = 0
    for img in imgs:
        x = (W - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + gap
    return canvas


def side_by_side(imgs, gap=GAP):
    """Place images side-by-side, aligned to top, white background."""
    W = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    H = max(i.height for i in imgs)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    x = 0
    for img in imgs:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    return canvas


def save(img, stem):
    rgb = img.convert("RGB")
    out_png = os.path.join(FINAL, stem + ".png")
    rgb.save(out_png, dpi=(DPI, DPI))
    # PDF: embed the PNG as a single-page PDF via PIL
    out_pdf = os.path.join(FINAL, stem + ".pdf")
    rgb.save(out_pdf, "PDF", resolution=DPI)
    print(f"  Saved: {stem}.png  ({img.width}x{img.height} px)")


# ── Fig 1: Q1 classification (1A confusion matrix / 1B holdout) ──────────────
print("Building Fig 1...")
fig1a = load("results/figures/rf/fig1a_q1_confusion_matrix.png")
fig1b = load("results/figures/rf/fig1b_holdout_validation.png")
# 1A: 1500x1350, 1B: 2400x1050 — stack 1A on top of 1B
save(stack_vertical([fig1a, fig1b]), "fig1_q1_classification")

# ── Fig 2: Q2 ARG prediction (2A AUROC forest / 2B driver dotplot) ───────────
print("Building Fig 2...")
fig2a = load("results/figures/q2/fig2a_q2_auroc_forest.png")
fig2b = load("results/figures/q2/fig2b_q2_driver_dotplot.png")
# 2A: 1950x1140, 2B: 2700x2100 — stack 2A on top of 2B
save(stack_vertical([fig2a, fig2b]), "fig2_q2_arg_prediction")

# ── Fig 3: Q3b block comparison (standalone) ─────────────────────────────────
print("Building Fig 3...")
fig3 = load("results/figures/q3b/fig3_q3b_block_comparison.png")
save(fig3, "fig3_q3b_block_comparison")

# ── Fig 4: SHAP interpretation (4A global beeswarm / 4B per-species heatmap) ─
print("Building Fig 4...")
fig4a = load("results/figures/interpretation/fig4a_shap_global_beeswarm.png")
fig4b = load("results/figures/interpretation/fig4b_shap_heatmap.png")
# 4A: 2519x2065, 4B: 1950x2100 — side by side (similar heights)
save(side_by_side([fig4a, fig4b]), "fig4_shap_interpretation")

print("\nAll combined figures written to results/figures/final/")
