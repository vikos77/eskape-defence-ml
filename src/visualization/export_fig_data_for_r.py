"""
Export figure data for R visualization scripts.

Run once from project root before running any R figure scripts:
    conda run -n eskape-ml python src/visualization/export_fig_data_for_r.py

Generates:
  results/fig1a_cm_for_r.csv          -- row-normalised confusion matrix (6x6)
  results/fig1a_stats_for_r.json      -- BA, 95% CI, fold_bas
  results/fig1b_holdout_for_r.json    -- per-species recall, holdout BA + CI, CV BA + CI
  results/fig4b_heatmap_for_r.csv     -- SHAP heatmap (20 rows x 6 species columns)
"""

import json, os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.utils import resample

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def p(rel): return os.path.join(ROOT, rel)

# ── Fig 1A: confusion matrix ──────────────────────────────────────────────────
print("Exporting Fig 1A data...")
data    = np.load(p("results/q1_367_oof_preds.npz"))
y_true  = data["y_true"]
y_pred  = data["y_pred"]
with open(p("results/q1_367_results.json")) as f:
    q1_res = json.load(f)

CLASS_NAMES = ["AB", "EC", "EF", "KP", "PA", "SA"]
n_classes   = len(CLASS_NAMES)
cm     = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

cm_df = pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
cm_df.index.name = "true_species"
cm_df.to_csv(p("results/fig1a_cm_for_r.csv"))

ci_lo, ci_hi = q1_res["ci95_ba"]
mean_ba = float(np.mean(q1_res["fold_bas"]))
with open(p("results/fig1a_stats_for_r.json"), "w") as f:
    json.dump({"mean_ba": mean_ba, "ci_lo": ci_lo, "ci_hi": ci_hi,
               "fold_bas": q1_res["fold_bas"]}, f, indent=2)
print("  -> fig1a_cm_for_r.csv, fig1a_stats_for_r.json")

# ── Fig 1B: holdout validation ────────────────────────────────────────────────
print("Exporting Fig 1B data (runs RF model on holdout)...")
hfm = pd.read_parquet(p("data/processed/holdout_feature_matrix.parquet"))
with open(p("results/q1_367_feat_cols.txt")) as f:
    feat_cols = [l.strip() for l in f if l.strip()]
with open(p("results/models/rf_q1_367.pkl"), "rb") as f:
    rf = pickle.load(f)

X_hold   = hfm[feat_cols].values
y_str    = hfm["species"].to_numpy(dtype=str)
y_pred_h = rf.predict(X_hold)
ba_h     = balanced_accuracy_score(y_str, y_pred_h)

rng = np.random.default_rng(42)
ba_boot = []
for _ in range(2000):
    idx = resample(np.arange(len(y_str)), stratify=y_str,
                   random_state=rng.integers(1_000_000))
    ba_boot.append(balanced_accuracy_score(y_str[idx], y_pred_h[idx]))
h_lo, h_hi = np.percentile(ba_boot, [2.5, 97.5])

SPECIES_ORDER = ["abaumannii", "ecloaceae", "efaecium",
                 "kpneumoniae", "paeruginosa", "saureus"]
recalls = {}
for sp in SPECIES_ORDER:
    mask = y_str == sp
    recalls[sp] = float((y_pred_h[mask] == sp).sum() / mask.sum())

cv_ba = float(np.mean(q1_res["fold_bas"]))
with open(p("results/fig1b_holdout_for_r.json"), "w") as f:
    json.dump({
        "recalls":   recalls,
        "holdout_ba": float(ba_h),
        "holdout_ci": [float(h_lo), float(h_hi)],
        "cv_ba":     cv_ba,
        "cv_ci":     [float(q1_res["ci95_ba"][0]), float(q1_res["ci95_ba"][1])],
    }, f, indent=2)
print(f"  -> fig1b_holdout_for_r.json  (holdout BA = {ba_h:.3f})")

# ── Fig 4B: SHAP heatmap ──────────────────────────────────────────────────────
print("Exporting Fig 4B SHAP heatmap data...")
shap_3d   = np.load(p("results/q4_shap_367_array.npy"))   # (3335, 359, 6)
global_df = pd.read_csv(p("results/q4_shap_367_global.csv"))
with open(p("results/q1_367_feat_cols.txt")) as f:
    feat_cols_all = [l.strip() for l in f if l.strip()]
fm = pd.read_parquet(p("data/processed/feature_matrix_3335.parquet"))
X_all = fm[feat_cols_all].to_numpy(dtype=float)

CLASSES    = ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]
COL_LABELS = ["AB", "EC", "EF", "KP", "PA", "SA"]
N_FEATS    = 20
top20      = global_df.head(N_FEATS).copy()
top20_idx  = [feat_cols_all.index(f) for f in top20["feature"]]

rows = []
for row_i, feat_i in enumerate(top20_idx):
    present_mask = X_all[:, feat_i] > 0
    if present_mask.sum() < 5:
        shap_vals = [0.0] * len(CLASSES)
    else:
        shap_vals = shap_3d[present_mask, feat_i, :].mean(axis=0).tolist()
    feat_name = top20.iloc[row_i]["feature"]
    category  = top20.iloc[row_i]["category"]
    name = feat_name.removeprefix("dp_").removeprefix("df_").removeprefix("padloc_")
    label = name + "†" if category != "named" else name
    row = {"rank": row_i + 1, "feature": feat_name, "label": label, "category": category}
    for col, val in zip(COL_LABELS, shap_vals):
        row[col] = val
    rows.append(row)

pd.DataFrame(rows).to_csv(p("results/fig4b_heatmap_for_r.csv"), index=False)
print("  -> fig4b_heatmap_for_r.csv")

print("\nAll exports complete.")
