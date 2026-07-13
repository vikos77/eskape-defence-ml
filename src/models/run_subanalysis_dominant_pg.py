"""
run_subanalysis_dominant_pg.py

Dominant-phylogroup exclusion subanalysis (Supplementary Table extension),
for KP, EF, PA, SA -- the four species not already covered by the existing
AB IC2-exclusion (S7) and EC E. hormaechei (S8) subanalyses in
run_subanalysis_367.py.

Rationale: AB's IC2 exclusion (S7) and EC's hormaechei-complex framing (S8)
test clonal dominance using species-appropriate units (IC2 is a multi-ST
clonal complex; E. hormaechei is a complex member). The correct generalised
unit for "is this species' Q2 signal an artefact of one dominant lineage" is
max single-phylogroup share, since phylogroups are the grouping unit Q2's
CV is already built on. Measured directly from the 309-phylogroup Mash
clustering (data/processed/feature_matrix_3335.parquet):

  AB  top phylogroup = 48.5% of species cohort  (S7 already covers this)
  EC  top phylogroup =  6.3%                    (flat; S8 covers complex axis)
  EF  top phylogroup = 21.4%
  KP  top phylogroup = 15.1%
  PA  top phylogroup =  7.5%                    (flat)
  SA  top phylogroup = 21.0%

EF and SA show the most concentration of the remaining four (21%); KP is
moderate (15%); PA is flat (7.5%, comparable to EC) and excluding its top
phylogroup is not expected to move results -- run for completeness/symmetry
since the reviewer asked for all 4.

Same Q2 logic as run_q2_367.py / run_subanalysis_367.py:
  - Per-subset tertile split of arg_count_unique
  - Per-subset prevalence filter (dp_* >= 5%)
  - StratifiedGroupKFold-5 RF Q2
  - AUROC + fold-bootstrap CI (2000 iters)
  - Spearman rho for top Gini features

Output: results/supplement_dominant_pg_exclusion_q2_367.csv
"""

import sys
sys.path.insert(0, "src")

import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
N_FOLDS      = 5
PREV_THRESH  = 0.05

with open("results/q1_367_results.json") as f:
    _q1 = json.load(f)
RF_PARAMS = dict(
    **_q1["best_params"],
    class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)


def fold_bootstrap_ci(fold_scores, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [
        float(np.mean(rng.choice(fold_scores, size=len(fold_scores), replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_q2_subanalysis(subset, label, feat_cols_pool):
    q33 = subset["arg_count_unique"].quantile(1 / 3)
    q67 = subset["arg_count_unique"].quantile(2 / 3)
    high = subset[subset["arg_count_unique"] >= q67].copy()
    low  = subset[subset["arg_count_unique"] <= q33].copy()
    q2   = pd.concat([high, low]).copy()
    q2["label"] = (q2["arg_count_unique"] >= q67).astype(int)

    n_high, n_low, n_q2 = len(high), len(low), len(q2)
    n_pgs = q2["phylogroup"].nunique()
    print(f"\n{label}: n_total={len(subset)}, Q2 high={n_high}, low={n_low}, total={n_q2}, phylogroups={n_pgs}")

    prev = q2[feat_cols_pool].mean()
    feat_cols = [c for c in feat_cols_pool if prev[c] >= PREV_THRESH]
    print(f"  Features after prevalence filter: {len(feat_cols)}")

    if n_pgs < N_FOLDS:
        print(f"  WARNING: fewer phylogroups ({n_pgs}) than folds ({N_FOLDS}) -- skipping")
        return None

    X      = q2[feat_cols].to_numpy(dtype=float)
    y      = q2["label"].to_numpy()
    groups = q2["phylogroup"].to_numpy(dtype=str)

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_auroc = []
    fold_ba    = []
    gini_sum   = np.zeros(len(feat_cols))

    for tr, te in cv.split(X, y, groups=groups):
        if len(np.unique(y[te])) < 2:
            continue
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[tr], y[tr])
        proba = rf.predict_proba(X[te])[:, 1]
        fold_auroc.append(roc_auc_score(y[te], proba))
        fold_ba.append(balanced_accuracy_score(y[te], rf.predict(X[te])))
        gini_sum += rf.feature_importances_

    mean_auroc = float(np.mean(fold_auroc))
    ci_lo, ci_hi = fold_bootstrap_ci(fold_auroc)
    mean_ba = float(np.mean(fold_ba))

    gini_avg = gini_sum / len(fold_auroc)
    top_idx  = np.argsort(gini_avg)[::-1][:10]
    top_feats = [(feat_cols[i], float(gini_avg[i])) for i in top_idx]

    spearman_rows = []
    for feat, gini in top_feats[:5]:
        rho, p = spearmanr(subset[feat], subset["arg_count_unique"])
        spearman_rows.append({"feature": feat, "gini": round(gini, 5),
                               "rho_arg": round(rho, 4), "p_arg": round(p, 6)})

    print(f"  AUROC={mean_auroc:.3f} [{ci_lo:.3f}-{ci_hi:.3f}]  BA={mean_ba:.3f}")
    for r in spearman_rows:
        print(f"    {r['feature']:<25}  gini={r['gini']:.4f}  rho_ARG={r['rho_arg']:+.3f}  p={r['p_arg']:.4f}")

    return {
        "label":       label,
        "n_total":     len(subset),
        "n_q2":        n_q2,
        "n_features":  len(feat_cols),
        "n_phylogroups": n_pgs,
        "mean_auroc":  mean_auroc,
        "ci95_lo":     ci_lo,
        "ci95_hi":     ci_hi,
        "mean_ba":     mean_ba,
        "top5_spearman": spearman_rows,
    }


# ── Load feature matrix ───────────────────────────────────────────────────

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
dp_cols = [c for c in fm.columns if c.startswith("dp_")]
feat_cols_pool = dp_cols

q2_stored = json.load(open("results/q2_367_results.json"))

SPECIES = ["kpneumoniae", "efaecium", "paeruginosa", "saureus"]

all_rows = []
summary = {}

for sp in SPECIES:
    sp_all = fm[fm["species"] == sp].copy()
    vc = sp_all["phylogroup"].value_counts()
    top_pg = vc.index[0]
    top_share = vc.iloc[0] / len(sp_all)

    print("\n" + "=" * 60)
    print(f"{sp}: excluding top phylogroup {top_pg} ({top_share*100:.1f}% of cohort, n={vc.iloc[0]})")
    print("=" * 60)

    sub = sp_all[sp_all["phylogroup"] != top_pg].copy()
    res_full = {"mean_auroc": q2_stored[sp]["mean_auroc"], "mean_ba": q2_stored[sp]["mean_ba"]}

    res_excl = run_q2_subanalysis(sub, f"{sp} (top-PG excluded)", feat_cols_pool)

    if res_excl is None:
        summary[sp] = {"top_phylogroup": top_pg, "top_pg_share": round(top_share, 4),
                        "status": "skipped (insufficient phylogroups after exclusion)"}
        continue

    delta_auroc = res_excl["mean_auroc"] - res_full["mean_auroc"]
    print(f"  Full-cohort AUROC={res_full['mean_auroc']:.3f}  ->  top-PG-excluded AUROC={res_excl['mean_auroc']:.3f}  (delta={delta_auroc:+.3f})")

    summary[sp] = {
        "top_phylogroup": top_pg,
        "top_pg_share": round(top_share, 4),
        "n_excluded": int(vc.iloc[0]),
        "full_cohort_auroc": res_full["mean_auroc"],
        "excluded_cohort_auroc": res_excl["mean_auroc"],
        "delta_auroc": round(delta_auroc, 4),
        "excluded_cohort_n": res_excl["n_total"],
        "excluded_cohort_n_q2": res_excl["n_q2"],
        "excluded_cohort_n_phylogroups": res_excl["n_phylogroups"],
        "excluded_cohort_ci95": [round(res_excl["ci95_lo"], 4), round(res_excl["ci95_hi"], 4)],
    }

    for r in res_excl["top5_spearman"]:
        all_rows.append({"species": sp, "top_phylogroup": top_pg,
                          "top_pg_share": round(top_share, 4), **r})

print("\n" + "=" * 60)
print("SUMMARY: dominant-phylogroup exclusion, full vs excluded Q2 AUROC")
print("=" * 60)
for sp, s in summary.items():
    if s.get("status"):
        print(f"  {sp:<14} {s['status']}")
        continue
    print(f"  {sp:<14} top_pg_share={s['top_pg_share']*100:5.1f}%  "
          f"full_AUROC={s['full_cohort_auroc']:.3f}  excl_AUROC={s['excluded_cohort_auroc']:.3f}  "
          f"delta={s['delta_auroc']:+.3f}")

with open("results/supplement_dominant_pg_exclusion_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
pd.DataFrame(all_rows).to_csv("results/supplement_dominant_pg_exclusion_q2_367.csv", index=False)
print("\nSaved: results/supplement_dominant_pg_exclusion_summary.json")
print("Saved: results/supplement_dominant_pg_exclusion_q2_367.csv")
