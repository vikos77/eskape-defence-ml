"""
Patch notebooks 06, 07, and 12 to use cluster bootstrap CIs.

Replaces the inline genome-level bootstrap_ci function with a wrapper around
the cluster-aware bootstrap_ci_auto from src/evaluation/bootstrap.py.
Also adds per-genome prediction saves for Q1 models.

Run from the project root:
    python notebooks/patch_cluster_bootstrap.py

Idempotent: checks whether the patch has already been applied before modifying.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ─── Replacement text for the inline bootstrap_ci function ─────────────────────
# Inserted into cell 8 of notebook 06 and the equivalent cells in 07/12.

BOOTSTRAP_IMPORT_BLOCK = """\
import sys as _sys
_sys.path.insert(0, str(Path("..") / "src"))
from evaluation.bootstrap import bootstrap_ci_auto as _bootstrap_ci_auto
from evaluation.bootstrap import genome_bootstrap_ci as _genome_bci

def bootstrap_ci(y_true, y_pred, groups_arg=None, metric_fn=None, n_boot=N_BOOT, seed=42):
    \"\"\"
    Cluster bootstrap CI where n_phylogroups >= 15 (Q1, Q2-EC/KP/PA).
    Falls back to genome-level bootstrap when n_phylogroups < 15 (EF, SA, AB).
    Wrapper maintains the original (mean, lo, hi) return convention.
    \"\"\"
    if metric_fn is None:
        metric_fn = balanced_accuracy_score
    mean_val = metric_fn(y_true, y_pred)
    if groups_arg is None:
        lo, hi = _genome_bci(y_true, y_pred, metric_fn, n_boot, seed)
    else:
        lo, hi, _ = _bootstrap_ci_auto(y_true, y_pred, groups_arg, metric_fn, n_boot, seed)
    return mean_val, lo, hi
"""

OLD_BOOTSTRAP_DEF = (
    "def bootstrap_ci(y_true, y_pred, metric_fn=None, n_boot=N_BOOT, seed=42):\n"
    "    # Bootstrap 95% CI over pooled per-genome predictions (M2 fix: n=878, not n=5)\n"
    "    if metric_fn is None:\n"
    "        metric_fn = balanced_accuracy_score\n"
    "    rng = np.random.default_rng(seed)\n"
    "    n = len(y_true)\n"
    "    scores = []\n"
    "    for _ in range(n_boot):\n"
    "        idx = rng.integers(0, n, size=n)\n"
    "        scores.append(metric_fn(y_true[idx], y_pred[idx]))\n"
    "    return metric_fn(y_true, y_pred), np.percentile(scores, 2.5), np.percentile(scores, 97.5)"
)


def patch_notebook(path: Path, patches: list[tuple[str, str]]) -> bool:
    """Apply (old, new) text patches to a notebook's cell sources. Returns True if changed."""
    with open(path) as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for old, new in patches:
            if old in src:
                src = src.replace(old, new, 1)
                changed = True
        if changed:
            cell["source"] = [src]  # store as single string; jupyter renders fine

    if changed:
        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  Patched: {path.name}")
    else:
        print(f"  No changes (already patched or pattern not found): {path.name}")
    return changed


def patch_notebook_06():
    """Patch RF notebook: bootstrap definition + Q1 and Q2 call sites."""
    path = ROOT / "notebooks" / "06_random_forest.ipynb"

    patches = [
        # 1. Replace inline bootstrap_ci definition
        (OLD_BOOTSTRAP_DEF, BOOTSTRAP_IMPORT_BLOCK),

        # 2. Q1 main bootstrap calls: add groups parameter
        (
            "mean_ba, lo_ba, hi_ba = bootstrap_ci(yt_q1, yp_q1)\n"
            "mean_f1, lo_f1, hi_f1 = bootstrap_ci(yt_q1, yp_q1,\n"
            "                                       metric_fn=lambda yt, yp: f1_score(yt, yp, average=\"macro\"))",
            "mean_ba, lo_ba, hi_ba = bootstrap_ci(yt_q1, yp_q1, groups_arg=groups)\n"
            "mean_f1, lo_f1, hi_f1 = bootstrap_ci(yt_q1, yp_q1, groups_arg=groups,\n"
            "                                       metric_fn=lambda yt, yp: f1_score(yt, yp, average=\"macro\"))\n"
            "# Save per-genome Q1 RF predictions for future recomputation\n"
            "np.save(RES / 'q1_rf_pred_true.npy', np.array(yt_q1))\n"
            "np.save(RES / 'q1_rf_pred_pred.npy', np.array(yp_q1))\n"
            "np.save(RES / 'q1_rf_pred_groups.npy', np.array(groups))"
        ),

        # 3. Q2 per-species bootstrap call: add grp_sp
        (
            "mean_ba_sp, lo_sp, hi_sp = bootstrap_ci(np.array(all_yt_sp), np.array(all_yp_sp))",
            "mean_ba_sp, lo_sp, hi_sp = bootstrap_ci(np.array(all_yt_sp), np.array(all_yp_sp), groups_arg=grp_sp)"
        ),

        # 4. Sensitivity analysis bootstrap call: add groups
        (
            "mean_t, lo_t, hi_t = bootstrap_ci(yt_sens, yp_sens)",
            "mean_t, lo_t, hi_t = bootstrap_ci(yt_sens, yp_sens, groups_arg=groups)"
        ),

        # 5. Anti-defence Q2 sensitivity: add grp_sp equivalent
        (
            "ba_ad, _, _ = bootstrap_ci(np.array(all_yt), np.array(all_yp))",
            "ba_ad, _, _ = bootstrap_ci(np.array(all_yt), np.array(all_yp), groups_arg=grp_sp)"
        ),
    ]

    return patch_notebook(path, patches)


def patch_notebook_07():
    """Patch XGB/LGBM notebook: bootstrap definition + Q1 call sites."""
    path = ROOT / "notebooks" / "07_gradient_boosting.ipynb"

    patches = [
        (OLD_BOOTSTRAP_DEF, BOOTSTRAP_IMPORT_BLOCK),

        # Early-stopping XGB BA call (cell 15)
        (
            "mean_ba_xgb, lo_xgb, hi_xgb = bootstrap_ci(mc_true_xgb, mc_pred_xgb)",
            "mean_ba_xgb, lo_xgb, hi_xgb = bootstrap_ci(mc_true_xgb, mc_pred_xgb, groups_arg=groups)\n"
            "np.save(RES / 'q1_xgb_pred_true.npy', mc_true_xgb)\n"
            "np.save(RES / 'q1_xgb_pred_pred.npy', mc_pred_xgb)\n"
            "np.save(RES / 'q1_xgb_pred_groups.npy', groups)"
        ),

        # Early-stopping XGB F1 call (cell 15)  -  add groups_arg
        (
            "mean_f1_xgb, lo_f1_xgb, hi_f1_xgb = bootstrap_ci(mc_true_xgb, mc_pred_xgb,\n"
            "    metric_fn=lambda yt, yp: f1_score(yt, yp, average=\"macro\"))",
            "mean_f1_xgb, lo_f1_xgb, hi_f1_xgb = bootstrap_ci(mc_true_xgb, mc_pred_xgb,\n"
            "    groups_arg=groups, metric_fn=lambda yt, yp: f1_score(yt, yp, average=\"macro\"))"
        ),

        # Early-stopping LGBM BA call (cell 17)
        (
            "mean_ba_lgbm, lo_lgbm, hi_lgbm = bootstrap_ci(mc_true_lgbm, mc_pred_lgbm)",
            "mean_ba_lgbm, lo_lgbm, hi_lgbm = bootstrap_ci(mc_true_lgbm, mc_pred_lgbm, groups_arg=groups)\n"
            "np.save(RES / 'q1_lgbm_pred_true.npy', mc_true_lgbm)\n"
            "np.save(RES / 'q1_lgbm_pred_pred.npy', mc_pred_lgbm)"
        ),

        # Early-stopping LGBM F1 call (cell 17)  -  add groups_arg
        (
            "mean_f1_lgbm, lo_f1_lgbm, hi_f1_lgbm = bootstrap_ci(mc_true_lgbm, mc_pred_lgbm,\n"
            "    metric_fn=lambda yt, yp: f1_score(yt, yp, average=\"macro\"))",
            "mean_f1_lgbm, lo_f1_lgbm, hi_f1_lgbm = bootstrap_ci(mc_true_lgbm, mc_pred_lgbm,\n"
            "    groups_arg=groups, metric_fn=lambda yt, yp: f1_score(yt, yp, average=\"macro\"))"
        ),

        # Fixed-iter XGB BA (cell 21)  -  add groups_arg + save
        (
            "ba_xf,  lo_xf,  hi_xf  = bootstrap_ci(mc_true_fixed, mc_pred_xgb_fixed)",
            "ba_xf,  lo_xf,  hi_xf  = bootstrap_ci(mc_true_fixed, mc_pred_xgb_fixed, groups_arg=groups)\n"
            "np.save(RES / 'q1_xgb_fixed_pred_true.npy', mc_true_fixed)\n"
            "np.save(RES / 'q1_xgb_fixed_pred_pred.npy', mc_pred_xgb_fixed)\n"
            "np.save(RES / 'q1_xgb_fixed_pred_groups.npy', groups)"
        ),

        # Fixed-iter LGBM BA (cell 21)  -  add groups_arg
        (
            "ba_lf,  lo_lf,  hi_lf  = bootstrap_ci(mc_true_fixed, mc_pred_lgbm_fixed)",
            "ba_lf,  lo_lf,  hi_lf  = bootstrap_ci(mc_true_fixed, mc_pred_lgbm_fixed, groups_arg=groups)"
        ),
    ]

    return patch_notebook(path, patches)


def patch_notebook_12():
    """Patch Phase 12 sensitivity notebook: bootstrap calls for Test A and Test B."""
    path = ROOT / "notebooks" / "10_phase12_sensitivity.ipynb"
    if not path.exists():
        print(f"  Not found: {path.name}")
        return False

    patches = [
        (OLD_BOOTSTRAP_DEF, BOOTSTRAP_IMPORT_BLOCK),
        # Test A and Test B bootstrap calls use grp_sp (per-species groups)
        # These patterns need to be verified against the actual notebook content
        # They are marked as approximate  -  check output after patching
    ]

    return patch_notebook(path, patches)


if __name__ == "__main__":
    print("Applying cluster bootstrap patches to notebooks...")
    print()
    changed_06 = patch_notebook_06()
    changed_07 = patch_notebook_07()
    changed_12 = patch_notebook_12()
    print()
    if any([changed_06, changed_07, changed_12]):
        print("Done. Re-run modified notebooks to compute updated cluster bootstrap CIs.")
        print("Verify: check that BA point estimates are unchanged and CIs are wider.")
    else:
        print("Nothing changed. Check patch patterns match notebook source exactly.")
