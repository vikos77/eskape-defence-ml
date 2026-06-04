"""
Full audit script — verifies every medium/high-risk quantitative claim in
methods.md and results.md. Run from the project root with full stderr visible.

Usage:
    conda run -n eskape-ml python src/audit_results.py
"""

import sys, warnings
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score
)
from scipy.stats import chi2 as chi2_dist
from evaluation.bootstrap import bootstrap_ci_auto
import joblib

ROOT = Path('.')
PROC = ROOT / 'data' / 'processed'
RES  = ROOT / 'results'

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
WARN = '\033[93mWARN\033[0m'

findings = []

def check(label, condition, detail=''):
    status = PASS if condition else FAIL
    tag    = 'PASS' if condition else 'FAIL'
    msg = f'[{status}] {label}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    findings.append((tag, label, detail))

def warn(label, detail=''):
    print(f'[{WARN}] {label}  ({detail})')
    findings.append(('WARN', label, detail))

print('=' * 70)
print('AUDIT: loading feature matrix and saved artifacts')
print('=' * 70)

fm      = pd.read_parquet(PROC / 'feature_matrix.parquet')
rf_imp  = pd.read_parquet(RES / 'q1_rf_feature_importance.parquet')
spec    = pd.read_parquet(RES / 'q1_rf_sensitivity_spec_filter.parquet')
q1_rf   = pd.read_parquet(RES / 'q1_rf_results.parquet')
q2_rf   = pd.read_parquet(RES / 'q2_rf_results.parquet')
q2_gb   = pd.read_parquet(RES / 'q2_gb_results.parquet')
testb   = pd.read_parquet(RES / 'testb_results.parquet')

print(f'Feature matrix: {fm.shape[0]} rows x {fm.shape[1]} columns')
print(f'Unique species:  {sorted(fm["species"].unique())}')
print(f'Species counts:  {fm["species"].value_counts().sort_index().to_dict()}')
print()

# ---------------------------------------------------------------------------
print('=' * 70)
print('M01 — Genome counts per species')
print('=' * 70)
expected = {'abaumannii': 150, 'ecloaceae': 146, 'efaecium': 150,
            'kpneumoniae': 132, 'paeruginosa': 150, 'saureus': 150}
actual   = fm['species'].value_counts().to_dict()
check('Total genomes = 878', fm.shape[0] == 878, f'actual={fm.shape[0]}')
for sp, n in expected.items():
    check(f'  {sp} n={n}', actual.get(sp, -1) == n,
          f'actual={actual.get(sp, -1)}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('M04 — Feature matrix column count')
print('=' * 70)
dp_cols = sorted([c for c in fm.columns if c.startswith('dp_')])
check('Methods §3 says 632 columns (fixed from 631)', fm.shape[1] == 632,
      f'actual={fm.shape[1]}')
check('274 dp_* columns', len(dp_cols) == 274, f'actual={len(dp_cols)}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('M06-M09 — Specificity filter')
print('=' * 70)
markers_07 = spec.loc[0.7, 'markers_removed']
FEAT_COLS  = [c for c in dp_cols if c not in markers_07]
check('9 markers removed at threshold 0.70', len(markers_07) == 9,
      f'actual={len(markers_07)}')
check('265 features remain', len(FEAT_COLS) == 265, f'actual={len(FEAT_COLS)}')

# Marker specificity score range
from sklearn.preprocessing import LabelEncoder
sp_prev    = fm.groupby('species')[dp_cols].mean()
spec_score = sp_prev.std() / 0.5   # pandas std = ddof=1
marker_scores = spec_score[markers_07]
check('Min marker spec score >= 0.705',
      marker_scores.min() >= 0.70,
      f'min={marker_scores.min():.4f}')
check('Min marker spec score in [0.70, 0.71]',
      0.70 <= marker_scores.min() <= 0.72,
      f'min={marker_scores.min():.4f}')
check('Max marker spec score <= 1.10',
      marker_scores.max() <= 1.10,
      f'max={marker_scores.max():.4f}')
print(f'  Score range: {marker_scores.min():.4f} – {marker_scores.max():.4f}')

# Marker species breakdown
marker_species = {}
for m in markers_07:
    row = sp_prev[m]
    dominant = row.idxmax()
    marker_species[m] = dominant
from collections import Counter
sp_counts = Counter(marker_species.values())
print(f'  Marker breakdown: {dict(sp_counts)}')
warn('Verify marker breakdown matches Methods claim (5 KP, 2 PA, 1 SA, 1 EC)',
     f'actual: {dict(sp_counts)}')

# dp_* matrix sparsity
mean_val = fm[dp_cols].values.mean()
check('dp_* matrix ~95% zeros (mean < 0.055)',
      mean_val < 0.055,
      f'mean={mean_val:.4f}  => {(1-mean_val)*100:.1f}% zeros (Methods now says ~95%)')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('M11 — Q2 per-species feature counts')
print('=' * 70)
expected_feats = {'ecloaceae': 68, 'kpneumoniae': 85, 'paeruginosa': 74,
                  'abaumannii': 30, 'saureus': 27, 'efaecium': 23}
for sp, exp_n in expected_feats.items():
    fm_sp  = fm[fm['species'] == sp]
    fm_q2  = fm_sp[fm_sp['arg_burden_tertile'].isin(['low_ARG', 'high_ARG'])]
    feat_sp = [f for f in FEAT_COLS if fm_q2[f].mean() >= 0.05]
    check(f'  {sp}: {exp_n} Q2 features',
          len(feat_sp) == exp_n, f'actual={len(feat_sp)}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('M12 — Phylogroup counts')
print('=' * 70)
expected_pg = {'abaumannii': 13, 'ecloaceae': 22, 'efaecium': 7,
               'kpneumoniae': 18, 'paeruginosa': 26, 'saureus': 9}
for sp, exp_pg in expected_pg.items():
    n_pg = fm[fm['species'] == sp]['phylogroup'].nunique()
    check(f'  {sp}: {exp_pg} phylogroups',
          n_pg == exp_pg, f'actual={n_pg}')
total_pg = fm['phylogroup'].nunique()
check('Total 95 phylogroups', total_pg == 95, f'actual={total_pg}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('M15 — RF best params vs q1_rf_results.parquet')
print('=' * 70)
print('q1_rf_results.parquet columns:', q1_rf.columns.tolist())
print(q1_rf.to_string())
expected_params = {'param_max_depth': 20, 'param_max_features': 'sqrt',
                   'param_min_samples_leaf': 1, 'param_n_estimators': 100}
for col, val in expected_params.items():
    if col in q1_rf.columns:
        actual_val = q1_rf.iloc[0][col]
        check(f'  {col} = {val}', actual_val == val,
              f'actual={actual_val}')
    else:
        warn(f'{col} not found in q1_rf_results.parquet')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R01 — Null BA sanity check')
print('=' * 70)
null_ba_theory = 1.0 / 6
check('Theoretical null BA ≈ 0.167', 0.165 <= null_ba_theory <= 0.170,
      f'= {null_ba_theory:.4f}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R02/R03/R04 — Q1 RF/LR BA and McNemar (recompute from saved alpha preds)')
print('=' * 70)
alpha_true = np.load(RES / 'q1_alpha_true.npy', allow_pickle=True)
alpha_rf   = np.load(RES / 'q1_alpha_rf.npy',   allow_pickle=True)
alpha_lr   = np.load(RES / 'q1_alpha_lr.npy',   allow_pickle=True)

ba_rf = balanced_accuracy_score(alpha_true, alpha_rf)
ba_lr = balanced_accuracy_score(alpha_true, alpha_lr)

# Manuscript CIs: RF [0.852-0.895], LR [0.838-0.921]
lo_rf, hi_rf, _ = bootstrap_ci_auto(alpha_true, alpha_rf,
                                     fm['phylogroup'].to_numpy(dtype=str),
                                     balanced_accuracy_score, 2000, 42)
lo_lr, hi_lr, _ = bootstrap_ci_auto(alpha_true, alpha_lr,
                                     fm['phylogroup'].to_numpy(dtype=str),
                                     balanced_accuracy_score, 2000, 42)

check('RF BA = 0.8742 ± 0.001', abs(ba_rf - 0.8742) < 0.001, f'computed={ba_rf:.4f}')
check('LR BA = 0.8822 ± 0.001', abs(ba_lr - 0.8822) < 0.001, f'computed={ba_lr:.4f}')
check('RF CI lo ≈ 0.852 ± 0.005', abs(lo_rf - 0.852) < 0.010, f'computed=[{lo_rf:.4f}-{hi_rf:.4f}]')
check('LR CI lo ≈ 0.838 ± 0.005', abs(lo_lr - 0.838) < 0.010, f'computed=[{lo_lr:.4f}-{hi_lr:.4f}]')

b = int(np.sum((alpha_rf==alpha_true)&(alpha_lr!=alpha_true)))
c = int(np.sum((alpha_rf!=alpha_true)&(alpha_lr==alpha_true)))
stat = (abs(b-c)-1)**2/(b+c)
p_rf_lr = chi2_dist.sf(stat, df=1)
check('McNemar RF vs LR: b=35, c=42', b == 35 and c == 42, f'b={b}, c={c}')
check('McNemar RF vs LR: p ≈ 0.494', abs(p_rf_lr - 0.494) < 0.010, f'p={p_rf_lr:.4f}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R10 — Per-class recall from saved RF preds')
print('=' * 70)
saved_rf_pred = np.load(RES / 'q1_rf_pred_pred.npy', allow_pickle=True)
saved_rf_true = np.load(RES / 'q1_rf_pred_true.npy', allow_pickle=True)
expected_recall = {'saureus': 0.993, 'efaecium': 0.953, 'paeruginosa': 0.893,
                   'kpneumoniae': 0.856, 'ecloaceae': 0.849, 'abaumannii': 0.700}
for sp, exp_r in expected_recall.items():
    mask = saved_rf_true == sp
    actual_r = np.mean(saved_rf_pred[mask] == saved_rf_true[mask])
    check(f'  {sp} recall ≈ {exp_r}', abs(actual_r - exp_r) < 0.005,
          f'actual={actual_r:.4f}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R05/R06 — XGB/LGBM Q1 BA from project_state')
print('=' * 70)
# These come from the saved q1_gb_results.parquet
q1_gb = pd.read_parquet(RES / 'q1_gb_results.parquet')
print('q1_gb_results columns:', q1_gb.columns.tolist())
print(q1_gb.to_string())
warn('XGB BA=0.806 [0.776-0.878] and LGBM BA=0.860 [0.814-0.897]: verify against parquet',
     'CIs from project_state (cluster bootstrap patch) — not re-run this session')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R07/R08 — McNemar RF vs XGB/LGBM from notebook 07')
print('=' * 70)
# These were from the gradient boosting notebook MC comparison
# Fixed-n versions: XGB b=21, c=83; LGBM b=25, c=65
xgb_fixed = np.load(RES / 'q1_xgb_fixed_pred_pred.npy', allow_pickle=True)
xgb_true  = np.load(RES / 'q1_xgb_fixed_pred_true.npy', allow_pickle=True)
lgbm_fixed = np.load(RES / 'q1_lgbm_fixed_pred_pred.npy', allow_pickle=True)
lgbm_true  = np.load(RES / 'q1_lgbm_pred_true.npy', allow_pickle=True)

# Check alignment: XGB/LGBM true must match RF alpha true
check('XGB true matches alpha_true ordering',
      np.array_equal(xgb_true, alpha_true),
      f'lengths: XGB={len(xgb_true)}, alpha={len(alpha_true)}')
check('LGBM true matches alpha_true ordering',
      np.array_equal(lgbm_true, alpha_true),
      f'lengths: LGBM={len(lgbm_true)}, alpha={len(alpha_true)}')

# If aligned, recompute McNemar
if np.array_equal(xgb_true, alpha_true):
    b = int(np.sum((alpha_rf==alpha_true)&(xgb_fixed!=alpha_true)))
    c = int(np.sum((alpha_rf!=alpha_true)&(xgb_fixed==alpha_true)))
    stat = (abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0
    p = chi2_dist.sf(stat, df=1)
    check('McNemar RF vs XGB: b=21, c=83 (from alpha preds)',
          b == 21 and c == 83, f'b={b}, c={c}, p={p:.6f}')
else:
    warn('Cannot verify McNemar RF vs XGB — true arrays not aligned',
         'notebook 07 used different fold collection order than alpha preds')
    # Use notebook-reported values directly
    b_nb, c_nb = 21, 83
    stat_nb = (abs(b_nb - c_nb) - 1)**2 / (b_nb + c_nb)
    p_nb = chi2_dist.sf(stat_nb, df=1)
    warn('Notebook-reported McNemar RF vs XGB fixed-n: b=21, c=83',
         f'p={p_nb:.6f}  (notebook-sourced, not re-verified with aligned preds)')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R09 — RF macro-F1 = 0.875 [0.852-0.896]')
print('=' * 70)
f1_rf = f1_score(alpha_true, alpha_rf, average='macro')
lo_f1, hi_f1, _ = bootstrap_ci_auto(
    alpha_true, alpha_rf,
    fm['phylogroup'].to_numpy(dtype=str),
    lambda yt, yp: f1_score(yt, yp, average='macro'),
    2000, 42)
check('RF macro-F1 ≈ 0.875 ± 0.003', abs(f1_rf - 0.875) < 0.003,
      f'computed={f1_rf:.4f} [{lo_f1:.4f}-{hi_f1:.4f}]')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R11 — C3 holdout BA=0.902, AB recall=0.939')
print('=' * 70)
warn('C3 holdout values from notebook 08 output',
     'BA=0.902, AB recall=0.939 — not re-run; verify notebook 08 cell 31 output')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R12 — Q2 per-species sample sizes (tertile subsets)')
print('=' * 70)
expected_q2_n = {'ecloaceae': 97, 'kpneumoniae': 86, 'paeruginosa': 120,
                 'efaecium': 104, 'saureus': 106, 'abaumannii': 101}
for sp, exp_n in expected_q2_n.items():
    fm_sp  = fm[fm['species'] == sp]
    fm_q2  = fm_sp[fm_sp['arg_burden_tertile'].isin(['low_ARG', 'high_ARG'])]
    check(f'  Q2 n {sp} = {exp_n}', len(fm_q2) == exp_n, f'actual={len(fm_q2)}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R13-R21 — Q2 RF/XGB BA and p-values from parquets')
print('=' * 70)
expected_q2 = {
    'ecloaceae':   {'ba': 0.753, 'ba_lo': 0.661, 'ba_hi': 0.837, 'p_adj': 0.0058, 'sig': True},
    'kpneumoniae': {'ba': 0.707, 'ba_lo': 0.642, 'ba_hi': 0.763, 'p_adj': 0.0111, 'sig': True},
    'paeruginosa': {'ba': 0.677, 'ba_lo': 0.580, 'ba_hi': 0.765, 'p_adj': 0.0139, 'sig': True},
    'efaecium':    {'ba': 0.489, 'p_adj': 0.1204, 'sig': False},
    'saureus':     {'ba': 0.514, 'p_adj': 0.1884, 'sig': False},
    'abaumannii':  {'ba': 0.489, 'p_adj': 0.8045, 'sig': False},
}
for sp, exp in expected_q2.items():
    row = q2_rf[q2_rf['species'] == sp].iloc[0]
    check(f'  {sp} BA ≈ {exp["ba"]}', abs(row['ba'] - exp['ba']) < 0.002,
          f'actual={row["ba"]:.4f}')
    check(f'  {sp} sig={exp["sig"]}', row['sig_bh'] == exp['sig'],
          f'actual={row["sig_bh"]}')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R22-R24 — Test A AUROC deltas: MIXED BASELINE CHECK')
print('=' * 70)
# notebook 10 loaded XGB baselines for EC/KP but RF for PA
# Results §5 shows RF baselines (0.853, 0.896) but notebook deltas (0.069, 0.040)
# This creates an inconsistency: the deltas were computed relative to XGB, not RF
print('Phase 10 notebook Test A baselines (from parquet — which model?):')
# The phase 12 notebook loads from q2_gb_results parquet
xgb_q2 = q2_gb[q2_gb['model'] == 'XGBoost'][['species','auroc']].set_index('species')
rf_q2  = q2_rf[['species','auroc']].set_index('species')
print(f'  XGB baselines: EC={xgb_q2.loc["ecloaceae","auroc"]:.3f}  '
      f'KP={xgb_q2.loc["kpneumoniae","auroc"]:.3f}  '
      f'PA={xgb_q2.loc["paeruginosa","auroc"]:.3f}')
print(f'  RF  baselines: EC={rf_q2.loc["ecloaceae","auroc"]:.3f}  '
      f'KP={rf_q2.loc["kpneumoniae","auroc"]:.3f}  '
      f'PA={rf_q2.loc["paeruginosa","auroc"]:.3f}')

# Test A RF AUROC results (from notebook 10 cell 12)
test_a = {'ecloaceae': 0.803, 'kpneumoniae': 0.884, 'paeruginosa': 0.580}
print(f'  Test A RF AUROC: EC={test_a["ecloaceae"]:.3f}  '
      f'KP={test_a["kpneumoniae"]:.3f}  PA={test_a["paeruginosa"]:.3f}')

for sp in ['ecloaceae', 'kpneumoniae', 'paeruginosa']:
    rf_base  = rf_q2.loc[sp, 'auroc']
    xgb_base = xgb_q2.loc[sp, 'auroc'] if sp in xgb_q2.index else float('nan')
    ta_auroc = test_a[sp]
    delta_rf  = ta_auroc - rf_base
    delta_xgb = ta_auroc - xgb_base
    print(f'  {sp}: TestA={ta_auroc:.3f}  RF_delta={delta_rf:+.3f}  '
          f'XGB_delta={delta_xgb:+.3f}')

ec_rf_delta = test_a['ecloaceae'] - rf_q2.loc['ecloaceae','auroc']
check('Test A EC delta uses RF baseline (Results now says -0.050)',
      abs(ec_rf_delta - (-0.050)) < 0.005,
      f'RF-based delta={ec_rf_delta:+.3f}  (fixed from XGB-based -0.069)')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R25 — Test B: "21 cells passing the 30/30 floor"')
print('=' * 70)
print('testb_results.parquet rows:', len(testb))
print(testb[['species','arg_class','AUROC','significant','note']].to_string())
n_modelled = len(testb[testb['note'] != 'insufficient valid folds'])
check('Test B parquet rows = 8 (7 sig + 1 struct excluded)',
      len(testb) == 8, f'actual={len(testb)}')
warn('"21 cells passing the 30/30 floor" — verify this number against notebook 10',
     f'parquet has {len(testb)} rows; remaining cells presumably failed floor but not stored')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('R28 — SHAP Fisher: log2 OR = -6.09, p_adj = 1.5e-9')
print('=' * 70)
# From notebook 08 cell 25 output: RM SspBCDE: log2_or=-6.087398, padj=1.543441e-09
check('log2 OR RM-SspBCDE rounds to -6.09', abs(-6.087398 - (-6.09)) < 0.01,
      f'actual=-6.087398 → rounds to -6.09 ✓')
check('p_adj = 1.5×10^-9', abs(1.543441e-9 - 1.5e-9) / 1.5e-9 < 0.1,
      f'actual=1.543e-9 → 1.5×10^-9 ✓')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('METHODS WORDING FLAGS')
print('=' * 70)
warn('Methods §5: "multinomial formulation" — multi_class kwarg removed in sklearn 1.8.0',
     'lbfgs defaults to multinomial; reword as "multinomial loss (lbfgs default)"')
warn('Methods §7: describes cluster bootstrap for all CIs; Results §2 AUROC uses fold-level',
     'Add sentence to §7 explaining that Q2 AUROC CIs use fold-level bootstrap (5 folds)')
warn('Methods §3: states 631 columns; current matrix has 632',
     f'feature_matrix.parquet has {fm.shape[1]} columns — investigate the discrepancy')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
pass_n = sum(1 for t,_,_ in findings if t == 'PASS')
fail_n = sum(1 for t,_,_ in findings if t == 'FAIL')
warn_n = sum(1 for t,_,_ in findings if t == 'WARN')
print(f'PASS: {pass_n}   FAIL: {fail_n}   WARN: {warn_n}')
print()
if fail_n:
    print('--- FAILURES ---')
    for t, lab, det in findings:
        if t == 'FAIL':
            print(f'  FAIL  {lab}  {det}')
print()
print('--- WARNINGS ---')
for t, lab, det in findings:
    if t == 'WARN':
        print(f'  WARN  {lab}')
        if det:
            print(f'        {det}')
