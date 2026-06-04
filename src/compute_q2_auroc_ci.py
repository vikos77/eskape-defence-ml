"""Compute Q2 per-fold mean AUROC with fold-level bootstrap CIs, matching notebook method."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/src')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = '/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml'
fm = pd.read_parquet(f'{ROOT}/data/processed/feature_matrix.parquet')
spec_filt = pd.read_parquet(f'{ROOT}/results/q1_rf_sensitivity_spec_filter.parquet')
dp_cols = sorted([c for c in fm.columns if c.startswith('dp_')])
markers = spec_filt.loc[0.7, 'markers_removed']
FEAT_COLS = [c for c in dp_cols if c not in markers]
best_params = dict(max_depth=20, max_features='sqrt', min_samples_leaf=1, n_estimators=100)

print('Species               AUROC    95% CI (fold-bootstrap)    BA       95% CI              n_folds')
print('-' * 100)

stored_auroc = {
    'ecloaceae': 0.853, 'kpneumoniae': 0.896, 'paeruginosa': 0.722,
    'efaecium': 0.892,  'saureus': 0.566,     'abaumannii': 0.487
}
results = {}

for sp in sorted(fm['species'].unique()):
    fm_sp = fm[fm['species'] == sp]
    fm_q2 = fm_sp[fm_sp['arg_burden_tertile'].isin(['low_ARG', 'high_ARG'])]
    if len(fm_q2) < 20 or fm_q2['arg_burden_tertile'].nunique() < 2:
        continue
    y_sp = (fm_q2['arg_burden_tertile'] == 'high_ARG').astype(int).to_numpy()
    feat_sp = [f for f in FEAT_COLS if fm_q2[f].mean() >= 0.05]
    X_sp = fm_q2[feat_sp].to_numpy(dtype=float)
    grp_sp = fm_q2['phylogroup'].to_numpy(dtype=str)
    cv_sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    auc_folds, ba_folds = [], []
    for tr, te in cv_sp.split(X_sp, y_sp, grp_sp):
        if len(set(y_sp[te])) < 2:
            continue
        rf = RandomForestClassifier(**best_params, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
        rf.fit(X_sp[tr], y_sp[tr])
        pred = rf.predict(X_sp[te])
        prob = rf.predict_proba(X_sp[te])[:, 1]
        auc_folds.append(roc_auc_score(y_sp[te], prob))
        ba_folds.append(balanced_accuracy_score(y_sp[te], pred))

    if not auc_folds:
        continue

    mean_auroc = np.mean(auc_folds)
    mean_ba    = np.mean(ba_folds)

    rng = np.random.default_rng(42)
    n_boot = 2000
    auc_boot = [np.mean(rng.choice(auc_folds, len(auc_folds))) for _ in range(n_boot)]
    ba_boot  = [np.mean(rng.choice(ba_folds,  len(ba_folds)))  for _ in range(n_boot)]
    auc_lo, auc_hi = np.percentile(auc_boot, [2.5, 97.5])
    ba_lo,  ba_hi  = np.percentile(ba_boot,  [2.5, 97.5])

    results[sp] = dict(auroc=mean_auroc, auc_lo=auc_lo, auc_hi=auc_hi,
                       ba=mean_ba, ba_lo=ba_lo, ba_hi=ba_hi,
                       n_folds=len(auc_folds))
    stored = stored_auroc.get(sp, float('nan'))
    diff = mean_auroc - stored
    print(f'{sp:<20}  {mean_auroc:.3f}  [{auc_lo:.3f}-{auc_hi:.3f}]  '
          f'{mean_ba:.3f}  [{ba_lo:.3f}-{ba_hi:.3f}]  {len(auc_folds)}  (stored={stored:.3f} diff={diff:+.3f})')

print('\nDone.')
