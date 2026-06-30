# Expected reviewer comment: feature importance for gradient boosting models

## Anticipated comment

> "You report SHAP values for Random Forest but not for XGBoost or LightGBM.
> Did you examine which features the gradient boosting models rely on, and do
> they agree with the RF interpretation?"

## Pre-emptive response

XGBoost gain-based importance was computed on a full-dataset fit (n=3,335,
best params from Section 6, n_estimators=387 from Section 7 early-stopping median).

**Top 20 features by XGBoost gain importance (Q1):**

| Rank | Feature              | Gain  |
|------|----------------------|-------|
| 1    | DS-32                | 68.3  |
| 2    | AbiO-Nhi_family      | 46.6  |
| 3    | PDC-S16              | 33.8  |
| 4    | CapRel               | 27.6  |
| 5    | PDC-S01              | 25.1  |
| 6    | FS_Sma               | 23.7  |
| 7    | PDC-S13              | 23.4  |
| 8    | PDC-M30              | 22.7  |
| 9    | PD-Lambda-3          | 21.4  |
| 10   | PDC-S37              | 19.8  |
| 11   | SspBCDE              | 19.3  |
| 12   | Sirona               | 18.5  |
| 13   | JukAB                | 18.5  |
| 14   | Abi2                 | 17.8  |
| 15   | PD-T4-3              | 17.1  |
| 16   | VP1840               | 16.6  |
| 17   | AbiC                 | 14.6  |
| 18   | AbiP2                | 13.6  |
| 19   | PDC-S05              | 12.7  |
| 20   | RM_type_HNH          | 10.9  |

Features with non-zero gain: 232 / 359. Features never split on: 127.

**Why these rankings are not directly comparable to RF SHAP:**

Gain importance and SHAP are fundamentally different quantities:
- **Gain** sums the reduction in training loss each time a feature is used to split
  a node, aggregated across all 387 trees. It is a training-set metric, is known to
  favour features that appear in many trees (frequency bias), and gives no information
  about the direction or magnitude of the feature's effect on individual predictions.
- **SHAP** (TreeSHAP) computes each feature's marginal contribution to each
  individual prediction, averaged across all possible feature orderings. It is
  theoretically grounded (Shapley values), direction-aware, and locally consistent.

The manuscript's biological interpretation is anchored to RF SHAP, which is the
more principled method. XGBoost gain rankings are not reported as a primary result
because cross-model comparison of non-equivalent importance metrics would be
methodologically misleading.

**What the rankings do show:**
SspBCDE (rank 11) and RM_type_HNH (rank 20) both appear in the XGBoost top 20,
consistent with the RF SHAP findings. The higher ranks for PDC-family systems
reflect XGBoost's tendency to exploit features that create clean splits across
species boundaries — PDC systems are highly species-specific in ESKAPE, making
them effective splitting features even if they are not biologically central to
the RESTRICT/FACILITATE question.

## Analysis location

Code recoverable from git history: commit prior to
`NB07 S9c: remove gain importance section; move to reviewer comments`.
Plot saved at: `results/figures/gb/q1_xgb_gain_importance.png`
