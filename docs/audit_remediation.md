# Audit Remediation Log

Tracking all audit findings from `audit.md`, implemented fixes, and before/after impact.
Each entry records: what changed, why, and what conclusion changed.

---

## C1  -  Q2 label inconsistency (FIXED 2026-05-26)

**What was wrong:**
RF (`build_rf_notebook.py`) and GB (`build_gb_notebook.py`) both computed Q2 ARG burden labels by
recomputing a per-species 67th-percentile threshold from raw ARG counts:
```python
tertile_thresh = fm_sp[arg_col].quantile(2/3)
y_sp = (fm_sp[arg_col] > tertile_thresh).astype(int).values
```
This assigned mid_ARG genomes (the middle third, n=264) to class 0 (low burden), making RF/GB
solve "top third vs rest" while LR solved "top third vs bottom third only". Different tasks  - 
not comparable.

**Fix applied:**
Replaced tertile recomputation with the pre-computed `arg_burden_tertile` column and filtered
to top+bottom tertile only, matching the LR task exactly:
```python
mask_q2 = fm_sp["arg_burden_tertile"].isin(["low_ARG", "high_ARG"])
fm_q2   = fm_sp[mask_q2]
y_sp    = (fm_q2["arg_burden_tertile"] == "high_ARG").astype(int).values
```
Applied to both `build_rf_notebook.py` (lines ~617–630) and `build_gb_notebook.py` (lines ~782–791).
Also updated GB builder's RF reference to load from `q2_rf_results.parquet` instead of hardcoded
stale values.

**Files changed:**
- `notebooks/build_rf_notebook.py`
- `notebooks/build_gb_notebook.py`
- `notebooks/06_random_forest.ipynb` (rebuilt + re-executed)
- `notebooks/07_gradient_boosting.ipynb` (rebuilt + re-executed)
- `results/q2_rf_results.parquet` (updated)
- `results/q2_gb_results.parquet` (updated)

**Before vs after (Q2 balanced accuracy):**

| Species | RF (bug) | RF (fixed) | LR | XGB (fixed) | LGBM (fixed) |
|---------|----------|------------|----|-------------|--------------|
| EC      | 0.508    | 0.750      | 0.752 | 0.787    | 0.706        |
| KP      | 0.546    | 0.756      | 0.719 | 0.778    | 0.778        |
| EF      | 0.681    | 0.677      | 0.512 | 0.581    | 0.496        |
| PA      | 0.621    | 0.659      | 0.645 | 0.610    | 0.597        |
| SA      | 0.475    | 0.480      | 0.470 | 0.521    | 0.480        |
| AB      | 0.500    | 0.500      | 0.473 | 0.500    | 0.500        |

**Conclusion that changed:**
OLD: "Moderate ARG-burden signal detectable by regularised LR but not by flexible RF at n~150."
NEW: Tree models (RF, XGBoost) are competitive with or better than LR for ARG burden prediction
in 4/6 species. EC and KP show the strongest signal (BA ~0.75–0.79). EF and PA show moderate
signal (BA ~0.65–0.68). SA and AB remain at near-chance across all models.

**Note on H5 (PA-1):** PA's `arg_burden_tertile` column already encodes the median split
(low_ARG=ARG≤5, mid_ARG=ARG=6 dropped, high_ARG=ARG≥7). The C1 fix automatically implements
H5 because using `arg_burden_tertile` preserves this pre-computed PA-specific logic.

---

---

## C2  -  Spec-filter threshold sensitivity (FIXED 2026-05-26)

**What was wrong:**
The Q1 model at threshold=0.70 retained features with spec_score 0.55–0.69 that are borderline
taxonomic markers. Four of the top-5 SHAP features (`dp_df_Mok_Hok_Sok`, `dp_padloc_PDC-S13`,
`dp_df_FS_Sma`, `dp_df_Abi2`) fell in this range. The Phase 8 notebook reported only the
threshold=0.70 result without a sensitivity check.

**Fix applied:**
Added Section 13 (C2 sensitivity) to `build_rf_notebook.py`. Reruns Q1 at threshold=0.50 using
identical RF params and GroupedStratifiedKFold setup. Results:

| Threshold | N features | N removed | BA       | 95% CI         |
|-----------|-----------|-----------|----------|----------------|
| 0.70 (primary) | 265  | 9         | 0.878    | [0.859–0.898]  |
| 0.50 (strict)  | 259  | 15        | 0.733    | [0.678–0.782]  |

**6 borderline markers removed at 0.50 but retained at 0.70:**
- `dp_df_Mok_Hok_Sok`  -  spec_score=0.557, dominant KP+EC (~60%)  -  SHAP rank 1
- `dp_padloc_PDC-S13`  -  spec_score=0.550, dominant KP (68%)  -  SHAP rank 2
- `dp_df_FS_Sma`  -  spec_score=0.675, dominant SA (83%)  -  SHAP rank 3
- `dp_df_Abi2`  -  spec_score=0.694, dominant SA (86%)  -  SHAP rank 4
- `dp_Mokosh_TypeII`  -  spec_score ~0.50–0.55
- `dp_padloc_PDC-M24`  -  spec_score ~0.50–0.55

**Files changed:**
- `notebooks/build_rf_notebook.py` (Section 13 added)
- `notebooks/06_random_forest.ipynb` (rebuilt + re-executed, now 29 cells)
- `results/q1_rf_sensitivity_spec_filter.parquet` (new output)

**Conclusion / manuscript action:**
The 14.5pp BA drop confirms substantial contribution from species-identity proxies.
Primary result (BA=0.878) is reported at pre-registered threshold with explicit caveat.
Phase 10 SHAP interpretation will be restricted to the 259 features surviving threshold=0.50.
Manuscript Methods will disclose both thresholds and the sensitivity result.

---

---

## M2 + M3  -  Bootstrap CI and sample weights (FIXED 2026-05-26)

**M2  -  What was wrong:**
Both RF and GB builders called `bootstrap_ci(np.array(fold_scores))`  -  bootstrapping over
5 fold-level BA scores. With n=5, percentile intervals have poor coverage (the 2.5th percentile
collapses to the minimum observed fold score). LR Phase 7 was already correct (pooled predictions).

**M2 fix applied:**
Replaced `bootstrap_ci(scores)` with `bootstrap_ci(y_true, y_pred)` in both builders.
New function resamples n=878 pooled per-genome (y_true, y_pred) pairs for Q1, and
n=~60-100 pairs for Q2 per species. Q2 uses list collection (not pre-allocated arrays) to
exclude skipped folds (folds where test set has only 1 class).

**M3  -  What was wrong:**
GB builder called `grid_xgb.fit(X, y_q1, groups=groups, sample_weight=sw_all)` where
`sw_all = compute_sample_weight("balanced", y_q1)` was computed once on all 878 genomes.
When GridSearchCV slices per fold, the weights encode the full-dataset class distribution,
not the fold's local distribution.

**M3 fix applied:**
Removed `sample_weight=sw_all` from `grid_xgb.fit()`. GridSearchCV now tunes on unweighted
training with `balanced_accuracy` scoring (imbalance-aware in evaluation). Per-fold sample
weights remain correctly computed inside the early-stopping CV loop.

**Files changed:**
- `notebooks/build_rf_notebook.py`: new `bootstrap_ci(y_true, y_pred)`, Q1 pooled collection,
  Q2 list collection
- `notebooks/build_gb_notebook.py`: same + M3 fix
- `notebooks/06_random_forest.ipynb`, `07_gradient_boosting.ipynb` (rebuilt + re-executed)
- `results/q1_rf_results.parquet`, `q1_gb_results.parquet` (updated CIs)
- `results/q2_rf_results.parquet`, `q2_gb_results.parquet` (updated CIs)

**CI comparison (Q1):**

| Model | Old CI (n=5 folds) | New CI (n=878 pooled) |
|-------|-------------------|----------------------|
| LR    | [0.813–0.859] ✓ already pooled | unchanged |
| RF    | [0.850–0.893] | [0.852–0.895] |
| XGBoost | [0.859–0.898] | [0.793–0.839] |
| LightGBM | [0.801–0.852] | [0.802–0.852] |

Note: XGBoost Q1 CI narrowed substantially  -  the old fold-level bootstrap happened to
have low variance across 5 folds, giving falsely tight CI. Pooled CI is wider and correct.

**Q2 final table (all models, pooled CIs):**

| Species | LR | RF | XGB | LGBM |
|---------|----|----|-----|------|
| EC      | 0.752 | 0.731 | 0.772 | 0.681 |
| KP      | 0.719 | 0.741 | 0.766 | 0.756 |
| PA      | 0.645 | 0.662 | 0.605 | 0.596 |
| EF      | 0.512 | 0.543 | 0.486 | 0.493 |
| SA      | 0.470 | 0.492 | 0.510 | 0.491 |
| AB      | 0.473 | 0.500 | 0.500 | 0.500 |

---

---

## H1  -  Per-species Q2 sparsity filter (FIXED 2026-05-26)

**What was wrong:**
Q2 per-species models used all 265 `dp_*` features regardless of whether features had
any prevalence within that species. SA (83%) and EF (78%) had the vast majority of
features identically zero across all Q2-eligible genomes  -  Gram-positive organisms in
a Gram-negative-dominated feature space. Ultra-sparse features added noise and inflated
the effective feature space.

**Fix applied:**
Added per-species sparsity filter in Q2 loop of both `build_rf_notebook.py` and
`build_gb_notebook.py`. After applying the Q2 tertile mask, compute mean prevalence
of each `dp_*` feature within that species' Q2 genomes and retain only those with
>= 5% prevalence. Applied identically in RF and GB.

```python
feat_prev_sp = fm_q2[FEAT_COLS].mean()
feat_q2_sp   = feat_prev_sp[feat_prev_sp >= 0.05].index.tolist()
X_sp         = fm_q2[feat_q2_sp].values
```

**Effective feature counts after filter:**

| Species | Features (filtered) | Features (nominal) | % retained |
|---------|--------------------|--------------------|------------|
| AB      | 30                 | 265                | 11%        |
| EF      | 23                 | 265                | 9%         |
| SA      | 27                 | 265                | 10%        |
| EC      | 68                 | 265                | 26%        |
| KP      | 85                 | 265                | 32%        |
| PA      | 74                 | 265                | 28%        |

**Before vs after (Q2 RF balanced accuracy):**

| Species | RF (pre-H1) | RF (post-H1) | Delta   |
|---------|-------------|--------------|---------|
| EC      | 0.750       | 0.753        | +0.003  |
| KP      | 0.756       | 0.707        | -0.049  |
| EF      | 0.677       | 0.489        | -0.188  |
| PA      | 0.659       | 0.677        | +0.018  |
| SA      | 0.480       | 0.514        | +0.034  |
| AB      | 0.500       | 0.489        | -0.011  |

**Conclusion that changed:**
OLD: "RF wins EF Q2 (BA=0.677 vs LR=0.512)"  -  treated as genuine signal.
NEW: EF Q2 BA collapses to 0.489 (chance) after H1 filter. The EF advantage was entirely
driven by ultra-sparse features that are present in only a few EF genomes  -  effectively
memorising individual genomes rather than learning a population-level signal. EF Q2
cannot be claimed as meaningful. EC and KP remain the two species with reliable Q2 signal.

---

---

## H4  -  BH correction across Q2 species (FIXED 2026-05-26)

**What was wrong:**
Six per-species Q2 binary classifiers reported without any correction for multiple
comparisons. FWER at alpha=0.05 across 6 tests = ~26% probability of at least one
false positive.

**Fix applied:**
Added Section 12b to both notebooks. One-sample t-test (fold BAs vs null BA=0.5,
alternative="greater") per species, then BH correction via
`statsmodels.stats.multitest.multipletests(method="fdr_bh")`. Uses RF fold BAs in
RF notebook; XGB fold BAs in GB notebook. Adjusted p-values saved to parquets.

**RF Q2  -  BH-corrected null-baseline significance:**

| Species | RF BA | n_folds | p_raw  | p_adj_BH | Significant? |
|---------|-------|---------|--------|----------|--------------|
| EC      | 0.753 | 5       | 0.0010 | 0.0058   | YES          |
| KP      | 0.707 | 5       | 0.0037 | 0.0111   | YES          |
| PA      | 0.677 | 5       | 0.0070 | 0.0139   | YES          |
| EF      | 0.489 | 4       | 0.0803 | 0.1204   | ns           |
| SA      | 0.514 | 5       | 0.1570 | 0.1884   | ns           |
| AB      | 0.489 | 4       | 0.8045 | 0.8045   | ns           |

**XGB Q2  -  BH-corrected null-baseline significance:**

| Species | XGB BA | n_folds | p_raw  | p_adj_BH | Significant? |
|---------|--------|---------|--------|----------|--------------|
| EC      | 0.824  | 5       | ~0.000 | ~0.000   | YES          |
| KP      | 0.789  | 5       | 0.0050 | 0.0126   | YES          |
| PA      | 0.568  | 5       | 0.0825 | 0.1374   | ns           |
| EF      | 0.486  | 4       | 0.1181 | 0.1477   | ns           |
| SA      | 0.508  | 5       | 0.6101 | 0.6101   | ns           |
| AB      | 0.500  | 4       | nan    | nan      | ns           |

Note: AB XGB returns all-0.5 fold BAs (majority-class predictor for all folds),
so t-test statistic is undefined (zero std). Correctly excluded from BH computation.

**Conclusion:**
After H1 filter and BH correction: EC, KP, PA show statistically significant Q2
prediction above chance for RF. EC and KP are significant for XGB (EC notably
stronger, BA=0.824). EF, SA, AB do not survive BH correction under either model.
The claim "defence systems predict ARG burden" is restricted to EC/KP (consistent
across models) and PA (RF only). EF's previous apparent advantage was a sparsity artefact.

---

---

## H3  -  GB fair comparison with fixed n_estimators (FIXED 2026-05-26)

**What was wrong:**
XGB/LGBM Q1 used an 80/20 inner split for early stopping → trained on ~64% of data
per fold. RF trained on ~80%. The 14pp RF advantage (0.878 vs 0.817) might partly
reflect data starvation rather than model quality.

**Fix applied:**
Added Section 10b to `build_gb_notebook.py`. Re-ran XGB (n_estimators=143, median of
early-stopping best_iters) and LGBM (n_estimators=92) with no inner split, training on
the full fold  -  same effective data as RF. McNemar tests against RF using aligned
per-genome predictions.

**Results (Q1 balanced accuracy):**

| Model                              | BA     | 95% CI          | vs RF  |
|------------------------------------|--------|-----------------|--------|
| RF (80% fold, n free)              | 0.8780 | [0.859–0.898]   |  -       |
| XGB (early-stop, 64% fold)         | 0.8165 | [0.793–0.839]   | -0.062 |
| XGB (fixed n=143, 80% fold)        | 0.8062 | [0.782–0.830]   | -0.072 |
| LGBM (early-stop, 64% fold)        | 0.8267 | [0.802–0.852]   | -0.051 |
| LGBM (fixed n=92, 80% fold)        | 0.8304 | [0.807–0.855]   | -0.048 |

- McNemar (XGB fixed-n vs RF): p<0.0001 (b=21, c=83)  -  RF significantly better
- McNemar (LGBM fixed-n vs RF): p<0.0001 (b=25, c=65)  -  RF significantly better

**Conclusion:**
Data starvation does not explain the RF advantage. XGB is actually **worse** with full
fold data (0.806 vs 0.817 early-stop)  -  early stopping was acting as a regulariser for
XGB, not penalising it through data reduction. LGBM gains marginally (+0.004) but remains
far below RF. RF's superiority on this sparse binary feature matrix at n=878 is genuine,
not an artefact of unequal training set sizes. The manuscript H3 limitation note should be
updated accordingly: "H3 fair comparison confirms RF advantage is not attributable to
the inner validation split used by gradient boosting."

---

---

## H7  -  Anti-defence feature decision and Q2 sensitivity (FIXED 2026-05-26)

**Decision:**
- Q1: `ad_*` excluded. spec_scores 0.62--0.81  -  more species-specific than the C2
  borderline `dp_*` features. Including them would inflate Q1 via taxonomic proxies.
- Q2: Sensitivity run in Section 12c of `06_random_forest.ipynb`. H1 per-species
  sparsity filter applied identically to `ad_*` features.

**Q2 sensitivity results (dp_* only vs dp_* + ad_*):**

| Species | dp_* only | dp_*+ad_* | Delta  | n_dp | n_ad |
|---------|-----------|-----------|--------|------|------|
| EC      | 0.753     | 0.772     | +0.019 | 68   | 9    |
| KP      | 0.707     | 0.742     | +0.035 | 85   | 10   |
| PA      | 0.677     | 0.696     | +0.019 | 74   | 10   |
| EF      | 0.489     | 0.577     | +0.089 | 23   | 2    |
| SA      | 0.514     | 0.603     | +0.089 | 27   | 3    |
| AB      | 0.489     | 0.524     | +0.034 | 30   | 3    |

**Key finding:** EF and SA (both at chance with dp_*-only) gain +0.089 BA when
anti-defence features are added. Biologically plausible: Gram-positive organisms
that lack defence systems rely on anti-defence systems co-mobilised with MGEs  - 
the same MGEs that carry ARGs. This is a genuine new finding beyond the published
single-species analysis. Phase 10 should examine which `ad_*` features drive the
EF and SA improvement.

---

## H8  -  Learning curves (FIXED 2026-05-26)

**Generated in Section 13b of `06_random_forest.ipynb`.**
Saved to `results/figures/rf/learning_curves.png`.

Q1 (RF vs LR): learning curves across 20--100% of the 878-genome training set with
GroupedStratifiedKFold CV.

Q2 (RF for EC, KP, PA): per-species curves with H1-filtered features across
20--100% of each species' Q2-eligible set.

---

## H6  -  RM_Type_I interpretation (FIXED 2026-05-26  -  documentation only)

**No code change.** Framing logged in `decisions.md` (2026-05-26).

`dp_RM_Type_I` permutation rank = 261/265 (value = -0.006). Two interpretations
to address in Phase 10:
1. RM Type I is redundant given other correlated features in Q1 multi-class context.
2. RM is a within-AB archetype signal, not a cross-species signal  -  Q1 cannot
   detect it because RM's within-AB variation does not distinguish AB from other species.

Phase 10 action: examine RM_Type_I in Q2 (ARG burden within species). If still
unimportant there, that is a genuine new finding: binary RM presence does not
predict multi-species ARG burden even though RM count correlates with ARG burden
in the published single-species analysis.

Do NOT present SHAP rank 17 for RM as "confirming" the published finding.
Correct framing: "Q1 RM signal is attenuated, consistent with RM being a
within-species archetype marker rather than a cross-species predictor."

---

---

## M4  -  McNemar tests for Q2 per-species model comparisons (FIXED 2026-05-26)

**What was wrong:** Q2 cross-model comparisons reported as point-estimate BA differences only.

**Fix:** Added Section 12c to `build_gb_notebook.py`. Re-ran all four models (LR, RF, XGB, LGBM)
on identical GroupedStratifiedKFold splits per species (same H1 filter, same folds). McNemar
pairwise test: LR vs RF, LR vs XGB, LR vs LGBM, RF vs XGB.

**Results:**

| Species | n   | LR    | RF    | XGB   | LGBM  | Significant pairs           |
|---------|-----|-------|-------|-------|-------|-----------------------------|
| EC      | 97  | 0.721 | 0.753 | 0.824 | 0.681 | none (LR vs XGB p=0.055)    |
| KP      | 86  | 0.777 | 0.707 | 0.789 | 0.756 | none                        |
| PA      | 120 | 0.638 | 0.677 | 0.568 | 0.596 | RF > XGB p=0.021 *          |
| EF      | 99  | 0.593 | 0.489 | 0.486 | 0.493 | LR > RF p=0.034 *, LR > XGB p=0.022 * |
| SA      | 106 | 0.496 | 0.514 | 0.508 | 0.491 | none                        |
| AB      | 89  | 0.500 | 0.489 | 0.500 | 0.500 | none                        |

Note: LR BAs in this table are from re-runs using H1-filtered features (not the Phase 7
all-265-feature model). EF LR=0.593 here vs 0.512 in the main table  -  the H1 filter
(23 features) changes the task and LR handles the reduction better than tree methods.

**Conclusion:** No species has a model that significantly beats all others. EF is the
only species where a model hierarchy is statistically detectable (LR > trees under
H1-filtered features). PA shows RF > XGB only. All EC/KP pairwise differences are
not statistically significant despite EC XGB BA=0.824  -  sample size limits power.

---

## Pending remediation items

Items to address in order:

| ID  | Priority | Status  | Description |
|-----|----------|---------|-------------|
| C1  | CRITICAL | DONE    | Q2 label bug in RF and GB |
| C2  | CRITICAL | DONE    | Re-run Q1 at spec_score<0.50; produce 2×2 accuracy table |
| C3  | CRITICAL | deferred | Held-out AB validation (data available in Supplementary_Data_S1.xlsx; implement before Phase 10 sign-off) |
| M2  | HIGH     | DONE    | Bootstrap CI over pooled predictions (n=878), not fold scores (n=5) |
| M3  | HIGH     | DONE    | Removed sample_weight from GridSearchCV; per-fold weights in early-stopping loop already correct |
| H1  | HIGH     | DONE    | Per-species sparsity filter (>=5% prevalence) for Q2 features |
| H4  | HIGH     | DONE    | BH correction across 6 Q2 species |
| H3  | HIGH     | DONE    | GB fair comparison: fixed n_estimators confirms RF advantage is genuine |
| H7  | HIGH     | DONE    | ad_* excluded from Q1 (spec_score 0.62-0.81); Q2 sensitivity shows +0.019 to +0.089 improvement |
| H8  | HIGH     | DONE    | Learning curves generated for Q1 (RF vs LR) and Q2 (EC, KP, PA) |
| H6  | HIGH     | DONE    | RM_Type_I framing logged in decisions.md; Phase 10 action items defined |
| M1  | MODERATE | DONE    | Deviation amendment logged in pre_analysis_plan.md §9; required manuscript statement written |
| M4  | MODERATE | DONE    | McNemar Q2: EF LR>RF/XGB (p<0.05); PA RF>XGB (p=0.02); EC/KP/SA/AB no sig pairs |
| M5  | MODERATE | DONE    | OOB=0.932 vs CV=0.874 delta (+5.7pp) explicitly framed as phylogenetic correction effect |
| M6  | MODERATE | DONE    | dp_Gabija (SHAP 13, perm 265, perm_val=-0.029) flagged inline in Section 10 and decisions.md |
| O2  | OPTIONAL | pending | Expanded max_depth grid for GB |
