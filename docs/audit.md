# ESKAPE Defence ML — Comprehensive Audit Report

- **Reviewer:** ML audit, 2026-05-26
- **Scope:** All phases (data through Phase 9 GB), `decisions.md`, `pre_analysis_plan.md`, all notebook builder scripts
- **Format:** Critical → High Priority → Moderate → Optional. Each finding includes what is wrong, why it matters biologically/statistically, and required remediation.

---

## CRITICAL — Findings that invalidate stated results or pre-registered claims

### C1 — Q2 label construction is wrong in RF and GB: all cross-model comparisons are invalid

**What the code does:**

In both `build_rf_notebook.py` and `build_gb_notebook.py`, Q2 labels are constructed as:

```python
tertile_thresh = fm_sp[arg_col].quantile(2/3)
y_sp = (fm_sp[arg_col] > tertile_thresh).astype(int).values
```

This encodes every genome as “high ARG” (1) or “not high ARG” (0), where “not high ARG” includes both the low-tertile and the mid-tertile genomes. The mid-tertile is included in the negative class.

**What the pre-registered protocol requires:**

The pre-analysis plan (and the LR notebook in Phase 7) explicitly use the `arg_burden_tertile` column with mid-tertile genomes (`mid_ARG`) excluded from the analysis entirely. Q2 is defined as top vs bottom tertile. The LR notebook:

- Filters to `y_sp.isin(["high_ARG", "low_ARG"])`
- Drops `mid_ARG = 264` genomes from the training set

**Why this matters:**

The two setups are different classification tasks:

- **LR (correct):** top 33% vs bottom 33% — this is the sharp contrast that tests whether high-burden genomes have a distinct defence signature
- **RF/GB (incorrect):** top 33% vs remaining 67% — the negative class is a mixture of clean-low-ARG and messy-mid-ARG genomes with overlapping defence profiles

The immediate consequence: the “EF RF wins” finding (`RF BA=0.681 vs LR BA=0.512`, `delta=+0.169`) is an artefact. When LR excludes the ambiguous middle, `EF LR = 0.512`. When RF includes that middle in the negative class, it changes the class boundary entirely. You are not measuring the same thing. The delta is meaningless. The claim that RF is superior for EF is not supported by the data as analysed.

The same applies to all 6 species in the Q2 RF comparison and the Q2 GB comparison. None of the cross-model Q2 comparisons are valid.

**Remediation:** Rerun RF and GB Q2 for all 6 species using the correct label: exclude `mid_ARG` genomes, use `arg_burden_tertile.isin(["high_ARG","low_ARG"])` as the filter. Re-derive all Q2 performance tables. The Q2 “model selection” conclusions will change.

---

### C2 — The top SHAP features (Phase 8/9, Q1) are near-threshold taxonomic markers — Phase 10 SHAP interpretation is built on a contaminated feature set

**What the data shows:**

| SHAP Rank | Feature | Species specificity (`spec_score`) | Dominant species | Prevalence in dominant species |
|---|---|---:|---|---:|
| 1 | `dp_df_Mok_Hok_Sok` | 0.557 | KP + EC | ~60% |
| 2 | `dp_padloc_PDC-S13` | 0.550 | KP | 68% |
| 3 | `dp_df_FS_Sma` | 0.675 | SA | 83% |
| 4 | `dp_df_Abi2` | 0.694 | SA | 86% |

Your specificity filter retains features with `spec_score < 0.70`. Features 3 and 4 are at `0.675` and `0.694` — within `0.03` of the cutoff. The cutoff itself was chosen as `std(per-species prevalence) / 0.5 ≥ 0.70`. This is an arbitrary threshold, and the data shows it is porous: four of your top SHAP features are borderline taxonomic markers.

**Quantified impact of filter stringency:**

At `spec_score < 0.50` (a more conservative filter):

- 15 features removed instead of 9
- Features removed include ranks 1, 2, 3, 4, 7, 9 in the current SHAP ranking
- Q1 BA collapses from `0.878 [0.859–0.898]` to `0.733 [0.678–0.782]`
- That is a 14.5 percentage-point drop from tightening the taxonomic filter by 0.20 units

A model whose performance drops 14.5pp from a minor threshold adjustment on the filter used to prevent taxonomic classification is not robustly classifying by defence repertoire — it is partly classifying by species identity encoded in near-threshold markers.

**Why this matters for the biology:**

The entire scientific value of Q1 is: “defence system composition can predict species, and the features driving this are biologically interpretable defence systems, not species-identity markers in disguise.” If `Abi2` (86% SA) is your top-4 SHAP feature, you have not shown that defence architecture distinguishes species — you have shown that having SA-specific defence systems predicts SA. That is a tautology.

Furthermore, Phase 10 is planned as the SHAP interpretation phase. The current plan compares SHAP ranks from this Q1 model to the published AB Fisher’s exact ranks (`RM`, `SspBCDE`, `Gao_Qat`). That comparison is invalid if the SHAP ranking is dominated by SA-identity (`Abi2`, `FS_Sma`) and KP-identity (`PDC-S13`, `Mok_Hok_Sok`) features rather than pan-ESKAPE defence architecture features.

**Remediation:**

1. Before Phase 10, rerun Q1 at both `spec_score` thresholds (`0.70` and `0.50`) and report the 2×2 accuracy table as pre-registered.
2. In Phase 10, restrict SHAP interpretation to features that survive the `0.50` filter — these are the features that are genuinely not species-identity proxies.
3. Acknowledge explicitly in the manuscript that the `0.878` accuracy is partly driven by features with `spec_score 0.55–0.70` that are borderline taxonomic markers.

---

### C3 — The pre-registered held-out AB validation set has not been executed

**What was pre-registered:**

`pre_analysis_plan.md` states: “Held-out validation: published 132 AB genomes from Muthuraman et al. 2026, never used in training.” This is listed as mandatory.

**What was found:**

A search across all executed notebooks (`.ipynb`), builder scripts (`.py`), and result files found no evidence of:

- Loading the 132 published AB genomes as a held-out set
- Computing performance metrics on this set
- Any file referencing `holdout`, `published_ab`, or `validation_set` in the context of the AB held-out protocol

All 9 notebooks are fully executed (all code cells have outputs), but none contains AB held-out validation.

**Why this matters:**

The published 132 AB genomes are the only external validation you have. The ESKAPE dataset includes AB genomes drawn from the same NCBI corpus, and while you excluded the exact 132 accessions from the analysis, this held-out set provides the single most defensible claim: “Our classifier correctly assigns AB IC2 genomes — the same genomes on which the published statistical analysis found depauperate defence repertoires — to the AB class.” Without this, your Q1 model has no external validation at all: 5-fold `GroupedStratifiedKFold` CV is internal resampling, not an independent test set.

For a paper arguing that ML reproduces and extends the published *Acinetobacter* statistical findings, the absence of the held-out check is a direct contradiction of the pre-registered validation plan.

**Remediation:** Run the held-out AB validation before any Phase 10 work. This is non-negotiable given that it was pre-registered.

---

## HIGH PRIORITY — Findings that change magnitude, framing, or methodological defensibility

### H1 — Q2 feature sparsity within species: models are trained on near-empty feature spaces

**What the data shows:**

For Q2, each species is modelled independently using all 265 `dp_*` features. Per-species feature usage:

| Species | Features always zero (Q2-eligible genomes) | % of feature space that is zero |
|---|---:|---:|
| SA | 219 / 265 | 83% |
| EF | 207 / 265 | 78% |
| EC | 194 / 265 | 73% |
| KP | 168 / 265 | 63% |
| PA | 161 / 265 | 61% |
| AB | 143 / 265 | 54% |

For SA and EF, Q2 models are trained on a feature matrix where 4 in every 5 features are identically zero across all training samples. These features carry no information whatsoever for those species. They add noise to the distance calculations, inflate the feature space for `GridSearchCV`, and make `max_features="sqrt"` over 265 features equivalent to `"sqrt"` over ~45 informative features (for SA).

**Biological reason this matters:**

SA and EF are Gram-positive organisms. The 265 `dp_*` features include many Gram-negative defence systems (Type I RM prevalent in *Enterobacteriaceae*, PDC systems, etc.) that are taxonomically absent in Gram-positives. When you force SA into a feature space designed for all ESKAPE, you are asking it to express itself in a vocabulary it does not speak.

**Remediation:**

- For within-species Q2 analysis, pre-filter features to those with >5% prevalence in that species’ Q2-eligible genomes before training.
- Report per-species effective feature count (not nominal 265).
- This should be a species-specific preprocessing step, not a single global filter.

---

### H2 — EF and SA Q2 have single-phylogroup dominance: grouped CV is statistically unreliable

**What the data shows:**

EF Q2-eligible genomes (removing `mid_ARG` after fix): largest phylogroup accounts for ~71% of all EF Q2 genomes. When `GroupedStratifiedKFold` assigns this group to a single fold, the remaining 4 training folds contain mostly one phylogroup, and the test fold is either all from one phylogroup or stripped of the dominant group entirely.

This is the structural limitation of grouped CV at small n with uneven phylogroup sizes. The CI you computed by bootstrap over 5 fold-level scores inherits this instability.

**Why this matters:**

The EF Q2 BA of `0.512` (LR) is computed over CV folds where the training and test set phylogroup composition varies dramatically fold-to-fold. The CI reflects this instability but does not fix it. You cannot make a clean “LR BA = 0.512 for EF” claim — the fold-to-fold variance is dominated by which phylogroup lands in the test fold, not by the model’s true generalisation.

**Remediation:**

- Flag EF and SA Q2 results as “statistically unreliable due to phylogroup imbalance” in all tables.
- Report per-fold BA for these species (not just mean and bootstrap CI) so readers can see the instability.
- Consider a leave-one-phylogroup-out (LOGO) scheme for EF and SA, or acknowledge this as a hard limitation.

---

### H3 — XGBoost and LightGBM are trained on less data than RF due to early stopping inner split — the McNemar comparison is confounded

**What the code does:**

XGBoost/LightGBM use early stopping with an inner 80/20 `StratifiedShuffleSplit` of the training fold. In a 5-fold `GroupedStratifiedKFold`:

- RF trains on 80% of data (~702 genomes)
- XGBoost trains on 80% × 80% = 64% of data (~562 genomes)

The McNemar test shows RF significantly outperforms XGBoost (`p < 0.0001`, `BA delta=+0.058`). Before concluding “RF is a better model for this problem,” you must acknowledge that XGBoost was trained on 140 fewer genomes per fold due to the early stopping split.

**Why this matters:**

With `n=878`, the difference between 702 and 562 training samples is substantial for a sparse binary feature matrix. Gradient boosting is known to be sample-hungry compared to RF when the feature space is wide and sparse. The performance gap may reflect data availability, not model quality.

This does not invalidate the McNemar result as an operational conclusion (“use RF for this dataset at this sample size”), but it does invalidate the conclusion “RF is better than XGBoost on ESKAPE defence data in general.” That requires equivalent training data.

**Remediation:**

- Add this confound as an explicit limitation in the manuscript: “The early stopping inner split used by XGBoost and LightGBM reduced effective training set size by ~20% relative to RF, which may partly explain the observed performance gap.”
- Optionally rerun GB without early stopping, using a fixed number of estimators from a preliminary scan, to generate an apples-to-apples comparison for at least one tree count.

---

### H4 — No multiple-testing correction across 6 species for Q2

**What the pre-analysis plan requires (§5):**

“Multiple testing correction (BH) mandatory whenever more than one test is run against the same dataset.”

**What was done:**

Q2 runs 6 independent binary classifiers (one per species) and reports BA for each. No BH correction is applied. If you are making the claim “defence systems predict high-ARG burden in KP (`p < 0.05` vs null baseline),” running 6 such tests simultaneously inflates the family-wise error rate.

**Why this matters:**

At `α = 0.05` and 6 tests, the probability of at least one false positive purely by chance is 26%. Given that several Q2 species show BA only marginally above 0.5 (`SA: 0.470`, `AB: 0.473`), the risk of reporting a spurious result as significant is real.

**Remediation:**

- Apply BH correction to the Q2 null-baseline comparison p-values across 6 species.
- Report adjusted p-values alongside raw BA estimates.

---

### H5 — PA-1 protocol amendment (PA binary split) not implemented in RF or GB Q2

**What PA-1 requires:**

PA-1, logged in `decisions.md`, specifies that for *P. aeruginosa* Q2, the ARG burden label should use a binary median split rather than the tertile split, because PA has 37% of genomes at `ARG=5` (minimum value), making the bottom tertile uninformative (all minimum-ARG genomes grouped together).

**What the code does:**

Both `build_rf_notebook.py` and `build_gb_notebook.py` apply the same tertile logic to all 6 species including PA. There is no conditional branch for PA using a median split.

**Why this matters:**

PA Q2 LR BA = `0.645`. If this was computed with the correct binary split (PA-1), then the RF and GB PA comparisons — which used the tertile split — are comparing different tasks. If LR also silently used tertiles (which the LR code may or may not have implemented correctly — this needs verification), all three models are wrong but consistently wrong. Either way, the pre-registered PA-1 amendment was not applied in RF/GB.

**Remediation:**

- Verify whether LR Phase 7 PA Q2 also used tertile (if so, PA-1 was never implemented for any model — re-implement and rerun).
- Implement the conditional PA binary split in RF and GB Q2 code.
- After C1 (Q2 label fix) is applied, implement PA-1 simultaneously.

---

### H6 — Negative permutation importance for RM_Type_I directly contradicts the paper’s headline finding

**What the data shows:**

`dp_RM_Type_I`: SHAP rank 17, permutation importance rank 261 of 265, permutation importance value = `-0.006`.

A negative permutation importance means that randomly shuffling this feature improves model performance on held-out folds. This is the statistical signature of a feature that adds noise, not signal, to the classifier.

**Why this matters for the biology:**

The published *Acinetobacter baumannii* paper’s central finding is that RM systems are restrictive gatekeepers — they negatively correlate with ARG/IME burden and define the RESTRICT archetype. `SspBCDE` is facilitative. This RESTRICT/FACILITATE dichotomy is the paper’s contribution.

If `RM_Type_I` has negative permutation importance in the Q1 multi-class classifier, there are two interpretations:

1. RM systems are not independently informative for species classification once other defence systems are conditioned on — they are correlated with other features (e.g., with defence-system count, with CRISPR-Cas, with `Mok_Hok_Sok`).
2. RM is a within-species signal (high RM genomes are non-IC2 AB), not a between-species signal. Q1 is a species classifier, so within-AB variation in RM may not distinguish AB from other species.

Both interpretations must be explicitly addressed in Phase 10. The current trajectory — SHAP comparison of Q1 features to the published AB Fisher’s exact ranks — will be scientifically misleading if it presents SHAP rank 17 for RM as evidence the ML “confirms” the published finding. It does not confirm it. It is silent on Q2 ARG-prediction relevance (because Q2 results are currently invalid for the reasons in C1). This needs careful framing.

**Remediation:**

- Flag this in Phase 10 as a key interpretive conflict requiring resolution.
- After C1 fix, examine `RM_Type_I` permutation importance in Q2 (where the biological question is ARG burden, not species identity).
- If RM is still low-importance in Q2 after the label fix, that is a genuine finding: RM system presence at the binary level may not predict ARG burden in the multi-species ESKAPE context even though it does in the published single-species analysis.

---

### H7 — Anti-defence features (`ad_*`, 29 columns) are entirely absent from all ML models

**What was found:**

The feature matrix contains 29 `ad_*` columns derived from AntiDefenseFinder. These encode the presence of anti-defence systems that phage and MGEs use to overcome host defences. They were computed as part of the Phase 2 pipeline and are present in the feature matrix.

None of the Phase 7 (LR), Phase 8 (RF), or Phase 9 (GB) notebooks include `ad_*` features in any analysis — not even in a sensitivity run.

**Why this matters:**

Anti-defence systems are encoded in MGEs. In the published paper, the RESTRICT archetype has high RM density and low MGE burden. Anti-defence systems are the MGE strategy to overcome RM. If `SspBCDE`-positive, RM-low genomes (IC2 AB) also carry anti-defence systems, the `ad_*` features would be informative for Q2 (they would positively predict high-ARG burden) and potentially for Q1.

Excluding these entirely is not a methodological flaw if justified, but the `CLAUDE.md` and pre-analysis plan do not exclude them — they are listed as pipeline deliverables. If they are excluded, a decision must be logged. If they are not excluded, they must be included.

**Remediation:**

- Make an explicit decision: include or exclude `ad_*` features, with justification in `decisions.md`.
- At minimum, run a sensitivity analysis with `ad_*` included (this is `CLAUDE.md` Phase 9 Optional O3 in your own framework).

---

### H8 — Learning curves were never computed

**What `CLAUDE.md` Phase 6 requires:**

“Learning curves: accuracy vs training-set size” listed as a Phase 6 (baseline classifier) deliverable.

**Why this matters at `n=878`:**

You have 6 species × ~150 genomes. In Q2, after removing `mid_ARG` tertile, per-species n drops to ~100 for the extreme tertiles. A learning curve would directly answer: are these models sample-saturated (plateau reached) or still on the steep part of the learning curve (more genomes would substantially improve performance)?

This is particularly important for justifying the Q2 LR-wins-over-RF finding for small-n species (KP, EC, EF). Learning curves would show that RF hasn’t saturated and LR’s apparent superiority may be a small-n artefact.

**Remediation:**

- Generate learning curves for Q1 (RF and GB vs LR) and for the per-species Q2 models with the highest uncertainty.
- This can be done post-hoc without rerunning the full pipeline.

---

## MODERATE — Findings that affect rigour but not core conclusions

### M1 — Post-hoc RF model selection violates the pre-registered criterion by a small but documented amount

**What was pre-registered:**

The pre-analysis plan specifies: “Primary Q1 model is the model with the highest BA when compared to LR reference, provided the CI does not overlap.” The intent is that a model wins only if its improvement is detectable above sampling noise.

**What was done:**

RF BA CI `[0.859–0.898]` and LR BA CI `[0.813–0.859]` overlap by `0.0002` units (LR upper = `0.859`, RF lower = `0.859`). The post-hoc decision was that this overlap is “below bootstrap CI precision from 5 fold-level scores” and the delta (`0.041`) justifies selecting RF. This is logged in `decisions.md`.

**Why this is moderate, not critical:**

The delta of `0.041` is real and substantial; the CI overlap of `0.0002` is genuinely within the precision limit of a 2000-resample bootstrap over 5 fold scores. The decision is defensible. However, it is a post-hoc deviation from a pre-registered criterion, and it must be presented as such in any manuscript. Reviewers who read the pre-analysis plan will see the criterion and the deviation immediately.

**Remediation:**

- In the manuscript, state explicitly: “Primary Q1 model selection criterion (non-overlapping CI) was technically violated by 0.0002 units; we treat this as below the precision floor of a 5-fold bootstrap CI and select RF on the basis of the 0.041 BA improvement.” The transparency is the fix.

---

### M2 — Bootstrap CI over 5 fold scores has poor coverage properties

**What was done:**

Bootstrap CI computed as percentile interval from 2000 resamples of the 5 fold-level BA scores.

**The statistical problem:**

Bootstrapping over 5 data points does not produce reliable confidence intervals. With `n=5`, the 2.5th percentile of a bootstrap distribution often collapses to a single observed value (the minimum fold score). The nominal 95% CI may have actual coverage well below 95%. This is not a matter of the number of bootstrap resamples (2000 is fine) — it is a matter of the number of base observations (5 is too few for reliable percentile intervals).

**Why this matters:**

The CIs presented in all phase results (`[0.850–0.893]`, etc.) may be systematically too narrow. They give a false precision to the performance estimates.

**Remediation:**

- Use the “bootstrap on predictions” approach instead: pool all held-out predictions from the 5 folds, then bootstrap over individual genome predictions. This gives `n=878` base observations for the bootstrap, producing reliable CI.
- Alternatively, switch to a t-interval over the 5 fold scores, which at least has correct Student’s-t coverage for small n (though with wide intervals).

---

### M3 — GB GridSearchCV uses sample weights computed on the full dataset, not per-fold

**What the code does:**

```python
sw_all = compute_sample_weight("balanced", y_q1)
grid.fit(X_q1, y_q1, sample_weight=sw_all)
```

The sample weights are computed once on all 878 genomes. Inside `GridSearchCV`'s cross-validation, each fold receives a subset of genomes, but the sample weights were derived from the full-data class distribution. This means each fold’s weights encode the class imbalance of the full dataset, not the fold’s local class distribution.

**Why this matters:**

With `GroupedStratifiedKFold`, phylogroup-correlated classes may be unevenly distributed across folds. The “balanced” weight correction applied to a training fold may under- or over-correct relative to what that specific fold’s class distribution requires.

**Remediation:**

- Move sample weight computation inside the CV loop: compute `compute_sample_weight("balanced", y_train_fold)` for each fold’s training set, not on the full dataset.

---

### M4 — No McNemar tests for Q2 within-species model comparison

McNemar’s test was correctly implemented for Q1. The Q2 cross-model comparisons (RF vs LR, GB vs LR, RF vs GB) are reported as point-estimate differences in BA only. Given the small per-species sample sizes (`n=~100` Q2-eligible genomes per species after label fix), point estimates without statistical tests are uninformative.

**Remediation:** After C1 label fix, run McNemar for each Q2 species × model pair.

---

### M5 — The OOB-to-CV gap is not explicitly framed as the phylogenetic correction effect

**The data:**

- RF OOB score (all genomes, no phylogenetic CV): `0.932`
- RF grouped CV BA: `0.878`
- Delta: `0.054`

This 5.4pp gap directly quantifies the magnitude of phylogenetic signal leakage: standard CV (implicit in OOB) gives `0.932` because related genomes bleed between training and test. Grouped CV removes that leak and gives `0.878`.

This is the single most pedagogically important number in Phase 8 — it is empirical proof that phylogenetic CV correction matters in your dataset. It is not currently presented this way in the notebook.

**Remediation:** Add explicit framing: “OOB=0.932 vs CV BA=0.878: the 5.4pp difference represents the contribution of phylogenetic signal to apparent performance when clone-level isolation is not enforced.”

---

### M6 — `dp_df_Gabija`: SHAP rank 13, permutation importance rank 265, negative permutation value — a spurious Gini feature

Gabija is a defence system found across Gram-negative bacteria, with no known MGE association and no published relevance to ARG burden or ESKAPE ecological niches. Its high Gini importance (rank 32) with negative permutation importance (rank 265, value negative) is a textbook case of Gini inflation for a feature correlated with other informative features but not independently predictive.

This must be flagged in Phase 10 — citing SHAP rank 13 for Gabija as a “key Q4 feature” without noting the permutation importance contradiction would be a misrepresentation.

---

## OPTIONAL — Improvements to rigour or interpretability

### O1 — Calibration analysis absent for Q2 models

Reliability diagrams were computed for GB Q1 (fold 0 only, selected classes). No calibration analysis exists for any Q2 model. For binary classifiers predicting ARG burden (a clinically relevant outcome), uncalibrated probabilities are a limitation. Flag this in the manuscript if probability outputs are discussed.

---

### O2 — GB `max_depth` grid may be constraining

The GB grids tested `max_depth=[4, 6]`. For 878 genomes with 265 sparse binary features, these depths may be insufficient. XGBoost and LightGBM documentation recommends `max_depth=3–8` for tabular data, but the optimal value depends on interaction order. Given that RF’s best depth was 20, the GB trees at depth 6 may be substantially underfitting relative to RF. This does not change the current conclusions but may explain part of the RF-vs-GB gap beyond the training data confound identified in H3.

---

### O3 — No sensitivity analysis with `dc_*` (count) features or `ad_*` features

The 274 `dc_*` (defence count per system) features and 29 `ad_*` features were computed but never used. The `CLAUDE.md` framework lists a sensitivity analysis with count features as an expected deliverable. This is in scope for Phase 11 but should not be deferred past manuscript draft.

---

### O4 — *E. cloacae* genomospecies sensitivity analysis not done

The pre-analysis plan documents *E. cloacae* complex as a known heterogeneous class (EC embraces at least 4 genomospecies). A sensitivity analysis treating EC as 3+ classes vs one class was pre-registered. Not yet done.

---

### O5 — No sample size power analysis

Given Q2 per-species `n≈100` after label fix, a retrospective power analysis (what BA effect size is detectable at `n=100` with 5-fold `GroupedStratifiedKFold`?) would contextualise the null or near-null Q2 results for SA and AB. This is optional but would strengthen the “insufficient power” narrative for negative Q2 findings.

---

## Summary table

| ID | Finding | Priority | Affects |
|---|---|---|---|
| C1 | Q2 label inconsistency (RF/GB `mid_ARG` included) | CRITICAL | All Q2 RF/GB results |
| C2 | Top SHAP features are near-threshold taxonomic markers | CRITICAL | Q1 interpretation, Phase 10 |
| C3 | Held-out AB validation not executed | CRITICAL | Pre-registered protocol |
| H1 | Within-species feature sparsity dominates Q2 training | HIGH | Q2 model quality |
| H2 | EF/SA phylogroup dominance makes grouped CV unreliable | HIGH | EF and SA Q2 CI reliability |
| H3 | XGBoost/LightGBM trained on 64% data vs RF 80% | HIGH | McNemar GB-vs-RF comparison |
| H4 | No BH correction across 6 Q2 species | HIGH | Q2 significance claims |
| H5 | PA-1 binary split not implemented in RF/GB | HIGH | PA Q2 cross-model comparison |
| H6 | `RM_Type_I` negative permutation importance | HIGH | Phase 10 biological narrative |
| H7 | Anti-defence features (`ad_*`) unused without decision log | HIGH | Feature completeness |
| H8 | Learning curves not computed | HIGH | Sample size justification |
| M1 | Post-hoc RF model selection (0.0002 CI overlap) | MODERATE | Transparency |
| M2 | Bootstrap CI over 5 folds has poor coverage | MODERATE | CI precision |
| M3 | GB sample weights computed on full dataset | MODERATE | GB CV validity |
| M4 | No McNemar tests for Q2 | MODERATE | Q2 model comparison |
| M5 | OOB-CV gap not framed as phylogenetic correction | MODERATE | Pedagogical/manuscript |
| M6 | Gabija spurious Gini inflation not flagged | MODERATE | Phase 10 |
| O1 | No Q2 calibration | OPTIONAL | Probability outputs |
| O2 | GB `max_depth` grid may underfit | OPTIONAL | GB vs RF gap |
| O3 | `dc_*`/`ad_*` sensitivity analysis missing | OPTIONAL | Completeness |
| O4 | EC genomospecies sensitivity analysis | OPTIONAL | Pre-registration |
| O5 | Q2 power analysis | OPTIONAL | Negative results framing |

---

## Recommended remediation order

**Do first (before any Phase 10 work):**

1. Fix C1 (Q2 label bug) and PA-1 (H5) simultaneously — one rerun fixes both.
2. Run C3 (held-out AB validation) — this is a pre-registered mandatory check.
3. Rerun Q1 at `spec_score < 0.50` filter and document 2×2 accuracy table (C2 partial fix).

**Do before manuscript draft:**

4. H4 (BH correction on Q2).
5. H6 framing in Phase 10 (RM negative permutation — cannot be ignored in interpretation).
6. M1 (transparent post-hoc model selection language).
7. M2 (bootstrap on predictions, not on 5 fold scores).

**Do alongside Phase 10:**

8. H7 (`ad_*` decision log).
9. H8 (learning curves — can be run post-hoc).
10. M5 (OOB-CV gap framing).
11. M6 (Gabija flagged in Phase 10 narrative).

**Before submission:**

12. O3 (`dc_*`/`ad_*` sensitivity).
13. O4 (EC sensitivity).
14. H3 (GB equal-data comparison, or explicit limitation statement).
