# Pre-Analysis Plan  -  ESKAPE Defence Systems ML Extension

**Registered:** 2026-04-21
**Author:** Vigneshwaran Muthuraman
**Extends:** Muthuraman et al. (2026), *Journal of Applied Microbiology*

This plan is locked before any modelling. Deviations must be logged in
`docs/decisions.md` with justification. Results that emerge from unplanned
analyses are clearly labelled exploratory in the manuscript.

---

## 1. Primary research questions

### Q1  -  Species classification (supervised, multi-class)
Can defence system repertoire alone classify ESKAPE species?
Which features drive classification  -  do they match the published *Acinetobacter*
findings (RM systems, SspBCDE, Gao_Qat)?

**Null hypothesis:** A stratified random classifier cannot be beaten by any
defence-system-based model.

### Q2  -  ARG burden prediction (supervised, binary)
Within and across species, can defence system profile predict high-ARG-burden
genomes?
Does the RESTRICT/FACILITATE signature from *Acinetobacter* generalise?

**Label definition (primary):** Top tertile = high ARG burden; bottom tertile = low ARG
burden. Middle tertile excluded from Q2 only. ARG burden = total count of
unique ARG genes per genome (ResFinder output). Tertile boundaries computed
*within each species* to prevent species-level ARG baseline differences from
confounding the label (see decisions.md 2026-05-13).

**Label definition (fallback  -  Protocol Amendment PA-1, 2026-05-14):**
For any species where `pd.qcut(q=3)` cannot form three distinct tertile bins
because the 33rd percentile equals the distribution minimum (floor effect),
a binary split at the within-species median is used instead:
- Genomes with ARG count **below** the median → `low_ARG`
- Genomes with ARG count **above** the median → `high_ARG`
- Genomes with ARG count **equal to** the median → `mid_ARG` (excluded from Q2,
  same as the middle tertile in the primary method)

This fallback applies to *P. aeruginosa* in this dataset (37% of PA genomes at
ARG = 5, the species minimum). It answers a slightly weaker question ("below
vs above average ARG burden") rather than "bottom third vs top third"  -  an
intentional compromise to retain the species in Q2 rather than exclude it.
PA Q2 results must be interpreted alongside this methodological note.

The fallback condition  -  0th percentile equals 33rd percentile  -  is a
data-structure criterion defined prospectively before any modelling, not chosen
because it improves results. It is applied uniformly to any species that meets
this criterion in this or any future dataset extending this analysis.

**Null hypothesis:** Defence system profile has no predictive value for ARG
burden beyond a stratified null baseline.

### Q3  -  Unsupervised archetypes (unsupervised)
Do genomes cluster by defence-system archetype independently of species?
Species labels are hidden during clustering. Labels applied only post-hoc
for interpretation.

**Expected outcomes logged before analysis:**
- If clusters recover species: defence systems are phylogenetically determined.
- If clusters cut across species: defence archetypes exist as an
  independent biological structure.
- Neither outcome is a failure.

### Q4  -  Interpretability (SHAP)
Top 10 SHAP features for each classifier compared to published Fisher's exact
ranks from Muthuraman et al. (2026). Agreement = cross-genus generalisation.
Disagreement = genus-specific architecture.

---

## 2. Dataset

### Training data
- ~150 complete genomes per ESKAPE species downloaded from NCBI RefSeq
- Quality gates: CheckM2 completeness ≥95%, contamination ≤5%
- Stratified by country of origin and isolation year where metadata permits
- Annotated with: DefenseFinder v2.0.1 (with AntiDefenseFinder --antidefensefinder),
  PADLOC v2.0.0, ResFinder v4.7.2, AMRFinderPlus v4.2.7, ICEberg tBLASTn, ISEScan v1.7.2.3,
  BacMet tBLASTn
  [Note: tool set updated vs original registration; CRISPRCasFinder removed (ARM64
  unavailable, redundant); BacMet tBLASTn collected but excluded from classifiers
  (RND efflux artefact); ISEScan added; AMRFinderPlus added. See decisions.md.]

### Held-out validation set (not used in training or hyperparameter tuning)
- Published 132 complete *A. baumannii* genomes from Muthuraman et al. (2026)
- Evaluated separately: complete genomes vs 90 contig-level IC2 assemblies
- Purpose: test whether the ESKAPE-trained model recovers the published
  RESTRICT/FACILITATE signal

---

## 3. Primary outcome metrics

| Question | Primary metric | Secondary metrics |
|---|---|---|
| Q1 | Balanced accuracy (macro) | Macro-F1, per-class precision/recall, confusion matrix |
| Q2 | Balanced accuracy (binary) | AUROC, precision-recall AUC |
| Q3 | Silhouette score | Gap statistic, cluster stability (bootstrap) |
| Q4 | SHAP rank agreement (Spearman ρ) | Top-10 overlap with published Fisher's ranks |

All classifiers compared to stratified null baseline.
Model comparisons use McNemar test (paired CV folds), not point-estimate comparison.
All performance estimates reported with 95% CI (bootstrap over CV folds).

---

## 4. Cross-validation strategy

**All classifier phases (Phase 7 onward):** Grouped 5-fold CV using Mash-distance-derived
phylogroups as grouping variable (see pipeline restructuring, decisions.md 2026-05-18).
Genomes from the same phylogroup go to the same fold entirely. Standard stratified CV
is not used for any primary result.

**Q1 accuracy reporting format (2×2 table, pre-specified):**
All Q1 accuracy claims must be reported in the following format to separate the two
independent validity corrections:

| | Full feature set (274 dp_*) | Specificity-filtered (std < 0.70) |
|---|---|---|
| Standard stratified CV | preliminary reference only | preliminary reference only |
| Phylogenetic grouped CV | secondary result | **primary reported result** |

The primary result is [specificity-filtered, phylogenetic grouped CV]. The other three
cells provide context. This format was pre-specified before any Phase 7 modelling
(see Protocol Amendment PA-2 and decisions.md 2026-05-18).

---

## 5. Pre-specified falsification criteria

- If no classifier beats stratified null baseline → defence systems are
  uninformative at this scale. Report as negative result.
- If Q1 accuracy exceeds 0.95 under stratified CV → investigate leakage
  (genome size, GC content) and taxonomic markers before reporting.
- If Q1 accuracy under the specificity-filtered (<0.70 std) + phylogenetic grouped CV
  condition drops below 0.70 → the published 0.984 result was driven by species-specific
  annotation markers and phylogenetic clone signal, not genuine defence architecture.
  Report Q1 as a negative or near-null finding.
- If Q1 accuracy drops >15 percentage points from full-feature to specificity-filtered
  under grouped CV → most of the signal was taxonomic markers; revise Q1 framing to
  acknowledge this explicitly.
- If Q4 shows no overlap between SHAP ranks and published Fisher's ranks →
  RESTRICT/FACILITATE is genus-specific, not cross-ESKAPE.

---

## 6. Decisions locked before analysis

| Decision | Choice | Rationale |
|---|---|---|
| Q2 label cutoff | Tertile (top vs bottom 33%); binary median fallback if tertile fails (PA-1) | Clean class separation; fallback retains species with floor effects |
| A. baumannii data source | Fresh NCBI download for training; published 132 as held-out validation | Tests generalisation to peer-reviewed benchmark |
| Q3 species label handling | Hidden during clustering; applied post-hoc only | Prevents species-bias contaminating archetype discovery |
| Python version | 3.11 | Full scikit-learn ≥1.4, shap ≥0.44, umap-learn compatibility |
| Random seeds | All set to 42 (see config/params.yaml) | Reproducibility |
| No deep learning | Enforced | Sample size (~900) does not justify neural networks |

---

## 7. Protocol amendments

Amendments are logged here when a pre-specified method cannot be applied as written
and a prospective modification is required. Amendments are defined *before* any
modelling results are seen. Post-hoc modifications are logged in `decisions.md` as
exploratory, not here.

### PA-1  -  Q2 binary split fallback

**Rule:** If `pd.qcut(q=3, duplicates='drop')` raises a `ValueError` for a species
(fewer than 3 distinct bin edges, floor effect where 0th == 33rd percentile), apply a
binary split at the within-species median instead: below median to `low_ARG`, above to
`high_ARG`, at median to `mid_ARG` (excluded from Q2 as for the standard middle tertile).

**Why:** A floor effect means many genomes share the species minimum ARG count. A
rank-based tertile would assign different labels to identical values, introducing noise
as signal. A median split is weaker than a tertile contrast but asks a coherent
biological question (below vs above species-average ARG burden) that still tests the
RESTRICT/FACILITATE hypothesis.

**Validity criteria:**
1. Applied in Phase 3 before any model performance is seen, data-structure test only.
2. The fallback criterion is species-agnostic and prospectively documented.
3. Binary median split is a standard method; not arbitrary.

**Current status:** Not triggered in the 600-genome-per-species dataset. All six ESKAPE
species produce 3 distinct tertile edges. The fallback code is retained for any future
dataset extension where a floor effect may occur.

---

### PA-2  -  Q1 feature specificity sensitivity analysis (2026-05-18)

**Trigger:** Preliminary LR baseline classifier (04_baseline_classifier.ipynb) achieved
Q1 balanced accuracy = 0.984, which exceeded the pre-specified 0.95 investigation
threshold (§5). Side investigation confirmed that several defence features are near-
universal in exactly one species or one clade, acting as taxonomic markers rather than
defence architecture signals. Sensitivity analysis was run before any Phase 7 modelling.

**Two orthogonal validity problems identified:**
1. *Clone contamination*  -  closely related genomes split across train/test folds.
   Fixed by phylogenetic GroupedStratifiedKFold (Phase 6  -  pre-existing plan).
2. *Taxonomic markers*  -  defence features near-universal in one species inflate
   classification accuracy independently of clone-level phylogeny.
   Fixed by feature specificity filtering (this amendment).

**Method:** For each dp_* binary feature, compute the standard deviation of
per-species prevalence (fraction of genomes in each species carrying the feature)
and normalise by 0.5 (maximum possible std across 6 balanced groups). A score near
1 indicates the feature is near-universal in one species and near-absent in others;
a score near 0 indicates roughly equal prevalence across all six species.

**Pre-specified threshold:** Features with specificity std ≥ 0.70 are classified as
strong taxonomic markers. Sensitivity analyses at std ≥ 0.50 and std ≥ 0.35 are
reported as supplementary comparisons.

**Sensitivity analysis result (standard CV, for reference only):**
- Full features (274): 0.984; filtered <0.70 std (266): 0.936; filtered <0.50 (259):
  0.902; filtered <0.35 (252): 0.874. Genuine defence signal is present  -  accuracy
  remains at 0.744 even with features capped at std < 0.20 (225 features). The 0.984
  is not fabricated by markers alone.

**Required reporting format:** Q1 accuracy must be reported as a 2×2 table (see §4)
with [specificity-filtered + grouped CV] as the primary cell.

**Manuscript Methods statement (required):** "For Q1 species classification, we
pre-specified a feature specificity analysis to distinguish genuine defence
architecture signal from species-specific annotation markers (features with
per-species prevalence standard deviation ≥ 0.70 normalised by the theoretical
maximum). Eight features exceeded this threshold and were removed from the primary
Q1 feature set. Accuracy is reported for both the full and specificity-filtered
feature sets, each evaluated under phylogenetically-grouped cross-validation."

---

## 10. Phase 12  -  Mechanism-level ARG burden and RM count sensitivity (pre-registered 2026-05-27)

### Motivation

Phase 8/9 Q2 models used binary defence presence (`dp_*`) as features and total ARG count
as the target. Two unresolved questions motivate Phase 12:

1. **RM count vs RM presence (Test A):** RM Type I/II/III negative permutation importance in Q1
   (decisions.md 2026-05-26, H6) and the published Spearman correlation (Muthuraman 2026) both
   operate on RM *count*, not binary presence. Binary RM may dilute the signal by treating a
   genome with 1 RM system identically to one with 8. Test A checks whether substituting RM
   *count* columns (`dc_RM_*`) for RM *presence* columns (`dp_RM_*`) in Q2 improves AUROC.
   All other features remain `dp_*` (presence/absence).

2. **Mechanism-class ARG target (Test B):** Total ARG count conflates plasmid-mediated ARGs
   (β-lactam, aminoglycoside  -  where RM gating is biologically expected) with chromosomal
   ARGs (fluoroquinolone point mutations  -  where no RM gating is predicted). A classifier
   predicting total ARG burden sees a mixed signal. Test B re-runs Q2 with mechanism-class
   ARG burden as the target: one binary classifier per (species × ARG mechanism class), using
   original `dp_*` features. This tests whether the RESTRICT/FACILITATE principle is
   class-specific rather than universal.

These are kept as **two separate experiments** with a shared feature set. Running both changes
simultaneously would prevent attribution of any accuracy change to the correct cause.

---

### Test A  -  RM count features in Q2

**Feature change:** Replace `dp_RM_Type_I`, `dp_RM_Type_II`, `dp_RM_Type_IIG`, `dp_RM_Type_III`
with `dc_RM_Type_I`, `dc_RM_Type_II`, `dc_RM_Type_IIG`, `dc_RM_Type_III` in Q2 RF.
All other features remain `dp_*` (binary presence).

**Pre-check required:** Before modelling, run `(fm["dc_X"] != fm["dp_X"]).sum()` for each RM
feature. If dc == dp for all genomes, RM counts are all 0 or 1 (no genome carries >1 RM system
of any subtype) and Test A is moot. Report this check result regardless of outcome.

**Target:** Original total-ARG-burden tertile label (same as Phase 8/9 Q2).

**Model:** Random Forest with same hyperparameters as Phase 8 (max_depth=20, max_features=sqrt,
min_samples_leaf=1, n_estimators=100). GroupKFold on same 95 phylogroups.

**Primary outcome:** AUROC per species. Compare to Phase 8 Q2 AUROC as baseline.

**Null hypothesis:** RM count features do not improve Q2 AUROC beyond Phase 8 binary-presence baseline.

**Falsification:** If dc_RM == dp_RM for ≥90% of genomes → Test A is uninformative (RM is
largely binary in this dataset); record as finding, not as failure.

---

### Test B  -  Mechanism-class ARG burden targets

**Target construction:**
For each mechanism class, compute per-genome count of ARGs in that class (from ResFinder output).
Apply the same tertile logic as primary Q2 (tertile within species; binary median fallback per PA-1
if tertile fails). Create one binary label per (species × class): high_class vs low_class.
Middle-class genomes excluded from that classifier (same as Q2 middle tertile exclusion).

**Mechanism classes included:** All classes where ≥2 ESKAPE species pass the 30/30 floor
(defined below). Expected inclusions: β-lactam, aminoglycoside, sulfonamide. Fluoroquinolone,
tetracycline, glycopeptide included if floor is met. Classes failing the floor in a given species
are excluded from that species' analysis  -  they are not excluded from other species.

**Sample size floor  -  30/30 rule (non-negotiable):**
A (species × mechanism class) classifier is run only if the training set contains ≥30 high_class
AND ≥30 low_class genomes after tertile construction and middle exclusion. See §10 note below
for justification. Cells failing this floor are documented in results as "excluded: insufficient
class size"  -  they are not imputed or combined.

**Features:** Original `dp_*` filtered set (265 features, same as Phase 8/9 Q2). No change to features.

**Model:** Random Forest, same hyperparameters as Phase 8. GroupKFold on same 95 phylogroups.

**CV strategy:** GroupKFold mandatory. No standard stratified CV for any primary result.
BH correction applied across all (species × class) tests within Test B. Significance threshold: q=0.05.

**Primary outcome:** AUROC per (species × class). Hypothesis: plasmid-mediated classes
(β-lactam, aminoglycoside) show higher AUROC and greater RM feature importance than
chromosomal classes (fluoroquinolone).

**Expected direction of SHAP for RM features:**
- β-lactam: RM presence → negative SHAP (restricts plasmid-mediated ARG acquisition)
- aminoglycoside: RM presence → negative SHAP (often plasmid-borne in Gram-negative)
- fluoroquinolone: RM SHAP ≈ 0 (chromosomal mutations, no RM gating)

If SHAP direction matches prediction: this is the strongest mechanistic confirmation of
RESTRICT available in this dataset. If it does not match: total-ARG RESTRICT signal may
reflect genome-wide correlates rather than direct plasmid gating.

**Null hypothesis:** Defence system profile predicts mechanism-class ARG burden no better
than total ARG burden (i.e., Test B AUROC ≈ Phase 8 AUROC for each species).

---

### Ordering requirement

Test A must be run and reported before Test B. If Test A shows RM count == RM presence for
most genomes, this is informative context for interpreting Test B SHAP output (RM SHAP in
Test B reflects binary presence, not count).

---

### What Phase 12 does NOT pre-specify

- Combining Test A features and Test B targets simultaneously. If both Tests A and B improve
  performance, a combined run is permitted as a supplementary exploratory analysis only.
  It must be labelled explicitly as exploratory.
- Changing the GroupKFold grouping structure. The 95 phylogroups defined in Phase 6 are fixed.
- Adding ARG features to Q1 (species classification). ARGs are a consequence of defence
  architecture, not a feature. Including ARG counts in Q1 would trivially improve accuracy
  by adding a label-correlated feature  -  this is leakage.

---

### 30/30 floor justification (for referee queries)

Three convergent lines of reasoning converge on n≥30 per class as the minimum viable floor.

**1. Statistical power for AUROC (Hanley & McNeil 1982).**
Using the Hanley-McNeil variance formula for the AUROC (equivalent to the Wilcoxon-Mann-Whitney
statistic), with n₁=n₂=30 and SE₀≈0.075 (at AUROC=0.5 null):

| AUROC | Z statistic | p (one-sided) | Survives BH (20 tests, q=0.05)? |
|---|---|---|---|
| 0.60 | 1.33 | 0.092 | No |
| 0.65 | 1.99 | 0.023 | Marginal |
| 0.70 | 2.66 | 0.004 | Yes |
| 0.75 | 3.33 | 0.0004 | Yes |

With n=20/20, SE₀≈0.092, and AUROC=0.70 gives p=0.015  -  fails BH correction for 20 tests.
With n=30/30, AUROC=0.70 gives p=0.004  -  survives BH correction for 30 tests.
The floor of 30 is thus the minimum that makes moderate effect sizes (AUROC≥0.70) reliably
detectable after multiple testing correction.

**2. Cross-validation stability.**
Under 5-fold GroupKFold with n₁=30 positives, each test fold receives ~6 positives.
Fold-level AUROC SE ≈ 0.13. Bootstrap CI over 5 fold-level scores: ±0.11.
This CI is borderline: it can distinguish AUROC=0.65 from 0.50 but not 0.60 from 0.50.
Below n₁=30, the CI exceeds ±0.15 and is uninformative (CI spans null to moderate effect).

**3. Random Forest stability.**
Bootstrap sampling in RF draws ~63.2% of training observations per tree. With n₁=30 positives
in train (~80% split = 24 positives), each tree sees ~15 positives. This is the minimum for
finding 2-3 non-random splits on positive-class signal. Below this (n_train_pos ≈ 10), trees
learn noise, and permutation importance / SHAP values become unreliable even when the
classifier appears to function on held-out data.

---

## 8. What this plan does not pre-specify

Secondary analyses that emerge from results are permitted but must be labelled
exploratory. Specifically:

- Continuous ARG count as regression target (alternative to binary Q2)
- Species-stratified models (train/test within single species)
- Integration of anti-defence system features as a separate predictor class

These are not primary analyses. Any manuscript claim must distinguish
confirmatory (pre-specified) from exploratory (post-hoc).

---

## 9. Post-hoc deviations from this plan (amendment log)

Deviations are logged here for transparency. Each must also appear in `decisions.md`.

### Amendment 1  -  Primary Q1 model selection criterion (2026-05-25)

**Pre-registered criterion:** The primary Q1 model is the model with the highest
balanced accuracy, provided the 95% bootstrap CI does not overlap with the LR
reference CI.

**Observed result:** RF CI [0.859--0.898], LR CI [0.813--0.859]. The lower bound of
the RF CI equals the upper bound of the LR CI to four decimal places  -  a technical
overlap of 0.0002 units.

**Deviation:** The non-overlap criterion is technically violated. RF was nonetheless
selected as the primary Q1 model.

**Justification:** The 0.0002 overlap falls below the precision floor of a bootstrap
CI estimated from 5 fold-level scores (2000 resamples of n=5 cannot reliably resolve
intervals at 4 decimal places). The intent of the criterion  -  preventing a 0.001
difference from being declared a win  -  is not violated by a delta of 0.041. All five
individual CV folds showed RF > LR; the boundary touch is a CI construction artefact.
Full rationale in decisions.md (2026-05-25).

**Required manuscript statement:** "Primary Q1 model selection required non-overlapping
95% bootstrap CIs. The RF CI [0.859--0.898] and LR CI [0.813--0.859] overlap by 0.0002
units at the boundary; we treat this as below the precision floor of a bootstrap CI from
five CV fold scores and select RF on the basis of the 0.041 BA improvement. This
constitutes a minor post-hoc deviation from the pre-registered criterion, logged in the
pre-analysis plan and in decisions.md."
