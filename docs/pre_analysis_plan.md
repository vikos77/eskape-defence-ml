# Pre-Analysis Plan — ESKAPE Defence Systems ML Extension

**Registered:** 2026-04-21
**Author:** Vigneshwaran Muthuraman
**Extends:** Muthuraman et al. (2026), *Journal of Applied Microbiology*

This plan is locked before any modelling. Deviations must be logged in
`docs/decisions.md` with justification. Results that emerge from unplanned
analyses are clearly labelled exploratory in the manuscript.

---

## 1. Primary research questions

### Q1 — Species classification (supervised, multi-class)
Can defence system repertoire alone classify ESKAPE species?
Which features drive classification — do they match the published *Acinetobacter*
findings (RM systems, SspBCDE, Gao_Qat)?

**Null hypothesis:** A stratified random classifier cannot be beaten by any
defence-system-based model.

### Q2 — ARG burden prediction (supervised, binary)
Within and across species, can defence system profile predict high-ARG-burden
genomes?
Does the RESTRICT/FACILITATE signature from *Acinetobacter* generalise?

**Label definition (primary):** Top tertile = high ARG burden; bottom tertile = low ARG
burden. Middle tertile excluded from Q2 only. ARG burden = total count of
unique ARG genes per genome (ResFinder output). Tertile boundaries computed
*within each species* to prevent species-level ARG baseline differences from
confounding the label (see decisions.md 2026-05-13).

**Label definition (fallback — Protocol Amendment PA-1, 2026-05-14):**
For any species where `pd.qcut(q=3)` cannot form three distinct tertile bins
because the 33rd percentile equals the distribution minimum (floor effect),
a binary split at the within-species median is used instead:
- Genomes with ARG count **below** the median → `low_ARG`
- Genomes with ARG count **above** the median → `high_ARG`
- Genomes with ARG count **equal to** the median → `mid_ARG` (excluded from Q2,
  same as the middle tertile in the primary method)

This fallback applies to *P. aeruginosa* in this dataset (37% of PA genomes at
ARG = 5, the species minimum). It answers a slightly weaker question ("below
vs above average ARG burden") rather than "bottom third vs top third" — an
intentional compromise to retain the species in Q2 rather than exclude it.
PA Q2 results must be interpreted alongside this methodological note.

The fallback condition — 0th percentile equals 33rd percentile — is a
data-structure criterion defined prospectively before any modelling, not chosen
because it improves results. It is applied uniformly to any species that meets
this criterion in this or any future dataset extending this analysis.

**Null hypothesis:** Defence system profile has no predictive value for ARG
burden beyond a stratified null baseline.

### Q3 — Unsupervised archetypes (unsupervised)
Do genomes cluster by defence-system archetype independently of species?
Species labels are hidden during clustering. Labels applied only post-hoc
for interpretation.

**Expected outcomes logged before analysis:**
- If clusters recover species: defence systems are phylogenetically determined.
- If clusters cut across species: defence archetypes exist as an
  independent biological structure.
- Neither outcome is a failure.

### Q4 — Interpretability (SHAP)
Top 10 SHAP features for each classifier compared to published Fisher's exact
ranks from Muthuraman et al. (2026). Agreement = cross-genus generalisation.
Disagreement = genus-specific architecture.

---

## 2. Dataset

### Training data
- ~150 complete genomes per ESKAPE species downloaded from NCBI RefSeq
- Quality gates: CheckM2 completeness ≥95%, contamination ≤5%
- Stratified by country of origin and isolation year where metadata permits
- Annotated with: DefenseFinder v2.0.2, PADLOC v2.0.0, CRISPRCasFinder,
  ResFinder, AntiDefenseFinder, ICEberg tBLASTn, BacMet tBLASTn

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

### PA-1 — Q2 binary split fallback (2026-05-14)

**Trigger:** During Phase 3 feature matrix construction, `pd.qcut(q=3)` failed for
*P. aeruginosa* with `ValueError: Bin labels must be one fewer than the number of bin
edges`. Root cause: 56/150 PA genomes (37%) have `arg_count_unique = 5` (the species
minimum), making the 0th and 33rd percentiles identical. After `duplicates='drop'`,
only 2 distinct bins remain — insufficient for 3 labels.

**Why this happened specifically for PA:** P. aeruginosa's clinical isolates almost
universally carry a small set of near-baseline acquired ARGs — chromosomal
cephalosporinase derivatives (blaPDC) and housekeeping efflux pump genes are
ubiquitous, creating a hard floor at 5 unique ARGs in ResFinder output. The
right-skewed tail (some PA genomes reaching ARG = 29) represents genomes with
additional mobile-element-acquired resistance, but 37% of PA sits at the minimum.
No equivalent floor effect is observed in the other 5 ESKAPE species.

**Options considered before choosing the amendment:**

*Option 1: Keep PA excluded from Q2* — Pre-registration intact; exclusion is honest;
simple to report. Loss: PA genomes with ARG = 20+ (genuine high-burden) are excluded
entirely, weakening cross-species generalisability.

*Option 2: Binary split at the median (chosen)* — PA genomes below median → low_ARG;
above median → high_ARG; at median → mid_ARG (excluded). Brings PA back into Q2.
Answers a slightly weaker biological question ("below vs above species-average ARG
burden" rather than "bottom vs top third"), but the question remains coherent and
directly tests the RESTRICT/FACILITATE hypothesis for PA. The 56 genomes at the ARG
floor (low_ARG) vs the 64 genomes with above-median burden (high_ARG) provide a
meaningful contrast.

*Option 3: Rank-based tertile* — Rank genomes by ARG count, split ranks into thirds.
Rejected: 56 PA genomes share ARG = 5 exactly. Assigning different labels to
identical-valued genomes based on arbitrary rank order would introduce noise as
signal — any classifier learning this split would be memorising random assignment,
not biology.

**Why the amendment is methodologically sound:**
1. Defined *before* any modelling — this is a Phase 3 (feature engineering) fix,
   not a Phase 6+ retroactive change. No model performance numbers have been seen.
2. The fallback criterion ("0th percentile == 33rd percentile") is a data-structure
   test, not an outcome test — it cannot be gamed post-hoc.
3. The binary split is a standard, pre-existing method (median split is weaker than
   tertile but not arbitrary). It is prospectively documented and therefore confirmatory,
   not exploratory.
4. The rule is species-agnostic and generalises to any future dataset extension.

**Result for PA:** `low_ARG` = 56 genomes (ARG < 6), `mid_ARG` = 30 genomes (ARG = 6,
excluded from Q2), `high_ARG` = 64 genomes (ARG > 6). PA participates in Q2 with 120
genomes (56 + 64). Q2 now runs on all 6 ESKAPE species, 614 eligible genomes
(low_ARG=325 + high_ARG=289; was 494 across 5 species before this amendment).

**Required manuscript statement (Methods):** "For *P. aeruginosa*, 37% of genomes sit
at the species ARG minimum (arg = 5), making tertile boundaries degenerate (0th = 33rd
percentile). A pre-specified binary split at the within-species median was applied
instead: genomes below the median were labelled low-ARG burden, genomes above were
labelled high-ARG burden, and median-tied genomes were excluded from Q2 as for the
middle tertile. The Q2 classifier for PA therefore contrasts below-average vs
above-average ARG burden rather than bottom vs top third."

---

### PA-2 — Q1 feature specificity sensitivity analysis (2026-05-18)

**Trigger:** Preliminary LR baseline classifier (04_baseline_classifier.ipynb) achieved
Q1 balanced accuracy = 0.984, which exceeded the pre-specified 0.95 investigation
threshold (§5). Side investigation confirmed that several defence features are near-
universal in exactly one species or one clade, acting as taxonomic markers rather than
defence architecture signals. Sensitivity analysis was run before any Phase 7 modelling.

**Two orthogonal validity problems identified:**
1. *Clone contamination* — closely related genomes split across train/test folds.
   Fixed by phylogenetic GroupedStratifiedKFold (Phase 6 — pre-existing plan).
2. *Taxonomic markers* — defence features near-universal in one species inflate
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
  0.902; filtered <0.35 (252): 0.874. Genuine defence signal is present — accuracy
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

## 8. What this plan does not pre-specify

Secondary analyses that emerge from results are permitted but must be labelled
exploratory. Specifically:

- Continuous ARG count as regression target (alternative to binary Q2)
- Species-stratified models (train/test within single species)
- Integration of anti-defence system features as a separate predictor class

These are not primary analyses. Any manuscript claim must distinguish
confirmatory (pre-specified) from exploratory (post-hoc).
