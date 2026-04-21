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

**Label definition:** Top tertile = high ARG burden; bottom tertile = low ARG
burden. Middle tertile excluded from Q2 only. ARG burden = total count of
unique ARG genes per genome (ResFinder output).

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

**Phase 6–8 (standard):** Stratified 5-fold CV, `random_state=42`.

**Phase 9 onward (phylogenetically corrected):** Grouped 5-fold CV using
Mash-distance-derived phylogroups as grouping variable. Genomes from the same
phylogroup go to the same fold entirely.

Any accuracy claim in the manuscript uses the Phase 9 grouped CV estimate.
Phase 6–8 stratified CV results are reported as preliminary only.

---

## 5. Pre-specified falsification criteria

- If no classifier beats stratified null baseline → defence systems are
  uninformative at this scale. Report as negative result.
- If Q1 accuracy exceeds 0.95 under stratified CV → investigate leakage
  (genome size, GC content) before reporting.
- If Q1 accuracy drops >15 percentage points under grouped CV → standard
  CV was capturing phylogenetic signal, not defence architecture.
- If Q4 shows no overlap between SHAP ranks and published Fisher's ranks →
  RESTRICT/FACILITATE is genus-specific, not cross-ESKAPE.

---

## 6. Decisions locked before analysis

| Decision | Choice | Rationale |
|---|---|---|
| Q2 label cutoff | Tertile (top vs bottom 33%) | Clean class separation; middle genomes used in Q1/Q3 |
| A. baumannii data source | Fresh NCBI download for training; published 132 as held-out validation | Tests generalisation to peer-reviewed benchmark |
| Q3 species label handling | Hidden during clustering; applied post-hoc only | Prevents species-bias contaminating archetype discovery |
| Python version | 3.11 | Full scikit-learn ≥1.4, shap ≥0.44, umap-learn compatibility |
| Random seeds | All set to 42 (see config/params.yaml) | Reproducibility |
| No deep learning | Enforced | Sample size (~900) does not justify neural networks |

---

## 7. What this plan does not pre-specify

Secondary analyses that emerge from results are permitted but must be labelled
exploratory. Specifically:

- Continuous ARG count as regression target (alternative to binary Q2)
- Species-stratified models (train/test within single species)
- Integration of anti-defence system features as a separate predictor class

These are not primary analyses. Any manuscript claim must distinguish
confirmatory (pre-specified) from exploratory (post-hoc).
