# ESKAPE Defence Systems — ML Extension

Machine learning extension of Muthuraman et al. (2026), *Journal of Applied Microbiology*.

**Research question:** Does the RESTRICT/FACILITATE defence-system dichotomy identified in *Acinetobacter* generalise across ESKAPE pathogens?

## Published baseline

- 132 complete *Acinetobacter* genomes
- Key finding: RM systems restrict ARG/MGE acquisition; SspBCDE and Gao_Qat facilitate it
- Published pipeline: [acinetobacter-defence-pipeline](https://github.com/vikos77/acinetobacter-defence-pipeline)

## This extension

| Question | Method | Notebook |
|---|---|---|
| Q1: Can defence repertoire classify species? | Logistic Regression, Random Forest, XGBoost | 05, 06, 07 |
| Q2: Can defence profile predict ARG burden? | Binary classifier, species-stratified | 05, 06, 07 |
| Q3: Do defence archetypes exist across species? | UMAP, K-means | 03, 09 |
| Q4: Which features drive each model? | SHAP | 08 |

## Current dataset

- **878 genomes** across 6 ESKAPE species (900 downloaded, 22 excluded for MLST mismatch)
- Species: KP=132, EC=146, AB=150, EF=150, PA=150, SA=150
- Features: 274 defence system presence/absence (dp_\*) + 274 counts (dc_\*) + 48 other
- Q2 eligible: 614 genomes (low_ARG=325, high_ARG=289)

## Phase progress

| # | Notebook | Topic | Status |
|---|---|---|---|
| 0 | — | Environment + repo scaffolding | **COMPLETE** |
| 1 | — | Literature ramp-up + pre-analysis plan | **COMPLETE** |
| 2 | — | Data acquisition (NCBI, 878 genomes × 6 species) | **COMPLETE** |
| 3 | 02_feature_engineering | Feature matrix construction (878 × 631) | **COMPLETE** |
| 4 | 01_eda | Exploratory data analysis | **COMPLETE** |
| 5 | 03_dimensionality_reduction | PCA, UMAP, t-SNE | **COMPLETE** |
| **6** | **04_phylogenetic_grouping** | **Mash distances → phylogroups (GroupedKFold)** | **IN PROGRESS** |
| 7 | 05_baseline_classifier | Null baseline, Logistic Regression, KNN | pending |
| 8 | 06_random_forest | Random Forest + feature importance | pending |
| 9 | 07_gradient_boosting | XGBoost / LightGBM + calibration | pending |
| 10 | 08_model_interpretation | SHAP + biological synthesis | pending |
| 11 | 09_unsupervised_archetypes | K-means archetypes + publication figures | pending |

### Key findings so far

- **EDA (Phase 4):** RESTRICT/FACILITATE is AB-specific, not pan-ESKAPE. In KP and PA,
  RM systems are plasmid-encoded alongside ARGs (co-carriage), not chromosomal gatekeepers.
  IME-ARG co-acquisition is universal across all 6 species (ρ = +0.65–0.77).
- **Dimensionality reduction (Phase 5):** Species separate cleanly in UMAP (Jaccard metric).
  AB clusters far left (depauperate IC2 profile); EF far bottom-right (gram-positive); KP
  and EC consistently adjacent (Enterobacterales). 103 PCs required for 80% variance.

### Pipeline restructuring note (2026-05-18)

Phylogenetic control was originally Phase 9 (after modelling). This has been corrected:
**Mash-based phylogroup assignment now precedes all classifier training.** All cross-validation
uses `GroupedStratifiedKFold` with phylogroups as the grouping variable from Phase 6 onwards.
This ensures results are not inflated by relatedness between training and test genomes.
See `docs/decisions.md` (2026-05-18 entry) for full rationale.

## Setup

```bash
conda env create -f environment.yml
conda activate eskape-ml
pre-commit install
```

## Citation

Muthuraman V et al. (2026) Niche-specific defence system selection in *Acinetobacter* spp.
*Journal of Applied Microbiology*.
