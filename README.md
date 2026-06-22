# ESKAPE Defence Systems - ML Extension

> Machine learning extension of Muthuraman et al. (2026), *Journal of Applied Microbiology*.
> Does the RESTRICT/FACILITATE defence-system dichotomy generalise across ESKAPE pathogens?

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![conda](https://img.shields.io/badge/env-eskape--ml-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Background

A prior analysis of *Acinetobacter* spp. identified a RESTRICT/FACILITATE dichotomy in
defence system biology: restriction-modification (RM) systems negatively correlated with
ARG and mobile genetic element (MGE) counts, while SspBCDE and Gao_Qat were positively
correlated, consistent with co-acquisition on MGEs. Whether this dichotomy is specific to
*Acinetobacter* or reflects a general feature of defence-resistance co-evolution in
clinical pathogens was untested.

This repository applies supervised machine learning to 3,335 complete and high-quality
genomes spanning all six ESKAPE species to test generalisation of that pattern. Four
research questions (Q1-Q4) were pre-registered before any model training; see the
manuscript's Methods section for the full analysis plan.

---

## Pipeline

```
NCBI genomes (ncbi-datasets-cli)
        |
        v
Quality control (CheckM2 >= 95% completeness; MLST species confirmation)
        |
        v
Defence annotation          ARG detection       MGE / IS detection
DefenseFinder + PADLOC      ResFinder           ISEScan + ICEberg tBLASTn
        |                         |                      |
        +-------------------------+----------------------+
                                  |
                                  v
                     Feature matrix  (01_feature_engineering)
                     3,335 genomes x 367 dp_* features
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
        Phylogenetic grouping          Exploratory analysis
        Mash distances                 PCA / UMAP / t-SNE
        309 phylogroups                (02_eda, 03_dimensionality_reduction)
        (04_phylogenetic_grouping)
                    |
                    v
        Grouped 5-fold CV (GroupedStratifiedKFold on phylogroups)
                    |
         +----------+----------+----------+
         |          |          |          |
         v          v          v          v
        Q1         Q2         Q3         Q4
   Species      ARG burden  Archetypes  SHAP
classification  prediction  K-means     attribution
  LR / RF /     RF per      (09)        (08)
  XGB / LGBM    species
  (05-07)       (05-07)
```

---

## Key results

| Question | Result |
|----------|--------|
| Q1: Species classification from defence profiles | RF balanced accuracy = 0.895 [95% CI: 0.871-0.923]; null baseline = 0.167 |
| Q2: ARG burden prediction within species | Significant in 4/6 species (EC, KP, PA, EF); AUROC 0.739-0.822; facilitative direction in all four |
| Q3: Pan-ESKAPE defence archetypes | Continuum, not discrete clusters; K-means ARI vs species = -0.004 on dereplicated representatives |
| Q4: SHAP generalisation of Acinetobacter findings | 3/4 published systems in global SHAP top 30; RM restriction specific to AB; facilitative direction conserved |

All models evaluated under phylogenetically grouped 5-fold cross-validation
(309 Mash-derived phylogroups; MLST concordance 92.4%).

---

## Dataset

| Property | Value |
|----------|-------|
| Genomes | 3,335 (3,460 downloaded; 125 excluded by MLST quality filter) |
| Species | AB n=600, EC n=507, EF n=524, KP n=504, PA n=600, SA n=600 |
| Defence features (dp_*) | 367 binary presence/absence columns |
| Full feature matrix | 3,335 x 806 (defence + ARG + MGE + genomic context) |
| Phylogroups | 309 (Mash k=21, s=1000; average linkage; per-species thresholds 0.005-0.010) |
| Cross-validation | GroupedStratifiedKFold(5) on phylogroup labels |

---

## Quick start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate eskape-ml

# 2. Install pre-commit hooks
pre-commit install

# 3. Download genomes and run annotation pipeline (requires NCBI datasets CLI)
snakemake --cores 8 --use-conda

# 4. Run notebooks in order (01 through 09)
jupyter lab
```

Raw genomes and interim tool outputs are gitignored. The processed feature matrix and
CV groups are released via Zenodo on publication (see Data availability below).

**External data dependency:** `08_model_interpretation.ipynb` requires
`Supplementary_Data_S1.xlsx` at the project root. This file contains the published
*Acinetobacter* holdout cohort from Muthuraman et al. (2026) and is available as
journal supplementary material or via Zenodo.

---

## Notebooks

| Notebook | Phase | Contents |
|----------|-------|----------|
| `01_feature_engineering` | 3 | Parse DefenseFinder, PADLOC, ResFinder, ISEScan, BacMet outputs into tidy feature matrix |
| `02_eda` | 4 | Per-species summary statistics, within/between-species variance, RESTRICT/FACILITATE correlation matrices |
| `03_dimensionality_reduction` | 5 | PCA (scree, loadings), UMAP (Jaccard metric, n_neighbors sweep), t-SNE |
| `04_phylogenetic_grouping` | 6 | Mash sketching, pairwise distances, hierarchical clustering, MLST concordance validation |
| `05_baseline_classifier` | 7 | Null baseline, Logistic Regression Q1 + Q2, learning curves |
| `06_random_forest` | 8 | RF Q1 + Q2, hyperparameter tuning, Gini + permutation + SHAP importance |
| `07_gradient_boosting` | 9 | XGBoost and LightGBM Q1 + Q2, early stopping, calibration, McNemar tests |
| `08_model_interpretation` | 10 | Per-class SHAP summary plots, holdout validation, alignment with published findings |
| `09_unsupervised_archetypes` | 11 | K-means, hierarchical clustering, silhouette + gap statistic, archetype profiles |

---

## Repository structure

```
eskape-defence-ml/
├── notebooks/              # Jupyter notebooks (01-09, numbered by phase)
├── src/                    # Python modules (features, models, evaluation, viz)
├── workflow/               # Snakemake pipeline extending the published Acinetobacter pipeline
├── config/
│   ├── species.yaml        # NCBI accession lists per species
│   └── params.yaml         # Pipeline and ML parameters
├── data/
│   ├── raw/                # Downloaded genomes (gitignored)
│   ├── interim/            # Tool outputs: DefenseFinder, PADLOC, ResFinder, Mash (gitignored)
│   └── processed/          # feature_matrix_3335.parquet, cv_groups_3335.parquet
│                           # (gitignored; deposited to Zenodo on release)
├── results/
│   ├── figures/            # PNG figures per phase (tracked in git)
│   └── models/             # Serialised RF/XGB/LGBM models (gitignored; Zenodo on release)
├── docs/
│   ├── mash_s10000_sensitivity.md # Mash sketch-size sensitivity check
│   └── expected_reviewer_comments/ # Rebuttal-ready notes on anticipated review points
└── environment.yml
```

---

## Data availability

Raw genomes are available from NCBI RefSeq under the accessions listed in
`config/species.yaml`. Processed data (feature matrix, CV groups, serialised models)
will be deposited to Zenodo on publication and linked here.

---

## Citation

If you use this repository, please cite:

**This work (in submission):**
> Muthuraman V et al. (2026) Defence system repertoires encode species identity and
> antibiotic resistance burden across ESKAPE pathogens: a machine learning analysis.
> *In submission.*

**Parent study:**
> Muthuraman V et al. (2026) Niche-specific defence system selection in *Acinetobacter* spp.
> *Journal of Applied Microbiology.*

**Published pipeline:**
> [acinetobacter-defence-pipeline](https://github.com/vikos77/acinetobacter-defence-pipeline)
