# ESKAPE Defence Systems - ML Perspective

> Machine learning reveals defence systems as one component of a broader genomic
> architecture underlying species identity and resistance-gene ecology across ESKAPE
> pathogens.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![conda](https://img.shields.io/badge/env-eskape--ml-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Background

Bacterial defence systems are proposed barriers to horizontal antibiotic resistance
gene (ARG) acquisition, yet some co-occur with ARGs on mobile elements, suggesting
facilitation rather than restriction. How much defence-system content alone explains
about species identity and ARG burden, relative to a genome's broader ARG and
mobile-element content, has not been tested across multiple pathogens.

This repository applies supervised and unsupervised machine learning to 3,335
complete and high-quality genomes spanning all six ESKAPE species. Four research
questions (Q1–Q4) were pre-registered before any model training. A fifth question
(Q3b) arose post hoc during analysis: whether IS element family composition adds
independent clustering signal to the defence feature space, after applying the same
taxonomic marker filter used in Q1/Q3.

---

## Pipeline

```
NCBI genomes (ncbi-datasets-cli)
        |
        v
Quality control (CheckM2 >= 95% completeness; MLST species confirmation)
        |
        v
Defence annotation       ARG detection    HMRG detection       IS detection
DefenseFinder + PADLOC    ResFinder       AMRFinderPlus         ISEScan
                                          (--plus, METAL only)
        |                     |                |                   |
        +---------------------+----------------+-------------------+
                                  |
                                  v
                     Feature matrix  (01_feature_engineering)
                     3,335 genomes x 828 columns
                     (367 dp_* defence presence, 367 dc_* copy number,
                      41 anti-defence, ARG/HMRG/IS counts, genomic context)
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
     +--------+-----------+-----------+------------+
     |        |           |           |            |           
     v        v           v           v            v           
    Q1       Q2          Q3          Q3b           Q4
 Species   ARG burden  Defence-   IS element     SHAP
classif.   prediction  only       block          attribution
 LR/RF/    RF per      archetype  comparison     (08)
XGB/LGBM   species     K-means    (09 §5)
 (05-07)   (05-07)     (09)
```

---

## Key results

| Question | Result |
|----------|--------|
| Q1: Species classification from defence profiles | RF balanced accuracy = 0.900 [95% CI: 0.871-0.923] under grouped CV (LR 0.911, XGB 0.904, LGBM 0.909); null baseline = 0.167. McNemar's test found all four classifiers statistically indistinguishable. External holdout validation (n=180, 30 ST-diverse assemblies per species, zero training overlap): BA = 0.944 [0.911-0.972]. |
| Q2: ARG burden prediction within species | BH-significant in 4/6 species (*E. cloacae*, *K. pneumoniae*, *E. faecium*, *P. aeruginosa*; AUROC 0.781-0.824); null in *A. baumannii* and *S. aureus*. Robust drivers are predominantly facilitative, from named and uncharacterised systems at comparable rates. |
| Q3: Pan-ESKAPE defence archetypes | Continuum, not discrete clusters; silhouette < 0.10 throughout, gap statistic favours K=1; full-dataset ARI vs species (0.362) collapses to 0.102 on dereplicated phylogroup representatives, confirming clonal-inflation artefact. |
| Q3b: IS element block comparison | IS element family composition (19 families after spec_score filter) adds genuine clustering signal to the defence feature space. Defence-only ARI = 0.219 [0.059-0.320]; IS-only ARI = 0.237 [0.150-0.341]; defence + IS ARI = 0.392 [0.192-0.484] (gain +0.172 over defence alone). All three conditions p < 0.01 (1000-permutation test, 309 dereplicated genomes, K=6 forced). |
| Q4: SHAP attribution | RM_Type_IV is the top global classification driver; uncharacterised PADLOC candidate clusters are the single strongest per-species driver in 3 of 6 species (EC, KP, PA), reported as statistically real with no mechanistic claim attached. |

All models evaluated under phylogenetically grouped 5-fold cross-validation
(309 Mash-derived phylogroups; MLST concordance 92.4%).

---

## Dataset

| Property | Value |
|----------|-------|
| Genomes | 3,335 (3,460 downloaded; 125 excluded by MLST quality filter) |
| Species | AB n=600, EC n=507, EF n=524, KP n=504, PA n=600, SA n=600 |
| Defence features (dp_*) | 367 binary presence/absence columns (359 used in Q1/Q3 after removing 8 near-exclusive species markers) |
| Full feature matrix | 3,335 x 828 (defence presence + copy number + anti-defence + ARG/HMRG/IS counts + genomic context) |
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

Raw genomes and interim tool outputs are gitignored. The processed feature matrix,
CV groups, holdout feature matrix, and serialised models are released via Zenodo
on publication (see Data availability below).

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `01_feature_engineering` | Parse DefenseFinder, PADLOC, ResFinder, AMRFinderPlus, ISEScan outputs into tidy feature matrix |
| `02_eda` | Per-species summary statistics, within/between-species variance, defence-ARG correlation matrices |
| `03_dimensionality_reduction` | PCA (scree, loadings), UMAP (Jaccard metric, n_neighbors sweep), t-SNE |
| `04_phylogenetic_grouping` | Mash sketching, pairwise distances, hierarchical clustering, MLST concordance validation |
| `05_baseline_classifier` | Null baseline, Logistic Regression Q1 + Q2, learning curves |
| `06_random_forest` | RF Q1 + Q2, hyperparameter tuning, Gini + permutation + SHAP importance |
| `07_gradient_boosting` | XGBoost and LightGBM Q1 + Q2, early stopping, calibration, McNemar tests |
| `08_model_interpretation` | Per-class SHAP summary plots, feature dependence plots; external 6-species holdout validation (n=180, BA = 0.944 [0.911-0.972]) |
| `09_unsupervised_archetypes` | K-means, hierarchical clustering, silhouette + gap statistic, defence archetype profiles (Q3); IS element block comparison (Q3b, §5) |

---

## Repository structure

```
eskape-defence-ml/
├── notebooks/              # Jupyter notebooks (01-09 primary; 10 supplementary, gitignored)
├── src/                    # Python modules (features, models, evaluation, viz)
├── workflow/                # Snakemake pipeline for genome download, QC, and annotation
├── config/
│   ├── species.yaml        # NCBI accession lists per species
│   └── params.yaml         # Pipeline and ML parameters
├── data/
│   ├── raw/                # Downloaded genomes (gitignored)
│   ├── interim/             # Tool outputs: DefenseFinder, PADLOC, ResFinder, Mash (gitignored)
│   └── processed/          # feature_matrix_3335.parquet, cv_groups_3335.parquet
│                           # (gitignored; deposited to Zenodo on release)
├── results/
│   ├── figures/             # PNG figures per phase (tracked in git)
│   └── models/             # Serialised RF/XGB/LGBM models (gitignored; Zenodo on release)
├── docs/
└── environment.yml
```

---

## Data availability

Raw genomes are available from NCBI RefSeq under the accessions listed in
`config/species.yaml`. Processed data (feature matrix, CV groups, holdout feature
matrix, serialised models) will be deposited to Zenodo on publication and linked here.

---

## Citation

If you use this repository, please cite:

> Muthuraman V et al. (2026) Machine learning reveals defence systems as one component
> of a broader genomic architecture underlying species identity and resistance-gene
> ecology across ESKAPE pathogens.
> *In submission.*
