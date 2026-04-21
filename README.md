# ESKAPE Defence Systems — ML Extension

Machine learning extension of Muthuraman et al. (2026), *Journal of Applied Microbiology*.

**Research question:** Does the RESTRICT/FACILITATE defence-system dichotomy identified in *Acinetobacter* generalise across ESKAPE pathogens?

## Published baseline

- 132 complete *Acinetobacter* genomes
- Key finding: RM systems restrict ARG/MGE acquisition; SspBCDE and Gao_Qat facilitate it
- Published pipeline: [acinetobacter-defence-pipeline](https://github.com/vikos77/acinetobacter-defence-pipeline)

## This extension

| Question | Method | Phase |
|---|---|---|
| Q1: Can defence repertoire classify species? | Random Forest, XGBoost | 6–8 |
| Q2: Can defence profile predict ARG burden? | Binary classifier, cross-species | 6–8 |
| Q3: Do defence archetypes exist across species? | UMAP, K-means | 5, 11 |
| Q4: Which features drive each model? | SHAP | 10 |

## Phases

0. Environment + repo scaffolding ← *current*
1. Literature ramp-up + pre-analysis plan
2. Data acquisition (NCBI, ~150 genomes × 6 species)
3. Feature matrix construction
4. Exploratory data analysis
5. Dimensionality reduction
6. Baseline classifiers
7. Random Forest
8. Gradient boosting
9. Phylogenetic control (critical)
10. Model interpretation + biological synthesis
11. Unsupervised archetypes + publication figures

## Setup

```bash
conda env create -f environment.yml
conda activate eskape-ml
pre-commit install
```

## Citation

Muthuraman V et al. (2026) Niche-specific defence system selection in *Acinetobacter* spp. *Journal of Applied Microbiology*.
