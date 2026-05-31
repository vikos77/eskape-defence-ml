# ESKAPE Defence Systems - ML Extension

Machine learning extension of Muthuraman et al. (2026), *Journal of Applied Microbiology*.

**Research question:** Does the RESTRICT/FACILITATE defence-system dichotomy identified in *Acinetobacter* generalise across ESKAPE pathogens?

## Published baseline

- 132 complete *Acinetobacter* genomes
- Key finding: RM systems restrict ARG/MGE acquisition; SspBCDE and Gao_Qat facilitate it
- Published pipeline: [acinetobacter-defence-pipeline](https://github.com/vikos77/acinetobacter-defence-pipeline)

## This extension

| Question | Method | Notebook |
|---|---|---|
| Q1: Can defence repertoire classify ESKAPE species? | Logistic Regression, Random Forest, XGBoost | 05, 06, 07 |
| Q2: Can defence profile predict high-ARG-burden genomes? | Binary RF/XGB, species-stratified, GroupKFold | 05, 06, 07 |
| Q3: Do pan-ESKAPE defence archetypes exist? | K-means, hierarchical clustering, silhouette | 09 |
| Q4: Which features drive classification? | SHAP, permutation importance | 08 |
| Q5 (sensitivity): Does RM count outperform binary presence? | Test A: dc_RM vs dp_RM in Q2 | 10 |
| Q6 (sensitivity): Is restriction ARG-class-specific? | Test B: mechanism-class RF per species | 10 |

## Dataset

- **878 genomes** across 6 ESKAPE species (900 downloaded; 22 excluded for MLST mismatch)
- Species breakdown: AB=150, EC=146, EF=150, KP=132, PA=150, SA=150
- Features: 265 defence system presence/absence (`dp_*`) columns used for classification
- Full feature matrix: 878 × 631 (dp_* + dc_* + genomic context columns)
- Phylogenetic control: 95 Mash-derived phylogroups (AB=13, EC=22, EF=7, KP=18, PA=26, SA=9)
- All CV: GroupKFold(5) grouping on phylogroups from Phase 6 onwards

## Phase progress

| # | Notebook | Topic | Status |
|---|---|---|---|
| 0 | — | Environment + repo scaffolding | **COMPLETE** |
| 1 | — | Literature ramp-up + pre-analysis plan | **COMPLETE** |
| 2 | — | Data acquisition (NCBI, 878 genomes × 6 species) | **COMPLETE** |
| 3 | 02_feature_engineering | Feature matrix construction (878 × 631) | **COMPLETE** |
| 4 | 01_eda | Exploratory data analysis | **COMPLETE** |
| 5 | 03_dimensionality_reduction | PCA, UMAP, t-SNE | **COMPLETE** |
| 6 | 04_phylogenetic_grouping | Mash distances → 95 phylogroups | **COMPLETE** |
| 7 | 05_baseline_classifier | Null baseline, Logistic Regression | **COMPLETE** |
| 8 | 06_random_forest | Random Forest Q1 + Q2 + SHAP | **COMPLETE** |
| 9 | 07_gradient_boosting | XGBoost Q1 + Q2 + calibration | **COMPLETE** |
| 10 | 08_model_interpretation | SHAP biological synthesis + holdout validation | **COMPLETE** |
| 11 | 09_unsupervised_archetypes | K-means archetypes + robustness (dereplicated) | **COMPLETE** |
| 12 | 10_phase12_sensitivity | RM count vs binary (Test A) + mechanism-class ARG (Test B) | **COMPLETE** |

## Key findings

### Phase 4 - EDA
- RESTRICT/FACILITATE is not pan-ESKAPE: in KP and PA, RM systems co-occur with ARGs on shared plasmids rather than acting as chromosomal gatekeepers.
- IME-ARG co-acquisition is universal across all 6 species (ρ = +0.65–0.77).

### Phase 5 - Dimensionality reduction
- Species separate cleanly in UMAP (Jaccard metric). AB far left (depauperate IC2 profile); EF far bottom-right (Gram-positive). KP and EC consistently adjacent (Enterobacterales).
- 103 PCs required for 80% variance — high-dimensional sparse feature space.

### Phase 6 - Phylogenetic grouping
- 95 phylogroups via Mash distance clustering. MLST concordance 99.1% (108/109 STs co-assigned).
- All subsequent CV uses GroupKFold(5) on phylogroups to prevent phylogenetic leakage.

### Phase 7 - Baseline classifiers (Logistic Regression)
- Q1 primary: BA = 0.837 [0.813–0.859].
- Q2 signal in EC and KP; AB Q2 AUROC inverted (low-ARG class predicted as high), replicating the published RESTRICT phenotype.

### Phase 8 - Random Forest
- Q1: RF BA = 0.878 [0.859–0.898]. Best params: max_depth=20, max_features=sqrt, min_samples_leaf=1, n_estimators=100.
- Per-class recall: SA=0.993, EF=0.953, PA=0.893, KP=0.856, EC=0.849, AB=0.700 (worst — IC2 clones confused with EF in feature space).
- Q2 significant species (RF, AUROC): PA=0.677 (RF wins over XGB).

### Phase 9 - Gradient boosting (XGBoost)
- Q1: XGB BA = 0.884 [0.863–0.901] — marginal gain over RF.
- Q2 significant species (XGB, AUROC): EC=0.872, KP=0.924. XGB primary for EC/KP; RF primary for PA.
- EC, KP significant under both models; PA significant under RF only (chromosomal ARG routes weaken XGB signal).

### Phase 10 — Model interpretation + holdout validation
- C3 holdout (33 published AB genomes): holdout BA=0.902, AB recall=0.939 vs CV recall=0.700. Gap explained by lower IC2 proportion in published cohort.
- SHAP Q1 rank 1: `dp_df_SspBCDE` (positive for AB, driven by IC2 dominance).
- Per-species SHAP rank 1: AB=SspBCDE, EF=AbiH (Gram-positive Abi), PA=CRISPR-Cas.
- Alignment with published Fisher's exact: agreement at top-3 (RM, SspBCDE, Gao_Qat). Divergence below rank 3 is expected — Fisher's estimates within-AB co-occurrence; SHAP estimates between-species discrimination. Different estimands, not a failure.

### Phase 11 — Unsupervised archetypes
- Full dataset (878 genomes): best K=10 by silhouette (score=0.068). ARI vs species=0.383 — ARTEFACTUAL (clonal inflation).
- Dereplicated (95 phylogroup representatives): best K=2, ARI vs species=−0.004 (near random).
- **Conclusion: ESKAPE defence profiles form a continuum, not discrete archetypes.**
- RESTRICT/FACILITATE recovered in full clustering: Cluster 2 (20 AB, RM-high/SspBCDE-low) = RESTRICT; Cluster 8 (79 AB, RM-low/SspBCDE-high) = FACILITATE (IC2). Requires within-AB clustering to study robustly.
- Q5b (defence + anti-defence + IS): K=2 split reflects genome complexity (large Gram-negatives vs small Gram-positives), not phage-permissive biology. IS burden nearly identical between clusters — phage-permissive hypothesis refuted; IS position (not count) is the correct proxy.

### Phase 12 — Sensitivity analyses (pre-registered 2026-05-27)

**Test A — RM count (dc_) vs binary presence (dp_) in Q2:**
- Only RM Type I has real count variation (31.2% of genomes have dc≠dp); Types II–IV are effectively binary (<5% differ).
- Substituting `dc_RM_Type_I` for `dp_RM_Type_I` **degraded** AUROC in all three species: EC=−0.069, KP=−0.040, PA=−0.142.
- Interpretation: binary RM presence captures the biologically relevant threshold effect. The RESTRICT signal saturates at presence/absence — a gate model, not a dose-response model.

**Test B — mechanism-class ARG burden (7 BH-significant cells, q=0.05):**

| Cell | AUROC [95% CI] | p_adj |
|---|---|---|
| KP / aminoglycoside | 0.803 [0.735–0.879] | 0.0077 |
| KP / sulfonamide | 0.817 [0.675–0.949] | 0.0202 |
| KP / beta-lactam | 0.676 [0.592–0.736] | 0.0116 |
| EC / beta-lactam | 0.750 [0.661–0.826] | 0.0107 |
| PA / beta-lactam | 0.793 [0.677–0.893] | 0.0112 |
| EF / macrolide_mlsb | 0.743 [0.587–0.912] | 0.0490 |
| EF / tetracycline | 0.814 | 0.0475 |

- Quinolone fails the 30/30 floor across all species (near-binary distribution) — consistent with predominantly chromosomal quinolone resistance and no RM gating.
- AB/aminoglycoside permanently excluded: GroupKFold(5) structurally infeasible due to IC2 clonal compression into 13 phylogroups.

**Test B SHAP direction — RM subtype finding:**
- Pre-registered prediction: `dp_RM_Type_I` negative in plasmid-mediated classes. Confirmed in 1/5 pre-specified cells (KP/aminoglycoside — via Type II, not Type I).
- **Restriction signal is in RM Type II and IIG, not Type I:** KP/aminoglycoside (Type II = −0.0039), EF/tetracycline (Type II = −0.0040), EF/macrolide_mlsb (Type IIG = −0.0052).
- PA/beta-lactam positive RM Type I SHAP (+0.0050) = genomic complexity confound; PA beta-lactam resistance is chromosomal (AmpC, OprD, efflux pumps) — no plasmid gate applies.
- **Exploratory finding:** RESTRICT principle extends to Gram-positive ESKAPE. EF tetracycline (tet(M) on Tn916) and macrolide/MLSB (erm(B) on Tn1545) are plasmid-borne in *E. faecium* — RM Type II/IIG restriction signal is biologically coherent.

## Setup

```bash
conda env create -f environment.yml
conda activate eskape-ml
pre-commit install
```

## Repository structure

```
eskape-defence-ml/
├── notebooks/          # Jupyter notebooks (01–10, numbered by phase)
├── data/
│   ├── raw/            # Downloaded genomes (gitignored — download via Snakemake)
│   ├── interim/        # Tool outputs (DefenseFinder, ResFinder, etc.; gitignored)
│   └── processed/      # feature_matrix.parquet, cv_groups.parquet (gitignored — Zenodo on release)
├── results/
│   ├── figures/        # PNG figures per phase (tracked in git)
│   ├── models/         # Serialised models (gitignored — Zenodo on release)
│   └── *.parquet       # Model output tables (gitignored — Zenodo on release)
├── docs/
│   ├── pre_analysis_plan.md
│   └── decisions.md    # All design decisions + audit log
└── config/             # species.yaml, params.yaml, arg_class_mapping.yaml
```

### External data dependency

`notebooks/08_model_interpretation.ipynb` (Phase 10 holdout validation) requires
`Supplementary_Data_S1.xlsx` at the project root. This file contains the published
*Acinetobacter* cohort metadata from Muthuraman et al. (2026) JAM and is available
as supplementary material from the journal or via the Zenodo release of this project.
It is not tracked in git (3.5 MB binary).

## Citation

Muthuraman V et al. (2026) Niche-specific defence system selection in *Acinetobacter* spp.
*Journal of Applied Microbiology*.
