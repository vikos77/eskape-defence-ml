# Methods

## 1. Genome selection and quality control

Complete genome assemblies for six ESKAPE species — *Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa*, and *Enterobacter cloacae* complex — were retrieved from NCBI RefSeq using the datasets CLI (v18.23.0) with a target of 150 genomes per species. Target sample size was chosen to match the scale of the published *Acinetobacter* cohort (n=132; Muthuraman et al., 2026) and to ensure balanced class representation in multi-class classifiers. For *E. cloacae* complex, accessions were drawn across six constituent taxids with per-member caps to avoid over-representation of any single member species; this complex was treated as a single classifier class throughout, following WHO priority pathogen classification (2024), with internal phylogenetic heterogeneity acknowledged as a pre-registered study limitation. Assembly quality was assessed with CheckM2 v1.0.2; genomes failing completeness <95% or contamination >5% thresholds were excluded. Genomes were additionally screened by multilocus sequence typing (MLST v2.23.0, PubMLST schemes); 22 accessions where MLST-derived species assignment was discordant with the deposited NCBI taxonomy were excluded. The final dataset comprised 878 complete genomes (*A. baumannii* n=150, *E. cloacae* complex n=146, *E. faecium* n=150, *K. pneumoniae* n=132, *P. aeruginosa* n=150, *S. aureus* n=150).

## 2. Defence system prediction and resistance gene annotation

Defence systems were predicted using DefenseFinder v2.0.1, with the AntiDefenseFinder module invoked via the `--antidefensefinder` flag, and PADLOC v2.0.0, applied to all 878 assemblies. The two tools employ partially overlapping but distinct HMM databases; using both maximises recall of the full defence repertoire. A non-redundant union of predictions was constructed: system types detected by only one tool were included once, and system types detected by both tools in the same genome were recorded as a single occurrence to prevent double-counting. CRISPR-Cas systems are detected independently by both tools, and a third dedicated tool was therefore not employed.

ARGs were identified using ResFinder v4.7.2 against the ResFinder database. Integrative and conjugative elements (ICEs) and integrative and mobilisable elements (IMEs) were detected by tBLASTn against the ICEberg2 protein database (threshold: percent identity 40%, query coverage 80%), identifying elements capable of horizontal ARG transfer between genomes. IS elements were detected using ISEScan v1.7.2.3 and annotated to IS family; IS elements act primarily as chromosomal transposition units and their burden reflects genome plasticity rather than directional ARG acquisition.

Acquired metal resistance genes were identified using AMRFinderPlus v4.2.7 (`--plus --organism` flags; database version 2026-03-24.1), retaining only hits with `Subtype == "METAL"`. BacMet2 tBLASTn was evaluated but excluded from analysis, as hits at any identity threshold were dominated by constitutive RND efflux pump homologs, producing a 6-fold Gram-negative/Gram-positive artefact unsuitable for cross-species comparison. A known crash in AMRFinderPlus v4.2.7 on specific HMM domain alignments affected 82 genomes (*S. aureus* 53/150, *E. faecium* 28/150, *A. baumannii* 1/150); these genomes were assigned zero metal resistance genes, representing missing data rather than confirmed absence, and metal resistance analyses for these two species should be interpreted with that caveat.

## 3. [Feature matrix construction — PENDING]

## 4. [Phylogenetic grouping — PENDING]

## 5. [Machine learning models and cross-validation — PENDING]

## 6. [Sensitivity analyses — PENDING]

## 7. [Statistical analysis and software — PENDING]
