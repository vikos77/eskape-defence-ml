# Results

## 1. Defence system profiles classify ESKAPE species with high balanced accuracy

All four supervised classifiers substantially exceeded the stratified null baseline
(balanced accuracy [BA] = 0.167 for six equiprobable classes). Random Forest (RF)
achieved a BA of 0.874 [95% CI: 0.852-0.895] under phylogenetically grouped
five-fold cross-validation (95 phylogroups, n=878 genomes). Logistic Regression (LR)
performed comparably (BA = 0.882 [0.838-0.921]); McNemar's test confirmed statistical
equivalence between RF and LR (b=35, c=42, p=0.494). RF was retained as the primary
classifier because its tree structure supports direct SHAP TreeExplainer attribution
(Q4). Both gradient boosting methods were inferior to RF: XGBoost BA = 0.806
[0.776-0.878], LightGBM BA = 0.860 [0.814-0.897]; McNemar tests showed RF
significantly outperformed XGBoost (b=21, c=83, p<0.001) and LightGBM (b=25, c=65,
p<0.001). The RF macro-F1 was 0.875 [0.852-0.896].

Per-class recall varied substantially across species (Figure X). *S. aureus* and
*E. faecium* were classified with near-perfect recall (SA: 0.993; EF: 0.953),
consistent with distinctive Gram-positive defence repertoires that are largely absent
from Gram-negative genomes. *A. baumannii* showed the lowest recall (0.700), explained
by the contraction of defence gene content in IC2 clones, which produces a
near-featureless profile that partially overlaps with other species at low defence
counts. Per-class recall for the remaining species was: PA 0.893, KP 0.856, EC 0.849
(Table X).

External validation on the 33 published *A. baumannii* genomes from Muthuraman et al.
(2026) that were held out from all training (C3 holdout) confirmed the primary result:
overall BA = 0.902, with AB recall = 0.939. The higher AB recall in the holdout
relative to the cross-validated estimate (0.700) reflects the holdout being drawn from
a broader set of *A. baumannii* lineages rather than exclusively IC2 clones.


## 2. Defence system profiles predict ARG burden in three of six ESKAPE species

For Q2, separate binary classifiers (RF and XGBoost) were trained per species to
distinguish high-ARG genomes (top tertile) from low-ARG genomes (bottom tertile),
using within-species defence system presence/absence features. Statistical significance
was assessed by one-sample t-test of fold-level balanced accuracies against a null of
BA = 0.5, with Benjamini-Hochberg correction applied across the six species (q = 0.05).

Defence system profiles yielded significantly above-chance prediction in three species.
For *E. cloacae* complex, RF BA = 0.753 [0.661-0.837] (p_adj = 0.0058) and XGBoost
BA = 0.824 (p_adj < 0.001), with RF AUROC = 0.853 and XGBoost AUROC = 0.872. For
*K. pneumoniae*, RF BA = 0.707 [0.642-0.763] (p_adj = 0.011) and XGBoost BA = 0.789
(p_adj = 0.013), AUROC = 0.896 and 0.924 respectively. For *P. aeruginosa*, RF BA =
0.677 [0.580-0.765] (p_adj = 0.014), AUROC = 0.722; XGBoost did not reach significance
(BA = 0.568, p_adj = 0.137).

The remaining three species did not exceed the null. For *E. faecium*, BA = 0.489
(p_adj = 0.120), indistinguishable from chance despite an AUROC of 0.892; the high
AUROC with near-chance BA reflects an unstable class boundary arising from the seven
available EF phylogroups, which constrains fold-level sample size and label balance.
For *S. aureus*, BA = 0.514 (p_adj = 0.188), AUROC = 0.566. For *A. baumannii*,
BA = 0.489 (p_adj = 0.805), AUROC = 0.487, consistent with IC2 clonal compression
eliminating the within-species ARG variance that would otherwise generate a
classifiable signal.

[Note for revision: AUROC 95% CIs for Q2 were not computed at the cluster bootstrap
level; fold-level bootstrap CIs should be added before submission.]


## 3. ESKAPE defence profiles form a continuum rather than discrete archetypes

K-means clustering of the full 878-genome dataset across K = 2 to 12 produced maximum
silhouette scores below 0.10 at all values of K (best K = 10, silhouette = 0.068). The
gap statistic identified K = 1 as optimal, confirming the absence of any meaningful
cluster structure. Adjusted Rand Index between the K-means K = 10 partition and species
labels was 0.383, but this was attributable to IC2 clonal inflation: when clustering
was repeated on the 95 phylogroup representatives (one genome per phylogroup),
the best K dropped to 2 (silhouette = 0.194) and ARI against species collapsed to
-0.004, indistinguishable from random. Agreement between K-means and hierarchical
Ward linkage at K = 10 was ARI = 0.468, below the 0.70 threshold considered evidence
of stable partition structure. ARI against ARG burden tertile was 0.031 for the full
dataset and near zero for the dereplicated set, confirming that whatever partitioning
structure exists does not capture ARG burden variation. ESKAPE defence profiles form a
continuum in feature space; there are no discrete defence archetypes recoverable across
species.


## 4. SHAP attribution identifies SspBCDE and RM as primary classification drivers

SHAP TreeExplainer was applied to the RF Q1 model to identify the features driving
species classification. Globally, the top-ranked feature by mean absolute SHAP value
was dp_df_Abi2, followed by dp_df_FS_Sma and dp_padloc_PDC-S13 (Figure X). Among
the published *Acinetobacter*-characterised systems, SspBCDE ranked 10th globally
(mean |SHAP| = 0.014) and first for the *A. baumannii* class specifically; RM Type I
ranked 17th globally and fourth for AB (Table X). The negative co-occurrence of RM and
SspBCDE (log2 OR = -6.09, p_adj = 1.5 x 10^-9) observed in Muthuraman et al. (2026)
was reproduced in the ESKAPE dataset: Fisher's exact test on all 878 genomes confirmed
the same direction and magnitude.

The four published systems with significant pairwise associations in the *Acinetobacter*
dataset (RM, SspBCDE, Gao_Qat, PD-T4-5) all appeared in the ML global SHAP top 30,
confirming partial cross-genus generalisation of the published signal. SspBCDE ranked
first for AB and GaoQat ranked 26th globally. The SHAP direction for SspBCDE was
positive for AB classification (SspBCDE presence increases the probability of
predicting *A. baumannii*), reflecting the dominance of IC2 SspBCDE-only genomes in
the AB class. RM direction was negative for AB (RM presence decreases AB probability),
consistent with the published finding that IC2 clones are RM-depleted.

For non-AB species, the primary SHAP drivers were species-specific systems not
previously characterised in *Acinetobacter*: AbiH was top-ranked for *E. faecium*,
and CRISPR-Cas was the primary driver for *P. aeruginosa*, consistent with the
unusually high CRISPR-Cas prevalence observed in clinical PA isolates in this dataset.


## 5. RM is effectively binary in this dataset and restriction predicts plasmid-mediated ARG classes

**Test A (RM copy number vs binary presence).** RM Type I was the only subtype with
substantial multi-copy variation (31% of genomes carrying more than one copy). In all
three species with significant Q2 signal, substituting the RM Type I copy-number
feature (dc_RM_Type_I) for the binary presence feature (dp_RM_Type_I) degraded AUROC:
EC -0.069, KP -0.040, PA -0.142. SHAP attribution confirmed that dc_RM_Type_I ranked
third among all 265 features for KP and first for PA when the count feature was active,
but the degradation in AUROC indicates that count introduces noise relative to the
binary signal. RM presence/absence is the operative encoding for this dataset; copy
number does not add predictive value.

**Test B (mechanism-class-specific ARG burden).** Seven (species x ARG class) cells
reached BH-corrected significance (q = 0.05) across 21 cells passing the 30/30 sample
floor (Table X). Significant cells were: *K. pneumoniae*/aminoglycoside (AUROC = 0.803
[0.735-0.879], p_adj = 0.0077), *P. aeruginosa*/beta-lactam (0.793 [0.677-0.893],
p_adj = 0.011), *E. cloacae*/beta-lactam (0.750 [0.661-0.826], p_adj = 0.011),
*K. pneumoniae*/beta-lactam (0.676 [0.592-0.736], p_adj = 0.012),
*K. pneumoniae*/sulfonamide (0.817 [0.675-0.949], p_adj = 0.020),
*E. faecium*/macrolide-MLSB (0.743 [0.587-0.912], p_adj = 0.049), and
*E. faecium*/tetracycline (0.814 [0.617-1.000], p_adj = 0.048). Significance in
*E. faecium* for macrolide-MLSB and tetracycline, two predominantly plasmid-mediated
resistance classes in Gram-positive bacteria, extends the restriction signal beyond
Gram-negative species.

For the pre-specified plasmid-mediated classes, SHAP attribution confirmed a
restrictive RM signal (negative SHAP on the high-ARG-burden class) in the
KP/aminoglycoside cell, driven by RM Type II and Type IIG subtypes rather than Type I.
For the beta-lactam cells, RM SHAP values were near zero or weakly positive, suggesting
that the beta-lactam signal is driven by features other than RM restriction. This
partial mismatch with the pre-specified directional prediction is reported without
post-hoc rationalisation; possible mechanistic explanations are addressed in the
Discussion.
