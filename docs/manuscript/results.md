# Results

## 1. Defence system profiles classify ESKAPE species with high balanced accuracy

All four supervised classifiers substantially exceeded the stratified null baseline
(balanced accuracy [BA] = 0.167 for six equiprobable classes; Table 1). Random Forest
(RF) achieved a BA of 0.874 [95% CI: 0.852-0.895] under phylogenetically grouped
five-fold cross-validation (95 phylogroups, n=878 genomes). Logistic Regression (LR)
performed comparably (BA = 0.882 [0.838-0.921]); McNemar's test confirmed statistical
equivalence between RF and LR (b=35, c=42, p=0.494). RF was retained as the primary
classifier because its tree structure supports direct SHAP TreeExplainer attribution
(Results Section 4). Both gradient boosting methods were inferior to RF: XGBoost
BA = 0.806 [0.776-0.878], LightGBM BA = 0.860 [0.814-0.897]; McNemar tests showed RF
significantly outperformed XGBoost (b=21, c=83, p<0.001) and LightGBM (b=25, c=65,
p<0.001). The RF macro-F1 was 0.875 [0.852-0.896].

Per-class recall varied substantially across species (Fig. 1; Table 2). *S. aureus*
and *E. faecium* were classified with near-perfect recall (SA: 0.993; EF: 0.953),
consistent with distinctive Gram-positive defence repertoires that are largely absent
from Gram-negative genomes. *A. baumannii* showed the lowest recall (0.700), explained
by the contraction of defence gene content in IC2 clones, which produces a
near-featureless profile that partially overlaps with other species at low defence
counts. Per-class recall for the remaining species was: PA 0.893, KP 0.856, EC 0.849.

External validation on the 33 published *A. baumannii* genomes from Muthuraman et al.
(2026) that were held out from all training (C3 holdout) confirmed the primary result, returning overall BA = 0.902 and AB recall = 0.939. The higher AB recall in the holdout
relative to the cross-validated estimate (0.700) reflects the holdout being drawn from
a broader set of *A. baumannii* lineages rather than exclusively IC2 clones.


## 2. Defence system profiles predict ARG burden in three of six ESKAPE species

Separate binary classifiers (RF and XGBoost) were trained per species to distinguish
high-ARG genomes (top tertile) from low-ARG genomes (bottom tertile), using
within-species defence system presence/absence features. Middle-tertile genomes were
excluded, leaving the following per-species Q2 sample sizes: EC n=97, KP n=86,
PA n=120, EF n=104, SA n=106, AB n=101. Statistical significance was assessed by
one-sample t-test of fold-level balanced accuracies against a null of BA = 0.5, with
Benjamini-Hochberg correction applied across the six species (q = 0.05).

Defence system profiles yielded significantly above-chance prediction in three of six
species (Fig. 2; Table 3). For *E. cloacae* complex, RF BA = 0.753 [0.661-0.837]
(p_adj = 0.0058), AUROC = 0.853 [0.778-0.917]. For *K. pneumoniae*, RF BA = 0.707
[0.642-0.763] (p_adj = 0.011), AUROC = 0.896 [0.797-0.969]. For *P. aeruginosa*,
RF BA = 0.677 [0.580-0.765] (p_adj = 0.014), AUROC = 0.722 [0.619-0.830]. XGBoost
exceeded RF in EC (BA = 0.824, p_adj < 0.001) and KP (BA = 0.789, p_adj = 0.013) but
not PA (BA = 0.568, p_adj = 0.137). AUROC CIs are 95% fold-level bootstrap
(2000 iterations); BA CIs are 95% cluster bootstrap on phylogroups.

The remaining three species did not exceed the null. For *S. aureus*, BA = 0.514
(p_adj = 0.188), AUROC = 0.566 [0.313-0.820]. For *A. baumannii*, BA = 0.489
(p_adj = 0.805), AUROC = 0.487 [0.362-0.679], consistent with IC2 clonal compression
eliminating the within-species ARG variance that would otherwise generate a
classifiable signal. For *E. faecium*, BA = 0.489 (p_adj = 0.120), AUROC = 0.892
[0.676-1.000]. The point estimate of 0.892 is arithmetically accurate but should be
read alongside its wide CI, as EF has only seven phylogroups, one fold was excluded
because it lacked both ARG-burden classes, and per-fold AUROC varied from 0.75 to 1.0
across the four valid folds. The fold-level bootstrap CI captures this variance; the
non-significant BA test (p_adj = 0.120) is the operative significance criterion.


## 3. ESKAPE defence profiles form a continuum rather than discrete archetypes

K-means clustering of the full 878-genome dataset across K = 2 to 12 produced maximum
silhouette scores below 0.10 at all values of K (best K = 10, silhouette = 0.068;
Fig. 3). The gap statistic independently identified K = 1 as optimal, confirming the
absence of meaningful cluster structure. Adjusted Rand Index between the K-means K = 10
partition and species labels was 0.383; however, this value was artefactual. When
clustering was repeated on the 95 phylogroup representatives (one genome per
phylogroup), best K dropped to 2 (silhouette = 0.194) and ARI against species collapsed
to -0.004, indistinguishable from random (Table 4). The full-dataset ARI of 0.383 was
driven by IC2 clonal inflation. The 150 *A. baumannii* genomes, dominated by near-
identical IC2 clones, formed a high-density cloud that was trivially separated by
K-means regardless of defence architecture. Agreement between K-means and hierarchical
Ward linkage at K = 10 was ARI = 0.468, below the 0.70 threshold considered evidence
of stable partition structure. ARI against ARG burden tertile was 0.031 for the full
dataset and near zero for the dereplicated set, confirming that the partitioning
structure does not capture ARG burden variation. ESKAPE defence profiles form a
continuum in feature space; there are no discrete defence archetypes recoverable after
phylogenetic dereplication.


## 4. SHAP attribution identifies SspBCDE and RM as primary classification drivers

SHAP TreeExplainer was applied to the primary RF Q1 model to identify the features
driving species classification (Fig. 4). Globally, the top-ranked feature by mean
absolute SHAP value was dp_df_Abi2 (0.023), followed by dp_df_FS_Sma (0.022) and
dp_padloc_PDC-S13 (0.020). Among the systems characterised in the published
*Acinetobacter* analysis (Muthuraman et al., 2026), SspBCDE ranked 10th globally
(mean |SHAP| = 0.014) and first for the *A. baumannii* class specifically; RM Type I
ranked 17th globally and fourth for AB (Table 5). The negative co-occurrence of RM and
SspBCDE observed in Muthuraman et al. (2026; log2 OR = -6.09, p_adj = 1.5 x 10^-9)
was reproduced in the ESKAPE dataset across all 878 genomes.

All four published systems with significant pairwise associations in the *Acinetobacter*
dataset (RM, SspBCDE, Gao_Qat, PD-T4-5) appeared in the ML global SHAP top 30,
confirming partial cross-genus generalisation of the published signal. The SHAP
direction for SspBCDE was positive for AB classification (SspBCDE presence increases
the probability of predicting *A. baumannii*), reflecting the dominance of IC2 clones
in the AB class. RM direction was negative for AB (RM presence decreases AB probability),
consistent with RM depletion in IC2 clones.

For non-AB species, the primary SHAP drivers were species-specific systems not
previously characterised in *Acinetobacter*: AbiH was the top-ranked driver for
*E. faecium*, and CRISPR-Cas for *P. aeruginosa*, consistent with the unusually high
CRISPR-Cas prevalence in clinical PA isolates in this dataset.


## 5. RM is effectively binary and restriction predicts plasmid-mediated ARG classes

**Test A (RM copy number vs binary presence).** RM Type I was the only subtype with
substantial multi-copy variation (31% of genomes carrying more than one copy, range
1-6). Substituting the RM Type I copy-number feature (dc_RM_Type_I) for binary
presence (dp_RM_Type_I) in the primary RF Q2 models degraded AUROC relative to the
primary RF AUROC baseline in all three significant species: PA -0.142 (0.722 to 0.580),
EC -0.050 (0.853 to 0.803), KP -0.012 (0.896 to 0.884). The degradation was strongest
in PA and moderate in EC; the KP delta (-0.012) was minimal. SHAP attribution in the
Test A models ranked dc_RM_Type_I third among 265 features for KP and first for PA,
yet AUROC still fell in both, indicating that copy-number adds noise relative to the
binary threshold signal. RM presence/absence is the operative encoding for this dataset;
the conclusion holds most strongly for PA and EC.

**Test B (mechanism-class-specific ARG burden).** Seven (species x ARG class) cells
reached BH-corrected significance (q = 0.05) across eight cells passing the 30/30
sample floor (Table S1; Fig. 5). One additional cell, *A. baumannii*/aminoglycoside,
passed the floor numerically but was excluded prior to modelling because IC2 clonal
lineages dominate the high-ARG tertile and their concentration in a subset of the 13 AB
phylogroups would confound tertile label with clonal ancestry. Quinolone resistance,
pre-specified as a negative control because it is predominantly chromosomal, failed
the 30/30 floor in all six species, consistent with near-binary count distributions.
Significant cells were: *K. pneumoniae*/aminoglycoside (AUROC = 0.803 [0.735-0.879],
p_adj = 0.0077), *P. aeruginosa*/beta-lactam (0.793 [0.677-0.893], p_adj = 0.011),
*E. cloacae*/beta-lactam (0.750 [0.661-0.826], p_adj = 0.011),
*K. pneumoniae*/beta-lactam (0.676 [0.592-0.736], p_adj = 0.012),
*K. pneumoniae*/sulfonamide (0.817 [0.675-0.949], p_adj = 0.020),
*E. faecium*/macrolide-MLSB (0.743 [0.587-0.912], p_adj = 0.049), and
*E. faecium*/tetracycline (0.814 [0.617-1.000], p_adj = 0.048). Significance in
*E. faecium* for macrolide-MLSB and tetracycline, two predominantly plasmid-mediated
resistance classes in Gram-positive bacteria, extends the restriction signal beyond
Gram-negative species.

SHAP attribution confirmed a restrictive RM signal (negative SHAP on the high-ARG-burden
class) in three of the seven significant cells: KP/aminoglycoside (RM Type II,
mean signed SHAP = -0.0039), EF/macrolide-MLSB (RM Type IIG, -0.0052), and
EF/tetracycline (RM Type II, -0.0040). In all three, the restriction signal was carried
by Type II or Type IIG rather than Type I, which showed SHAP values near zero. For the
four remaining significant cells (three beta-lactam cells and KP/sulfonamide), RM SHAP
values were near zero or weakly positive, indicating that the signal in these cells is
driven by features other than canonical RM restriction; possible mechanistic explanations
are addressed in the Discussion.
