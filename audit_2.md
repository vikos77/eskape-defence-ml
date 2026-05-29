Phase 10 & 11 Audit Report

  ---
  Phase 11 — Unsupervised Archetypes (09_unsupervised_archetypes.ipynb)

  ---
  CRITICAL — Findings that invalidate stated conclusions

  ---
  C1 — The dereplication robustness check shows the entire Q3 result is a clonal inflation artefact
  
  The notebook correctly executes the dereplication (1 representative per phylogroup, 878 → 95 genomes) and computes the verdict itself: SENSITIVE: ARI 
  difference ≥0.10 — clonal inflation affected full-dataset clustering.

  What the numbers actually show:

  ┌────────────────────────┬────────────────┬────────────────────┐
  │        Dataset         │ ARI vs species │ ARI vs ARG tertile │
  ├────────────────────────┼────────────────┼────────────────────┤
  │ Full 878-genome        │ 0.3829         │ 0.0306             │
  ├────────────────────────┼────────────────┼────────────────────┤
  │ Dereplicated 95-genome │ -0.0043        │ -0.0009            │
  └────────────────────────┴────────────────┴────────────────────┘

  ARI drops from 0.383 to -0.004 — not just a weakening, but to below-chance. The clustering on the dereplicated dataset is statistically indistinguishable from
  a random partition. The entire "10 archetypes with moderate species alignment" conclusion exists only because IC2 A. baumannii contributes 69 near-identical
  feature vectors, SA contributes redundant clonal lineages, and similarly for other species. Remove the clonal redundancy and there is nothing left.
  
  Biological reason this matters: The scientific claim in Q3 is "do genomes cluster by defence-system archetype independently of species?" The word
  "independently" requires that the result not be driven by clonal lineages, which are not independent observations. The robustness check correctly identifies
  the problem and then stops. It does not update the primary conclusion. The correct conclusion from this notebook — the one that belongs in the manuscript — is:
  
  ▎ "K-means clustering on the full 878-genome dataset identified 10 clusters (ARI=0.38). However, dereplication to one genome per phylogroup (n=95) reduced this
  ▎  ARI to essentially zero (ARI=-0.004), indicating that the cluster structure in the full dataset was driven entirely by clonal lineage redundancy rather than
  ▎  biologically meaningful defence-system archetypes."
  
  The manuscript language section in the notebook does not reflect this. It prepares language for an ARI of 0.3–0.7, which is appropriate only for the
  uncorrected result.

  ---
  C2 — The RESTRICT/FACILITATE "recovery" also disappears under dereplication
  
  The full-dataset result for AB looks compelling:
  - Cluster 8: 79 genomes, SspBCDE=0.949, RM_Type_I=0.038 → FACILITATE/IC2
  - Cluster 2: 20 genomes, SspBCDE=0.000, RM_Type_I=0.950 → RESTRICT
  
  But in the dereplicated dataset, all 13 AB representative genomes fall into a single cluster (cluster 1, SspBCDE=0.077, RM_Type_I=0.538 — no dichotomy). The 13
   AB phylogroup representatives do not split cleanly into RESTRICT vs FACILITATE because the IC2 lineage is now represented by a single genome, not 69.

  The full-dataset "recovery" is an artefact: the 69 near-identical IC2 vectors pulled a centroid to SspBCDE=0.949/RM=0.038, creating a visually convincing
  cluster. That is not biological signal — it is a population structure artefact.

  What this means for the manuscript: The claim "unsupervised clustering cross-validates the published RESTRICT/FACILITATE dichotomy using an orthogonal,
  label-free method" cannot be made as stated. The correct statement is: "The IC2 lineage is sufficiently numerous and distinct in defence profile that it forms
  its own cluster when clonal redundancy is present, consistent with the published depauperate-defence observation. However, this effect is a consequence of
  clonal overrepresentation, not an independent validation of the RESTRICT/FACILITATE principle."

  ---
  C3 — Q5b best K=2 is not a meaningful archetype discovery
  
  Q5b was designed to test whether "distinct phage-permissive archetypes emerge" when IS and anti-defence features are added. The result: K=2, ARI vs Q3=0.074.

  K=2 is not an archetype discovery — it is a binary split. The two clusters divide genomes into those with higher IS burden, more anti-defence systems, and
  higher ARG counts (n=211) vs the rest (n=667). This is structurally equivalent to "genomes with more mobile elements vs fewer." The question asked whether
  distinct archetypes emerge; a binary partition into "high MGE burden" and "low MGE burden" is not that.
  
  The 14pp jump in silhouette from Q3 (0.068) to Q5b K=2 (0.150) is misleading: K=2 always inflates silhouette relative to higher-K solutions because it captures
   the single dominant axis of variation (IS count is a continuous variable with one major gradient). This is the silhouette-maximising solution, not an
  ecologically meaningful one.
  
  The Q5b analysis cannot support the "phage-permissive archetype hypothesis" in its current form.

  ---
  HIGH PRIORITY

  ---
  H1 — Silhouette scores universally below 0.07: the data has no meaningful cluster structure

  Full K-sweep results:

  ┌─────┬───────────────┐
  │  K  │  Silhouette   │
  ├─────┼───────────────┤
  │ 2   │ 0.0227        │
  ├─────┼───────────────┤
  │ 3   │ 0.0394        │
  ├─────┼───────────────┤
  │ 6   │ 0.0681        │
  ├─────┼───────────────┤
  │ 10  │ 0.0684 (best) │
  └─────┴───────────────┘
  
  Silhouette below 0.25 is generally described as "no substantial cluster structure"; below 0.10 is effectively noise. The notebook correctly finds the
  silhouette peak but never states the absolute value of 0.068 indicates no meaningful structure. It computes, plots, and proceeds to interpret K=10 clusters as
  if they were real.
  
  The correct action upon seeing max silhouette = 0.068 is to note: "The silhouette scores across K=2–12 are all below 0.07, indicating that defence system
  repertoires do not naturally partition into distinct cluster groups in this feature space. The chosen K=10 represents the highest-scoring partition of an
  inherently unstructured dataset." This conclusion then leads directly to C1 (dereplication confirms this).
  
  H2 — K-means and hierarchical clustering agree on only 47% of genome assignments (ARI=0.47)

  Two clustering methods applied to the same data with the same K should agree substantially if the clusters are real. ARI=0.47 means K-means and Ward
  hierarchical clustering disagree on 53% of the genome assignments. For datasets with strong cluster structure, method agreement ARI typically exceeds 0.70. The
   0.47 here is another independent indicator (alongside silhouette) that the data lacks stable partition structure.
  
  H3 — ARI vs ARG tertile ≈ 0 across all three partitions

  ┌───────────────────┬────────────────────┐
  │     Partition     │ ARI vs ARG tertile │
  ├───────────────────┼────────────────────┤
  │ K-means K=10      │ 0.031              │
  ├───────────────────┼────────────────────┤
  │ Hierarchical K=10 │ 0.021              │
  ├───────────────────┼────────────────────┤
  │ K-means K=6       │ 0.030              │
  └───────────────────┴────────────────────┘
  
  All are essentially zero. Defence-system archetypes, as identified by unsupervised clustering, are completely uninformative for ARG burden. This is the most
  biologically direct answer to Q3's secondary question ("do archetypes map onto ARG burden?") and the answer is no.

  This finding must be prominently reported. It directly informs the Q4 synthesis: the SHAP feature importance identifies features that drive species
  classification (Q1) but those same features do not form ARG-burden-relevant archetypes in unsupervised space.

  H4 — Cluster 5 (192 genomes: AB/EC/PA mixed) has no biological interpretation

  Looking at the contingency table, Cluster 5 contains: 50 AB (33% of all AB), 51 EC (35% of EC), 7 EF, 15 KP, 69 PA (46% of all PA). This is the second-largest
  cluster (192/878 = 22% of all genomes). It encompasses the Gram-negative organisms that are neither IC2-type (cluster 8) nor purely PA (cluster 0) nor
  EC-dominant (cluster 4).
  
  Biologically, this cluster likely represents genomes with typical-for-Gram-negative but not species-specific defence profiles: common RM Type II, no CRISPR,
  moderate defence counts, no distinctive MGEs. These are "normal" Gram-negative genomes. The cluster has no distinctive defence signature — it is the default
  bucket. The notebook does not examine or interpret this cluster at all, yet it contains a large fraction of AB, EC, and PA genomes.
  
  ---
  MODERATE

  ---
  M1 — f-string bug in ARI comparison summary output
  
  In the ARI summary table, the second row shows literally Hierarchical Ward K={best_k} because the second list entry in the builder's code is a regular string,
  not an f-string (missing the f prefix). The first entry correctly evaluates to K-means K=10. This is a minor reproducibility bug — the table is technically
  correct in values but wrong in one label. Fix: add f prefix to the second entry.

  M2 — FEAT_COLS hardcoded in Phase 11 vs dynamically computed in Phase 10

  Phase 10 computes the taxonomic markers dynamically from the feature matrix:
  spec_score = sp_prev.std() / 0.5
  markers = spec_score[spec_score >= 0.70].index.tolist()
  
  Phase 11 hardcodes a 9-element list. If the feature matrix changes (e.g., a genome is added/removed, which shifts per-species prevalences slightly), Phase 10
  auto-recomputes the correct markers while Phase 11 silently uses stale ones. Since the notebook is fully executed and FEAT_COLS is 265 in both (consistent),
  this is currently not wrong — but it is fragile. Phase 11 should derive FEAT_COLS using the same dynamic computation.

  M3 — No gap statistic

  CLAUDE.md Phase 11 deliverables specify "Silhouette and gap-statistic for K selection." Gap statistic (comparing within-cluster variance to a null reference
  distribution of random data) was not computed. The gap statistic would have immediately indicated the absence of cluster structure (gap statistic peak at K=1
  for unstructured data). Its absence is particularly notable given that the silhouette scores are low enough that the gap statistic would likely have flagged
  K=1 as optimal.

  M4 — RESTRICT check uses only dp_RM_Type_I, not all RM subtypes

  The RESTRICT archetype is characterised by RM system presence broadly. The code takes rm_cols[0] which is dp_RM_Type_I. The feature matrix also contains
  dp_RM_Type_II, dp_RM_Type_IIG, dp_RM_Type_III. For a complete RESTRICT characterisation, the relevant measure is "any RM present" (logical OR across all four
  RM types, or a summary count). Reporting only Type I may undercount the RESTRICT signal in the RESTRICT cluster.
  
  ---
  OPTIONAL

  ---
  O1 — Euclidean distance on binary features is not the most appropriate metric
  
  K-means and Ward linkage both use Euclidean distance internally. For 265 binary presence/absence features, Jaccard distance (intersection over union of present
   features) is more appropriate: two genomes with 10 shared features but 200 absent ones would have near-zero Euclidean distance even if their defence profiles
  are completely different. CLAUDE.md Phase 4 comprehension check 1 explicitly asks about this: "Defence systems are sparse. What does this imply for distance
  metrics (Euclidean vs Jaccard vs Hamming)?" The choice of Euclidean is not justified anywhere.
  
  A sensitivity run with K-medoids using Jaccard distance would test whether the (already weak) cluster structure changes with a more appropriate metric.

  O2 — No random-seed sensitivity test for K-means

  K-means is stochastic. The notebook uses RANDOM_STATE=42 consistently but does not test whether the K=10 partition is stable across seeds. For unstructured
  data (which this appears to be), K-means results can vary substantially across seeds. Running K-means with seeds 0, 1, 2, 42, 99 and reporting the inter-seed
  ARI would quantify this instability.
  
  ---
  Phase 10 — Model Interpretation (08_model_interpretation.ipynb)

  ---
  HIGH PRIORITY
  
  ---
  H1 — Holdout AB recall = 0.939 vs CV recall = 0.700: the 33.9pp gap requires mechanistic explanation
  
  The C3 result on first reading looks like strong external validation. But the gap is so large it demands investigation before being interpreted as
  generalisation success.

  Looking at the holdout error pattern: 31/33 correct; 1 misclassified as EF, 1 as PA. In training CV, 22% of AB were predicted as EF (the IC2 confusion). In the
   holdout, only 1/33 (3%) is predicted as EF.

  The most likely explanation: the 33 published A. baumannii genomes in the holdout are predominantly non-IC2. IC2 genomes have SspBCDE-only profiles that look
  like EF to the classifier. If the holdout 33 are enriched for RESTRICT-archetype (RM+/SspBCDE-) AB, they would be easy to classify correctly — producing high
  recall without requiring the model to have "learned" the IC2 phenotype better.
  
  This is not a failure — it is a nuanced result. But the manuscript cannot state "holdout recall = 0.939, comparable to training CV" when it is actually 33.9pp
  higher. The correct framing is: "Holdout AB recall (0.939) exceeded training CV recall (0.700). Comparison of misclassification patterns suggests this reflects
   cohort composition: the holdout cohort appears enriched for non-IC2 A. baumannii with diverse defence repertoires, which are more readily classified than IC2
  clones."

  No statistical test was run. A simple binomial test (null: recall = 0.700, observed: 31/33) gives p < 0.01 — the holdout recall is significantly higher than CV
   recall. This should be reported and interpreted, not glossed over.

  H2 — Global beeswarm uses mean signed SHAP averaged over 6 classes — cancels class-specific signals

  The global beeswarm is generated as:
  shap_2d = shap_3d.mean(axis=2)   # (878, 265) — mean SHAP over 6 classes
  
  This averages signed SHAP values. A feature with +0.02 SHAP for AB and -0.02 SHAP for SA and near-zero for the other four classes averages to approximately
  zero — the feature appears unimportant globally even though it has strong opposing effects on two classes. SspBCDE, Gao_Qat, and RM are all expected to have
  opposing class effects (positive for AB, neutral or negative for others), so their apparent global importance is systematically suppressed.

  The correct approach for a global multiclass beeswarm is to average the absolute SHAP values over classes: shap_3d.mean(axis=2) → np.abs(shap_3d).mean(axis=2).
   This is what the global ranking (cell 2.2) correctly computes, but the beeswarm visual uses the signed average.

  H3 — PDC systems at global SHAP ranks 8–9 are absent from the alignment table — this is a missing finding

  The global SHAP top-20 includes dp_padloc_PDC-S02 (rank 8) and dp_padloc_PDC-M30 (rank 9). These are PADLOC-named phage defence systems that do not appear in
  the published paper's 20-system Fisher's exact analysis (the published analysis was AB-focused and used a different system naming scheme). They are not in the
  PUBLISHED_TO_DP mapping.
  
  This is not a bug — they genuinely have no counterpart in the published analysis. But the current alignment table silently omits them. The correct framing is:
  "The ML identified PDC-S02 and PDC-M30 (global SHAP ranks 8–9) as significant Q1 discriminators. These systems were not included in the published Fisher's
  exact analysis. Their biological role in ESKAPE defence architecture warrants follow-up." This is the most directly additive finding the ML layer makes — new
  candidate systems beyond what the published paper tested.

  H4 — Alignment scatter uses global SHAP rank but the published analysis was AB-only — the AB-class SHAP rank is the fair comparator

  The alignment scatter plot (x-axis: n significant Fisher's pairs from published AB analysis, y-axis: global SHAP rank) compares AB-specific published Fisher's
  to pan-ESKAPE SHAP. The global SHAP rank for SspBCDE is 10 because it matters globally across ESKAPE; for RM it is 17.

  But the published paper only ever looked at AB. Comparing the published AB result to the global SHAP rank introduces a category mismatch — the published result
   captures "is this system important in AB?" while the global SHAP captures "does this system discriminate any ESKAPE species?" The AB-class SHAP rank column
  (SspBCDE=1, RM=4) is the appropriate comparator for the published Fisher's exact analysis. The alignment scatter should use AB-class SHAP rank on the y-axis.
  
  ---
  MODERATE

  ---
  M1 — SHAP computed on full 878-genome dataset with clonal inflation, not quantified
  
  The limitation is acknowledged (limitation #4 in biological synthesis): "SHAP captures lineage-level signals alongside biology-level signals." But it is not
  quantified. The SspBCDE SHAP signal for AB is partly inflated by the fact that 69/150 AB genomes are near-identical IC2 vectors all contributing SHAP≈+X for
  SspBCDE. A comparison between SHAP computed on full dataset vs dereplicated 95-genome dataset would quantify this inflation. This was not done.

  M2 — 2 zero-defence holdout genomes not investigated

  The output notes: "Missing features (all zeros per genome): 2 genomes have zero defence calls." What are these genomes? A genome with all-zero features will be
   predicted by the model as whichever ESKAPE class has the highest prior for zero-defence profiles. If these are two of the 33 true AB (which they probably
  aren't, given 31/33 are correct), their predictions would be unreliable. The accessions should be identified and checked against the S2 metadata.
  
  M3 — Self-OR = 100 for all published systems: truncation artefact not flagged

  The published S16 Fisher's exact table shows self_or = 100.0 for every system. This is an Excel display artefact — odds ratios above some threshold are capped
  at 100 in the source table (the true self-comparison OR is infinite or undefined). The code processes this column as-is. This doesn't affect the significant
  pairs analysis (which only uses non-self comparisons), but the self_or column in the published summary table is meaningless and should be removed or annotated.
  
  ---
  Summary Table

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────┬──────────┐
  │ ID  │                                                Finding                                                 │ Phase │ Priority │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ C1  │ Dereplication collapses ARI from 0.383 to -0.004: Q3 is clonal artefact                                │ Ph11  │ CRITICAL │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ C2  │ RESTRICT/FACILITATE recovery disappears post-dereplication: all 13 AB representatives cluster together │ Ph11  │ CRITICAL │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ C3  │ Q5b K=2 is a binary MGE-burden split, not an archetype discovery                                       │ Ph11  │ CRITICAL │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H1  │ Silhouette scores all <0.07: no meaningful cluster structure — never stated explicitly                 │ Ph11  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H2  │ KM vs HC agreement ARI=0.47: no stable partition structure                                             │ Ph11  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H3  │ ARI vs ARG tertile ≈ 0: archetypes uninformative for ARG burden                                        │ Ph11  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H4  │ Cluster 5 (192 genomes: AB/EC/PA mixed) not biologically interpreted                                   │ Ph11  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H5  │ Holdout AB recall 0.939 >> CV 0.700: gap requires mechanistic explanation, no binomial test            │ Ph10  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H6  │ Global beeswarm uses signed mean SHAP: cancels opposing class effects                                  │ Ph10  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H7  │ PDC-S02, PDC-M30 at SHAP ranks 8–9 absent from alignment table: missing finding                        │ Ph10  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H8  │ Alignment scatter uses global SHAP rank vs AB-only Fisher's: AB-class SHAP rank is the fair comparator │ Ph10  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M1  │ f-string bug: 'Hierarchical Ward K={best_k}' outputs literally                                         │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M2  │ FEAT_COLS hardcoded in Ph11 vs dynamically computed in Ph10                                            │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M3  │ No gap statistic (pre-registered deliverable)                                                          │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ H8  │ Alignment scatter uses global SHAP rank vs AB-only Fisher's: AB-class SHAP rank is the fair comparator │ Ph10  │ HIGH     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M1  │ f-string bug: 'Hierarchical Ward K={best_k}' outputs literally                                         │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M2  │ FEAT_COLS hardcoded in Ph11 vs dynamically computed in Ph10                                            │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M3  │ No gap statistic (pre-registered deliverable)                                                          │ Ph11  │ MODERATE │
  │ M1  │ f-string bug: 'Hierarchical Ward K={best_k}' outputs literally                                         │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M2  │ FEAT_COLS hardcoded in Ph11 vs dynamically computed in Ph10                                            │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M3  │ No gap statistic (pre-registered deliverable)                                                          │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M4  │ RESTRICT check uses dp_RM_Type_I only, not all RM subtypes                                             │ Ph11  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M5  │ SHAP clonal inflation not quantified (acknowledged but unaddressed)                                    │ Ph10  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M6  │ 2 zero-defence holdout genomes not investigated                                                        │ Ph10  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ M7  │ Self-OR=100 truncation artefact in S16 not flagged                                                     │ Ph10  │ MODERATE │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ O1  │ Euclidean distance on binary features (Jaccard is more appropriate)                                    │ Ph11  │ OPTIONAL │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────┼──────────┤
  │ O2  │ No random-seed sensitivity test for K-means                                                            │ Ph11  │ OPTIONAL │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────┴──────────┘

  ---
  Recommended action order

  Before any manuscript writing on Q3:
  1. The CRITICAL findings change the Q3 narrative fundamentally. The correct result is a negative finding: "defence repertoires do not form robust archetypes
   independent of clonal structure." This should be stated clearly in the biological synthesis section. The manuscript language section needs to be rewritten.
  2. The RESTRICT/FACILITATE claim in the notebook must be re-framed as C2 specifies: it is an observation about IC2 clonal dominance, not an independent
  cross-validation.
  3. Q5b: either increase the K-sweep range to K=2–15 and argue for a higher-K interpretation, or reframe Q5b as "IS/anti-defence burden defines a gradient,
  not discrete archetypes" — which is equally publishable as a negative result.

  Phase 10 before submission:
  4. H5: Run binomial test for holdout recall. Examine IC2 vs non-IC2 composition in the holdout 33 AB.
  5. H7: Add PDC-S02/PDC-M30 to the alignment table with a note that these are ML-identified systems beyond the published 20-system analysis.
  6. H8: Regenerate the alignment scatter using AB-class SHAP rank on y-axis.
  7. H6: Change the global beeswarm to use np.abs(shap_3d).mean(axis=2) instead of shap_3d.mean(axis=2).
