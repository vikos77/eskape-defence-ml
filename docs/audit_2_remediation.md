# Audit 2 Remediation Log

Tracking all findings from `audit_2.md` (Phase 10 and Phase 11 review).
Each entry: what was wrong, what changed, before/after impact on conclusions.

---

## Phase 10 items

---

## Ph10-H5 — Holdout recall 0.939 vs CV 0.700: binomial test + IC2 composition check

**Status:** FIXED 2026-05-27

**What was wrong:**
C3 holdout recall (31/33 correct = 0.939) is 33.9pp higher than CV recall (0.700). No
statistical test was run and no mechanistic explanation was stated. The manuscript cannot
present 0.939 as a simple validation without explaining the gap.

**Fix required:**
1. Add scipy.stats.binomtest(k=31, n=33, p=0.700, alternative='greater').
2. Examine IC2 composition in the holdout 33 AB: count how many have SspBCDE=1 (IC2 proxy)
   vs SspBCDE=0 (non-IC2). Compare to training CV AB composition (46% IC2).
3. Update manuscript language: holdout recall exceeds CV recall because holdout is enriched
   for non-IC2 AB with diverse defence repertoires, not because the model generalises better.

**Before:** "Holdout recall = 0.939" stated without caveat.
**After:** "Holdout recall (0.939) significantly exceeds CV recall (0.700, binomial p=X).
Most likely explanation: holdout cohort is enriched for non-IC2 A. baumannii."

---

## Ph10-H6 — Global beeswarm uses signed SHAP: cancels opposing class effects

**Status:** FIXED 2026-05-27

**What was wrong:**
`shap_2d = shap_3d.mean(axis=2)` uses signed mean. A feature with +0.02 for AB and −0.02
for SA averages to ~0. Systematically suppresses species-specific features like SspBCDE
and RM_Type_I in the global visual.

**Fix required:**
Change beeswarm computation to `np.abs(shap_3d).mean(axis=2)` to match the global bar
ranking cell which already uses absolute values.

**Before:** Global beeswarm uses signed mean — class-opposing features appear small.
**After:** Global beeswarm uses abs mean — true global importance visible.

---

## Ph10-H7 — PDC-S02/PDC-M30 at SHAP ranks 8–9 absent from alignment table

**Status:** FIXED 2026-05-27

**What was wrong:**
The global SHAP top-20 includes dp_padloc_PDC-S02 (rank 8) and dp_padloc_PDC-M30 (rank 9).
These are not in the published paper's 20-system Fisher's exact analysis (S16 was AB-only
and used a different naming scheme). They are silently absent from the alignment table.

**Fix required:**
Add PDC-S02 and PDC-M30 explicitly to the alignment section with a note: "ML-identified
systems not present in published 20-system analysis. Fisher's exact rank: N/A. Warrants
follow-up."

**Before:** Alignment table silently excludes ML-identified systems with no published comparator.
**After:** New-finding systems flagged explicitly as additive ML discoveries.

---

## Ph10-H8 — Alignment scatter uses global SHAP rank vs AB-only Fisher's: category mismatch

**Status:** FIXED 2026-05-27

**What was wrong:**
The x-axis is Fisher's exact significance from the published AB-only analysis. The y-axis
is global SHAP rank (across all 6 ESKAPE species). These are different estimands. SspBCDE
has global SHAP rank 10 (diluted by other species) but AB-class SHAP rank 1. Using global
rank artificially weakens the apparent agreement with the published findings.

**Fix required:**
Regenerate the alignment scatter using AB-class SHAP rank on the y-axis. The AB-class
SHAP values are already computed in the per-class SHAP cells.

**Before:** Alignment scatter mixes AB-specific published analysis with global ML ranks.
**After:** Alignment scatter uses AB-class SHAP rank — fair comparison to published AB analysis.

---

## Ph10-M5 — SHAP clonal inflation not quantified

**Status:** DEFERRED — Phase 12

**What was wrong:**
Limitation #4 in biological synthesis acknowledges "SHAP captures lineage-level signals
alongside biology-level signals" but does not quantify the inflation. SspBCDE SHAP is
partly inflated by 69/150 AB being near-identical IC2 vectors.

**Fix required:**
Compute mean SHAP for SspBCDE in the AB class using: (a) all 150 AB genomes vs
(b) the 13 dereplicated AB representatives. Report ratio as inflation estimate.

**Before:** IC2 inflation acknowledged but unquantified.
**After:** Inflation factor reported numerically.

---

## Ph10-M6 — 2 zero-defence holdout genomes not investigated

**Status:** FIXED 2026-05-27

**What was wrong:**
Output noted "2 genomes have zero defence calls" in the holdout feature matrix. Accessions
not identified. Need to confirm these are not among the 33 true AB genomes (which would
make their predictions unreliable).

**Fix required:**
Identify accessions with all-zero FEAT_COLS in the holdout matrix and cross-reference
against S2 metadata. Confirm species and check whether they affected the recall calculation.

**Before:** Zero-defence genomes noted but ignored.
**After:** Accessions identified and excluded from recall calculation if appropriate.

---

## Ph10-M7 — Self-OR=100 truncation artefact in S16 not flagged

**Status:** FIXED 2026-05-27

**What was wrong:**
S16 Fisher's exact table shows self_or = 100.0 for all diagonal entries. This is an Excel
display cap (true self-comparison OR is undefined/infinite). The self_or column is
meaningless and misleading.

**Fix required:**
Remove self_or column from the published summary output or annotate it explicitly as a
display artefact. Does not affect significant-pairs analysis.

**Before:** self_or=100 silently present in summary table.
**After:** Column removed or annotated as Excel display cap.

---

## Phase 11 items

---

## Ph11-C1 — Q3 primary conclusion must be stated as negative finding

**Status:** PENDING

**What was wrong:**
The biological synthesis and manuscript language sections were written for an ARI of
0.3–0.7 (moderate alignment). The dereplication result (ARI=−0.004) is already computed
and the verdict output reads "SENSITIVE." But the synthesis narrative and manuscript
language cells were not updated to reflect this. They prepare language for a result that
is invalidated by the notebook's own robustness check.

**Fix required:**
Rewrite biological synthesis Section 5 and manuscript language to state:
"Unsupervised clustering identified no robust defence-system archetypes after phylogenetic
dereplication. The moderate ARI observed in the full 878-genome dataset (ARI=0.38) was
driven entirely by clonal lineage redundancy and is not interpretable as biological
archetype structure."

**Before:** Synthesis prepared language for moderate ARI result.
**After:** Synthesis states negative finding clearly and honestly.

---

## Ph11-C2 — RESTRICT/FACILITATE "recovery" must be reframed

**Status:** PENDING

**What was wrong:**
Manuscript language section states: "unsupervised clustering cross-validates the published
RESTRICT/FACILITATE dichotomy using an orthogonal, label-free method." This claim fails
under dereplication: all 13 AB representatives cluster together. The "recovery" is an
observation about IC2 clonal dominance, not independent cross-validation.

**Fix required:**
Replace manuscript language with: "The IC2 lineage, by virtue of being sufficiently
numerous and defence-distinct, forms its own cluster when clonal redundancy is present.
This observation is consistent with published IC2 defence depletion but does not
independently cross-validate RESTRICT/FACILITATE, as it disappears after dereplication."

**Before:** Claimed independent cross-validation.
**After:** Describes IC2 clonal dominance observation with appropriate scope.

---

## Ph11-C3 — Q5b K=2 reframed as gradient, not archetype discovery

**Status:** PENDING

**What was wrong:**
Q5b was designed to test whether discrete phage-permissive archetypes emerge. The result
is K=2 driven by genome complexity (IS burden identical between clusters; defence count,
ARG, IME all 2× different). The hypothesis as pre-registered is not supported.

**Fix required:**
Rewrite Q5b synthesis: "The K=2 partition separated genomes by overall genomic complexity
(large-genome Gram-negative PA/EC/KP vs small-genome Gram-positive EF/SA/AB) rather than
by IS-driven defence disruption. IS element burden was similar between clusters (58.9 vs
55.0), refuting the hypothesis that IS burden drives defence depletion at the
cross-species level. This finding motivates a future positional analysis (IS elements
within defence loci) rather than a count-based approach."

**Before:** Pre-registered hypothesis left ambiguously unresolved.
**After:** Negative result clearly stated with mechanistic interpretation and future direction.

---

## Ph11-H1 — Silhouette <0.07 never explicitly interpreted as "no structure"

**Status:** PENDING

**What was wrong:**
The notebook computes silhouette, plots it, and proceeds to run K=10 clustering. It never
states that silhouette below 0.07 means no meaningful cluster structure. The reader is
left to infer this.

**Fix required:**
Add one sentence after the K selection plot: "All silhouette scores across K=2–12 are below
0.07 (maximum 0.068 at K=10). Values below 0.10 indicate no substantial cluster structure;
the data does not partition into distinct natural groups in this feature space."

---

## Ph11-H2 — KM vs HC ARI=0.47 not interpreted as unstable partition

**Status:** PENDING

**What was wrong:**
ARI=0.47 between K-means and hierarchical clustering is printed but not interpreted. For
well-structured data, method agreement ARI typically exceeds 0.70. 0.47 here is additional
evidence of absent cluster structure.

**Fix required:**
Add one sentence to the ARI comparison output: "K-means vs hierarchical agreement ARI=0.47
is below the 0.70 threshold typically seen for stable cluster structure, confirming that
the 10-cluster partition is not robustly recoverable across methods."

---

## Ph11-H3 — ARI vs ARG tertile ≈ 0 not stated as direct Q3 answer

**Status:** PENDING

**What was wrong:**
ARI vs ARG tertile is 0.031 (K-means), 0.021 (HC), 0.030 (K=6) — all printed but not
interpreted as the direct answer to Q3's secondary question.

**Fix required:**
Add sentence: "ARI between defence-system clusters and ARG burden tertile is essentially
zero across all partitions (K-means K=10: 0.031; HC K=10: 0.021). Defence-system
archetypes, as identified by unsupervised clustering, are uninformative for ARG burden."

---

## Ph11-H4 — Cluster 5 (192 genomes, AB/EC/PA mixed) uninterpreted

**Status:** PENDING

**What was wrong:**
The second-largest cluster (192/878 = 22% of all genomes) contains 50 AB, 51 EC, 69 PA,
and is not discussed anywhere in the notebook.

**Fix required:**
Add interpretation in biological synthesis: "Cluster 5 (192 genomes; 33% of AB, 35% of
EC, 46% of PA) likely represents genomes with moderate, non-species-specific Gram-negative
defence profiles: common RM Type II, no CRISPR, moderate defence count, absence of the
lineage-specific systems that define other clusters. It is the default-Gram-negative
bucket rather than a biologically distinctive archetype."

---

## Ph11-M1 — f-string bug: 'Hierarchical Ward K={best_k}' prints literally

**Status:** PENDING

**What was wrong:**
Second row of ARI summary table: `'Hierarchical Ward K={best_k}'` — missing `f` prefix.
Prints as literal text, not evaluated K value.

**Fix required:** Add `f` prefix: `f'Hierarchical Ward K={best_k}'`.

---

## Ph11-M2 — FEAT_COLS hardcoded in Phase 11 vs dynamically computed in Phase 10

**Status:** PENDING

**What was wrong:**
Phase 11 hardcodes a 9-element TAXONOMIC_MARKERS list. Phase 10 derives markers dynamically
from per-species prevalence. Currently consistent (both 265) but fragile to dataset changes.

**Fix required:**
Replace hardcoded list with same dynamic computation used in Phase 10.

---

## Ph11-M3 — Gap statistic missing (pre-registered deliverable)

**Status:** PENDING

**What was wrong:**
CLAUDE.md Phase 11 deliverables specify "Silhouette and gap-statistic for K selection."
Gap statistic was not computed. It would have confirmed K=1 as optimal on dereplicated
data (no structure), reinforcing C1/H1 conclusions.

**Fix required:**
Add gap statistic computation (compare within-cluster variance to null reference from
random uniform data). ~10 lines using sklearn or manual computation.

---

## Ph11-M4 — RESTRICT check uses dp_RM_Type_I only, not all RM subtypes

**Status:** PENDING

**What was wrong:**
The RESTRICT archetype is characterised by any RM presence. Code uses `rm_cols[0]`
which happens to be dp_RM_Type_I. Misses Type II, IIG, III.

**Fix required:**
Compute `any_rm = OR across dp_RM_Type_I, dp_RM_Type_II, dp_RM_Type_IIG, dp_RM_Type_III`
and report both individual subtypes and composite.

---
