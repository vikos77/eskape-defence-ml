# Pending Task — Learning Document: Unsupervised Learning + Phase 12

**Requested:** 2026-05-27 (Session 16)
**Status:** PENDING — create at start of next session BEFORE comprehension check re-attempt
**Output path:** `docs/learning/unsupervised_and_phase12_guide.md`
**Tone:** Assume zero prior knowledge. Use real-world analogies throughout. No jargon without
         definition. Grounded in actual biological results from this project — no hallucination.

---

## What the user explicitly asked for

A "very elaborate learning file" covering:

1. **Unsupervised learning fundamentals** — from scratch, with analogies
   - How K-means clustering works (algorithm, not just definition)
   - How hierarchical clustering works (agglomerative, dendrogram)
   - How to choose K: silhouette score, gap statistic
   - ARI (Adjusted Rand Index) — what it measures, concrete numeric example
   - When unsupervised results are real vs artefactual

2. **ARG burden** — what the Q2 question is exactly, grounded in project biology
   - What an ARG is (resistance gene, not just "something that makes bacteria resistant")
   - What "burden" means (count, distribution, why tertiles)
   - How total ARG burden fits into Phase 8/9 Q2
   - Why class-specific ARG burden is different from total ARG burden
   - The plasmid-mediated vs chromosomal distinction (critical for Phase 12)

3. **Phase 12 — full rationale and results**
   - Why Phase 12 exists (what Phase 8 Q2 left unanswered)
   - Test A: dc vs dp RM — what dc and dp mean, why the substitution matters
   - Test B: mechanism-class breakdown — the airport customs analogy
   - 30/30 floor — full explanation with clinical trial analogy (already drafted in session, reproduce here)
   - The SHAP directionality pre-registration — what was predicted and why
   - Actual results: Test A KP +0.098, EC −0.012, PA −0.054
   - Actual results: Test B 7 significant cells, which species/classes, what AUROC means
   - SHAP direction finding: RM Type II restricts (not Type I); PA/beta_lactam positive (genomic
     complexity confound); quinolone near-zero (expected negative control)
   - How to interpret a MISMATCH between pre-registered prediction and observed SHAP

---

## Key facts to include (verified from actual notebook outputs — DO NOT deviate)

### Phase 11 data (unsupervised section)
- 878 genomes, 6 ESKAPE species
- Full dataset K-means: best K = 10 by silhouette (score = 0.068)
- ARI vs species (full dataset): 0.383 — ARTEFACTUAL (clonal inflation)
- ARI vs species (dereplicated 95 phylogroups): −0.004 — near random, true biology
- K-means vs HC agreement (ARI): 0.468
- Conclusion: ESKAPE defence profiles form a CONTINUUM, not discrete archetypes
- RESTRICT/FACILITATE in full clustering: Cluster 2 (20 AB) = RM-high/SspBCDE-low = RESTRICT;
  Cluster 8 (79 AB) = RM-low/SspBCDE-high = FACILITATE (IC2)
- Q5b (defence + anti-defence + IS): K=2, IS burden nearly identical across clusters → genome
  complexity split (large Gram-negative vs small Gram-positive), not phage-permissive split

### Phase 12 data (actual results from notebook execution)
**dc/dp pre-check:**
- RM Type I: 274/878 genomes differ (31.2%), max dc = 6
  Distribution: {0:311, 1:293, 2:226, 3:38, 4:7, 5:2, 6:1}
- RM Type II: 38/878 differ (4.3%) — effectively binary
- RM Type IIG: 9/878 differ (1.0%) — effectively binary
- RM Type III: 2/878 differ (0.2%) — effectively binary
- RM Type IV: 43/878 differ (4.9%) — effectively binary

**Test A results:**
- Swapped features: dc_RM_Type_I, dc_RM_Type_II, dc_RM_Type_IIG, dc_RM_Type_III
- KP: Ph8 AUROC 0.789 → Test A 0.887 (+0.098) ★
- EC: Ph8 AUROC 0.824 → Test A 0.812 (−0.012)
- PA: Ph8 AUROC 0.677 → Test A 0.623 (−0.054)
- Test A SHAP top features KP: dc_RM_Type_II rank 1 (|SHAP|=0.0569), dc_RM_Type_I rank 3 (0.0399)
- Test A SHAP top features EC: dc_RM_Type_II rank 1 (|SHAP|=0.0780)
- Test A SHAP top features PA: dc_RM_Type_I rank 1 (|SHAP|=0.0538)

**Test B 30/30 floor passing cells:**
- EC/beta_lactam: passes (n_high=43, n_low=61)
- PA/beta_lactam: passes (n_high=44, n_low=66)
- AB/aminoglycoside: passes floor but fails GroupKFold (insufficient valid folds — IC2 clonal compression)
- KP/aminoglycoside: passes (n_high=44, n_low=52)
- KP/beta_lactam: passes (n_high=41, n_low=54)
- KP/sulfonamide: passes (n_high=31, n_low=48)
- EF/macrolide_mlsb: passes (n_high=46, n_low=32)
- EF/tetracycline: passes
- Quinolone: FAILS all species (near-binary distribution)

**Test B BH-significant cells (q=0.05):**
- KP/aminoglycoside: AUROC=0.803 [0.735–0.879] p_adj=0.0077 ★
- PA/beta_lactam: AUROC=0.793 [0.677–0.893] p_adj=0.0112 ★
- EC/beta_lactam: AUROC=0.750 [0.661–0.826] p_adj=0.0107 ★
- KP/beta_lactam: AUROC=0.676 [0.592–0.736] p_adj=0.0116 ★
- KP/sulfonamide: AUROC=0.817 [0.675–0.949] p_adj=0.0202 ★
- EF/macrolide_mlsb: AUROC=0.743 [0.587–0.912] p_adj=0.0490 ★
- EF/tetracycline: AUROC=0.814, p_adj=0.0475 ★

**Test B SHAP direction (dp_RM_Type_I signed mean, key cells):**
- EC/beta_lactam: dp_RM_Type_I = +0.0016 (≈ 0, MISMATCH — predicted negative)
- PA/beta_lactam: dp_RM_Type_I = +0.0050 → FACILITATES (MISMATCH — predicted negative)
- KP/aminoglycoside: dp_RM_Type_I = −0.0013 (≈ 0, MISMATCH), dp_RM_Type_II = −0.0039 ← RESTRICTS
- KP/beta_lactam: dp_RM_Type_I = +0.0009 (≈ 0, MISMATCH)
- KP/sulfonamide: dp_RM_Type_II = −0.0016 ← RESTRICTS; dp_RM_Type_IIG = −0.0001 ≈ 0
- EF/macrolide_mlsb: dp_RM_Type_IIG = −0.0052 ← RESTRICTS
- EF/tetracycline: dp_RM_Type_II = −0.0040 ← RESTRICTS, dp_RM_Type_IIG = −0.0032 ← RESTRICTS

**Interpretation of SHAP direction finding:**
Restriction signal EXISTS but is in RM Type II/IIG, NOT Type I. Pre-registration specified
Type I because Phase 8 total-ARG pointed there. Finding: Type II is the gatekeeping subtype
in KP/EF. PA/beta_lactam positive RM Type I = genomic complexity confound (larger complex PA
genomes have both more RM AND more chromosomal beta-lactam resistance; no plasmid gate applies).

---

## Analogies already used in session (reuse and expand, don't invent new conflicting ones)

- **Airport customs for RM gating:** RM systems = customs officers. Plasmid = smuggler carrying
  resistance gene (contraband). Customs officers check incoming luggage. Chromosomal mutation =
  person inside the country spontaneously becoming a criminal. No customs stop applies.
- **Clinical trial for 30/30 floor:** Need ≥30 patients per arm (high/low class) to have
  statistical power. Below 30 = underpowered = AUROC CI too wide to interpret.
- **Genomic complexity confound (PA/beta_lactam):** Large festivals have both more security guards
  AND more drugs — not because guards cause drugs, but because both are features of a large/complex
  event. In PA: complex genomes have both more RM and more chromosomal beta-lactam resistance.

---

## Structure suggested for the output document

```
docs/learning/unsupervised_and_phase12_guide.md

1. What is unsupervised learning? (5 min read)
2. K-means clustering — step by step with concrete example (10 min read)
3. Hierarchical clustering — how it differs, when to use each (8 min read)
4. How we choose K: silhouette and gap statistic (6 min read)
5. ARI — what it measures, worked numeric example (8 min read)
6. Phase 11 results — what we actually found and what it means (10 min read)
7. What is ARG burden? (the Q2 question, from scratch) (8 min read)
8. Plasmid-mediated vs chromosomal resistance — the biology you need for Phase 12 (8 min read)
9. Phase 12 — why it exists and what it asks (6 min read)
10. Test A — dc vs dp RM, with results (8 min read)
11. Test B — mechanism-class targets, 30/30 floor, with results (15 min read)
12. SHAP direction in Test B — pre-registration, observed results, interpretation (10 min read)
13. Putting it all together — what Phase 12 adds to the manuscript (5 min read)
```

---

## Comprehension check pending at start of next session

**Q4 (deferred):** After reading sections 11–12 of the learning document, attempt this again:

"Write one sentence for the Results section that: (a) states what we observed about SHAP
direction for RM features in the significant Test B cells, (b) names which RM subtype carried
the restriction signal, and (c) uses no causal language."

The model answer from Session 16 (for grading):
"Among the seven BH-significant (species × ARG class) cells, signed SHAP analysis revealed
that dp_RM_Type_II was negatively associated with high-class ARG burden in K. pneumoniae
aminoglycoside (mean SHAP = −0.0039), K. pneumoniae sulfonamide (−0.0016), and E. faecium
tetracycline (−0.0040), whereas dp_RM_Type_I showed near-zero or positive associations across
pre-specified plasmid-mediated classes, a pattern consistent with RM Type II rather than
Type I acting as the genomic correlate of reduced plasmid-mediated ARG acquisition."

---

## Instructions for next session startup

1. Read this file.
2. Create `docs/learning/unsupervised_and_phase12_guide.md` following the structure above.
   Use the verified data from the "Key facts" section — do not invent numbers.
   Use the analogies listed — do not introduce conflicting analogies.
   Assume reader knows nothing about ML, clustering, or ARG biology.
3. After creating the document, tell the user it is ready and ask them to:
   a. Read it
   b. Re-attempt Q4 from Section 16 comprehension check (quoted above)
4. Score Q4 and update docs/comprehension_review.md.
5. Do NOT push comprehension_review.md to GitHub (per standing instruction).
