Phase 12 Audit Report

---
  CRITICAL — Findings that invalidate stated conclusions

---
  C1 — phase8_q2_auroc baseline contains BA values not AUROC values: the headline Test A conclusion is wrong

The hardcoded baseline in Section 5:
  phase8_q2_auroc = {
    "ecloaceae":   0.824,   # XGB (primary)
    "kpneumoniae": 0.789,   # XGB (primary)
    "paeruginosa": 0.677,   # RF  (primary — XGB 0.568)
  }   

From q2_gb_results.parquet (post C1-fix run):
  - EC XGBoost: BA = 0.824, AUROC = 0.872
- KP XGBoost: BA = 0.789, AUROC = 0.924
- PA RF: BA = 0.677, AUROC = 0.722

The three hardcoded "baseline" values are the balanced accuracy values, not AUROCs. The Test A function q2_rf_per_species reports roc_auc_score — it returns
AUROC. So ΔAUROC is computed as AUROC_TestA − BA_Phase8, which is dimensionally wrong.

The ΔAUROC with correct AUROC baselines:
  
  ┌─────────┬──────────────┬────────────────────────┬─────────────┬─────────────────────────┐
│ Species │ Test A AUROC │ Correct AUROC baseline │ True ΔAUROC │ Reported ΔAUROC (wrong) │
├─────────┼──────────────┼────────────────────────┼─────────────┼─────────────────────────┤
│ EC      │ 0.812        │ 0.872 (XGB)            │ −0.060      │ −0.012                  │
├─────────┼──────────────┼────────────────────────┼─────────────┼─────────────────────────┤
│ KP      │ 0.887        │ 0.924 (XGB)            │ −0.037      │ +0.098 ← "improvement"  │
├─────────┼──────────────┼────────────────────────┼─────────────┼─────────────────────────┤
│ PA      │ 0.623        │ 0.722 (RF)             │ −0.099      │ −0.054                  │
└─────────┴──────────────┴────────────────────────┴─────────────┴─────────────────────────┘

The KP "improvement" of +0.098 is an artefact of subtracting the BA value 0.789 from the AUROC value 0.887. The correct comparison shows KP Test A AUROC =
  0.887 is 37pp lower than the correct KP XGB AUROC baseline of 0.924.

The synthesis conclusion "dc_RM improves Q2 prediction. Multi-copy RM signal is real." is factually incorrect. With the correct baseline, Test A degrades
AUROC in every single species. The correct conclusion is: "Replacing binary RM presence with RM count features does not improve Q2 AUROC in any species.
  Binary presence encoding was adequate."

This one error inverts the headline finding of Test A.

---
  C2 — Synthesis SHAP direction check uses only dp_RM_Type_I, producing incorrect "Matches: []" conclusion

The synthesis at the end of Section 11 checks:
  matched = [(sp, cls) for (sp, cls) in shap_sign_results
             if cls in ("beta_lactam", "aminoglycoside", "sulfonamide", "trimethoprim")
             and shap_sign_results[(sp, cls)].get("dp_RM_Type_I", 0) < -0.002]

This checks only dp_RM_Type_I. But the verbose SHAP output shows negative (RESTRICT) direction in other RM subtypes:
  
  ┌─────────────────────┬────────────────┬─────────────┬──────────────────────────┬─────────────────────┐
│        Cell         │    Feature     │ Signed SHAP │ Pre-specified prediction │  Status if checked  │
├─────────────────────┼────────────────┼─────────────┼──────────────────────────┼─────────────────────┤
│ KP / aminoglycoside │ dp_RM_Type_II  │ −0.0039     │ negative                 │ MATCH               │                             
├─────────────────────┼────────────────┼─────────────┼──────────────────────────┼─────────────────────┤
│ KP / aminoglycoside │ dp_RM_Type_I   │ −0.0013     │ negative                 │ near zero           │
├─────────────────────┼────────────────┼─────────────┼──────────────────────────┼─────────────────────┤
│ EF / tetracycline   │ dp_RM_Type_II  │ −0.0040     │ (ambiguous)              │ RESTRICT            │
├─────────────────────┼────────────────┼─────────────┼──────────────────────────┼─────────────────────┤
│ EF / tetracycline   │ dp_RM_Type_IIG │ −0.0032     │ (ambiguous)              │ RESTRICT            │
├─────────────────────┼────────────────┼─────────────┼──────────────────────────┼─────────────────────┤
│ EF / macrolide_mlsb │ dp_RM_Type_IIG │ −0.0052     │ (ambiguous)              │ RESTRICT            │
├─────────────────────┼────────────────┼─────────────┼──────────────────────────┼─────────────────────┤
│ KP / sulfonamide    │ dp_RM_Type_II  │ −0.0016     │ negative                 │ borderline RESTRICT │
├─────────────────────┼────────────────┼─────────────┼──────────────────────────┼─────────────────────┤
│ PA / beta_lactam    │ dp_RM_Type_I   │ +0.0050     │ negative                 │ MISMATCH            │
└─────────────────────┴────────────────┴─────────────┴──────────────────────────┴─────────────────────┘

The published paper's finding is about "RM systems" broadly — not specifically Type I. In Gram-positive E. faecium, RM Type II and IIG are the dominant
  subtypes; Type I is not detected. In K. pneumoniae, Type II is the dominant RM form. The check against only dp_RM_Type_I is biologically inappropriate for
  species where Type I is rare.
  
  The correct synthesis is: KP/aminoglycoside matches the pre-specified directional prediction via RM Type II (< −0.002). EF/tetracycline and EF/macrolide
  show RM Type II and IIG restriction (exploratory, biologically plausible for plasmid-borne resistance). Only PA/beta_lactam is a genuine mismatch. The
  current "Matches: []" report is factually wrong in a direction that understates support for the RESTRICT hypothesis.
  
  ---
  C3 — Test A swaps ALL dp_RM_* features despite 4 of 5 being classified as "moot," and the SHAP attribution for the KP result is to the wrong feature

  The pre-check logic identifies:
  - RM_Type_I: 31.2% differ → live
  - RM_Type_II: 4.3% differ → moot
  - RM_Type_IIG: 1.0% differ → moot
  - RM_Type_III: 0.2% differ → moot
  
  The feature-swap code (c.replace("dp_", "dc_") for all items in dp_rm_set) swaps all 4 non-filtered RM features regardless of the moot/live classification.
  The pre-check narrative implies only the live feature should be swapped, but the code does not implement that logic.

  Then in the KP Test A SHAP output:
  - dc_RM_Type_II: rank 1/85, mean|SHAP| = 0.0569 — this is the moot feature
  - dc_RM_Type_I: rank 3/85, mean|SHAP| = 0.0399 — the live feature
  
  The dominant SHAP driver of the KP "improvement" (which is actually a degradation, see C1) is dc_RM_Type_II — the feature that 95.7% of genomes already
  encode identically as binary presence. Even setting aside C1, the attribution "multi-copy RM signal is real" points to RM_Type_I (live), but SHAP says
  RM_Type_II (moot) is more important.

  ---
  HIGH PRIORITY

  ---
  H1 — PA β-lactam positive RM SHAP is not just a "mismatch" — it is biologically important and requires explicit manuscript discussion
  
  PA β-lactam AUROC = 0.793 (significant, BH q=0.011). dp_RM_Type_I signed SHAP = +0.0050 (classified "FACILITATES" — positive, predicts higher β-lactam
  burden when RM is present).

  This is counter-intuitive but mechanistically explainable. P. aeruginosa β-lactam ARG burden is dominated by chromosomal mechanisms: PDC AmpC
  overexpression, OprD porin loss, MexAB efflux upregulation — none of which are plasmid-borne, none of which RM can gate. When ResFinder identifies β-lactam
  ARGs in PA, it is partly detecting these chromosomal adaptation signatures. Clinical MDR PA strains, which are high-β-lactam-ARG, may also be RM-poor (they
  have acquired anti-defence systems, they are phage-susceptible IC2-equivalent PA lineages). This creates a confound: RM-poor PA = clinical MDR PA = high
  β-lactam count, producing a positive RM SHAP not because RM facilitates β-lactam acquisition but because both covary with clinical origin.

  If this is not addressed, the PA β-lactam result as written ("MISMATCH ✗ — RM facilitates β-lactam") will be used in a reviewer question to challenge the
  entire RESTRICT hypothesis. The manuscript must pre-empt this with the chromosomal mechanism explanation.

  ---
  H2 — SHAP in both Test A and Test B is computed in-sample on a deeply overfit RF
  
  The pattern in both Section 6 and Section 11:
  rf.fit(X_q2, y_q2)        # train on full Q2 data
  sv = explainer.shap_values(X_q2)   # SHAP on same data
  
  RF at max_depth=20, min_samples_leaf=1 on training sets of n=44–66 per class will achieve near-perfect training accuracy. SHAP values on training data for
  an overfit model reflect memorisation patterns, not generalisation patterns. The signed SHAP values at magnitude 0.001–0.005 are particularly unreliable
  because small values from an overfit model may not represent the actual population-level effect.

  For Test B in particular, the SHAP values for EF (n_low=32 for macrolide, n_high=47 for tetracycline) on a max_depth=20 RF are computed on essentially
  memorised data. The directionality may be correct — but it cannot be stated as confidently as the analysis implies.

  ---
  H3 — AB aminoglycoside "insufficient valid folds" is a structural dead end, not a transient failure
  
  AB aminoglycoside passed the 30/30 floor (n_high=30, n_low=50). Yet it fails with "insufficient valid folds." This is the third time in this project that AB
   Q2 fails for the same structural reason: 13 AB phylogroups with the largest containing approximately 43% of Q2-eligible AB genomes. GroupKFold cannot
  create 5 folds where every test fold contains both classes when one phylogroup dominates.

  The notebook logs this as "note": "insufficient valid folds" in the results table without commentary. The correct conclusion to draw and document is: AB Q2 
  analysis is structurally infeasible under GroupKFold(5) at any biologically meaningful phenotype, given the current AB phylogroup composition. This is not a
   transient failure to fix with more iterations — it is a sample-structure limitation. The manuscript should state this explicitly once, with the
  phylogroup-size distribution, rather than allowing it to appear as a missing cell in all Q2 tables.                                 

  ---
  H4 — EF tetracycline and EF macrolide RM restriction is a genuinely new biological finding, buried as "ambiguous" in the synthesis
  
  Test B shows EF tetracycline AUROC = 0.814 (p_adj=0.047) and EF macrolide AUROC = 0.743 (p_adj=0.049), both BH-significant. The SHAP shows:
  - EF/tetracycline: dp_RM_Type_II = −0.0040, dp_RM_Type_IIG = −0.0032 — both RESTRICT direction, both exceeding the −0.002 threshold
  - EF/macrolide: dp_RM_Type_IIG = −0.0052 — RESTRICT direction
  
  Tetracycline resistance in E. faecium is predominantly plasmid-mediated (tet(M) on Tn916, tet(L) on various plasmids). Macrolide/MLS_B resistance is
  similarly plasmid-borne (erm(B) on Tn1545-related elements). Both are legitimate targets for RM gatekeeping in Gram-positives. The fact that EF RM systems
  (particularly Type II and IIG, which are the dominant EF RM classes) associate negatively with these ARG burdens is consistent with the RESTRICT principle
  extending to Gram-positive organisms with plasmid-borne resistance.
  
  The synthesis labels these cells "ambiguous" (because they weren't in the pre-specified directional prediction list). The correct handling: flag as
exploratory findings with biological rationale. This is potentially the most publication-valuable result in Phase 12 — it extends the RESTRICT principle
from the Gram-negative ESKAPE species to E. faecium — and it is invisible in the current synthesis.

---
  H5 — Bootstrap CI on 5 fold AUROC scores has the same poor-coverage problem as previous phases

The bootstrap_ci function resamples 2000 times over 5 fold-level AUROC scores. This is the same methodology flagged in the previous audit. The resulting CIs
are particularly wide: KP/sulfonamide [0.675–0.949] has a width of 0.274. For significance calls, this CI width matters: if the lower bound approaches 0.5,
the cell's significance is questionable. The bootstrap-on-predictions approach (pool all held-out predictions, bootstrap on individual genomes) would give
  narrower, better-calibrated CIs.
  
  ---
  MODERATE

  ---
  M1 — No paired statistical test for ΔAUROC in Test A
  
  Test A reports ΔAUROC as point estimates only (after C1 is fixed, these will all be negative). A paired t-test on the 5 fold-level AUROC differences (Test A
   fold i vs Phase 8 fold i, using the same CV split) would give a formal p-value. This is the correct test for "does dc_RM improve over dp_RM?" and its
  absence means the "no improvement" conclusion, while correct, lacks a formal test.

  M2 — SHAP direction threshold of 0.002 has no statistical basis

  The classification < −0.002 = RESTRICTS, > 0.002 = FACILITATES, between = near zero is arbitrary. For signed mean SHAP values at magnitude 0.001–0.005, a
  genome-level t-test (is the mean signed SHAP for feature F significantly negative across the Q2 genome set?) would give a principled threshold. The current
  classification changes meaningfully between cells that differ by 0.001 (e.g., KP/aminoglycoside dp_RM_Type_I = −0.0013 classified as near zero vs the −0.002
   threshold).

  M3 — PA β-lactam label asymmetry (n_high=44, n_low=66) from median_fallback not discussed

  PA β-lactam uses the median fallback (same PA-1 amendment). n_high=44 vs n_low=66 means 33% more PA genomes are low-β-lactam than high. AUROC is insensitive
   to this imbalance (it's a rank statistic), but the asymmetry is biologically informative: most PA genomes are low-β-lactam burden relative to the median,
consistent with chromosomal resistance being a threshold effect (one OprD mutation = high resistance, regardless of ARG gene count).

M4 — "Genomes with ≥1 ARG in any class: 883" diagnostic count will confuse readers

The 883 > 878 count in the parsing section is due to 22 MLST-excluded genomes having ResFinder data. These are correctly dropped in the left-merge with
all_genomes. The left-merge is correct. But the 883 figure without explanation will cause a reviewer to question data integrity. A one-line comment
explaining "includes 22 MLST-excluded genomes, dropped in subsequent merge" removes the concern.

---
  OPTIONAL

---
  O1 — Learning curves not computed for Test B cells

With n_high as low as 30–44 per cell, learning curves (AUROC vs training set size) would directly show whether the significant cells are on the steep part
of the curve (adding 20 more genomes would substantially improve AUROC) or near saturation. This is particularly important for justifying the significance
of the EF cells (n_low=32 for macrolide).

O2 — No permutation importance for Test B                                                                                           

Phase 8 computed SHAP + permutation importance. Phase 12 Test B only computes in-sample SHAP. Adding even a one-run permutation importance (on CV test
                                                                                                                                            folds) would provide a cross-check: if permutation importance and SHAP agree on RM direction, the signed SHAP result is more credible.

O3 — Rifamycin class detected (55 genomes) but not modelled — worth documenting why

55 genomes have rifamycin ARGs. Rifamycin is not in the pre-specified Test B classes. It is effectively chromosomal in many species (rpoB mutations) but can
be plasmid-borne. A brief sentence in the synthesis explaining the exclusion (not enough genomes per species to pass 30/30 floor, or not in the
                                                                              pre-registration) would complete the audit trail.

---
  Summary Table

┌─────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────┐
│ ID  │                                                                Finding                                                                │ Priority │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ C1  │ phase8_q2_auroc hardcodes BA not AUROC — KP "improvement" of +0.098 is an artefact; correct comparison shows ΔAUROC < 0 for all       │ CRITICAL │
│     │ species                                                                                                                               │          │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ C2  │ SHAP direction check uses only dp_RM_Type_I; "Matches: []" is wrong — KP/aminoglycoside and EF/tetracycline match via Type II and IIG │ CRITICAL │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ C3  │ Test A swaps moot features; KP SHAP rank 1 is dc_RM_Type_II (moot, 4.3% differ), not the "live" dc_RM_Type_I that justified running   │ CRITICAL │
│     │ Test A                                                                                                                                │          │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ H1  │ PA β-lactam positive RM SHAP — chromosomal mechanism confound needs explicit manuscript discussion                                    │ HIGH     │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ H2  │ In-sample SHAP on max_depth=20 RF trained on n≤44 per class — overfit RF, unreliable SHAP magnitudes                                  │ HIGH     │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ H3  │ AB aminoglycoside structural failure — third occurrence; must be declared a permanent limitation                                      │ HIGH     │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ H4  │ EF tetracycline and macrolide RM restriction (Type II/IIG) is a new biological finding buried as "ambiguous"                          │ HIGH     │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ H5  │ Bootstrap CI from 5 fold AUROC scores — poor coverage, same issue as previous phases                                                  │ HIGH     │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ M1  │ No paired t-test for ΔAUROC significance in Test A                                                                                    │ MODERATE │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ M2  │ SHAP direction threshold 0.002 is arbitrary, no statistical basis                                                                     │ MODERATE │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ M3  │ PA β-lactam label asymmetry (44 vs 66) from median fallback not discussed                                                             │ MODERATE │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ M4  │ 883 > 878 ARG parsing count unexplained (explainable by 22 MLST-excluded genomes)                                                     │ MODERATE │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ O1  │ No learning curves for Test B cells                                                                                                   │ OPTIONAL │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ O2  │ No permutation importance alongside SHAP in Test B                                                                                    │ OPTIONAL │
├─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ O3  │ Rifamycin exclusion not documented                                                                                                    │ OPTIONAL │
└─────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────┘

---
  Remediation order

Fix immediately (changes the headline conclusion):
  1. C1: In Section 5, replace the hardcoded phase8_q2_auroc dict with values loaded from q2_gb_results.parquet (AUROC column for XGB primary species) and
q2_rf_results.parquet (AUROC for PA RF). Re-run ΔAUROC calculation. The correct conclusion will be "RM count does not improve AUROC — binary presence was
  adequate."
2. C2: In Section 11's synthesis checker, replace the dp_RM_Type_I-only check with: any RM feature (all dp_RM_* columns in the cell's feature set) showing
signed SHAP < −0.002 counts as a RESTRICT match. Update matched, mismatched, and quinolone_cells accordingly. Re-run the narrative verdict.
3. C3: Implement the moot/live swap logic — only swap dp_RM_Type_I to dc_RM_Type_I (the live feature). Retaining moot features as dp_* is the pre-registered
intent. After C1's baseline fix, re-run Test A with only RM_Type_I swapped.

  Add to synthesis before manuscript:

  4. H1: Add a paragraph in Section 13 explaining the PA β-lactam positive RM SHAP via chromosomal mechanism confound.
  5. H4: Elevate EF tetracycline and macrolide from "ambiguous" to "exploratory — consistent with RESTRICT principle in Gram-positive organisms." Add
  biological rationale (tet(M)/erm(B) on conjugative elements).
  6. H3: Add a documented conclusion for AB Q2: "Structurally infeasible under GroupKFold(5) due to phylogroup size imbalance. AB Q2 is excluded from all
  cross-species comparisons."
