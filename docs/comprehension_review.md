# Phase 3 Comprehension Check Review

All questions issued during Phase 3 notebook construction (Sessions 4–6, 2026-05-02 to 2026-05-13).  
Format: question → user attempt → model answer + key correction if any.

---

## Section 1 — DefenseFinder Parsing

**Q1. The defence count table is 97% sparse. Why is Euclidean distance a poor choice, and which metric handles sparse binary data better?**

User attempt: Euclidean works with continuous x/y data and tries to find shortest distance between points, so it won't work with binary data. Jaccard index is better — it does not inflate zero figures and completely collapses if both systems are absent.

Model answer: Euclidean distance suffers from the *double-zero problem* on sparse binary data — two genomes that both *lack* a system look artificially similar because they share a zero, which inflates their apparent similarity. Jaccard ignores double-zeros entirely: the denominator is the union (systems present in at least one genome), so shared absences contribute nothing. Jaccard = |A ∩ B| / |A ∪ B|. This is the standard distance metric for ecological presence/absence matrices, which have the same structure as defence system matrices.

Assessment: Correct. User correctly named Jaccard and the double-zero problem.

---

**Q2. What does `fill_value=0` do in `.unstack(fill_value=0)`, and what would the table look like without it?**

User attempt: fill_value=0 adds 0 wherever a system is absent. Without it, absent systems would be empty or NaN. Zero is important here because it means "system absent" — a semantically meaningful value.

Model answer: `.unstack()` by default fills missing (genome, system) combinations with NaN. `fill_value=0` replaces those NaN with 0, encoding absence explicitly. Without it, downstream arithmetic (mean, correlation) would ignore NaN or propagate it — either outcome is wrong. Zero is semantically correct: absence of a system is a measured fact, not missing data.

Assessment: Correct.

---

**Q3. A genome has 3 copies of RM_Type_I. What value appears in the count table? In the P/A table? Which do you use for "does this genome have the RM gateway?"**

User attempt: We deduplicate so they're represented just once. Systems like RM which are present almost everywhere multiple times would inflate comparisons if not deduplicated. So it should be just 1 for one type/subtype.

Model answer: Count table = 3 (three instances). P/A table = 1 (clipped at 1). We keep *both* representations. For the binary question "does this genome have RM?" use P/A = 1. But count = 3 is not discarded — multiple copies may reflect tandem MGE insertions or stronger restriction activity, and the ML step decides whether count adds signal over binary. The user incorrectly implied counts are always collapsed; they are stored separately.

Assessment: Partially correct. Right on P/A = 1, but missed that count = 3 is also kept in the matrix, not discarded.

---

**Q4. RM_Type_I is in 876/900 genomes. Is it more useful for Q1 (species classification) or Q2 (ARG burden prediction)?**

User attempt: It has nothing to do with species so Q1 is out. For Q2, it confirms/tests the paper's negative correlation with ARG. Only if it exclusively classifies a species would it be useful for Q1.

Model answer: RM_Type_I is in 97.3% of genomes — near-constant, so near-zero variance. Weak discriminator for ANY classifier, including Q2. But for Q2, the signal lives in the *minority*: the 24 RM-absent genomes. Whether those 24 genomes have systematically higher ARG burdens is exactly what Q2 tests. For Q1, it carries almost no information (every species has it). Correct direction, but the near-zero-variance argument was missing.

Assessment: Correct direction, missing the variance/minority-class nuance.

---

## Section 2 — PADLOC Parsing

**Q1. A PADLOC file has 12 rows; after `drop_duplicates(subset=["system", "system.number"])` it has 4 rows. What does this tell you?**

User attempt: 2 counts dropped per count retained. Could be a single protein system deduplicated or dual protein system dropped at system.number level.

Model answer: 4 rows = 4 system *instances* (unique combinations of system name × system number). 12 original rows = 12 individual protein/gene detections. Average 3 genes per system instance (12÷4). We don't know how many distinct system *types* — could be 4 instances of one type, or 4 different types. Nothing is "dropped" as incorrect; extra gene rows are collapsed into the representative instance row.

Assessment: Partially correct. Right that rows collapsed, but confused on "dropped" framing and didn't compute instances correctly.

---

**Q2. DMS_other is in 886/900 genomes. A reviewer says include it — it varies in 14 genomes. One-sentence rebuttal.**

User attempt: 14 genomes is too few to consider; it's present in >95% of genomes; DMS_other has no cited proof as it is a wide-net catch-all.

Model answer: DMS_other is a near-constant feature (variance ≈ 0 across 886/900 genomes) and an unannotated catch-all with no experimentally validated biological specificity, making it both statistically uninformative and biologically uninterpretable as an ML feature.

Assessment: Correct on both statistical and biological grounds.

---

**Q3. The cross-tool sanity check shows 113 systems in both tools. Does this match prove the mapping is correct? Give a coincidence scenario.**

User attempt: The mapping could be wrong based on naming issues between tools; if there's no error catching then it might never show up.

Model answer: The count match alone does not verify individual mappings. Coincidence example: system A is incorrectly labelled "both" in the map (DF detects it, PADLOC doesn't) and system B is also incorrectly labelled "both" (PADLOC detects it, DF doesn't). Two errors cancel out — "both" count is still 113. Correct verification requires checking that each individual "both" system actually co-occurs in both tools' outputs for at least some genomes.

Assessment: Correct direction but imprecise — didn't give the concrete cancel-out example.

---

**Q4. SoFic: PADLOC detects 410 instances, DF detects ~6. We kept them separate (`df_SoFic` + `padloc_SoFic`). What does separation preserve that merging would destroy?**

User attempt: High numbers from one tool might be tool/database artifact. Keeping separate lets us investigate deeply; merging would dilute the high number and assume they're the same biology.

Model answer: Keeping them separate preserves the tool-discordance signal — a 68× difference (410 vs 6) is itself informative about database sensitivity or HMM model specificity. Merging under one canonical name would assume both detections are equivalent, hiding the fact that one tool's calls are essentially uncorroborated. Additionally, separate names allow the sparsity filter to handle `df_SoFic` (6 genomes) correctly — it would be dropped as too rare, while `padloc_SoFic` (410 genomes) survives. Merging would allow the rare DF calls to ride on PADLOC's prevalence.

Assessment: Correct on the main point. Bonus insight on sparsity filter interaction was not raised.

---

## Section 3 — DF + PADLOC Merge

**Q1. 179 RM_Type_I detections are single-tool only (n_tools=1). Biological and technical explanations.**

User attempt: Technical: tool sensitivity and HMM database differences. Biological: some mutation or sequence change affecting the GFF and tool output.

Model answer: Technical — DF and PADLOC define system boundaries differently. DF requires a minimum gene complement within a defined genomic window; PADLOC may call a system from a partial complement. A genome where one mandatory gene is missing or diverged enough to fall below one tool's threshold but not the other's produces a single-tool detection. Biological — the locus may be genuinely degenerate (disrupted by insertion or deletion), producing a partial system that one tool's looser boundary definition still calls.

Assessment: Correct. Should add the boundary-definition distinction, not just sensitivity.

---

**Q2. DF says 3 copies of RM_Type_I; PADLOC says 1. We store max=3. Overestimate scenario vs correct scenario.**

User attempt: It boils down to tool sensitivity and database; we keep the higher count to hypothesize phage pressure. Invalidated if differences are too high or niche-specific.

Model answer: Overestimate — DF called 3 copies, but two are degenerate prophage remnants that disrupted the locus (partial genes only). PADLOC's stricter boundary definition correctly identified only 1 intact copy. max=3 overstates functional copy number. Correct — DF calls 3 complete intact loci at separate chromosomal positions. PADLOC missed 2 because its HMM profiles don't cover the divergent subtype. max=3 is right; min=1 would have underestimated. The practical implication: interpret count as "upper bound on copy number," not "confirmed copy number."

Assessment: User answered WHY counts differ (correct) but didn't answer the specific overestimate/correct SCENARIOS asked. Re-explained with concrete examples.

---

**Q3. 133 systems dropped by sparsity filter. Statistical rebuttal for including rare systems + tree-based problem.**

User attempt: Including rare systems causes severe imbalance and near-zero variance predictors adding noise. In tree-based classifiers they cause overfitting and inflated feature importance.

Model answer: Statistically, a feature present in <5/900 genomes has near-zero variance — it contributes essentially zero Gini information at any split, adding dimensionality without predictive signal. In tree-based classifiers specifically, rare features enable *spurious splits*: a split on a rare feature can perfectly separate 4 training genomes from 896, driving a large apparent Gini reduction that generalises to nothing on held-out data. This inflates that feature's importance score in the output, crowding out genuinely informative features.

Assessment: Correct. The specific term "spurious split" fills in the mechanism the user gestured at.

---

**Q4. Binary P/A vs integer count as primary RF input — which and why?**

User attempt: Binary P/A is primary. Presence/absence has stronger biological validation than copy number. Using copy numbers could inflate information via spurious numeric thresholds. RF feature randomness (max_features) property relevant here.

Model answer: Binary P/A is primary. RF's `max_features` parameter randomly samples columns at each split. If both `RM_Type_I` (binary) and `RM_Type_I_count` (integer) are in the pool, they compete, effectively doubling RM's probability of selection at each split — artificially inflating its importance and reducing diversity across trees. Binary features also avoid numeric threshold splits (e.g., "count ≥ 2.5") that have no biological meaning. Integer counts can be included in a secondary analysis if binary P/A leaves detectable signal on the table.

Assessment: Correct. User correctly identified max_features and spurious thresholds.

---

## Section 4 — ResFinder (ARG Counts)

**Q1. Mean ARG = 8.6, median = 7.0. Distribution shape + test implications.**

User attempt: Right-skewed (mean > median). Rules out two-sample t-test. Use Spearman ranked correlation.

Model answer: Right-skewed — correct. The t-test requires approximately normal distribution of means (by CLT this is weakened at large N, but count data with a hard floor at 0 is structurally non-normal). Corrected: Spearman is a *correlation* test (measures monotonic co-variation between two variables), not a group-comparison test. For *comparing ARG counts between two groups* (e.g., RM-present vs RM-absent), the correct non-parametric test is **Wilcoxon rank-sum** (Mann-Whitney U). For measuring correlation between ARG count and another variable (e.g., IME count), use Spearman. Both are in the paper's published toolkit — apply each to the right question.

Assessment: Partially correct. Right on distribution and rejecting t-test. Wrong test named for group comparison — Spearman is for correlation, Wilcoxon is for two-group comparison.

---

**Q2. K. pneumoniae median ARG = 15.5, S. aureus = 3.0. Pooled Q2 tertile label implications.**

User attempt: KP would be in the top tertile, SA in the bottom. Classifier would learn to distinguish KP from SA, not high-ARG vs low-ARG genomes.

Model answer: Correct and critically important. With pooled tertile labels, the "high-ARG" class becomes predominantly KP + A. baumannii; "low-ARG" becomes predominantly S. aureus + E. faecium. The classifier learns species identity rather than defence-ARG associations. Fix: define tertile labels *within each species* (high-ARG KP vs low-ARG KP; high-ARG SA vs low-ARG SA), or treat species as a covariate. Pre-registered as a design decision for Section 9.

Assessment: Correct and insightful.

---

**Q3. 16 genomes have zero ARGs. Biological and methodological explanations.**

User attempt: Biological — no antibiotic pressure so no selection for resistance. Methodological — database gaps or parameters too stringent.

Model answer: Biological — some genomes in the dataset may be older clinical isolates from before widespread antibiotic use, or environmental isolates without exposure to clinical antibiotics. Methodological — ResFinder uses a curated acquired-resistance gene database; chromosomally encoded intrinsic resistance mechanisms (e.g., efflux pump overexpression, porin loss) are not captured, so a genome can be clinically resistant yet score zero in ResFinder.

Assessment: Correct on both.

---

**Q4. 15 copies of blaTEM-1 + 1 copy of mecA. arg_count_unique=2, arg_count_total=16. Which captures clinical resistance vs MGE activity?**

User attempt: MGE acquisition activity — total count; clinical treatment failure — unique count.

Model answer: `arg_count_unique = 2` captures *resistance breadth* — how many distinct antibiotic classes this genome is resistant to (relevant to treatment: "we can't use penicillins or methicillin"). `arg_count_total = 16` captures *MGE acquisition load* — how many insertion events occurred, a proxy for how active horizontal gene transfer has been in this genome's history.

Assessment: Correct.

---

## Sections 5 & 6 — ICEberg and BacMet (IME and HMRG Counts)

**Q1. P. aeruginosa has median IME count = 2 (lowest). Biological and technical reasons.**

User attempt: Biological — PA depends on chromosomally encoded genes with limited MGE-mediated transfer. Technical — large genome with large proteins reduces coverage when using max_protein_length as denominator.

Model answer: Biological — P. aeruginosa mobilises genes primarily through genomic islands and Pf phages rather than canonical ICEs/IMEs captured in the ICEberg database. Its large genome (6–7 Mb) accommodates extensive chromosomal gene acquisition through non-canonical mechanisms. Technical — the max_protein_length denominator is conservative. For elements with several large proteins (>1000 aa), the 80% coverage bar is harder to reach from a partial alignment, meaning some genuine hits at the boundaries of ICEberg elements get filtered out.

Assessment: Correct on both. The technical explanation was right — large proteins raise the bar for the coverage filter.

---

**Q2. HMRG counts are high (all 900 genomes, median 221/753). What does `hmrg_count_unique` biologically encode, and a confounded Q1 scenario.**

User attempt: HMRGs encode protection against pollution/heavy metals. For Q1 confounding, all ESKAPE species might be heavy HMRG species, but unique ones might still differentiate.

Model answer: Biologically, `hmrg_count_unique` reflects the complement of metal-homeostasis and efflux proteins encoded in the genome — a mixture of constitutive housekeeping (chromosomally conserved RND pumps, metal-binding proteins) and acquired metal resistance (plasmid-borne arsenic, mercury, copper operons). For Q1 confounding: gram-negative bacteria (K. pneumoniae, P. aeruginosa, A. baumannii) constitutively encode RND-family efflux pumps that match many BacMet entries, while gram-positive (S. aureus, E. faecium) lack outer-membrane-dependent systems, producing a systematic gram-stain-level split in HMRG count that a classifier could exploit to distinguish species without learning anything about metal resistance specifically.

Assessment: Partially correct. User identified the right concern but didn't specify the gram-positive/gram-negative split as the concrete confounding mechanism.

---

**Q3. Why is pident ≥ 40% the standard threshold, and what is the zone below called?**

User attempt: 40% is the ideal mark above which false positive risk drops sharply because conserved positions exceed what random sequence achieves. Below this is the "twilight zone."

Model answer: At pident ≥ 40%, the number of conserved positions in an alignment exceeds what is achievable by convergent evolution or random sequence similarity alone — the alignment is almost certainly detecting true homology (shared evolutionary origin). Below ~30–35% identity, statistical significance becomes unreliable even with good e-values because the sequence space is dense enough for random alignments to achieve meaningful scores. This region (roughly 20–35% identity) is the **twilight zone** (Rost 1999). Our 40% threshold sits safely above it.

Assessment: Correct. Twilight zone correctly named.

---

**Q4. Which count vector (ARG, IME, HMRG) will show the strongest correlation with defence system features?**

User attempt: IME counts — because ARG and HMRG resistance depend on IMEs for survival. IME would be the strongest correlator.

Model answer: IME count (`ime_count_unique`) should show the strongest correlation with the defence system profile, because IMEs are the physical vehicles for both ARGs and HMRGs — the defence system landscape (RM as gatekeeper, SspBCDE as facilitator) acts on the incoming element as a whole. A genome where IMEs integrated successfully accumulated more of everything that IMEs carry (ARGs, HMRGs, anti-defence genes). ARG count correlates nearly as strongly (ARGs ride the same IMEs) but with added noise because ARGs also transfer on plasmids outside the ICEberg IME database. HMRG count is weakest because BacMet's constitutive housekeeping genes dilute the MGE-acquisition signal.

Assessment: Correct prediction. Reasoning was directionally right but needed the "vehicle vs cargo" framing made explicit.

---

## BacMet Investigation — Phase 3 Session, 2026-05-12

Questions issued during the BacMet confounding deep-dive and AMRFinderPlus design decision.

---

**Q1. You run AMRFinderPlus with `--organism Klebsiella_pneumoniae` and get 45 hits; without `--organism`, 38 hits. What causes the 7-hit difference, and why does this matter most for SA and EF?**

Score: 6/10

User attempt: Without `--organism`, the tool scans the full pan-bacterial database causing confidence to be diluted across proteins, with only top hits selected. With `--organism`, a targeted subset is used, giving higher confidence and therefore more hits.

Model answer: The mechanism is addition, not selection. `--organism` adds species-specific HMM profiles that do not exist in the pan-bacterial core database. Without `--organism Klebsiella_pneumoniae`, those Klebsiella-specific gene families (specific efflux pumps, carbapenemases in the METAL-adjacent STRESS category) simply are not searched for. The 7 extra hits are genes found only in the organism-specific profile set. No "top hit" selection occurs — all hits above the HMM score threshold are reported regardless. For SA and EF this is most critical: SA copper resistance (CopA, CopZ, Mco) and EF cadmium resistance are in Staphylococcus/Enterococcus-specific profiles. Without `--organism`, those resistance genes are completely absent from the search and will produce zero hits, not lower-confidence hits.

Correction: "diluted confidence / top hits selected" is the wrong mechanism. The correct framing is "organism-specific profiles not run at all."

---

**Q2. AMRFinderPlus gives KP ~30–40 METAL hits; BacMet gave ~330. The lower count from AMRFinderPlus — is this a loss of data or a better measurement? Justify.**

Score: 2/10

User attempt: It is a problem because the counts decreased multifold. BacMet hits could have added some say in the feature table. If the project were gram-negative only, this would be a loss of good data.

Model answer: The lower count is the correct result. BacMet's 330 hits were predominantly distant RND efflux pump structural homologs (40–65% identity) — not acquired metal resistance genes. Including those 330 hits in the feature matrix would train the ML model on RND pump diversity, which encodes gram-stain phylogenetic signal, not metal resistance burden. More features does not mean more information. Noisy features — features that measure the wrong thing — actively hurt ML models by: (1) adding dimensions that correspond to phylogenetic signal, causing classifiers to learn species identity instead of defence architecture; (2) diluting true signal features in tree-based models (RF/XGBoost) by competing for splits. For gram-negative-only analysis, BacMet at 40% would still capture structural homologs rather than acquired genes — the problem exists in any multi-species comparison. AMRFinderPlus's 30–40 hits are the actual acquired metal resistance genes. Fewer clean features outperform more noisy ones.

Correction: This is a fundamental misconception to revisit during Phase 7 (RF feature importance) — when noisy features appear in the SHAP output with spurious importance values, this is the underlying reason.

---

**Q3. Arsenic resistance count positively correlates with ARG burden across ESKAPE genomes. What is the correct causal interpretation and the paper phrasing?**

Score: 7/10

User attempt: If arsenic positively correlates with ARGs, these genes could be acquired via plasmids, meaning HGT happened. There's no other way this plays out.

Model answer: Co-carriage via HGT is one correct mechanism but not the only one. The distinction to preserve: (1) **co-carriage** — ars operon and ARGs are physically on the same plasmid or integrative element, transferred together in a single event; (2) **co-selection** — ars operon and ARGs are on separate mobile elements, but both are selected in environments with dual antibiotic + metal pressure (e.g., polluted clinical or agricultural settings). Positive correlation is consistent with both scenarios; you cannot distinguish them from correlation data alone — that requires genomic context (same plasmid? same transposon?). Causal direction is also undefined: you cannot say arsenic resistance drives ARG acquisition or vice versa. Correct paper phrasing: "Positive correlation between arsenic resistance gene count and ARG burden is consistent with co-carriage or co-selection on mobile genetic elements under dual antibiotic and heavy metal selective pressure."

Correction: "No other way this plays out" is too strong. Co-selection without co-carriage is a real and common mechanism in clinical and agricultural environments.

---

---

## Sections 6b and 7 — AMRFinderPlus + ISEScan (Session 6, 2026-05-13)

**Q1. AMRFinderPlus `--organism` mechanism: why do SA copper resistance genes (CopA/CopZ) produce zero hits without the flag?**

Score: 6/10

User attempt: The organism flag adds organism-specific proteins/genes during the run, which is not the case without the flag. Hence, we get targeted specified hits.

Model answer: `--organism` loads an entirely separate HMM profile library — the species-specific profile set — that is not part of the pan-bacterial core database. SA copper resistance genes (CopA, CopZ, Mco) do not exist in the pan-bacterial HMM library at all. They are defined exclusively in the SA-specific profile set. Without `--organism Staphylococcus_aureus`, AMRFinderPlus does not search these profiles — there is nothing to hit. The zero is not diluted signal at lower confidence; the search space simply does not contain those gene families. This is why adding `--organism` for gram-positive species is not optional — it is the only way to detect gram-positive-specific resistance gene families.

Correction: "Targeted specified hits" implies the same genes are found but at higher confidence without the flag. The correct framing: the genes are not in the pan-bacterial database at all and require organism-specific profiles to exist in the search.

---

**Q2. Why does AMRFinderPlus report copper and silver resistance as a compound COPPER/SILVER class rather than two separate entries?**

Score: 7/10

User attempt: The compound class has some shared locus or genetic data. To have two separate entries, some criteria would need to be met which is not happening here.

Model answer: The *pco/sil* gene cluster is a single co-resistance operon on IncH plasmids (and some integrons). The same set of proteins — PcoA, PcoB, PcoC, PcoD, PcoE on the copper side and SilA, SilB, SilC, SilE, SilP on the silver side — are encoded in a single chromosomal/plasmid region and transcribed together. Copper-transporting ATPases in this cluster have measurable silver efflux activity and vice versa; the two metals compete for the same binding sites on some Pco/Sil proteins. AMRFinderPlus's compound Class annotation reflects the reference gene family's biological reality — these genes evolved as a co-resistance unit, not two independent loci that happen to sit nearby.

Correction: User had the right idea ("shared locus") but needed to name the *pco/sil* operon specifically and explain that the compound annotation is a biological fact about the operon, not an annotation gap.

---

**Q3. Why should ISEScan's `new` family IS elements be included in the feature matrix despite not matching a known family?**

Score: 0/10 — user requested a detailed explanation of ISEScan.

ISEScan explanation provided in session:
ISEScan identifies IS elements using two independent criteria: (1) transposase ORF detection via HMM profile library built from all known IS transposase families in ISfinder; (2) Terminal Inverted Repeat (TIR) detection — ISEScan scans for the short reverse-complement IR sequences (8–40 bp) that flank every IS element in the correct orientation. A `new`-family element passes BOTH criteria — it has a transposase-like ORF (detected by HMM, just not confidently classifiable to a known family) AND correctly positioned flanking IRs. Failure to classify reflects incomplete reference database coverage of IS transposase diversity, not a structural false positive. In *A. baumannii*, novel IS elements outside known families have been documented carrying ARG cassettes.

Model 2-sentence rebuttal: ISEScan identifies `new`-family elements using two independent lines of evidence — a transposase ORF detected by HMM search and flanking terminal inverted repeats in the correct orientation — which are the defining structural features of all IS elements regardless of family. The `new` label reflects the limits of current reference databases, not a detection artefact; these elements have the same transposition machinery and genomic impact as classified IS elements and cannot be excluded without discarding real biology.

---

**Q4. Genome A (IS=90) vs Genome B (IS=10), both with IME=5. Which has higher predicted ARG burden and why?**

Score: 6/10

User attempt: Higher IS element count could be due to higher transposition activity around ARG genes, so genome A would have higher ARG burden.

Model answer: Genome A is predicted to have higher ARG burden. The mechanism is more specific than general transposition activity: IS6/IS26 family elements directly flank ARG cassettes to form composite transposons — the architecture is `IS26 → [ARG + promoter] → IS26`. Each IS26 insertion flanking an ARG cassette creates a discrete mobile unit that can circularise, transfer to a new replicon, and integrate. High IS26 copy number in KP is a validated clinical marker for extensive resistance cassette accumulation. Additionally, a genome with 90 IS elements has undergone far more transposition events, indicating a history of MGE-intensive HGT activity — the same genetic environment that selects for ARG accumulation. Caution: the causality is not directional from IS count alone — both high IS count and high ARG burden may reflect the same selective environment (clinical, high antibiotic and HGT pressure) rather than IS elements causing ARG acquisition.

Correction: The answer was directionally correct but needed the IS6/IS26 composite transposon architecture and the co-selection caveat.

---

## Summary of patterns across all questions

| Area | Correct | Partially correct | Missed/wrong |
|------|---------|-------------------|--------------|
| Distance metrics | ✓ Jaccard vs Euclidean | | |
| Data representation | ✓ fill_value=0; ✓ P/A vs count primary | Count=3 kept (not collapsed) | |
| Feature engineering | ✓ sparsity filter rationale | | SoFic sparsity interaction |
| Statistical tests | ✓ non-parametric justification | Wilcoxon vs Spearman confusion | |
| ML mechanics | ✓ max_features; ✓ spurious splits | Overestimate/correct scenarios | |
| Biological interpretation | ✓ IME as vehicle; ✓ Q2 label pooling | HMRG gram-stain confounding; pco/sil operon identity | |
| MGE ecology | ✓ IS vs ICE distinction (after explanation); IS count → ARG burden direction | IS6/IS26 composite transposon mechanism | ISEScan `new` family biology |

**Items to revisit:**
1. Wilcoxon rank-sum (group comparison) vs Spearman (correlation) — know when to use each
2. Count table kept alongside P/A — both go into the matrix, both have use cases
3. Coverage filter conservatism: max_protein_length underestimates coverage (harder to pass, not easier)
4. BacMet gram-stain confounding (RESOLVED — BacMet removed; AMRFinderPlus added)
5. **Noisy features hurt ML models** — more hits ≠ more information; wrong-biology features inflate dimensions and dilute true signal (revisit at Phase 7 SHAP)
6. **Co-carriage vs co-selection** — correlation consistent with both; cannot distinguish without genomic context
7. **AMRFinderPlus `--organism` mechanism** — organism-specific genes are absent from pan-bacterial DB entirely; `--organism` loads a separate profile set, not a confidence filter
8. **ISEScan `new` family** — structural IS element (transposase HMM + flanking IRs), not a false positive; unclassified by database coverage; include in feature matrix
9. **IS6/IS26 composite transposon architecture** — IS26 flanks ARG cassettes forming `IS26 → ARG → IS26` mobile units; high IS26 = clinical marker of resistance accumulation in KP
10. **pco/sil co-resistance operon** — COPPER/SILVER is a single genetic locus on IncH plasmids, not two separate genes that happen to co-occur
