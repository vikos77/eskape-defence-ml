# Expected Reviewer Comment: Defence system copy number vs presence/absence for Q2

## Anticipated criticism

> "Q2 uses binary presence/absence (dp_*) features to predict ARG burden tertile. Defence
> system copy number may carry more information than mere presence -- a genome with two RM
> systems is not the same as a genome with one. Did the authors check whether count data
> would have revealed a significant signal that the binary encoding missed?"

---

## Rebuttal

**Yes, checked, and it does not change the conclusion.**

### What "count" means here

`dc_*` (copy number per system) is not a raw or naively-summed count. It is computed as
`max(DefenseFinder instance count, PADLOC instance count)` per system per genome --
already deduplicated at the system-instance level by both tools independently before the
two are combined (DefenseFinder's `systems.tsv` is one row per complete system instance;
PADLOC's per-gene rows are collapsed via `(system, system.number)` before counting). We
verified this against raw tool output rather than assuming it: e.g. *A. baumannii*
`GCF_000505685.1` has two independently-confirmed RM_Type_I systems on two different
contigs (`NC_023028.1`, `NC_023031.1`), each a complete 3-gene operon; `GCF_003431865.1`
has four RM_Type_I instances across three replicons. `dc_RM_Type_I` correctly reports 2
and 4 respectively -- genuine copy number, not a parsing or deduplication artefact. We also
confirmed the identity `dp_* == (dc_* > 0)` holds exactly across all 236 named systems and
3,335 genomes, so swapping encodings does not change which systems are in the model, only
how each one is represented numerically.

### Sensitivity analysis: same feature set, dp_ vs dc_ encoding, 20 CV seeds

Per-species Q2 (RF, GroupedStratifiedKFold, same >=5% prevalence filter, same
hyperparameters as the primary named-236 Q2 analysis), run with 20 independent CV random
seeds per encoding to separate genuine effects from fold-split noise:

| Species | frac. of 20 seeds p<0.05 (dp_) | frac. (dc_) | mean AUROC (dp_) | mean AUROC (dc_) |
|---|---|---|---|---|
| AB | 0.00 | 0.00 | 0.741 | 0.734 |
| EC | 0.50 | 0.45 | 0.674 | 0.676 |
| EF | 0.40 | 0.50 | 0.695 | 0.726 |
| KP | 0.75 | 0.95 | 0.762 | 0.781 |
| PA | 0.50 | 0.50 | 0.714 | 0.769 |
| SA | 0.00 | 0.00 | 0.714 | 0.669 |

A single-seed spot check initially suggested *E. faecium* crossed the BH-corrected
significance threshold under `dc_` but not `dp_`. The 20-seed sweep shows this was fold-split
lottery: both encodings' EF p-values range from <0.01 to ~0.2-0.3 depending on the seed.

The real, seed-stable pattern: count encoding gives a modest, consistent AUROC gain in KP
(+0.019), PA (+0.055), and EF (+0.031); EC and AB are unchanged; SA gets consistently *worse*
with counts (-0.045). None of this is large enough, or consistent enough, to flip the
primary result -- KP, the strongest candidate either way, does not reliably survive
BH correction across species under either encoding.

### Conclusion for the manuscript

The Q2 null result (Option A: all 6 species non-significant after BH correction) is robust
to the choice between presence/absence and copy-number encoding. This was checked as an
internal robustness exercise; it does not change any reported number and is not required in
the manuscript text. If a reviewer raises this specific question, the analysis above is the
complete answer. Scripts: `src/models/run_q2_dc_sensitivity.py`,
`src/models/run_q2_seed_stability.py`. Results: `results/q2_dc_sensitivity_results.json`,
`results/q2_seed_stability.json` (untracked, matching the convention for sensitivity-analysis
outputs vs the primary `q1_named_results.json`/`q2_named_results.json`).
