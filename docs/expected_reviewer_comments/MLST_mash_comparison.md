# Expected Reviewer Comment: Why Mash distances rather than MLST sequence types for phylogrouping?

**Anticipated comment:** "Why did the authors derive phylogroups from Mash whole-genome
distances rather than using MLST sequence types directly, given that MLST concordance
was 92.4%? MLST is an established standard for defining bacterial lineages and would
have been simpler."

## Prepared response

Phylogroups were derived from Mash whole-genome distances rather than MLST sequence
types for three reasons. First, MLST coverage was incomplete across the 3,335-genome
dataset due to novel alleles and missing loci, while Mash operates on raw sequence
with no database dependency. Second, MLST groups genomes by lineage label regardless
of actual pairwise divergence, which would force genuinely diverged same-ST genomes
into the same CV fold and introduce, rather than prevent, leakage. Third, Mash distance
allows per-species threshold calibration, validated against MLST concordance (92.4%
overall), providing consistent grouping granularity across six species with different
MLST scheme resolutions.

## Supporting points if pressed

- High concordance (92.4%) validates that Mash groups capture real clone structure;
  it does not argue for replacing Mash with ST. The direction of the argument matters:
  concordance confirms Mash is right, not that ST would be equivalent.
- The 7.6% discordant cases are same-ST genomes that Mash correctly separated due to
  genuine within-ST divergence. Using ST would force these into the same fold, which
  is the wrong direction for leakage control.
- Each ESKAPE species has its own MLST scheme with different discriminatory power.
  ST resolution is not comparable across species. Mash with per-species thresholds
  gives consistent biological granularity across all six.
- Genomes with no valid ST (novel alleles, missing loci) would require exclusion or
  singleton treatment under ST-based grouping. Mash handles them without any special
  case.
