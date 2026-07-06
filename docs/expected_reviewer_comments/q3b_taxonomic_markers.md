# Expected Reviewer Comment: Q3b — Taxonomic Markers in ARG and HMRG Blocks

## Anticipated comment

> "The authors claim IS element family composition adds clustering signal beyond defence
> systems (Q3b). However, ARG and HMRG profiles are known to be partially species-specific.
> Were these blocks filtered for taxonomic markers before clustering? If not, the high ARI
> values for ARG and HMRG blocks may simply reflect species identity encoded via
> chromosomal resistance genes, not genuine mobile-element ecology."

## Response

This concern is valid and was addressed prospectively using the same spec_score filter
applied to defence features. The asymmetric-filtering problem — running filtered defence
against unfiltered mobile blocks — would inflate the mobile signal and is not a valid
comparison.

### Spec-score filter

For each feature in each block: `spec_score = std(per-species prevalence) / 0.5`.
Features with spec_score ≥ 0.70 are species-restricted markers, excluded from analysis.
This is the identical threshold used for the Q1/Q3 defence block (Cell 2 of NB09),
which removed 8 of 367 defence features.

### Results per block

**HMRG (12 features → 10 retained, 2 markers):**

| Feature | spec_score | Pattern |
|---|---|---|
| arsenic | 0.893 | ~99% in EC+KP, ~1% in EF+PA, 0% in AB |
| copper_silver | 0.767 | ~89% in EC, ~77% in EF, ~58% in KP, 0% in AB+PA |

These are gram-negative vs gram-positive metal resistance class splits —
taxonomic, not mobile-element ecology.

**IS (24 features → 21 retained, 3 markers):**

| Feature | spec_score | Pattern |
|---|---|---|
| IS1182 | 0.754 | 88% in EF, 98% in SA, <10% elsewhere |
| IS982 | 0.710 | 96% in EF only |
| IS1 | 0.708 | 72% in EC+KP, <9% elsewhere |

**ARG (143 features → 134 retained, 9 markers):**

| Gene | spec_score | Pattern |
|---|---|---|
| fosA | 0.808 | 99.7% in PA only (chromosomal) |
| blaADC-25 | 0.745 | 100% in AB only (chromosomal) |
| aac(6')-Ii | 0.744 | 99.8% in EF only (intrinsic) |
| blaPAO | 0.743 | 99.7% in PA only (chromosomal) |
| aph(3')-IIb | 0.742 | 99.5% in PA only (chromosomal) |
| catB7 | 0.734 | 98.5% in PA only (chromosomal) |
| msr(C) | 0.725 | 97.3% in EF only (intrinsic) |
| OqxA | 0.710 | 95.2% in KP only |
| OqxB | 0.710 | 95.2% in KP only |

These 9 genes are documented chromosomal/intrinsic resistance genes — not horizontally
acquired ARG. Their presence is a species identifier, not a mobile-element signal.

### Technical note on spec_score computation

The spec_score filter uses `pd.DataFrame.std()` which defaults to `ddof=1` (sample std,
N-1 denominator), consistent with the original defence-block filter applied in NB09 Cell 2.
An earlier analysis using numpy `std()` (ddof=0) gave lower spec_scores for borderline
features; those results are superseded by the ddof=1 computation.

### Impact of filtering on Q3b ARI (spec_score ddof=1, 309 dereplicated genomes, K=6)

| Block | Unfiltered n | Unfiltered ARI | Filtered n | Filtered ARI | Delta | Interpretation |
|---|---|---|---|---|---|---|
| defence_only | 359 | 0.219 | 359 | 0.219 | 0.000 | Baseline — already filtered |
| hmrg | 12 | 0.380 | 9 | 0.101 | −0.279 | Large collapse — 3 markers drove 73% of signal |
| IS | 24 | 0.304 | 19 | 0.237 | −0.067 | **Smallest decline — IS is the most mobile block** |
| arg | 143 | 0.557 | 134 | 0.069 | −0.488 | Near-complete collapse — 9 chromosomal markers |
| mobile combined | 179 | 0.684 | 162 | 0.226 | −0.458 | Driven by ARG markers |
| all combined | 538 | 0.691 | 521 | 0.355 | −0.336 | Best combination post-filter |

Filtered 95% CIs (500-bootstrap): defence [0.059, 0.320], IS [0.150, 0.341],
hmrg [0.048, 0.144], arg [0.011, 0.129], mobile [0.163, 0.337], all [0.161, 0.510].

### Conclusion

IS element family composition shows the smallest decline after marker filtering (ARI
0.304→0.237, ΔARI=−0.067) compared to ARG (ΔARI=−0.488) and HMRG (ΔARI=−0.279).
IS remains the closest non-defence block to the defence baseline (gain+0.018), consistent
with IS families being genuinely mobile across species. However, the gain over defence is
marginal and does not constitute a strong independent signal.

ARG clustering collapses almost entirely — 9 chromosomal/intrinsic resistance genes
drove 88% of the unfiltered ARG ARI. This is a methodologically important finding:
the original Q3b positive result was substantially inflated by features that are lineage
identifiers, not mobile-element ecology signals.

## Implication for manuscript framing

The Q3b claim cannot be "mobile element burden recovers species identity better than
defence systems" — that was an artefact of unfiltered ARG blocks. The defensible claim is:

> "After removing taxonomically restricted features (spec_score ≥ 0.70) from all blocks,
> ARG gene profiles lose nearly all clustering signal (ARI 0.557→0.069), demonstrating
> that the unfiltered ARG result was driven by 9 chromosomally encoded resistance genes
> specific to individual species. IS element family composition shows the smallest decline
> after filtering (ARI 0.304→0.237), consistent with IS families being genuinely mobile
> across species, but the gain over defence systems alone is marginal (+0.018). No single
> mobile-element block adds strong independent clustering signal above defence systems after
> symmetric marker filtering."

## Related reviewer comments
- See also: `q3_dereplication_n_reps_sensitivity.md` for the Q3 dereplication robustness check.
