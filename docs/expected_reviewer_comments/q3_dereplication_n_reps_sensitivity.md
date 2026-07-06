# Expected Reviewer Comment: Q3 Dereplication — Number of Representatives per Phylogroup

## Anticipated comment

> "The authors use one representative genome per phylogroup for dereplication. Was this
> choice validated? Using a single representative may discard genuine within-phylogroup
> defence variation, potentially over-deflating the clustering signal."

## Response

The 1-per-phylogroup choice is the standard conservative approach for phylogenetic
dereplication in genomic clustering studies. To verify that the NEGATIVE verdict for Q3
is not an artefact of this choice, we re-ran the dereplication analysis at 2 and 3
representatives per phylogroup (closest to group centroid in each case).

### Results

Full dataset (3,335 genomes): ARI vs species = 0.3624

| Representatives per phylogroup | Genomes | Best K | Max silhouette | ARI vs species | Delta ARI | Verdict |
|---|---|---|---|---|---|---|
| 1 | 309 | 2 | 0.100 | 0.102 | 0.261 | SENSITIVE |
| 2 | 618 | 2 | 0.040 | 0.174 | 0.189 | SENSITIVE |
| 3 | 858 | 2 | 0.040 | 0.158 | 0.205 | SENSITIVE |

Threshold for sensitivity: delta ARI ≥ 0.10.

All three choices produce the same SENSITIVE verdict. Delta ARI ranges from 0.19 to 0.26
across the three levels — all well above the 0.10 threshold. Notably, increasing from
1 to 2–3 representatives does not recover cluster structure: the maximum silhouette
actually drops from 0.100 to ~0.040 at 2 and 3 reps, indicating that adding more
representatives introduces noise rather than recovering signal.

The Q3 NEGATIVE conclusion is therefore robust to the choice of dereplication stringency.

## Implication

The full-dataset clustering (ARI 0.36) is driven by clonal inflation — most prominently
the over-representation of IC2 *A. baumannii* genomes with an identical SspBCDE-only
defence profile. Once clonal redundancy is removed at any level of stringency, no
discrete archetype structure remains.
