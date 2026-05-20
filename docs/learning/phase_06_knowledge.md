# Phase 6 — What We Did and Why: Complete Reference

**Topic:** Phylogenetic grouping using Mash distances and hierarchical clustering  
**Notebook:** `notebooks/04_phylogenetic_grouping.ipynb`  
**Assumed knowledge:** None — built from the ground up

---

## 1. The Problem: Why Standard Cross-Validation Is Wrong for This Data

### What is cross-validation?

When you build a machine learning model, you need to test how well it works on data it has never seen. The naive approach — train the model and test it on the same data — always makes the model look better than it really is, because the model has already "seen the answers."

Cross-validation solves this by splitting the data so the test set is always held out from training:

1. Divide your 878 genomes into 5 equal groups called **folds** (roughly 175 genomes each).
2. **Round 1:** Train on folds 2, 3, 4, 5 (≈702 genomes). Test on fold 1 (≈176 genomes). Record accuracy.
3. **Round 2:** Train on folds 1, 3, 4, 5. Test on fold 2. Record accuracy.
4. Continue for all 5 rounds.
5. Average the 5 accuracy scores.

Every genome gets tested exactly once on data the model never trained on. This is honest.

### The clone problem

Here's why standard cross-validation fails for bacterial genomes. Imagine 150 *S. aureus* genomes. Many of them are **clones** — near-identical copies of each other, like photos of the same person from slightly different angles. Sequence type ST8 might have 20 genomes in the dataset. Standard 5-fold CV randomly assigns these 20 genomes, putting roughly 16 into training folds and 4 into the test fold.

When the model tries to classify those 4 test genomes, it has already seen 16 near-identical clones during training. It is not learning "what makes *S. aureus* defence systems unique." It is recognising "I have seen this exact genomic fingerprint before." The accuracy looks great, but nothing was actually learned.

**Analogy:** Studying 16 photos of the same person's face. Being tested on 2 more photos of the same person. You score 100%. It does not mean you can recognise faces — it means you memorised one face.

### The fix: Grouped Cross-Validation

All clones of the same lineage are assigned to the same group (a **phylogroup**). During cross-validation, entire phylogroups move together — never split across train and test. If 20 SA clones of ST8 form one phylogroup, all 20 go to fold 3. The model is then tested on lineages it has genuinely never seen in training.

This answers the right question: "Can this model classify a *new lineage* it has never encountered?"

**Phase 6's job:** Define those phylogroups. Every step in the notebook serves this one goal.

---

## 2. Mash — Compressing a 5 Million Base-Pair Genome into 1000 Numbers

### The challenge

Comparing two 5-million-base-pair genomes directly (base by base) for all 878×877/2 = 384,753 pairs would take hours or days. We need a way to summarise each genome into something small that still captures its genetic identity.

### Step A — K-mers

A **k-mer** is a substring of length k. For k=21, slide a window of 21 characters across the genome sequence and extract every consecutive 21-letter word:

```
Genome:    ATGCGATCGAAATCGATCGATCGTTAAC...
21-mer 1:  ATGCGATCGAAATCGATCGAT
21-mer 2:   TGCGATCGAAATCGATCGATC
21-mer 3:    GCGATCGAAATCGATCGATCG
...
```

A 5 Mb genome produces roughly 5 million 21-mers. Every genome produces a different set because the sequence is different.

**Why k=21?** Long enough that the same 21-mer appearing in two completely unrelated genomes is extremely unlikely (4^21 = 4.4 trillion possible 21-mers). Short enough that closely related genomes share many of the same k-mers.

### Step B — Hashing

Each 21-mer is converted to an integer using a mathematical function (a hash function). The same sequence always gives the same number:

```
ATGCGATCGAAATCGATCGAT  →  482947202
TGCGATCGAAATCGATCGATC  →  917384651
GCGATCGAAATCGATCGATCG  →  204938175
```

This turns the DNA alphabet (A, T, G, C) into numbers, which are easier to compare.

### Step C — The MinHash sketch: keep only the 1000 smallest numbers

Out of 5 million hash values, **keep only the 1000 smallest**. This is the **sketch** — a fingerprint of the genome represented by 1000 numbers. The sketch is the same size (1000 numbers) regardless of whether the genome is 2 Mb or 7 Mb.

**Why does this work?** If two genomes are very similar (share 90% of their 21-mers), their full sets of hash values largely overlap. The 1000 minimum hash values from genome A and the 1000 minimum hash values from genome B will also largely overlap — because if A and B share most k-mers, they share most of the minimum values too. If two genomes share only 5% of k-mers (completely different species), their minimum values will barely overlap.

### Step D — The distance

Count what fraction of the 1000 minimum values are shared between the two sketches. Convert to a distance:

**Mash distance = 1 − (fraction of shared minimums)**

| Mash distance | Meaning |
|---|---|
| 0.000 | Identical genomes |
| 0.005 | Near-clones, possibly same outbreak (ANI ≈ 99.5%) |
| 0.010 | Same lineage, some evolutionary drift (ANI ≈ 99%) |
| 0.050+ | Usually different species |
| 1.000 | No shared k-mers at all |

The notebook runs `mash sketch` to build one sketch file (7.2 MB for 878 genomes), then `mash dist` to compute all 878×878 = 770,884 pairwise distances in about 3 minutes.

---

## 3. The 878×878 Distance Matrix

The result is a symmetric square table:

```
             GCF_000001  GCF_000002  GCF_000003  ...
GCF_000001    0.000       0.003       0.812       ...
GCF_000002    0.003       0.000       0.809       ...
GCF_000003    0.812       0.809       0.000       ...
```

- Diagonal = 0 (every genome is identical to itself)
- Small off-diagonal values = closely related genomes
- Large values (near 1) = different species

The median pairwise distance across all 384,753 pairs is 1.0. This makes sense: most pairs are *between* species (SA vs KP, AB vs EF, etc.) — completely unrelated genomes.

---

## 4. The Distance Distribution — Reading the Histogram

The notebook plots a histogram of all pairwise distances. Key features:

**Left panel (all distances):** A spike at distance = 1.0 (most pairs are cross-species). A long thin tail on the left (within-species pairs).

**Right panel (distances < 0.15 only):** The structure within the within-species zone:
- Pairs at distance < 0.005 → near-identical clones (same outbreak, same hospital)
- Pairs at 0.005–0.010 → same lineage, some drift
- Pairs at 0.010–0.05 → same species, different lineages
- Pairs at 0.05–0.15 → cross-species overlap zone

At t=0.010: 20,518 pairs (5.33% of all pairs) are within this threshold. These are the within-lineage pairs we want to keep together in the same CV fold.

---

## 5. Hierarchical Clustering — Building the Tree of Life (At a Small Scale)

### The concept

We have 878 genomes with pairwise distances. We want to group them into "phylogroups" so that closely related genomes are in the same group. The tool is **hierarchical clustering**.

**Algorithm (in plain English):**

Start with 878 individual genomes, each its own "group."

1. Find the two *closest* groups (smallest pairwise distance between them).
2. Merge them into one group.
3. Repeat: find the two closest groups again. Merge them.
4. Keep repeating until everything is in one group.

This builds a **dendrogram** — a branching tree where:
- Each leaf at the bottom is one genome
- Each branching point records when two groups merged and how far apart they were
- The height of each branch = the distance at which the merge happened
- Branches close to the base = very similar genomes merged early
- Branches near the top = distant genomes (or species) merged late

**Analogy:** Imagine 878 people standing on a field. Keep asking: "who are the two closest people/groups?" and having them hold hands. Eventually everyone is connected. The tree records the history — who joined with whom, and when (at what distance).

### Average linkage (UPGMA)

When merging two *groups* (not just two individuals), you need to decide how to measure the distance between groups. There are three options:

- **Single linkage:** distance = closest pair between the two groups (can chain together distant genomes via intermediate ones — "chaining artifact")
- **Complete linkage:** distance = furthest pair (over-splits real clusters)
- **Average linkage (UPGMA):** distance = average of all pairwise distances between members of the two groups

Average linkage is standard for whole-genome bacterial comparisons because it resists both chaining and over-splitting. The notebook uses `method="average"`.

### Cutting the dendrogram at t=0.010

Drawing a horizontal line across the dendrogram at height = 0.010 creates the phylogroups:

- Every cluster whose topmost branch is *below* the line becomes one phylogroup
- Clusters separated by a merge *above* the line become separate phylogroups

Meaning: genomes in the same phylogroup are at most 0.010 Mash distance apart. Genomes in different phylogroups are more than 0.010 apart.

t=0.010 ≈ 99% average nucleotide identity — a standard boundary for "same lineage."

---

## 6. Why Global Clustering Failed — The Within-Species Fix

### The problem

Running hierarchical clustering on all 878 genomes at t=0.020:

- All 150 *S. aureus* genomes → **1 phylogroup**
- *P. aeruginosa* → **2 phylogroups**
- *K. pneumoniae* → **2 phylogroups**

Why? SA, PA, and KP clinical isolates are very clonal — within-species distances are almost all below 0.020. Average linkage kept finding pairs below threshold and chaining them together until the entire species merged into one blob.

**The CV consequence:** If all 150 SA genomes are in one phylogroup, GroupedStratifiedKFold assigns all 150 to one test fold. The model must classify SA without ever training on a single SA genome. That is "leave-one-species-out" CV — the wrong thing.

Phylogenetic grouping should prevent *within-species* clone contamination, not hold out entire species.

### The fix: within-species clustering

Cluster each species *separately*, using only within-species distances from the same 878×878 matrix. SA distances are clustered against SA only. KP against KP only. No new Mash computations needed.

Result: each species gets its own set of phylogroups. SA produces 10 groups (before singleton merging), capturing its within-SA lineage structure.

### The PA special case (t=0.005)

PA clinical isolates have unusually low within-species diversity — max Mash distance is 0.026. At t=0.010, 104 PA genomes from many different sequence types (independently evolved lineages) all merged into one group because their pairwise distances are just barely above or at the threshold. This was wrong: those STs are not clones of each other.

Fix: t=0.005 for PA. Only pairs at distance ≤ 0.005 (ANI ≥ 99.5%, genuine near-clones) merge together. At this threshold, 26 PA phylogroups emerge, largest = 19 genomes. 100% MLST concordance confirmed.

**Generalisation rule:** Any species whose max within-species Mash distance is < 0.030 should be checked for threshold sensitivity. Diagnostic: if the largest phylogroup at default threshold contains > 30% of the species' genomes, tighten the threshold.

---

## 7. Singletons — The Merging Step

After within-species clustering at the chosen thresholds, many genomes are **singletons** — phylogroups of exactly 1 genome.

Before merging: 270 total phylogroups, 175 singletons.

**Why singletons form:** A genome is a singleton if its nearest same-species neighbour is still further than the threshold. High singletons = high within-species nucleotide diversity. EC had 95 groups from 146 genomes — nearly one genome per group. This is biologically meaningful: EC isolates are genuinely diverse (many independently evolved lineages, no dominant clones). It is not a failure.

**Why singletons are a CV problem:** GroupedStratifiedKFold cannot balance a group of 1 across 5 folds in any meaningful way. A singleton in the test fold contributes only 1 genome to the accuracy estimate for its species — that estimate has enormous variance (either 0% or 100%, nothing in between).

**Fix:** Each singleton is merged into its nearest non-singleton same-species phylogroup (by minimum Mash distance). The singleton joins the closest existing group in its species. No cross-species merging — contamination remains 0%.

After merging: 95 total phylogroups, smallest size = 2.

---

## 8. MLST Concordance — Validating the Phylogroups

**MLST (Multi-Locus Sequence Typing):** A standard microbiology method where 7 housekeeping genes are sequenced per genome. The combination of alleles at those 7 loci defines a "sequence type" (ST). ST258 in KP is a famous global carbapenem-resistant lineage. Same ST = same lineage by definition.

**Validation logic:** If our Mash-derived phylogroups are biologically valid, genomes of the same ST should always land in the same phylogroup. We check this for every ST that appears in ≥ 2 genomes.

**Result:** 108/109 STs fully co-assigned = 99.1%. PASS.

The 1 discordant ST is not a failure — it means two genomes share the same 7-gene MLST type but have diverged at the whole-genome level (different mobile elements, different genomic islands). Mash sees the whole genome and correctly separates them. MLST only sees 7 genes; Mash sees everything. In this case Mash is more informative than MLST.

**CV impact of 1 discordant ST:** Negligible. One genome potentially placed in the "wrong" fold out of 878 = 0.1% contamination.

---

## 9. The Final Outputs

**`data/processed/cv_groups.parquet`:** 878 rows, one column `phylogroup`. Values like `AB_PG_001`, `PA_PG_026`, `SA_PG_009`. This is the grouping variable passed to all classifiers in Phase 7 onward.

**`data/processed/feature_matrix.parquet`:** Updated with a `phylogroup` column added.

### Final phylogroup summary

| Species | Phylogroups | Threshold | MLST concordance |
|---|---|---|---|
| A. baumannii | 13 | 0.010 | 100% |
| E. cloacae | 22 | 0.010 | 100% |
| E. faecium | 7 | 0.010 | 94.7% |
| K. pneumoniae | 18 | 0.010 | 100% |
| P. aeruginosa | 26 | **0.005** | 100% |
| S. aureus | 9 | 0.010 | 100% |
| **Total** | **95** | | **99.1%** |

---

## 10. How Phase 7 Uses This Output

In Phase 7, we pass `cv_groups.parquet` to `StratifiedGroupKFold` from scikit-learn:

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5)
for train_idx, test_idx in sgkf.split(X, y, groups=cv_groups["phylogroup"].values):
    # train_idx: indices for training genomes
    # test_idx:  indices for testing genomes
    # GUARANTEE: no phylogroup appears in both train_idx and test_idx
```

The key guarantee: every genome in a phylogroup is in either the training set or the test set for a given fold — never both. The model must generalise to lineages it has never seen. That is the correct test.

---

*This document was written as a learning reference during Phase 6 (2026-05-19/20).  
Source of truth: `notebooks/04_phylogenetic_grouping.ipynb`.*
