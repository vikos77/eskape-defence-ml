"""
Q3 distance-metric sensitivity: reconstructed script for Supplementary Table S9.

Background: results.md/discussion.md cite a "Jaccard silhouette analysis" for K=2-8
(scores 0.039-0.044) as confirmation that the Euclidean-based Q3 null result (NB09,
K-means) is not an artefact of distance-metric choice on sparse binary defence data.
No script generating those numbers existed anywhere in the repo -- same migration-gap
pattern as the un-reconstructable feature_matrix_3335_named.parquet filter found
earlier in this audit. The CSV (results/supplement_q3_jaccard_sensitivity.csv) survived
but is not reproducible from current code, and its 7 values don't indicate whether it
was computed on the 367-feature or named-236 matrix.

This script rebuilds the sensitivity check from scratch on the named-236 matrix (231
FEAT_COLS, same spec_score>=0.70 filter as Q1/NB09), since K-means itself assumes
Euclidean centroids and is not meaningful under Jaccard: pairwise Jaccard distances are
computed, then agglomerative clustering (average linkage, precomputed distance) is used
in place of K-means, and silhouette is computed on the same precomputed distance matrix.

Outputs:
  results/supplement_q3_jaccard_sensitivity.csv  (k, silhouette_jaccard)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")

with open("results/q1_named_feat_cols.txt") as f:
    FEAT_COLS = f.read().splitlines()
assert len(FEAT_COLS) == 231

X = fm[FEAT_COLS].to_numpy(dtype=bool)
print(f"Feature matrix for Jaccard sensitivity: {X.shape}")

print("Computing pairwise Jaccard distance matrix (3335 x 3335)...")
D = squareform(pdist(X, metric="jaccard"))
print(f"Distance matrix: {D.shape}, dtype={D.dtype}")

rows = []
for k in range(2, 9):
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(D)
    sil = silhouette_score(D, labels, metric="precomputed")
    rows.append({"k": k, "silhouette_jaccard": float(sil)})
    print(f"  K={k}  silhouette(Jaccard)={sil:.4f}")

df = pd.DataFrame(rows)
df.to_csv("results/supplement_q3_jaccard_sensitivity.csv", index=False)
print("\nSaved: results/supplement_q3_jaccard_sensitivity.csv")
print(f"\nRange: {df['silhouette_jaccard'].min():.4f} - {df['silhouette_jaccard'].max():.4f}")
