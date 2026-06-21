"""
Q3 distance-metric sensitivity (S9): Jaccard vs Euclidean clustering, full
367-feature pool (359 FEAT_COLS from q1_367_feat_cols.txt).

Outputs:
  results/supplement_q3_jaccard_367.csv  (k, silhouette_jaccard)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")

with open("results/q1_367_feat_cols.txt") as f:
    FEAT_COLS = f.read().splitlines()

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
df.to_csv("results/supplement_q3_jaccard_367.csv", index=False)
print("\nSaved: results/supplement_q3_jaccard_367.csv")
print(f"\nRange: {df['silhouette_jaccard'].min():.4f} - {df['silhouette_jaccard'].max():.4f}")
