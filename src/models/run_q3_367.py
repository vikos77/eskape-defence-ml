"""
Q3: unsupervised clustering (K-means + Ward hierarchical) on the full
367-feature defence-system matrix.

Numbers-only port of notebooks/09_unsupervised_archetypes.ipynb's Sections
1/2/3/6 (build_nb09.py: K_SELECTION, GAP_STAT, KMEANS_RUN, WARD_LINKAGE,
DEREPLICATE, DEREP_KMEANS) -- no plots, no notebook regeneration (deferred to
the separate NB06-10 audit task). Uses FEAT_COLS from q1_367_feat_cols.txt
(359 features, same spec_score>=0.70 filter as Q1).

Outputs:
  results/q3_367_results.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster

RANDOM_STATE = 42

fm = pd.read_parquet("data/processed/feature_matrix_3335.parquet")
with open("results/q1_367_feat_cols.txt") as f:
    FEAT_COLS = f.read().splitlines()
print(f"FEAT_COLS (Q3 defence, 367 pool): {len(FEAT_COLS)}")

species_labels = fm["species"].to_numpy(dtype=str)
arg_labels = fm["arg_burden_tertile"].fillna("unknown").to_numpy(dtype=str)
X_q3 = fm[FEAT_COLS].values.astype(float)
print(f"Q3 feature matrix: {X_q3.shape}")

# ── Section 1: K selection (silhouette, CH, gap statistic) ─────────────────
k_range = range(2, 13)
sil_scores, ch_scores = [], []
print(f"\nK-means K=2..12 on {X_q3.shape} (n_init=20)...")
for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    lbl = km.fit_predict(X_q3)
    sil_scores.append(silhouette_score(X_q3, lbl))
    ch_scores.append(calinski_harabasz_score(X_q3, lbl))
    print(f"  K={k:2d}  silhouette={sil_scores[-1]:.4f}  CH={ch_scores[-1]:.1f}")

best_k = k_range.start + int(np.argmax(sil_scores))
print(f"\nBest K by silhouette: {best_k} ({max(sil_scores):.4f})")
print(f"Silhouette at K=6: {sil_scores[6-2]:.4f}")


def _wcss(X, labels):
    total = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        total += float(np.sum((Xc - Xc.mean(axis=0)) ** 2))
    return total


rng_gap = np.random.RandomState(RANDOM_STATE)
n_refs = 10
gap_vals, gap_errs = [], []
gap_ks = [1] + list(k_range)
print(f"\nGap statistic K=1..12 (n_refs={n_refs})...")
for k in gap_ks:
    lbl_k = (np.zeros(len(X_q3), dtype=int) if k == 1
              else KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_q3))
    log_Wk = np.log(_wcss(X_q3, lbl_k) + 1e-10)
    ref_log_Wks = []
    for _ in range(n_refs):
        X_ref = rng_gap.uniform(0, 1, X_q3.shape)
        lbl_r = (np.zeros(len(X_ref), dtype=int) if k == 1
                 else KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit_predict(X_ref))
        ref_log_Wks.append(np.log(_wcss(X_ref, lbl_r) + 1e-10))
    gap = np.mean(ref_log_Wks) - log_Wk
    gap_err = np.std(ref_log_Wks) * np.sqrt(1 + 1 / n_refs)
    gap_vals.append(gap)
    gap_errs.append(gap_err)
    print(f"  K={k:2d}  gap={gap:.4f}  se={gap_err:.4f}")

gap_optimal_k = gap_ks[0]
for i in range(len(gap_ks) - 1):
    if gap_vals[i] >= gap_vals[i + 1] - gap_errs[i + 1]:
        gap_optimal_k = gap_ks[i]
        break
print(f"\nGap-statistic optimal K: {gap_optimal_k}")

# ── Section 2: K-means at best_k and forced K=6 ─────────────────────────────
km_best = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE).fit(X_q3)
labels_best = km_best.labels_
km_6 = KMeans(n_clusters=6, n_init=30, random_state=RANDOM_STATE).fit(X_q3)
labels_6 = km_6.labels_

ari_best_species = adjusted_rand_score(species_labels, labels_best)
ari_6_species = adjusted_rand_score(species_labels, labels_6)
ari_best_arg = adjusted_rand_score(arg_labels, labels_best)
ari_arg_k6 = adjusted_rand_score(arg_labels, labels_6)

print(f"\nK-means K={best_k}: ARI vs species={ari_best_species:.4f}, ARI vs ARG={ari_best_arg:.4f}")
print(f"K-means K=6 (forced):  ARI vs species={ari_6_species:.4f}")

# ── Section 3: HC Ward agreement ────────────────────────────────────────────
print(f"\nWard linkage on {X_q3.shape}...")
Z = linkage(X_q3, method="ward", metric="euclidean")
hc_labels = fcluster(Z, t=best_k, criterion="maxclust") - 1
ari_hc_sp = adjusted_rand_score(species_labels, hc_labels)
ari_hc_km = adjusted_rand_score(labels_best, hc_labels)
ari_arg_hc = adjusted_rand_score(arg_labels, hc_labels)
print(f"Hierarchical K={best_k}: ARI vs species={ari_hc_sp:.4f}, ARI vs K-means={ari_hc_km:.4f}")

# ── Section 6: dereplication robustness (1 genome per phylogroup) ──────────
pg = pd.read_parquet("data/processed/cv_groups_3335.parquet")
fm_pg = fm.copy()
if "phylogroup" in fm_pg.columns:
    pg_col = fm_pg["phylogroup"].to_numpy(dtype=str)
else:
    fm_pg = fm_pg.join(pg.iloc[:, 0].rename("phylogroup"))
    pg_col = fm_pg["phylogroup"].fillna("UNASSIGNED").to_numpy(dtype=str)
n_pgs = len(np.unique(pg_col))
print(f"\nUnique phylogroups: {n_pgs}")

rep_indices = []
for pg_id in np.unique(pg_col):
    mask = pg_col == pg_id
    grp = X_q3[mask]
    cent = grp.mean(axis=0)
    best = np.argmin(np.linalg.norm(grp - cent, axis=1))
    rep_indices.append(np.where(mask)[0][best])
rep_indices = np.array(rep_indices)
X_derep = X_q3[rep_indices]
sp_derep = species_labels[rep_indices]
arg_derep = arg_labels[rep_indices]
print(f"Dereplicated dataset: {X_derep.shape[0]} genomes (1 representative per Mash phylogroup)")

sil_derep = []
for k in k_range:
    lbl = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X_derep)
    s = silhouette_score(X_derep, lbl) if len(np.unique(lbl)) > 1 else 0.0
    sil_derep.append(s)
best_k_d = k_range.start + int(np.argmax(sil_derep))
labels_d = KMeans(n_clusters=best_k_d, n_init=30, random_state=RANDOM_STATE).fit_predict(X_derep)
ari_d_sp = adjusted_rand_score(sp_derep, labels_d)
ari_d_arg = adjusted_rand_score(arg_derep, labels_d)
delta = abs(ari_d_sp - ari_best_species)

print(f"\nDereplicated K-means ({X_derep.shape[0]} genomes):")
print(f"  Best K: {best_k_d} (silhouette={max(sil_derep):.4f})")
print(f"  ARI vs species: {ari_d_sp:.4f} (full-dataset: {ari_best_species:.4f}), delta={delta:.4f}")
print(f"  ARI vs ARG: {ari_d_arg:.4f}")
print(f"  Verdict: {'SENSITIVE' if delta >= 0.10 else 'ROBUST'} (threshold 0.10)")

# HC Ward on dereplicated set too, for KM-vs-HC agreement check at the derep scale
Z_d = linkage(X_derep, method="ward", metric="euclidean")
hc_labels_d = fcluster(Z_d, t=best_k_d, criterion="maxclust") - 1
ari_hc_km_d = adjusted_rand_score(labels_d, hc_labels_d)
ari_hc_sp_d = adjusted_rand_score(sp_derep, hc_labels_d)
print(f"  Derep HC Ward K={best_k_d}: ARI vs species={ari_hc_sp_d:.4f}, ARI vs KM={ari_hc_km_d:.4f}")

results = {
    "n_features": len(FEAT_COLS),
    "full": {
        "n_genomes": int(X_q3.shape[0]),
        "sil_scores": sil_scores, "ch_scores": ch_scores, "k_range": list(k_range),
        "best_k": best_k, "best_k_silhouette": float(max(sil_scores)),
        "silhouette_at_k6": float(sil_scores[6 - 2]),
        "gap_ks": gap_ks, "gap_vals": gap_vals, "gap_errs": gap_errs,
        "gap_optimal_k": int(gap_optimal_k),
        "ari_kmeans_bestk_vs_species": float(ari_best_species),
        "ari_kmeans_bestk_vs_arg": float(ari_best_arg),
        "ari_kmeans_k6_vs_species": float(ari_6_species),
        "ari_kmeans_k6_vs_arg": float(ari_arg_k6),
        "ari_hc_ward_vs_species": float(ari_hc_sp),
        "ari_hc_ward_vs_kmeans": float(ari_hc_km),
        "ari_hc_ward_vs_arg": float(ari_arg_hc),
    },
    "dereplicated": {
        "n_genomes": int(X_derep.shape[0]), "n_phylogroups": int(n_pgs),
        "best_k": best_k_d, "best_k_silhouette": float(max(sil_derep)),
        "ari_kmeans_vs_species": float(ari_d_sp), "ari_kmeans_vs_arg": float(ari_d_arg),
        "delta_ari_vs_full": float(delta),
        "robustness_verdict": "SENSITIVE" if delta >= 0.10 else "ROBUST",
        "ari_hc_ward_vs_species": float(ari_hc_sp_d), "ari_hc_ward_vs_kmeans": float(ari_hc_km_d),
    },
}

with open("results/q3_367_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/q3_367_results.json")
print("Done.")
