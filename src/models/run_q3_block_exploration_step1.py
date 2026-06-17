"""
Q3 unsupervised feature-block exploration -- Step 1 (cheap diagnostic, future scope).

Question: does adding HMRG + IS element presence/absence to defence-only Q3
clustering move silhouette/ARI off the dereplicated anchor (silhouette=0.0631,
ARI-to-species=0.2002, ARI-to-ARG=0.0492; see NB09, re-verified 2026-06-17)?

Pre-registered expectation: minimal movement. HMRG (12 metal classes) and IS
(23 families) are both far lower-dimensional than defence (231 features), so
even if informative they cannot dominate a combined Euclidean distance: this
is a "do these blocks contain ANY signal at all" check, not yet a fair
contribution test (that needs each block reported alone, done here too).

ARG is deliberately NOT included in this step -- it is the block expected to
trivially inflate ARI-to-species (resistome content is a species barcode) and
should be looked at in isolation, not let loose into a first combined look.
IME is out of scope entirely (citability + tBLASTn noise, see prior decision).

Design:
  - Reuse the EXACT same 309 representative genomes NB09 selected (closest to
    phylogroup centroid in defence Euclidean space) for every block, so all
    results share the same genome sample and are directly comparable.
  - Binarize HMRG (12 per-class columns, excludes hmrg_metal_total/
    hmrg_metal_classes aggregates) and IS (23 per-family columns, excludes
    is_count_total aggregate) via count > 0 -- same presence/absence
    convention as defence dp_ columns.
  - Report each block ALONE (defence, HMRG, IS, HMRG+IS) and combined
    (defence+HMRG+IS), not just incremental stacking.
  - Euclidean K-means, same as the primary NB09 method (Jaccard sensitivity
    deferred for now -- this is a first look).

Outputs: printed comparison only. Nothing written to results/ yet -- this is
exploratory, not a tracked result.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
K_RANGE = range(2, 9)

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")

with open("results/q1_named_feat_cols.txt") as f:
    DEFENCE_COLS = f.read().splitlines()
assert len(DEFENCE_COLS) == 231

HMRG_EXCLUDE = {"hmrg_metal_total", "hmrg_metal_classes"}
HMRG_COLS = [c for c in fm.columns if c.startswith("hmrg_") and c not in HMRG_EXCLUDE]
IS_COLS = [c for c in fm.columns if c.startswith("is_") and c != "is_count_total"]
print(f"Defence cols: {len(DEFENCE_COLS)}, HMRG cols: {len(HMRG_COLS)}, IS cols: {len(IS_COLS)}")

# Binarize HMRG/IS -- same presence/absence convention as defence dp_
HMRG_BIN = fm[HMRG_COLS].gt(0).astype(float)
HMRG_BIN.columns = [f"hmrg_bin_{c}" for c in HMRG_COLS]
IS_BIN = fm[IS_COLS].gt(0).astype(float)
IS_BIN.columns = [f"is_bin_{c}" for c in IS_COLS]

fm_ext = pd.concat([fm, HMRG_BIN, IS_BIN], axis=1)

# --- Reproduce NB09's exact dereplication: 1 genome per phylogroup, closest to
#     centroid in DEFENCE Euclidean space (same selection for every block below,
#     so all results are directly comparable on the same genome sample) ---
X_defence = fm_ext[DEFENCE_COLS].to_numpy(dtype=float)
pg_col = fm_ext["phylogroup"].to_numpy(dtype=str)

rep_indices = []
for pg_id in np.unique(pg_col):
    mask = pg_col == pg_id
    grp = X_defence[mask]
    cent = grp.mean(axis=0)
    best = np.argmin(np.linalg.norm(grp - cent, axis=1))
    rep_indices.append(np.where(mask)[0][best])
rep_indices = np.array(rep_indices)
print(f"Dereplicated reps: {len(rep_indices)} (NB09 anchor: 309)")

fm_derep = fm_ext.iloc[rep_indices]
species_derep = fm_derep["species"].to_numpy(dtype=str)
arg_derep = fm_derep["arg_burden_tertile"].fillna("unknown").to_numpy(dtype=str)


def cluster_block(cols, label):
    X = fm_derep[cols].to_numpy(dtype=float)
    sils = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
        sils.append(silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0)
    best_k = list(K_RANGE)[int(np.argmax(sils))]
    best_labels = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE).fit_predict(X)
    ari_sp = adjusted_rand_score(species_derep, best_labels)
    ari_arg = adjusted_rand_score(arg_derep, best_labels)
    print(f"{label:<24} n_feat={len(cols):<4} best_K={best_k:<3} "
          f"silhouette={max(sils):.4f}  ARI_species={ari_sp:.4f}  ARI_arg={ari_arg:.4f}")
    return {"label": label, "n_feat": len(cols), "best_k": best_k,
            "silhouette": max(sils), "ari_species": ari_sp, "ari_arg": ari_arg}


print(f"\n{'='*90}")
print("Dereplicated (309 reps) -- Euclidean K-means, K=2..8")
print(f"{'='*90}")

results = []
results.append(cluster_block(DEFENCE_COLS, "Defence only (sanity vs anchor: K=3, sil=0.0631, ARI_sp=0.2002)"))
results.append(cluster_block(list(HMRG_BIN.columns), "HMRG only"))
results.append(cluster_block(list(IS_BIN.columns), "IS only"))
results.append(cluster_block(list(HMRG_BIN.columns) + list(IS_BIN.columns), "HMRG + IS (mobilome combo)"))
results.append(cluster_block(DEFENCE_COLS + list(HMRG_BIN.columns) + list(IS_BIN.columns), "Defence + HMRG + IS"))

print(f"\n{'='*90}")
print("DELTA vs defence-only anchor:")
anchor = results[0]
for r in results[1:]:
    print(f"  {r['label']:<30} d_silhouette={r['silhouette']-anchor['silhouette']:+.4f}  "
          f"d_ARI_species={r['ari_species']-anchor['ari_species']:+.4f}")
