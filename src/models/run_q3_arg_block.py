"""
Q3 unsupervised feature-block exploration -- Step 2: ARG presence/absence block.

Pre-registered expectation (see conversation / advisor note): ARG gene content is
the most species-endemic block of all -- expect the strongest silhouette and the
highest ARI-to-species, more extreme than even HMRG's near-categorical barcode
behaviour, because individual resistance genes (e.g. blaOXA-23/blaOXA-66 in AB) are
acquired and fixed within specific clonal lineages. This is checked, not assumed --
the per-gene prevalence table is inspected before any clustering claim is made.

New parsing required: ResFinder's per-genome ResFinder_results_tab.txt lists
detected genes by name (one row per hit, already at ResFinder's own internal
identity/coverage thresholds -- same detection criterion already used for
arg_count_unique in NB01). No file in the matrix currently captures per-gene
presence/absence (only aggregate arg_count_unique/total exist) -- this is built
fresh here, analogous to dp_ for defence systems.

Prevalence filter: keep genes present in >=1% of all 3,335 genomes (>=34 genomes).
646 distinct gene names exist across the dataset; 160 are singletons (1 genome only)
and contribute pure noise to a Euclidean clustering distance. 143 genes survive the
filter. This is a clustering-stability filter, not a Q2-style per-species filter --
no species-identity confound applies here since this is unsupervised, not predicting
species from features.

Reuses the same 309 dereplicated representative genomes (selected via defence
Euclidean centroid, see run_q3_block_exploration_step1.py) for direct comparability
with the HMRG/IS results already obtained.

Outputs: printed comparison only -- exploratory, nothing written to results/ yet.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
K_RANGE = range(2, 9)
PREV_THRESH = 0.01  # >=1% of 3335 genomes

SPECIES = ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]
INTERIM = Path("data/interim")


def norm_acc(s: str) -> str:
    prefix, version = s.rsplit("_", 1)
    return f"{prefix}.{version}"


# ── Build ARG presence/absence matrix from raw ResFinder output ─────────────
print("Parsing ResFinder per-gene presence across all genomes...")
arg_presence = {}  # genome_id -> set of gene names

for sp in SPECIES:
    res_dir = INTERIM / sp / "resfinder"
    for genome_dir in sorted(res_dir.iterdir()):
        if not genome_dir.is_dir():
            continue
        genome_id = norm_acc(genome_dir.name)
        results_file = genome_dir / "ResFinder_results_tab.txt"
        if not results_file.exists() or results_file.stat().st_size == 0:
            arg_presence[genome_id] = set()
            continue
        raw = pd.read_csv(results_file, sep="\t")
        if raw.empty:
            arg_presence[genome_id] = set()
            continue
        gene_col = "Resistance gene" if "Resistance gene" in raw.columns else raw.columns[0]
        arg_presence[genome_id] = set(raw[gene_col].unique())

all_genes = sorted(set.union(*arg_presence.values()))
print(f"Distinct ARG gene names: {len(all_genes)}")

arg_df = pd.DataFrame(
    {gene: [1.0 if gene in arg_presence[g] else 0.0 for g in arg_presence] for gene in all_genes},
    index=list(arg_presence.keys()),
)
print(f"ARG presence matrix (pre-filter): {arg_df.shape}")

gene_prev = arg_df.mean()
ARG_COLS = gene_prev[gene_prev >= PREV_THRESH].index.tolist()
print(f"Genes >= {PREV_THRESH:.0%} prevalence: {len(ARG_COLS)} (of {len(all_genes)})")

arg_df = arg_df[ARG_COLS]
arg_df.columns = [f"argp_{c}" for c in ARG_COLS]

# ── Load named-236 matrix + HMRG/IS blocks (same construction as step 1) ────
fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
fm = fm.join(arg_df)
assert fm[arg_df.columns].isna().sum().sum() == 0, "ARG presence join produced NaNs -- index mismatch"

with open("results/q1_named_feat_cols.txt") as f:
    DEFENCE_COLS = f.read().splitlines()

HMRG_EXCLUDE = {"hmrg_metal_total", "hmrg_metal_classes"}
HMRG_COLS_RAW = [c for c in fm.columns if c.startswith("hmrg_") and c not in HMRG_EXCLUDE]
IS_COLS_RAW = [c for c in fm.columns if c.startswith("is_") and c != "is_count_total"]

HMRG_BIN = fm[HMRG_COLS_RAW].gt(0).astype(float)
HMRG_BIN.columns = [f"hmrg_bin_{c}" for c in HMRG_COLS_RAW]
IS_BIN = fm[IS_COLS_RAW].gt(0).astype(float)
IS_BIN.columns = [f"is_bin_{c}" for c in IS_COLS_RAW]

fm_ext = pd.concat([fm, HMRG_BIN, IS_BIN], axis=1)
ARG_COLS_FINAL = list(arg_df.columns)
HMRG_COLS = list(HMRG_BIN.columns)
IS_COLS = list(IS_BIN.columns)

# ── Reproduce the same dereplication (1 genome per phylogroup, defence Euclidean centroid) ──
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
print(f"\nDereplicated reps: {len(rep_indices)}")

fm_derep = fm_ext.iloc[rep_indices]
species_derep = fm_derep["species"].to_numpy(dtype=str)
arg_tertile_derep = fm_derep["arg_burden_tertile"].fillna("unknown").to_numpy(dtype=str)

# ── Per-species ARG gene prevalence check (is this a clean barcode like HMRG, or graded like IS?) ──
print("\nTop 5 most prevalent ARG genes -- per-species presence rate:")
top5 = gene_prev.reindex(all_genes).sort_values(ascending=False).head(5).index
sp_for_check = fm["species"]
for g in top5:
    col = f"argp_{g}"
    if col not in fm.columns:
        continue
    rates = fm.groupby(sp_for_check)[col].mean()
    print(f"  {g:<20}" + "  ".join(f"{sp}={r:.2f}" for sp, r in rates.items()))


def cluster_block(cols, label):
    X = fm_derep[cols].to_numpy(dtype=float)
    sils = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
        sils.append(silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0)
    best_k = list(K_RANGE)[int(np.argmax(sils))]
    best_labels = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE).fit_predict(X)
    ari_sp = adjusted_rand_score(species_derep, best_labels)
    ari_arg = adjusted_rand_score(arg_tertile_derep, best_labels)
    print(f"{label:<28} n_feat={len(cols):<4} best_K={best_k:<3} "
          f"silhouette={max(sils):.4f}  ARI_species={ari_sp:.4f}  ARI_arg={ari_arg:.4f}")
    return {"label": label, "n_feat": len(cols), "best_k": best_k,
            "silhouette": max(sils), "ari_species": ari_sp, "ari_arg": ari_arg}


print(f"\n{'='*95}")
print("Dereplicated (309 reps) -- Euclidean K-means, K=2..8")
print(f"{'='*95}")

results = {}
results["defence"] = cluster_block(DEFENCE_COLS, "Defence only (anchor)")
results["hmrg"] = cluster_block(HMRG_COLS, "HMRG only")
results["is"] = cluster_block(IS_COLS, "IS only")
results["arg"] = cluster_block(ARG_COLS_FINAL, "ARG presence only")
results["arg_hmrg_is"] = cluster_block(ARG_COLS_FINAL + HMRG_COLS + IS_COLS, "ARG + HMRG + IS")
results["all"] = cluster_block(DEFENCE_COLS + ARG_COLS_FINAL + HMRG_COLS + IS_COLS, "Defence + ARG + HMRG + IS")

print(f"\n{'='*95}")
print("DELTA vs defence-only anchor:")
anchor = results["defence"]
for key, r in results.items():
    if key == "defence":
        continue
    print(f"  {r['label']:<32} d_silhouette={r['silhouette']-anchor['silhouette']:+.4f}  "
          f"d_ARI_species={r['ari_species']-anchor['ari_species']:+.4f}")
