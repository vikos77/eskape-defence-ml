"""
Q3b: Multi-block unsupervised species recovery (defence + ARG + HMRG + IS).

Fulfills an already-stated open direction from discussion.md (written before this
script existed): whether ARG/HMRG/IS content reveals genome-architecture structure
beyond defence composition. Built after Q1-Q4 were locked, so the specific choices
below (K selection, prevalence thresholds) were decided during construction rather
than pre-specified in pre_analysis_plan.md -- see docs/decisions.md 2026-06-17 entry
for the full framing. Tests whether combining defence system presence with three
other genomic-content blocks -- ARG
resistance gene presence, heavy-metal-resistance-gene (HMRG) class presence,
and insertion-sequence (IS) family presence -- recovers species identity in
unsupervised clustering, where defence alone does not (Q3 primary result:
ARI=0.193 at K=6, silhouette=0.035 -- continuum, no discrete archetypes).

IME (ICEberg) is deliberately excluded. It was tested and found to (a) carry
no independent species signal (ARI=0.195, no better than defence alone) and
(b) its apparent structure is a likely database-redundancy artifact -- the
five most prevalent ICEberg element IDs are near-identical presence patterns
across genomes, consistent with multiple reference entries representing the
same conserved IME backbone region rather than five independent signals.
Combined with the pre-existing citability concern (ICEberg element IDs are
internal database identifiers, not published named systems -- the same
problem that justified excluding PDC/DS-N from the defence matrix), IME adds
noise and measurably worsens the combined result (ARI 0.712 -> 0.488 if
included). See run_q3_ime_block.py for the full IME-inclusion test.

Method:
  - 309 dereplicated representative genomes (1 per Mash-defined phylogroup,
    closest to phylogroup centroid in DEFENCE Euclidean space) -- identical
    selection to NB09's primary Q3 robustness check, reused here so every
    block's result is directly comparable to the locked Q3 anchor.
  - K-means, Euclidean distance, K FORCED to 6 (the species count) for every
    block and combination. Silhouette-optimal K varies by block (K=3 to K=8)
    and is NOT a fair basis for cross-block comparison -- an earlier pass of
    this analysis compared blocks at their own optimal K and concluded
    combination destroyed signal; forcing K=6 reversed that conclusion
    entirely (logged in decisions.md as a corrected finding).
  - Each block reported ALONE (not just cumulative stacking) because raw
    concatenation under unweighted Euclidean distance lets whichever block
    has the most features dominate -- reporting blocks alone is the only way
    to see information content that combination can mask.

Statistical rigor (matching Q1/Q2 standards -- CI, null comparison, stability):
  - 20-seed K-means stability check (does the ARI depend on K-means init/seed?)
  - Bootstrap 95% CI on ARI: 2000 resamples (with replacement) of the 309
    representative genomes, K=6 K-means refit on each resample, ARI vs the
    resampled species labels.
  - Permutation test: 2000 species-label shuffles against the FIXED K=6
    partition from the real data, one-sided p-value (fraction of permuted
    ARI >= observed ARI).

Outputs:
  results/q3b_multiblock_results.json
  results/supplement_q3b_multiblock.csv
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SEEDS = 20
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 2000
SPECIES = ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]
INTERIM = Path("data/interim")


def norm_acc(s: str) -> str:
    prefix, version = s.rsplit("_", 1)
    return f"{prefix}.{version}"


# ── Build ARG presence/absence matrix (per-gene, >=1% prevalence) ───────────
print("Parsing ResFinder per-gene presence...")
arg_presence = {}
for sp in SPECIES:
    res_dir = INTERIM / sp / "resfinder"
    for genome_dir in sorted(res_dir.iterdir()):
        if not genome_dir.is_dir():
            continue
        genome_id = norm_acc(genome_dir.name)
        rf = genome_dir / "ResFinder_results_tab.txt"
        if not rf.exists() or rf.stat().st_size == 0:
            arg_presence[genome_id] = set()
            continue
        raw = pd.read_csv(rf, sep="\t")
        if raw.empty:
            arg_presence[genome_id] = set()
            continue
        gene_col = "Resistance gene" if "Resistance gene" in raw.columns else raw.columns[0]
        arg_presence[genome_id] = set(raw[gene_col].unique())

all_genes = sorted(set.union(*arg_presence.values()))
arg_df = pd.DataFrame(
    {g: [1.0 if g in arg_presence[gi] else 0.0 for gi in arg_presence] for g in all_genes},
    index=list(arg_presence.keys()),
)
gene_prev = arg_df.mean()
ARG_COLS_KEEP = gene_prev[gene_prev >= 0.01].index.tolist()
arg_df = arg_df[ARG_COLS_KEEP]
arg_df.columns = [f"argp_{c}" for c in ARG_COLS_KEEP]
print(f"ARG presence columns (>=1% prevalence): {len(arg_df.columns)} of {len(all_genes)} distinct genes")

# ── Load named-236 matrix, build HMRG/IS binarized blocks ───────────────────
fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
fm = fm.join(arg_df)
assert fm[arg_df.columns].isna().sum().sum() == 0

with open("results/q1_named_feat_cols.txt") as f:
    DEFENCE_COLS = f.read().splitlines()
assert len(DEFENCE_COLS) == 231

HMRG_EXCLUDE = {"hmrg_metal_total", "hmrg_metal_classes"}
HMRG_COLS_RAW = [c for c in fm.columns if c.startswith("hmrg_") and c not in HMRG_EXCLUDE]
IS_COLS_RAW = [c for c in fm.columns if c.startswith("is_") and c != "is_count_total"]
HMRG_BIN = fm[HMRG_COLS_RAW].gt(0).astype(float)
HMRG_BIN.columns = [f"hmrg_bin_{c}" for c in HMRG_COLS_RAW]
IS_BIN = fm[IS_COLS_RAW].gt(0).astype(float)
IS_BIN.columns = [f"is_bin_{c}" for c in IS_COLS_RAW]
fm_ext = pd.concat([fm, HMRG_BIN, IS_BIN], axis=1)

ARG_COLS = list(arg_df.columns)
HMRG_COLS = list(HMRG_BIN.columns)
IS_COLS = list(IS_BIN.columns)
print(f"Defence: {len(DEFENCE_COLS)}, ARG: {len(ARG_COLS)}, HMRG: {len(HMRG_COLS)}, IS: {len(IS_COLS)}")

# ── Dereplicate: 1 genome per phylogroup, closest to centroid in defence space ──
X_defence_full = fm_ext[DEFENCE_COLS].to_numpy(dtype=float)
pg_col = fm_ext["phylogroup"].to_numpy(dtype=str)
rep_indices = []
for pg_id in np.unique(pg_col):
    mask = pg_col == pg_id
    grp = X_defence_full[mask]
    cent = grp.mean(axis=0)
    best = np.argmin(np.linalg.norm(grp - cent, axis=1))
    rep_indices.append(np.where(mask)[0][best])
rep_indices = np.array(rep_indices)
print(f"Dereplicated reps: {len(rep_indices)} (NB09 anchor: 309)")

fm_derep = fm_ext.iloc[rep_indices].reset_index(drop=True)
species_derep = fm_derep["species"].to_numpy(dtype=str)

CONDITIONS = {
    "defence_only": DEFENCE_COLS,
    "hmrg_only": HMRG_COLS,
    "is_only": IS_COLS,
    "arg_only": ARG_COLS,
    "arg_hmrg_is": ARG_COLS + HMRG_COLS + IS_COLS,
    "all_combined": DEFENCE_COLS + ARG_COLS + HMRG_COLS + IS_COLS,
}


def analyze_condition(cols, label):
    X = fm_derep[cols].to_numpy(dtype=float)
    n = X.shape[0]

    # Primary fit (the headline number)
    primary_labels = KMeans(n_clusters=6, n_init=30, random_state=RANDOM_STATE).fit_predict(X)
    primary_ari = adjusted_rand_score(species_derep, primary_labels)
    primary_sil = silhouette_score(X, primary_labels)

    # Seed stability: 20 independent K-means fits, different random_state each
    seed_aris = []
    for seed in range(N_SEEDS):
        labels = KMeans(n_clusters=6, n_init=10, random_state=seed).fit_predict(X)
        seed_aris.append(adjusted_rand_score(species_derep, labels))
    seed_aris = np.array(seed_aris)

    # Bootstrap 95% CI: resample genomes with replacement, refit, recompute ARI
    rng = np.random.default_rng(RANDOM_STATE)
    boot_aris = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        sp_b = species_derep[idx]
        labels_b = KMeans(n_clusters=6, n_init=5, random_state=b).fit_predict(Xb)
        boot_aris[b] = adjusted_rand_score(sp_b, labels_b)
    ci_lo, ci_hi = np.percentile(boot_aris, [2.5, 97.5])

    # Permutation test: shuffle species labels against the FIXED primary partition
    perm_aris = np.empty(N_PERMUTATIONS)
    rng2 = np.random.default_rng(RANDOM_STATE + 1)
    for p in range(N_PERMUTATIONS):
        shuffled = rng2.permutation(species_derep)
        perm_aris[p] = adjusted_rand_score(shuffled, primary_labels)
    p_value = float((np.sum(perm_aris >= primary_ari) + 1) / (N_PERMUTATIONS + 1))  # add-one correction (Davison & Hinkley)

    print(f"{label:<16} n_feat={len(cols):<4} ARI={primary_ari:.4f} "
          f"[{ci_lo:.4f}-{ci_hi:.4f}]  silhouette={primary_sil:.4f}  "
          f"seed_mean={seed_aris.mean():.4f} (sd={seed_aris.std():.4f})  p={p_value:.4f}")

    return {
        "label": label, "n_features": len(cols),
        "ari_primary": float(primary_ari), "silhouette": float(primary_sil),
        "ari_ci95": [float(ci_lo), float(ci_hi)],
        "ari_seed_mean": float(seed_aris.mean()), "ari_seed_sd": float(seed_aris.std()),
        "ari_seed_values": seed_aris.tolist(),
        "permutation_p": p_value,
    }


print(f"\n{'='*100}")
print("Q3b: forced K=6, 309 dereplicated reps, full statistical rigor")
print(f"{'='*100}")

results = {}
for key, cols in CONDITIONS.items():
    results[key] = analyze_condition(cols, key)

with open("results/q3b_multiblock_results.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame([
    {"condition": k, "n_features": v["n_features"], "ari": v["ari_primary"],
     "ari_ci_lo": v["ari_ci95"][0], "ari_ci_hi": v["ari_ci95"][1],
     "silhouette": v["silhouette"], "seed_mean": v["ari_seed_mean"],
     "seed_sd": v["ari_seed_sd"], "permutation_p": v["permutation_p"]}
    for k, v in results.items()
]).to_csv("results/supplement_q3b_multiblock.csv", index=False)

print("\nSaved: results/q3b_multiblock_results.json")
print("Saved: results/supplement_q3b_multiblock.csv")
