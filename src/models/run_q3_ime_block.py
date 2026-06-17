"""
Q3 unsupervised feature-block exploration -- Step 3: IME presence/absence block.

Caution flagged before building this (user instruction): IME is the noisiest block.
Raw tBLASTn hits against the ICEberg_IME database are enormous before filtering --
one element ID alone produced 142,266 raw hits across ~600 A. baumannii genomes,
and the same element flooded every other species at similar volume. The existing
ime_count_unique column already applies the correct filter (pident>=40,
coverage>=80% of max protein length per element, see NB01 Section 5) and is far
saner: mean 9.6 distinct qualifying elements per genome, median 10, max 46, 336
genomes with zero. This script reuses that exact filter -- nothing looser -- to
build a per-element presence/absence matrix, the IME analogue of dp_/argp_.

Citability caveat (separate from noise, already logged in decisions.md): ICEberg
element IDs (e.g. ICEberg|185_IME) are internal database identifiers, not citable
biological names -- the same problem that justified excluding PDC/DS-N from the
defence matrix. Fine for an unsupervised clustering check (we are not naming SHAP
drivers here), but if this block's individual top contributors are ever reported,
they must be flagged as "ICEberg reference IDs," not named systems.

Reuses the same 309 dereplicated representative genomes and forced K=6 comparison
established in the corrected ARG/HMRG/IS analysis (K=2..8 model-selection was shown
to be misleading for cross-block comparison; species count is the principled K).

Outputs: printed comparison only -- exploratory, nothing written to results/ yet.
"""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SPECIES = ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]
INTERIM = Path("data/interim")
DB_DIR = Path("data/raw/databases")
PIDENT_MIN = 40.0
COVERAGE_MIN = 0.80
BLAST_COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
              "qstart", "qend", "sstart", "send", "evalue", "bitscore"]


def norm_acc(s: str) -> str:
    prefix, version = s.rsplit("_", 1)
    return f"{prefix}.{version}"


def parse_fasta_max_lengths(fasta_path: Path) -> dict:
    lengths = defaultdict(list)
    current_id, current_len = None, 0
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    lengths[current_id].append(current_len)
                current_id = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
    if current_id is not None:
        lengths[current_id].append(current_len)
    return {k: max(v) for k, v in lengths.items()}


print("Loading ICEberg protein length lookup...")
ime_lengths = parse_fasta_max_lengths(DB_DIR / "ICEberg_IME.fasta")
print(f"ICEberg element IDs: {len(ime_lengths)}")

print("Parsing per-genome qualifying IME element presence (pident>=40, coverage>=80%)...")
ime_presence = {}  # genome_id -> set of qualifying element IDs

for sp in SPECIES:
    ice_dir = INTERIM / sp / "iceberg"
    for tsv_file in sorted(ice_dir.glob("*_iceberg.tsv")):
        fn = tsv_file.stem.replace("_iceberg", "")
        genome_id = norm_acc(fn)
        if tsv_file.stat().st_size == 0:
            ime_presence[genome_id] = set()
            continue
        raw = pd.read_csv(tsv_file, sep="\t", header=None, names=BLAST_COLS)
        raw = raw[raw["pident"] >= PIDENT_MIN]
        raw = raw[raw["qseqid"].isin(ime_lengths)].copy()
        if raw.empty:
            ime_presence[genome_id] = set()
            continue
        raw["max_prot_len"] = raw["qseqid"].map(ime_lengths)
        raw["coverage"] = (raw["qend"] - raw["qstart"] + 1) / raw["max_prot_len"]
        raw = raw[raw["coverage"] >= COVERAGE_MIN]
        ime_presence[genome_id] = set(raw["qseqid"].unique())

n_qualifying = [len(v) for v in ime_presence.values()]
print(f"Genomes parsed: {len(ime_presence)}")
print(f"Qualifying elements per genome -- mean={np.mean(n_qualifying):.2f}, "
      f"median={np.median(n_qualifying):.1f}, max={max(n_qualifying)}, "
      f"zero={sum(1 for x in n_qualifying if x == 0)}")

all_elements = sorted(set.union(*ime_presence.values()))
print(f"Distinct qualifying element IDs across dataset: {len(all_elements)} (of {len(ime_lengths)} in DB)")

ime_df = pd.DataFrame(
    {e: [1.0 if e in ime_presence[g] else 0.0 for g in ime_presence] for e in all_elements},
    index=list(ime_presence.keys()),
)

elem_prev = ime_df.mean().sort_values(ascending=False)
print("\nPer-element prevalence (fraction of 3335 genomes) -- top 10:")
print(elem_prev.head(10).round(3).to_string())
print("\nPer-element prevalence -- bottom 10:")
print(elem_prev.tail(10).round(3).to_string())

PREV_THRESH = 0.01
IME_COLS_KEEP = elem_prev[elem_prev >= PREV_THRESH].index.tolist()
print(f"\nElements >= {PREV_THRESH:.0%} prevalence: {len(IME_COLS_KEEP)} (of {len(all_elements)})")
ime_df = ime_df[IME_COLS_KEEP]
ime_df.columns = [f"imep_{c}" for c in IME_COLS_KEEP]

# ── Per-species prevalence check on the top elements: barcode or graded? ────
print("\nTop 5 most prevalent IME elements -- per-species presence rate:")
fm0 = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
sp_lookup = fm0["species"]
for e in elem_prev.head(5).index:
    col = f"imep_{e}"
    if col not in ime_df.columns:
        continue
    joined = ime_df[col].reindex(sp_lookup.index)
    rates = joined.groupby(sp_lookup).mean()
    print(f"  {e:<22}" + "  ".join(f"{sp}={r:.2f}" for sp, r in rates.items()))

# ── Build combined matrix + rerun the established framework ─────────────────
fm = fm0.join(ime_df)
assert fm[ime_df.columns].isna().sum().sum() == 0

with open("results/q1_named_feat_cols.txt") as f:
    DEFENCE_COLS = f.read().splitlines()

# Rebuild ARG presence (same as run_q3_arg_block.py) for the full combined comparison
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
arg_df_full = pd.DataFrame(
    {g: [1.0 if g in arg_presence[gi] else 0.0 for gi in arg_presence] for g in all_genes},
    index=list(arg_presence.keys()),
)
gene_prev = arg_df_full.mean()
ARG_COLS_KEEP = gene_prev[gene_prev >= 0.01].index.tolist()
arg_df_full = arg_df_full[ARG_COLS_KEEP]
arg_df_full.columns = [f"argp_{c}" for c in ARG_COLS_KEEP]
fm = fm.join(arg_df_full)
ARG_COLS_FINAL = list(arg_df_full.columns)

HMRG_EXCLUDE = {"hmrg_metal_total", "hmrg_metal_classes"}
HMRG_COLS_RAW = [c for c in fm.columns if c.startswith("hmrg_") and c not in HMRG_EXCLUDE]
IS_COLS_RAW = [c for c in fm.columns if c.startswith("is_") and c != "is_count_total"]
HMRG_BIN = fm[HMRG_COLS_RAW].gt(0).astype(float)
HMRG_BIN.columns = [f"hmrg_bin_{c}" for c in HMRG_COLS_RAW]
IS_BIN = fm[IS_COLS_RAW].gt(0).astype(float)
IS_BIN.columns = [f"is_bin_{c}" for c in IS_COLS_RAW]
fm_ext = pd.concat([fm, HMRG_BIN, IS_BIN], axis=1)
HMRG_COLS = list(HMRG_BIN.columns)
IS_COLS = list(IS_BIN.columns)
IME_COLS = list(ime_df.columns)

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
fm_derep = fm_ext.iloc[rep_indices]
species_derep = fm_derep["species"].to_numpy(dtype=str)


def cluster_k6(cols, label):
    X = fm_derep[cols].to_numpy(dtype=float)
    labels = KMeans(n_clusters=6, n_init=30, random_state=RANDOM_STATE).fit_predict(X)
    sil = silhouette_score(X, labels)
    ari = adjusted_rand_score(species_derep, labels)
    print(f"{label:<32} n_feat={len(cols):<4} K=6  silhouette={sil:.4f}  ARI_species={ari:.4f}")
    return labels


print(f"\n{'='*95}")
print("Dereplicated (309 reps) -- forced K=6 (species count), Euclidean K-means")
print(f"{'='*95}")
cluster_k6(DEFENCE_COLS, "Defence only")
cluster_k6(IME_COLS, "IME presence only")
cluster_k6(ARG_COLS_FINAL + HMRG_COLS + IS_COLS, "ARG + HMRG + IS (prior anchor)")
cluster_k6(ARG_COLS_FINAL + HMRG_COLS + IS_COLS + IME_COLS, "ARG + HMRG + IS + IME")
cluster_k6(DEFENCE_COLS + ARG_COLS_FINAL + HMRG_COLS + IS_COLS, "Defence+ARG+HMRG+IS (prior anchor)")
cluster_k6(DEFENCE_COLS + ARG_COLS_FINAL + HMRG_COLS + IS_COLS + IME_COLS, "Defence+ARG+HMRG+IS+IME (all blocks)")
