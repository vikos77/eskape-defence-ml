"""
Recompute every rho_IME value cited in the manuscript using a deduplicated IME count,
to check whether ICEberg database redundancy (48 of 98 element IDs collapse into one
near-identical sequence cluster, ~5.5x mean inflation of ime_count_unique, only 0.54
correlation between original and deduplicated count -- see docs/decisions.md) changes
the reported Spearman correlations.

Deduplication: pairwise k-mer (k=8) Jaccard similarity between all protein sequences in
ICEberg_IME.fasta, element IDs single-linkage clustered at similarity >= 0.5. Per genome,
deduplicated count = number of distinct clusters represented among qualifying elements
(pident>=40, coverage>=80%, same filter as ime_count_unique), not raw element ID count.

Outputs: printed comparison table only.
"""

import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

SPECIES = ["abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus"]
INTERIM = Path("data/interim")
DB_DIR = Path("data/raw/databases")
PIDENT_MIN, COVERAGE_MIN = 40.0, 0.80
BLAST_COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
              "qstart", "qend", "sstart", "send", "evalue", "bitscore"]


def norm_acc(s: str) -> str:
    prefix, version = s.rsplit("_", 1)
    return f"{prefix}.{version}"


def parse_fasta_max_lengths(path):
    lengths = defaultdict(list)
    cid, clen = None, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cid is not None:
                    lengths[cid].append(clen)
                cid = line[1:].split()[0]
                clen = 0
            else:
                clen += len(line)
    if cid is not None:
        lengths[cid].append(clen)
    return {k: max(v) for k, v in lengths.items()}


def load_protein_seqs(path):
    seqs = defaultdict(list)
    cid, cseq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cid is not None:
                    seqs[cid].append("".join(cseq))
                cid = line[1:].split()[0]
                cseq = []
            else:
                cseq.append(line)
    if cid is not None:
        seqs[cid].append("".join(cseq))
    return seqs


def kmer_set(s, k=8):
    return set(s[i:i + k] for i in range(len(s) - k + 1))


print("Building ICEberg redundancy clusters (k-mer similarity >= 0.5, single-linkage)...")
seqs = load_protein_seqs(DB_DIR / "ICEberg_IME.fasta")
proteins_by_elem = defaultdict(list)
for eid, plist in seqs.items():
    for p in plist:
        if len(p) >= 20:
            proteins_by_elem[eid].append(kmer_set(p, 8))

elem_ids = sorted(seqs.keys())
n = len(elem_ids)
parent = list(range(n))


def find(x):
    while parent[x] != x:
        x = parent[x]
    return x


def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[rx] = ry


THRESH = 0.5
for i, j in itertools.combinations(range(n), 2):
    e1, e2 = elem_ids[i], elem_ids[j]
    best = 0.0
    for k1 in proteins_by_elem[e1]:
        for k2 in proteins_by_elem[e2]:
            inter = len(k1 & k2)
            if inter == 0:
                continue
            s = inter / len(k1 | k2)
            if s > best:
                best = s
            if best >= THRESH:
                break
        if best >= THRESH:
            break
    if best >= THRESH:
        union(i, j)

groups = defaultdict(list)
for i, eid in enumerate(elem_ids):
    groups[find(i)].append(eid)
group_list = list(groups.values())
elem_to_group = {m: gid for gid, members in enumerate(group_list) for m in members}
print(f"Redundancy groups: {len(group_list)} from {n} element IDs "
      f"({sum(1 for g in group_list if len(g) > 1)} groups with >1 member)")

ime_lengths = parse_fasta_max_lengths(DB_DIR / "ICEberg_IME.fasta")

print("Recomputing per-genome IME counts (original vs deduplicated)...")
records = []
for sp in SPECIES:
    ice_dir = INTERIM / sp / "iceberg"
    for tsv_file in sorted(ice_dir.glob("*_iceberg.tsv")):
        genome_id = norm_acc(tsv_file.stem.replace("_iceberg", ""))
        if tsv_file.stat().st_size == 0:
            records.append({"genome_id": genome_id, "species": sp,
                             "ime_orig": 0, "ime_dedup": 0})
            continue
        raw = pd.read_csv(tsv_file, sep="\t", header=None, names=BLAST_COLS)
        raw = raw[raw["pident"] >= PIDENT_MIN]
        raw = raw[raw["qseqid"].isin(ime_lengths)].copy()
        if raw.empty:
            records.append({"genome_id": genome_id, "species": sp,
                             "ime_orig": 0, "ime_dedup": 0})
            continue
        raw["max_prot_len"] = raw["qseqid"].map(ime_lengths)
        raw["coverage"] = (raw["qend"] - raw["qstart"] + 1) / raw["max_prot_len"]
        raw = raw[raw["coverage"] >= COVERAGE_MIN]
        qualifying = set(raw["qseqid"].unique())
        dedup_groups = set(elem_to_group[q] for q in qualifying)
        records.append({"genome_id": genome_id, "species": sp,
                         "ime_orig": len(qualifying), "ime_dedup": len(dedup_groups)})

ime_df = pd.DataFrame(records).set_index("genome_id")

fm = pd.read_parquet("data/processed/feature_matrix_3335_named.parquet")
fm = fm.join(ime_df[["ime_orig", "ime_dedup"]])
assert fm["ime_orig"].isna().sum() == 0
print(f"Sanity check -- ime_orig vs existing ime_count_unique match: "
      f"{(fm['ime_orig'] == fm['ime_count_unique']).all()}")

# ── Recompute every cited rho_IME value ──────────────────────────────────────
CITATIONS = [
    ("abaumannii", "dp_RM_Type_IV", -0.397),
    ("abaumannii", "dp_RM_Type_I", -0.312),
    ("abaumannii", "dp_SspBCDE", +0.565),
    ("abaumannii", "dp_Gao_Qat", +0.308),
    ("ecloaceae", "dp_RM_Type_II", +0.198),
    ("kpneumoniae", "dp_RM_Type_II", +0.343),
    ("paeruginosa", "dp_RM_Type_I", +0.367),
    ("efaecium", "dp_AbiE", +0.508),
    ("efaecium", "dp_df_AbiH", +0.426),
    ("efaecium", "dp_df_AbiJ", +0.271),
    ("efaecium", "dp_df_MazEF", +0.479),
]

print(f"\n{'='*100}")
print(f"{'Species':<14}{'System':<18}{'rho (orig, manuscript)':>24}{'rho (orig, recomputed)':>24}"
      f"{'rho (dedup)':>14}")
results = []
for sp, col, cited_rho in CITATIONS:
    sp_df = fm[fm.species == sp]
    if col not in sp_df.columns:
        print(f"{sp:<14}{col:<18}  COLUMN NOT FOUND")
        continue
    x = sp_df[col].to_numpy()
    rho_orig, p_orig = spearmanr(x, sp_df["ime_orig"].to_numpy())
    rho_dedup, p_dedup = spearmanr(x, sp_df["ime_dedup"].to_numpy())
    print(f"{sp:<14}{col:<18}{cited_rho:>24.3f}{rho_orig:>24.3f}{rho_dedup:>14.3f}")
    results.append({"species": sp, "system": col, "cited_rho": cited_rho,
                     "rho_orig_recomputed": rho_orig, "p_orig": p_orig,
                     "rho_dedup": rho_dedup, "p_dedup": p_dedup})

df_results = pd.DataFrame(results)
df_results.to_csv("results/supplement_ime_dedup_check.csv", index=False)
print("\nSaved: results/supplement_ime_dedup_check.csv")
