"""
Positive-control support: Mash distances from the 180-genome external holdout
(Table S16) to the 3,335-genome training cohort.

Needed for the Mash-kNN species classifier comparator (docs/manuscript/
revision_roadmap_2026-09-07.md, R1-4). The existing distance_matrix.parquet
only covers the 3,335 training genomes (Methods §4) -- holdout genomes were
never sketched against it. This script sketches the same 180 genomes RF's
holdout BA=0.944 was computed on (Table S16, not the full 60-per-species
candidate pool) and queries them against the existing training sketch,
using the same Mash parameters as Methods §4 (k=21, s=1000) for consistency
with the reported RF numbers.

Output: data/interim/mash/holdout_distance_matrix.parquet
  index   = holdout accession (query)
  columns = training accession (ref), matching distance_matrix.parquet's
            column order exactly
"""

import csv
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
S16_PATH = REPO_ROOT / "results/tables/S16_holdout_set.csv"
TRAIN_SKETCH = REPO_ROOT / "data/interim/mash/all_genomes.msh"
TRAIN_DIST_MATRIX = REPO_ROOT / "data/interim/mash/distance_matrix.parquet"
OUT_PATH = REPO_ROOT / "data/interim/mash/holdout_distance_matrix.parquet"


def load_holdout_accessions():
    rows = [r for r in csv.DictReader(open(S16_PATH)) if r["Accession"]]
    accessions = [r["Accession"] for r in rows]
    assert len(accessions) == 180, f"expected 180 holdout accessions, got {len(accessions)}"
    return accessions


def find_fna(accession):
    fname_acc = accession.replace(".", "_")
    hits = list((REPO_ROOT / "data/raw/holdout_genomes").rglob(f"{fname_acc}.fna"))
    assert len(hits) == 1, f"expected exactly one fna for {accession}, found {len(hits)}"
    return hits[0]


def main():
    accessions = load_holdout_accessions()
    fna_paths = [find_fna(a) for a in accessions]
    print(f"Resolved {len(fna_paths)} holdout FASTA files.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        sketch_prefix = tmpdir / "holdout_180"
        sketch_path = sketch_prefix.with_suffix(".msh")
        file_list_path = tmpdir / "holdout_fna_list.txt"
        file_list_path.write_text("\n".join(str(p) for p in fna_paths) + "\n")

        print("Sketching 180 holdout genomes (k=21, s=1000, matching Methods §4)...")
        subprocess.run(
            ["mash", "sketch", "-k", "21", "-s", "1000", "-l", str(file_list_path),
             "-o", str(sketch_prefix)],
            check=True, capture_output=True, text=True,
        )

        print("Querying against training sketch (all_genomes.msh, 3,335 genomes)...")
        dist_out = subprocess.run(
            ["mash", "dist", str(TRAIN_SKETCH), str(sketch_path)],
            check=True, capture_output=True, text=True,
        ).stdout

    # mash dist columns: ref-id, query-id, distance, p-value, shared-hashes
    rows = []
    for line in dist_out.strip().splitlines():
        ref_id, query_id, dist, pval, shared = line.split("\t")
        rows.append((ref_id, query_id, float(dist)))
    long_df = pd.DataFrame(rows, columns=["ref", "query", "dist"])
    print(f"Parsed {len(long_df)} pairwise distances "
          f"(expected {180 * 3335} = 180 x 3335: {len(long_df) == 180 * 3335}).")

    # ref/query ids come straight from the fna path -- normalise to bare
    # accessions matching feature_matrix_3335.parquet's index convention.
    def to_accession(path_str):
        stem = Path(path_str).stem  # e.g. GCF_000746645_1
        base, ver = stem.rsplit("_", 1)
        return f"{base}.{ver}"

    long_df["ref"] = long_df["ref"].map(to_accession)
    long_df["query"] = long_df["query"].map(to_accession)

    wide = long_df.pivot(index="query", columns="ref", values="dist")

    # column order must match the training distance matrix exactly, since
    # downstream code indexes species labels positionally off that matrix.
    train_dm = pd.read_parquet(TRAIN_DIST_MATRIX)
    missing_cols = set(train_dm.columns) - set(wide.columns)
    assert not missing_cols, f"missing training genomes in holdout dist output: {missing_cols}"
    wide = wide[train_dm.columns]

    missing_rows = set(accessions) - set(wide.index)
    assert not missing_rows, f"missing holdout genomes in dist output: {missing_rows}"
    wide = wide.loc[accessions]  # order = S16 order, for easy joining later

    assert wide.shape == (180, 3335), wide.shape
    assert wide.notna().all().all(), "unexpected NaNs in holdout distance matrix"

    wide.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}, shape {wide.shape}")
    print(f"Distance range: [{wide.values.min():.4f}, {wide.values.max():.4f}]")


if __name__ == "__main__":
    main()
