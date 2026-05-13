"""
Phase 3 parsers — one function per tool, all return a tidy DataFrame indexed by genome_id.

Each parser follows the same contract:
    - Input:  path to the tool's output for one species (or a glob pattern)
    - Output: pd.DataFrame with 'genome_id' as the index and tool-specific columns
    - Empty output files → zero counts (not NaN); missing files → NaN with a warning
    - No parser modifies global state or reads the system_name_map — that join
      happens in the notebook after all parsers have run.

Column naming conventions:
    Defence binary:  {canonical_name}_present   (bool)
    Defence count:   {canonical_name}_count      (int)
    Tool agreement:  {canonical_name}_n_tools    (0/1/2)
    Anti-defence:    adf_{subtype}_present / _count
    ARG:             arg_count                   (int, unique gene-level)
    IME:             ime_hit_count / ime_unique_count
    HMRG:            hmrg_hit_count / hmrg_unique_count
    IS element:      is_{family}_count / is_total_count
    MLST:            mlst_st (str)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _acc_fn_to_genome_id(path: Path) -> str:
    """Extract genome_id from any file whose stem starts with a GCF/GCA accession.

    Examples
    --------
    GCF_000505685_1_padloc.csv   → GCF_000505685_1
    GCF_000505685_1              → GCF_000505685_1   (directory name)
    """
    stem = path.stem if path.is_file() else path.name
    # Strip known suffixes
    for suffix in ("_padloc", "_iceberg", "_bacmet", "_isescan", "_isescan.sum"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _read_protein_lengths(fasta_path: Path) -> dict[str, int]:
    """Return {sequence_id: aa_length} for a protein FASTA.

    Used to compute query coverage for tBLASTn outputs.
    """
    lengths: dict[str, int] = {}
    current_id: str | None = None
    current_len = 0
    with fasta_path.open() as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    lengths[current_id] = current_len
                # Take the first whitespace-delimited token as the ID
                current_id = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
    if current_id is not None:
        lengths[current_id] = current_len
    return lengths


BLAST_COLS = [
    "qseqid", "sseqid", "pident", "length",
    "mismatch", "gapopen", "qstart", "qend",
    "sstart", "send", "evalue", "bitscore",
]


# ── DefenseFinder parser ──────────────────────────────────────────────────────

def parse_defensefinder(interim_dir: Path) -> pd.DataFrame:
    """Parse all defense_finder_systems.tsv files under interim_dir/defensefinder/.

    Returns one row per genome with columns:
        df_systems: list[str]           — defence subtype names detected
        df_anti_systems: list[str]      — anti-defence subtype names detected
        df_raw: pd.DataFrame (attached) — full systems table (for merging later)

    For the feature matrix we return a long-format DataFrame:
        genome_id | df_subtype | df_type | activity | count
    which the notebook pivots into wide format after joining with system_name_map.
    """
    records: list[dict] = []
    df_dir = interim_dir / "defensefinder"
    if not df_dir.exists():
        logger.warning("DefenseFinder directory not found: %s", df_dir)
        return pd.DataFrame()

    for acc_dir in sorted(df_dir.iterdir()):
        if not acc_dir.is_dir():
            continue
        genome_id = acc_dir.name
        tsv = acc_dir / "defense_finder_systems.tsv"
        if not tsv.exists():
            logger.warning("Missing DF systems file: %s", tsv)
            continue

        try:
            d = pd.read_csv(tsv, sep="\t")
        except Exception as exc:
            logger.error("Failed to read %s: %s", tsv, exc)
            continue

        if d.empty:
            # Tool ran, found zero systems — this is a valid result, not missing data
            records.append({
                "genome_id": genome_id,
                "df_subtype": "__none__",
                "df_type": "__none__",
                "activity": "Defense",
                "count": 0,
            })
            continue

        # Count instances per subtype (a genome may have >1 copy of the same system)
        for (subtype, typ, activity), grp in d.groupby(["subtype", "type", "activity"]):
            records.append({
                "genome_id": genome_id,
                "df_subtype": str(subtype),
                "df_type": str(typ),
                "activity": str(activity),
                "count": len(grp),
            })

    return pd.DataFrame(records)


# ── PADLOC parser ─────────────────────────────────────────────────────────────

def parse_padloc(interim_dir: Path) -> pd.DataFrame:
    """Parse all *_padloc.csv files under interim_dir/padloc/.

    De-duplicates to system level: each unique system.number within a genome
    counts as one system instance (a system.number groups all genes of one
    system copy). Returns long-format:
        genome_id | padloc_system | count
    """
    records: list[dict] = []
    padloc_dir = interim_dir / "padloc"
    if not padloc_dir.exists():
        logger.warning("PADLOC directory not found: %s", padloc_dir)
        return pd.DataFrame()

    for csv_path in sorted(padloc_dir.glob("*_padloc.csv")):
        genome_id = _acc_fn_to_genome_id(csv_path)

        try:
            d = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", csv_path, exc)
            continue

        if d.empty:
            records.append({"genome_id": genome_id, "padloc_system": "__none__", "count": 0})
            continue

        # One row per unique system instance (deduplicate by system.number)
        system_instances = d.drop_duplicates(subset=["system.number", "system"])
        for system_name, grp in system_instances.groupby("system"):
            records.append({
                "genome_id": genome_id,
                "padloc_system": str(system_name),
                "count": len(grp),  # number of copies of this system in the genome
            })

    return pd.DataFrame(records)


# ── ResFinder parser ──────────────────────────────────────────────────────────

def parse_resfinder(interim_dir: Path) -> pd.DataFrame:
    """Parse ResFinder_results_tab.txt for each genome.

    Counts unique resistance gene names (not alleles). Returns:
        genome_id | arg_count | arg_genes (comma-joined list)
    """
    rows: list[dict] = []
    rf_dir = interim_dir / "resfinder"
    if not rf_dir.exists():
        logger.warning("ResFinder directory not found: %s", rf_dir)
        return pd.DataFrame()

    for acc_dir in sorted(rf_dir.iterdir()):
        if not acc_dir.is_dir():
            continue
        genome_id = acc_dir.name
        tab = acc_dir / "ResFinder_results_tab.txt"

        if not tab.exists():
            logger.warning("Missing ResFinder results: %s", tab)
            rows.append({"genome_id": genome_id, "arg_count": float("nan"), "arg_genes": ""})
            continue

        try:
            d = pd.read_csv(tab, sep="\t")
        except Exception as exc:
            logger.error("Failed to read %s: %s", tab, exc)
            rows.append({"genome_id": genome_id, "arg_count": float("nan"), "arg_genes": ""})
            continue

        if d.empty:
            rows.append({"genome_id": genome_id, "arg_count": 0, "arg_genes": ""})
            continue

        # Column name varies slightly across ResFinder versions; first column = gene name
        gene_col = d.columns[0]
        # Strip allele suffix: "aph(3')-VI" is the gene; variants have same base name
        # Use the gene name as-is (unique gene-level, not allele-level)
        genes = d[gene_col].dropna().unique()
        rows.append({
            "genome_id": genome_id,
            "arg_count": len(genes),
            "arg_genes": ",".join(sorted(str(g) for g in genes)),
        })

    return pd.DataFrame(rows).set_index("genome_id")


# ── BLAST tBLASTn parser (shared for ICEberg and BacMet) ─────────────────────

def parse_blast_hits(
    tsv_path: Path,
    query_lengths: dict[str, int],
    pident_min: float = 40.0,
    coverage_min: float = 0.80,
) -> pd.DataFrame:
    """Filter one BLAST outfmt6 TSV and return passing hits.

    Applies:
        pident >= pident_min
        alignment_length / query_protein_length >= coverage_min

    An empty file is valid (zero hits) and returns an empty DataFrame.
    A missing file raises FileNotFoundError — callers must handle this.

    Parameters
    ----------
    tsv_path      : path to a BLAST outfmt6 TSV (12 standard columns)
    query_lengths : dict mapping qseqid → protein length in aa
    pident_min    : minimum percent identity (default 40.0)
    coverage_min  : minimum fraction of query covered (default 0.80)
    """
    if tsv_path.stat().st_size == 0:
        return pd.DataFrame(columns=BLAST_COLS)

    hits = pd.read_csv(tsv_path, sep="\t", header=None, names=BLAST_COLS)

    # pident filter
    hits = hits[hits["pident"] >= pident_min]
    if hits.empty:
        return hits

    # Coverage filter: alignment_length / query_protein_length >= coverage_min
    hits["qlen"] = hits["qseqid"].map(query_lengths)
    missing_qlens = hits["qlen"].isna().sum()
    if missing_qlens:
        warnings.warn(
            f"{tsv_path.name}: {missing_qlens} hits have qseqid not found in "
            "query FASTA — these hits are dropped (check FASTA matches DB used for BLAST).",
            stacklevel=2,
        )
    hits = hits.dropna(subset=["qlen"])
    hits["query_coverage"] = hits["length"] / hits["qlen"]
    hits = hits[hits["query_coverage"] >= coverage_min]

    return hits.drop(columns=["qlen"])


def parse_iceberg(interim_dir: Path, iceberg_fasta: Path) -> pd.DataFrame:
    """Parse ICEberg tBLASTn hits for all genomes in interim_dir.

    Returns:
        genome_id | ime_hit_count | ime_unique_count
    where ime_unique_count = number of distinct ICE/IME database entries (qseqid)
    with at least one passing hit (pident>=40, coverage>=80%).
    """
    query_lengths = _read_protein_lengths(iceberg_fasta)
    rows: list[dict] = []
    iceberg_dir = interim_dir / "iceberg"
    if not iceberg_dir.exists():
        logger.warning("ICEberg directory not found: %s", iceberg_dir)
        return pd.DataFrame()

    for tsv in sorted(iceberg_dir.glob("*_iceberg.tsv")):
        genome_id = _acc_fn_to_genome_id(tsv)
        try:
            hits = parse_blast_hits(tsv, query_lengths)
        except FileNotFoundError:
            logger.warning("ICEberg TSV missing (tool not yet run?): %s", tsv)
            rows.append({"genome_id": genome_id, "ime_hit_count": float("nan"), "ime_unique_count": float("nan")})
            continue
        rows.append({
            "genome_id": genome_id,
            "ime_hit_count": len(hits),
            "ime_unique_count": hits["qseqid"].nunique() if not hits.empty else 0,
        })

    return pd.DataFrame(rows).set_index("genome_id")


def parse_bacmet(interim_dir: Path, bacmet_fasta: Path) -> pd.DataFrame:
    """Parse BacMet tBLASTn hits for all genomes in interim_dir.

    Returns:
        genome_id | hmrg_hit_count | hmrg_unique_count
    where hmrg_unique_count = distinct BacMet protein entries with a passing hit.
    """
    query_lengths = _read_protein_lengths(bacmet_fasta)
    rows: list[dict] = []
    bacmet_dir = interim_dir / "bacmet"
    if not bacmet_dir.exists():
        logger.warning("BacMet directory not found: %s", bacmet_dir)
        return pd.DataFrame()

    for tsv in sorted(bacmet_dir.glob("*_bacmet.tsv")):
        genome_id = _acc_fn_to_genome_id(tsv)
        try:
            hits = parse_blast_hits(tsv, query_lengths)
        except FileNotFoundError:
            logger.warning("BacMet TSV missing: %s", tsv)
            rows.append({"genome_id": genome_id, "hmrg_hit_count": float("nan"), "hmrg_unique_count": float("nan")})
            continue
        rows.append({
            "genome_id": genome_id,
            "hmrg_hit_count": len(hits),
            "hmrg_unique_count": hits["qseqid"].nunique() if not hits.empty else 0,
        })

    return pd.DataFrame(rows).set_index("genome_id")


# ── ISEScan parser ────────────────────────────────────────────────────────────

def parse_isescan(interim_dir: Path) -> pd.DataFrame:
    """Parse ISEScan TSVs for all genomes in interim_dir.

    Empty TSV = zero IS elements (valid zero, not missing).
    Returns:
        genome_id | is_total_count | is_{family}_count  (one column per IS family)
    """
    rows: list[dict] = []
    isescan_dir = interim_dir / "isescan"
    if not isescan_dir.exists():
        logger.warning("ISEScan directory not found: %s", isescan_dir)
        return pd.DataFrame()

    for tsv in sorted(isescan_dir.glob("*_isescan.tsv")):
        genome_id = _acc_fn_to_genome_id(tsv)

        if tsv.stat().st_size == 0:
            # Legitimate zero: ISEScan ran and found nothing
            rows.append({"genome_id": genome_id, "is_total_count": 0})
            continue

        try:
            d = pd.read_csv(tsv, sep="\t")
        except Exception as exc:
            logger.error("Failed to read %s: %s", tsv, exc)
            rows.append({"genome_id": genome_id, "is_total_count": float("nan")})
            continue

        if d.empty:
            rows.append({"genome_id": genome_id, "is_total_count": 0})
            continue

        # Count IS elements per family; each row in the TSV is one IS element copy
        family_counts = d["family"].value_counts().to_dict()
        row = {"genome_id": genome_id, "is_total_count": len(d)}
        for fam, cnt in family_counts.items():
            row[f"is_{fam}_count"] = cnt
        rows.append(row)

    df = pd.DataFrame(rows).set_index("genome_id")
    # Fill missing IS family columns with 0 (genome has that family = 0 copies)
    is_cols = [c for c in df.columns if c.startswith("is_") and c != "is_total_count"]
    df[is_cols] = df[is_cols].fillna(0).astype(int)
    return df


# ── MLST parser ───────────────────────────────────────────────────────────────

def parse_mlst(interim_dir: Path) -> pd.DataFrame:
    """Parse MLST results TSV for a species.

    Returns:
        genome_id | mlst_scheme | mlst_st
    """
    mlst_file = interim_dir / "mlst" / "mlst_results.tsv"
    if not mlst_file.exists():
        logger.warning("MLST results not found: %s", mlst_file)
        return pd.DataFrame()

    try:
        d = pd.read_csv(mlst_file, sep="\t", header=None)
    except Exception as exc:
        logger.error("Failed to read %s: %s", mlst_file, exc)
        return pd.DataFrame()

    # mlst tool output: FILE SCHEME ST allele1 allele2 ...
    # Column 0 = filename (path), column 1 = scheme, column 2 = ST
    d.columns = ["filepath", "mlst_scheme", "mlst_st"] + [f"allele_{i}" for i in range(len(d.columns) - 3)]
    # Extract genome_id from the filename column
    d["genome_id"] = d["filepath"].apply(lambda p: Path(p).stem.replace(".fna", ""))
    return d[["genome_id", "mlst_scheme", "mlst_st"]].set_index("genome_id")


# ── Metadata parser ───────────────────────────────────────────────────────────

def parse_metadata(interim_dir: Path) -> pd.DataFrame:
    """Parse the per-species metadata TSV.

    Returns:
        genome_id | species | genome_size_bp | gc_content | country |
        isolation_source | collection_year | ...
    """
    meta_file = interim_dir / "metadata.tsv"
    if not meta_file.exists():
        logger.warning("Metadata TSV not found: %s", meta_file)
        return pd.DataFrame()

    try:
        d = pd.read_csv(meta_file, sep="\t")
    except Exception as exc:
        logger.error("Failed to read %s: %s", meta_file, exc)
        return pd.DataFrame()

    if "genome_id" in d.columns:
        d = d.set_index("genome_id")
    return d
