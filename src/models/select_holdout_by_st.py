#!/usr/bin/env python
"""
Stage 1b – Run MLST on holdout candidates and down-select to 30 per species
           by maximising ST novelty relative to the training set.

Reads
-----
config/holdout_candidates.yaml        – 60 candidates per species
data/raw/holdout_genomes/{sp}/*.fna   – downloaded genomes
data/processed/feature_matrix_3335.parquet  – training STs (sequence_type col)

Writes
------
data/interim/holdout/{sp}/mlst/mlst_results.tsv   – raw MLST output per species
config/holdout_accessions.yaml                     – final 30 per species (no-overlap)
results/holdout_st_selection_report.json           – ST overlap audit for manuscript
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT     = Path(__file__).parent.parent.parent
CONF     = ROOT / "config"
PROC     = ROOT / "data" / "processed"
RAWDIR   = ROOT / "data" / "raw" / "holdout_genomes"
INTERIM  = ROOT / "data" / "interim" / "holdout"
RES      = ROOT / "results"

MLST_ENV = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/.snakemake/conda/79ad6a7652dddecf09240fc0f44ce5ee_"
N_FINAL  = 30   # final holdout size per species

# MLST scheme names used by mlst tool (matches training mlst_scheme column)
MLST_SCHEMES = {
    "abaumannii":  "abaumannii",
    "ecloaceae":   "ecloacae",
    "efaecium":    "efaecium",
    "kpneumoniae": "klebsiella",
    "paeruginosa": "paeruginosa",
    "saureus":     "saureus",
}


def acc_to_fname(acc: str) -> str:
    return acc.replace(".", "_")


def run_mlst(sp: str, genome_dir: Path, out_tsv: Path) -> pd.DataFrame:
    """Run mlst on all .fna files in genome_dir; return parsed results."""
    fna_files = sorted(genome_dir.glob("*.fna"))
    if not fna_files:
        print(f"  WARN: no .fna files in {genome_dir}", file=sys.stderr)
        return pd.DataFrame()

    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    scheme = MLST_SCHEMES.get(sp, "")
    scheme_arg = ["--scheme", scheme] if scheme else []

    cmd = (
        ["conda", "run", "--prefix", MLST_ENV, "mlst"]
        + scheme_arg
        + [str(f) for f in fna_files]
    )

    print(f"  Running mlst on {len(fna_files)} genomes ...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARN: mlst error: {result.stderr[:200]}", file=sys.stderr)

    with open(out_tsv, "w") as f:
        f.write(result.stdout)

    print("done")

    # Parse: FILE  SCHEME  ST  ALLELES...
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        fpath  = Path(parts[0].strip())
        fname  = fpath.stem                          # e.g. GCF_000015425_1
        # Reconstruct accession: replace last _ with .
        # stem format: GCF_000015425_1 → GCF_000015425.1
        acc = fname.rsplit("_", 1)
        acc = acc[0] + "." + acc[1] if len(acc) == 2 else fname
        try:
            st = int(parts[2].strip())
        except (ValueError, IndexError):
            st = None
        rows.append({"accession": acc, "st": st})

    return pd.DataFrame(rows)


def select_by_st_novelty(
    mlst_df: pd.DataFrame,
    training_sts: set,
    n: int,
) -> tuple[list[str], dict]:
    """
    Down-select to n accessions maximising ST novelty.
    Priority 1: novel STs (not in training_sts)
    Priority 2: unknown ST (mlst returned '-' or None)
    Priority 3: shared STs (last resort, documented)
    Returns (selected_accessions, audit_dict)
    """
    novel   = mlst_df[~mlst_df["st"].isin(training_sts) & mlst_df["st"].notna()]
    unknown = mlst_df[mlst_df["st"].isna()]
    shared  = mlst_df[ mlst_df["st"].isin(training_sts)]

    selected = []
    # Deduplicate by ST within novel (take one per novel ST)
    seen_st: set = set()
    for _, row in novel.iterrows():
        if row["st"] not in seen_st:
            seen_st.add(row["st"])
            selected.append(row["accession"])
        if len(selected) == n:
            break

    # Fill from unknown ST
    if len(selected) < n:
        for _, row in unknown.iterrows():
            selected.append(row["accession"])
            if len(selected) == n:
                break

    # Fill from shared ST if still needed
    filled_from_shared = []
    if len(selected) < n:
        sel_set = set(selected)
        for _, row in shared.iterrows():
            if row["accession"] not in sel_set:
                selected.append(row["accession"])
                filled_from_shared.append((row["accession"], row["st"]))
            if len(selected) == n:
                break

    audit = {
        "n_novel_st":         len(novel),
        "n_unknown_st":       len(unknown),
        "n_shared_st":        len(shared),
        "n_selected_novel":   len(novel[novel["accession"].isin(selected)]),
        "n_selected_unknown": len(unknown[unknown["accession"].isin(selected)]),
        "n_filled_shared":    len(filled_from_shared),
        "filled_shared_detail": [
            {"accession": a, "st": st} for a, st in filled_from_shared
        ],
    }
    return selected[:n], audit


def main() -> None:
    print("=" * 60)
    print("Stage 1b – MLST + ST-based holdout down-selection")
    print("=" * 60)

    cand_path = CONF / "holdout_candidates.yaml"
    if not cand_path.exists():
        sys.exit("ERROR: config/holdout_candidates.yaml not found. Run Stage 1 first.")

    with open(cand_path) as f:
        candidates: dict[str, list[str]] = yaml.safe_load(f)

    # Load training STs per species
    fm = pd.read_parquet(PROC / "feature_matrix_3335.parquet")

    final_holdout: dict[str, list[str]] = {}
    full_audit: dict[str, dict] = {}

    for species, acc_list in candidates.items():
        print(f"\n── {species}")

        # Training STs for this species
        train_sts = set(
            fm[fm["species"] == species]["sequence_type"].dropna().astype(int)
        )
        print(f"  Training STs: {len(train_sts)} unique")

        # Run MLST
        genome_dir = RAWDIR / species
        out_tsv    = INTERIM / species / "mlst" / "mlst_results.tsv"
        mlst_df    = run_mlst(species, genome_dir, out_tsv)

        if mlst_df.empty:
            print(f"  WARN: empty MLST result for {species}; using all candidates")
            final_holdout[species] = acc_list[:N_FINAL]
            full_audit[species]    = {"error": "mlst empty"}
            continue

        print(f"  MLST results: {len(mlst_df)} rows "
              f"({mlst_df['st'].notna().sum()} with typed ST)")

        # Down-select
        selected, audit = select_by_st_novelty(mlst_df, train_sts, N_FINAL)
        print(f"  Selected {len(selected)}/{N_FINAL}: "
              f"{audit['n_selected_novel']} novel ST, "
              f"{audit['n_selected_unknown']} unknown ST, "
              f"{audit['n_filled_shared']} shared ST (fill)")

        if audit["n_filled_shared"] > 0:
            print(f"  WARN: {audit['n_filled_shared']} genomes share STs with training "
                  f"(documented in audit):")
            for item in audit["filled_shared_detail"]:
                print(f"    {item['accession']}: ST{item['st']}")

        # Final accession overlap assertion (belts and braces)
        training_acc = set(fm.index.astype(str))
        overlap = training_acc & set(selected)
        assert len(overlap) == 0, f"ACCESSION LEAK in {species}: {overlap}"
        print(f"  Accession overlap with training: 0 (verified)")

        final_holdout[species] = selected
        full_audit[species]    = audit

    # Write final holdout accessions
    out_yaml = CONF / "holdout_accessions.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(final_holdout, f, default_flow_style=False)
    print(f"\nSaved: {out_yaml}")

    # Write audit report
    audit_path = RES / "holdout_st_selection_report.json"
    RES.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(full_audit, f, indent=2)
    print(f"Saved: {audit_path}")

    total = sum(len(v) for v in final_holdout.values())
    print(f"\nFinal holdout: {total} genomes across {len(final_holdout)} species")
    for sp, accs in final_holdout.items():
        print(f"  {sp}: {len(accs)}")
    print("\nNext: run Stage 3 (DF + PADLOC) on final holdout accessions")


if __name__ == "__main__":
    main()
