#!/usr/bin/env python
"""
Stage 3 – Run DefenseFinder 2.0.1 and PADLOC v2.0.0 on holdout genomes.

Reads config/holdout_accessions.yaml (30 per species, ST-filtered).
Writes DF outputs to data/interim/holdout/{sp}/defensefinder/{acc_fn}/
Writes PADLOC outputs to data/interim/holdout/{sp}/padloc/{acc_fn}_padloc.csv

Both tools are invoked via conda run on their Snakemake conda environments to
guarantee the same tool + model/DB versions used for training data.

Safe to re-run: skips any genome whose output already exists.
"""

import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
CONF = ROOT / "config"

RAWDIR  = ROOT / "data" / "raw" / "holdout_genomes"
INTERIM = ROOT / "data" / "interim" / "holdout"

# Snakemake conda env paths (verified at start of C3 overhaul session)
DF_ENV    = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/.snakemake/conda/22b28198c2bd510e5fa4e6abd853fd19_"
PADLOC_ENV = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml/.snakemake/conda/5103cc8bb9918aaeace824bae11e6843_"

# DF models dir (same symlink used for training)
DF_MODELS = str(ROOT / "data" / "raw" / "databases" / "defensefinder_models")


def acc_to_fname(acc: str) -> str:
    return acc.replace(".", "_")


def run_defensefinder(acc: str, fna: Path, outdir: Path) -> bool:
    """
    Run DefenseFinder on a single genome.  Returns True on success.
    outdir is the per-genome directory.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    systems_tsv = outdir / "defense_finder_systems.tsv"

    # Already done if the prefixed file (actual DF output) exists, OR the canonical
    # file has content (training-style output).
    acc_fn = outdir.name
    prefixed = outdir / f"{acc_fn}_defense_finder_systems.tsv"
    if prefixed.exists() and prefixed.stat().st_size > 0:
        return True
    if systems_tsv.exists() and systems_tsv.stat().st_size > 0:
        return True

    cmd = [
        "conda", "run", "--prefix", DF_ENV,
        "defense-finder", "run",
        "--models-dir", DF_MODELS,
        "--out-dir",    str(outdir),
        str(fna),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    DF FAIL {acc}: {result.stderr[-200:]}", file=sys.stderr)
        # Touch empty output so we don't retry indefinitely
        systems_tsv.touch()
        return False

    if not systems_tsv.exists():
        systems_tsv.touch()
    return True


def run_padloc(acc: str, fna: Path, padloc_dir: Path) -> bool:
    """
    Run PADLOC on a single genome.  Returns True on success.
    Output CSV: padloc_dir/{acc_fn}_padloc.csv
    """
    padloc_dir.mkdir(parents=True, exist_ok=True)
    acc_fn = acc_to_fname(acc)
    out_csv = padloc_dir / f"{acc_fn}_padloc.csv"

    if out_csv.exists():
        return True

    cmd = [
        "conda", "run", "--prefix", PADLOC_ENV,
        "padloc",
        "--fna",    str(fna),
        "--outdir", str(padloc_dir),
        "--cpu",    "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    PADLOC FAIL {acc}: {result.stderr[-200:]}", file=sys.stderr)
        out_csv.touch()
        return False

    if not out_csv.exists():
        out_csv.touch()
    return True


def main() -> None:
    print("=" * 60)
    print("Stage 3 – DefenseFinder 2.0.1 + PADLOC v2.0.0 on holdout")
    print("=" * 60)

    acc_path = CONF / "holdout_accessions.yaml"
    if not acc_path.exists():
        sys.exit("ERROR: config/holdout_accessions.yaml not found. "
                 "Run Stages 1+1b first.")

    with open(acc_path) as f:
        holdout: dict[str, list[str]] = yaml.safe_load(f)

    total_genomes = sum(len(v) for v in holdout.values())
    print(f"Total holdout genomes to process: {total_genomes}")
    print(f"DF env:     {DF_ENV}")
    print(f"PADLOC env: {PADLOC_ENV}")
    print(f"DF models:  {DF_MODELS}")

    df_ok = df_fail = padloc_ok = padloc_fail = 0

    for species, accessions in holdout.items():
        sp_genome_dir = RAWDIR / species
        sp_interim    = INTERIM / species

        print(f"\n── {species}: {len(accessions)} genomes")

        for i, acc in enumerate(accessions, 1):
            acc_fn = acc_to_fname(acc)
            fna    = sp_genome_dir / f"{acc_fn}.fna"

            if not fna.exists():
                print(f"  [{i:2d}/{len(accessions)}] SKIP (no .fna): {acc}")
                continue

            print(f"  [{i:2d}/{len(accessions)}] {acc}", end="  ", flush=True)

            # DefenseFinder
            df_outdir = sp_interim / "defensefinder" / acc_fn
            ok_df = run_defensefinder(acc, fna, df_outdir)
            status_df = "DF:OK" if ok_df else "DF:FAIL"
            if ok_df:
                df_ok += 1
            else:
                df_fail += 1

            # PADLOC
            padloc_dir = sp_interim / "padloc"
            ok_pl = run_padloc(acc, fna, padloc_dir)
            status_pl = "PADLOC:OK" if ok_pl else "PADLOC:FAIL"
            if ok_pl:
                padloc_ok += 1
            else:
                padloc_fail += 1

            print(f"{status_df}  {status_pl}")

    print("\n── Pipeline complete ──")
    print(f"DefenseFinder : {df_ok} OK, {df_fail} failed")
    print(f"PADLOC        : {padloc_ok} OK, {padloc_fail} failed")
    print("\nNext: Stage 4 – build holdout feature matrix")


if __name__ == "__main__":
    main()
