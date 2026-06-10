"""
copy_overlap_annotations.py — Copy 900 annotation outputs from the original project.

For each of the 900 original accessions (150/species × 6 species), copies
annotation outputs from eskape-defence-ml into eskape-defence-ml_2, so the
Snakemake pipeline only needs to annotate the ~2,560 new genomes fresh.

Tool outputs copied:
  defensefinder  → data/interim/{sp}/defensefinder/{acc_fn}/   (full dir)
  padloc         → data/interim/{sp}/padloc/{acc_fn}_padloc.{csv,gff}
                                            {acc_fn}_prodigal.faa
  resfinder      → data/interim/{sp}/resfinder/{acc_fn}/       (full dir)
  iceberg        → data/interim/{sp}/iceberg/{acc_fn}_iceberg.tsv
  isescan        → data/interim/{sp}/isescan/{acc_fn}_isescan.{tsv,sum}
  amrfinderplus  → data/interim/{sp}/amrfinderplus/{acc_fn}_amrfinder.tsv

Filename convention: GCF_043738335.1 → GCF_043738335_1  (dot → underscore)

Safety:
  - Every destination path is asserted to start with NEW_ROOT before any write.
  - Existing destinations are skipped (idempotent; re-runnable safely).
  - Missing sources are logged per tool, not raised — genome gets re-annotated fresh.

Usage:
    cd eskape-defence-ml_2
    conda run -n eskape-ml python src/features/copy_overlap_annotations.py
"""

import os
import shutil
from pathlib import Path

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────

NEW_ROOT  = Path("/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml_2").resolve()
ORIG_ROOT = Path("/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml").resolve()

ORIG_CONFIG  = ORIG_ROOT / "config/species.yaml"
ORIG_INTERIM = ORIG_ROOT / "data/interim"
NEW_INTERIM  = NEW_ROOT  / "data/interim"

TOOLS = ["defensefinder", "padloc", "resfinder", "iceberg", "isescan", "amrfinderplus"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def acc_fn(accession: str) -> str:
    """GCF_043738335.1 → GCF_043738335_1"""
    return accession.replace(".", "_")


def safe_dest(path: Path) -> Path:
    """Assert destination is inside NEW_ROOT — hard block on any accidental writes to original."""
    resolved = path.resolve()
    assert str(resolved).startswith(str(NEW_ROOT)), (
        f"SAFETY VIOLATION: destination {resolved} is outside {NEW_ROOT}"
    )
    return resolved


def source_paths(sp: str, acc: str, tool: str) -> list[Path]:
    """Return list of source files/dirs for one accession + tool."""
    fn = acc_fn(acc)
    base = ORIG_INTERIM / sp / tool
    if tool == "defensefinder":
        return [base / fn]                             # directory
    if tool == "padloc":
        return [base / f"{fn}_padloc.csv",
                base / f"{fn}_padloc.gff",
                base / f"{fn}_prodigal.faa"]
    if tool == "resfinder":
        return [base / fn]                             # directory
    if tool == "iceberg":
        return [base / f"{fn}_iceberg.tsv"]
    if tool == "isescan":
        return [base / f"{fn}_isescan.tsv",
                base / f"{fn}_isescan.sum"]
    if tool == "amrfinderplus":
        return [base / f"{fn}_amrfinder.tsv"]
    raise ValueError(f"Unknown tool: {tool}")


def dest_paths(sp: str, acc: str, tool: str) -> list[Path]:
    """Return list of destination files/dirs for one accession + tool."""
    fn = acc_fn(acc)
    base = NEW_INTERIM / sp / tool
    if tool == "defensefinder":
        return [safe_dest(base / fn)]
    if tool == "padloc":
        return [safe_dest(base / f"{fn}_padloc.csv"),
                safe_dest(base / f"{fn}_padloc.gff"),
                safe_dest(base / f"{fn}_prodigal.faa")]
    if tool == "resfinder":
        return [safe_dest(base / fn)]
    if tool == "iceberg":
        return [safe_dest(base / f"{fn}_iceberg.tsv")]
    if tool == "isescan":
        return [safe_dest(base / f"{fn}_isescan.tsv"),
                safe_dest(base / f"{fn}_isescan.sum")]
    if tool == "amrfinderplus":
        return [safe_dest(base / f"{fn}_amrfinder.tsv")]
    raise ValueError(f"Unknown tool: {tool}")


def copy_item(src: Path, dst: Path) -> str:
    """
    Copy src → dst. Returns 'copied', 'skipped' (already exists), or 'missing'.
    Uses copytree for directories, copy2 for files.
    """
    if not src.exists():
        return "missing"
    if dst.exists():
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copied"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(ORIG_CONFIG) as f:
        orig_cfg = yaml.safe_load(f)

    print("Copying annotation outputs for 900 original accessions")
    print("=" * 60)
    print(f"Source:      {ORIG_ROOT}")
    print(f"Destination: {NEW_ROOT}")
    print()

    grand = {"copied": 0, "skipped": 0, "missing": 0}

    for sp_key, sp_cfg in orig_cfg["species"].items():
        accessions = sp_cfg.get("accessions", [])
        print(f"{'─'*60}")
        print(f"{sp_key} ({len(accessions)} accessions)")

        sp_stats: dict[str, dict] = {t: {"copied": 0, "skipped": 0, "missing": 0}
                                     for t in TOOLS}

        for acc in accessions:
            for tool in TOOLS:
                srcs = source_paths(sp_key, acc, tool)
                dsts = dest_paths(sp_key, acc, tool)
                for src, dst in zip(srcs, dsts):
                    result = copy_item(src, dst)
                    sp_stats[tool][result] += 1
                    grand[result] += 1

        # Per-tool summary for this species
        for tool in TOOLS:
            s = sp_stats[tool]
            missing_note = f"  ← {s['missing']} missing (will be re-annotated)" if s["missing"] else ""
            print(f"  {tool:<16} copied={s['copied']:>4}  skipped={s['skipped']:>4}  "
                  f"missing={s['missing']:>3}{missing_note}")

    print()
    print("=" * 60)
    print(f"TOTAL  copied={grand['copied']}  skipped={grand['skipped']}  "
          f"missing={grand['missing']}")
    if grand["missing"]:
        print(f"  → {grand['missing']} missing source files: these genomes will be "
              f"re-annotated fresh by Snakemake (identical tool versions → identical output).")
    print("\nDone. Verify counts above before proceeding to Snakefile setup.")


if __name__ == "__main__":
    main()
