#!/usr/bin/env python
"""
Stage 2 – Download holdout candidate genomes from NCBI.

Reads config/holdout_candidates.yaml and downloads each genome to
data/raw/holdout_genomes/{species}/{GCF_accession}.fna

Uses ncbi-datasets-cli in batches of 20 to stay within NCBI rate limits.
Skips any accession whose .fna already exists (safe to re-run).
"""

import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

ROOT    = Path(__file__).parent.parent.parent
CONF    = ROOT / "config"
RAWDIR  = ROOT / "data" / "raw" / "holdout_genomes"
BATCH   = 20    # accessions per datasets download call
RETRIES = 3


def acc_to_fname(acc: str) -> str:
    return acc.replace(".", "_")


def download_batch(accessions: list[str], outdir: Path) -> list[str]:
    """
    Download a batch of accessions into outdir.
    Returns list of accessions that failed after all retries.
    """
    failed = []
    tmpzip = outdir / "_batch.zip"

    for attempt in range(1, RETRIES + 1):
        cmd = [
            "datasets", "download", "genome", "accession",
            *accessions,
            "--filename",    str(tmpzip),
            "--include",     "genome",
            "--no-progressbar",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            break
        print(f"     attempt {attempt}/{RETRIES} failed: {result.stderr[:120]}",
              file=sys.stderr)
        time.sleep(5 * attempt)
    else:
        print(f"  WARN: batch failed after {RETRIES} attempts", file=sys.stderr)
        return accessions

    # Extract and rename .fna files
    try:
        with zipfile.ZipFile(tmpzip, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".fna") and "/data/" in member:
                    # path: ncbi_dataset/data/{accession}/.../*.fna
                    parts = member.split("/")
                    if len(parts) >= 4:
                        acc  = parts[2]                     # e.g. GCF_000015425.1
                        dest = outdir / f"{acc_to_fname(acc)}.fna"
                        if not dest.exists():
                            dest.write_bytes(zf.read(member))
        tmpzip.unlink(missing_ok=True)
    except zipfile.BadZipFile as exc:
        print(f"  WARN: bad zip: {exc}", file=sys.stderr)
        tmpzip.unlink(missing_ok=True)
        return accessions

    return failed


def main() -> None:
    print("=" * 60)
    print("Stage 2 – Holdout genome download")
    print("=" * 60)

    cand_path = CONF / "holdout_candidates.yaml"
    if not cand_path.exists():
        sys.exit("ERROR: config/holdout_candidates.yaml not found. Run Stage 1 first.")

    with open(cand_path) as f:
        candidates: dict[str, list[str]] = yaml.safe_load(f)

    all_failed = []

    for species, accessions in candidates.items():
        sp_dir = RAWDIR / species
        sp_dir.mkdir(parents=True, exist_ok=True)

        # Which accessions need downloading?
        needed = [
            acc for acc in accessions
            if not (sp_dir / f"{acc_to_fname(acc)}.fna").exists()
        ]
        already = len(accessions) - len(needed)
        print(f"\n── {species}: {len(accessions)} total, {already} already present, "
              f"{len(needed)} to download")

        if not needed:
            continue

        # Batch download
        for i in range(0, len(needed), BATCH):
            batch = needed[i : i + BATCH]
            print(f"   batch {i//BATCH + 1}: {len(batch)} accessions ...", end=" ")
            failed = download_batch(batch, sp_dir)
            n_ok = len(batch) - len(failed)
            print(f"{n_ok} OK, {len(failed)} failed")
            all_failed.extend([(species, a) for a in failed])
            if i + BATCH < len(needed):
                time.sleep(2)   # polite rate-limiting

        # Count what we have
        fna_files = list(sp_dir.glob("*.fna"))
        print(f"   {len(fna_files)} .fna files now in {sp_dir.relative_to(ROOT)}")

    print("\n── Download summary ──")
    for species, accessions in candidates.items():
        sp_dir = RAWDIR / species
        n_present = len(list(sp_dir.glob("*.fna")))
        print(f"  {species}: {n_present}/{len(accessions)} genomes present")

    if all_failed:
        print(f"\nWARN: {len(all_failed)} accessions failed to download:")
        for sp, acc in all_failed:
            print(f"  {sp}: {acc}")
    else:
        print("\nAll downloads succeeded.")

    print("\nNext: run MLST on candidates → Stage 1b (ST-based selection)")


if __name__ == "__main__":
    main()
