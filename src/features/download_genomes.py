"""
download_genomes.py — Download/copy all 3,460 genomes for the ESKAPE 4× study.

Strategy:
  Original 900 accessions: copy FASTA from eskape-defence-ml (already on disk).
  New 2,560 accessions: download from NCBI using ncbi-datasets-cli.

Output: data/raw/genomes/{sp}/{acc_fn}.fna
  (same path convention Snakemake expects — annotation pipeline picks up from here)

Parallelism: ThreadPoolExecutor with WORKERS threads (default 6).
Retries: up to 5 per genome, with exponential back-off.
Idempotent: skips genomes whose .fna already exists.

Usage:
    cd eskape-defence-ml_2
    conda run -n eskape-ml python src/features/download_genomes.py [--workers N]
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────

NEW_ROOT  = Path("/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml_2").resolve()
ORIG_ROOT = Path("/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml").resolve()

NEW_CONFIG  = NEW_ROOT  / "config/species.yaml"
ORIG_CONFIG = ORIG_ROOT / "config/species.yaml"
NEW_GENOMES = NEW_ROOT  / "data/raw/genomes"
ORIG_GENOMES= ORIG_ROOT / "data/raw/genomes"

LOG_DIR = NEW_ROOT / "logs"


def acc_fn(accession: str) -> str:
    return accession.replace(".", "_")


def safe_dest(path: Path) -> Path:
    resolved = path.resolve()
    assert str(resolved).startswith(str(NEW_ROOT)), (
        f"SAFETY: {resolved} is outside {NEW_ROOT}"
    )
    return resolved


# ── Genome download (mirrors Snakefile download_genome rule) ──────────────────

def download_one(acc: str, dest: Path, max_retries: int = 5) -> tuple[str, str]:
    """
    Download one genome from NCBI; return (accession, status).
    status: 'ok' | 'skipped' | 'failed'
    """
    safe_dest(dest)
    if dest.exists():
        return acc, "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        with tempfile.TemporaryDirectory(dir=dest.parent,
                                         prefix=f"tmp_{acc_fn(acc)}_") as tmpdir:
            archive = Path(tmpdir) / "genome.zip"
            try:
                result = subprocess.run(
                    ["datasets", "download", "genome", "accession", acc,
                     "--include", "genome", "--filename", str(archive)],
                    capture_output=True, timeout=300,
                )
                if result.returncode != 0 or not archive.exists():
                    raise RuntimeError(f"datasets exit {result.returncode}")

                # Verify zip integrity
                check = subprocess.run(
                    ["unzip", "-tq", str(archive)],
                    capture_output=True,
                )
                if check.returncode != 0:
                    raise RuntimeError("corrupt zip")

                # Extract
                subprocess.run(
                    ["unzip", "-oq", str(archive), "-d", tmpdir],
                    check=True, capture_output=True,
                )

                # Find the .fna file; fall back to paired accession if missing
                fna_files = list(Path(tmpdir).rglob("*.fna"))
                if not fna_files:
                    paired = _paired_accession(Path(tmpdir))
                    if paired:
                        return download_one(paired, dest, max_retries=max_retries)
                    raise RuntimeError("no .fna found")

                shutil.move(str(fna_files[0]), dest)
                return acc, "ok"

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    return acc, f"failed: {e}"

    return acc, "failed: max retries"


def _paired_accession(tmpdir: Path) -> str:
    """Extract paired assembly accession from the NCBI report, if present."""
    report = tmpdir / "ncbi_dataset" / "data" / "assembly_data_report.jsonl"
    if not report.exists():
        return ""
    try:
        with report.open() as f:
            data = json.loads(next(f))
        info = data.get("assemblyInfo", {})
        paired = info.get("pairedAssembly", {}) or {}
        return paired.get("accession") or data.get("pairedAccession") or ""
    except Exception:
        return ""


# ── FASTA copy for original accessions ────────────────────────────────────────

def copy_original_fasta(acc: str, sp: str) -> tuple[str, str]:
    fn = acc_fn(acc)
    src  = ORIG_GENOMES / sp / f"{fn}.fna"
    dest = safe_dest(NEW_GENOMES / sp / f"{fn}.fna")
    if dest.exists():
        return acc, "skipped"
    if not src.exists():
        # Original FASTA missing — will fall through to download
        return acc, "src_missing"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return acc, "copied"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(workers: int = 6):
    LOG_DIR.mkdir(exist_ok=True)

    with open(NEW_CONFIG) as f:
        new_cfg = yaml.safe_load(f)
    with open(ORIG_CONFIG) as f:
        orig_cfg = yaml.safe_load(f)

    # Build original accession sets per species
    orig_accs: dict[str, set[str]] = {
        sp: set(cfg.get("accessions", []))
        for sp, cfg in orig_cfg["species"].items()
    }

    # Collect all tasks: (acc, sp, is_original)
    copy_tasks: list[tuple[str, str]] = []
    download_tasks: list[tuple[str, str]] = []

    for sp, sp_cfg in new_cfg["species"].items():
        for acc in sp_cfg.get("accessions", []):
            if acc in orig_accs.get(sp, set()):
                copy_tasks.append((acc, sp))
            else:
                download_tasks.append((acc, sp))

    print(f"Genomes to copy  (originals): {len(copy_tasks)}")
    print(f"Genomes to download (new):    {len(download_tasks)}")
    print(f"Workers: {workers}")
    print()

    # ── Phase 1: copy originals ───────────────────────────────────────────────
    print("Phase 1 — Copying original FASTAs...")
    results_copy = {"copied": 0, "skipped": 0, "src_missing": 0}
    to_download_after_missing: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(copy_original_fasta, acc, sp): (acc, sp)
                   for acc, sp in copy_tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            acc, status = fut.result()
            results_copy[status] = results_copy.get(status, 0) + 1
            if status == "src_missing":
                _, sp = futures[fut]
                to_download_after_missing.append((acc, sp))
            if i % 100 == 0 or i == len(copy_tasks):
                print(f"  {i}/{len(copy_tasks)} originals processed", flush=True)

    print(f"  copied={results_copy['copied']}  "
          f"skipped={results_copy['skipped']}  "
          f"src_missing={results_copy['src_missing']}")
    if results_copy["src_missing"]:
        print(f"  {results_copy['src_missing']} originals without local FASTA "
              f"added to download queue.")

    # ── Phase 2: download new genomes ─────────────────────────────────────────
    all_downloads = download_tasks + to_download_after_missing
    print(f"\nPhase 2 — Downloading {len(all_downloads)} new genomes "
          f"({workers} parallel workers)...")

    failed_accs = []
    counts = {"ok": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_one, acc, safe_dest(NEW_GENOMES / sp / f"{acc_fn(acc)}.fna")
            ): acc
            for acc, sp in all_downloads
        }
        for i, fut in enumerate(as_completed(futures), 1):
            acc, status = fut.result()
            if status.startswith("failed"):
                counts["failed"] += 1
                failed_accs.append((acc, status))
            else:
                counts[status] = counts.get(status, 0) + 1
            if i % 100 == 0 or i == len(all_downloads):
                print(f"  {i}/{len(all_downloads)}  "
                      f"ok={counts['ok']} skipped={counts['skipped']} "
                      f"failed={counts['failed']}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("DOWNLOAD SUMMARY")
    print(f"  Originals copied:  {results_copy['copied']}")
    print(f"  New downloaded:    {counts['ok']}")
    print(f"  Skipped (exist):   {results_copy['skipped'] + counts['skipped']}")
    print(f"  Failed:            {counts['failed']}")

    if failed_accs:
        fail_log = LOG_DIR / "download_failures.txt"
        with open(fail_log, "w") as f:
            for acc, reason in failed_accs:
                f.write(f"{acc}\t{reason}\n")
        print(f"\n  Failed accessions written to: {fail_log}")
        print("  Re-run this script to retry — it skips already-downloaded genomes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel download threads (default: 6)")
    args = parser.parse_args()
    main(args.workers)
