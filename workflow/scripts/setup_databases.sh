#!/usr/bin/env bash
# setup_databases.sh — Download and prepare reference databases for Phase 2
#
# Run once before the Snakemake pipeline. Safe to re-run (skips existing files).
#
# Usage:
#   conda activate eskape-ml
#   bash workflow/scripts/setup_databases.sh
#
# Downloads to: data/raw/databases/
# Total size: ~150 MB

set -euo pipefail

DB_DIR="data/raw/databases"
mkdir -p "$DB_DIR"

echo "Setting up reference databases in ${DB_DIR}/"
echo "================================================"

# ── 1. ResFinder database ─────────────────────────────────────────────────────
# Official repository from the Centre for Genomic Epidemiology (CGE)
RESFINDER_DB="${DB_DIR}/db_resfinder"
if [ -d "$RESFINDER_DB" ]; then
    echo "[SKIP] ResFinder DB already exists at ${RESFINDER_DB}"
else
    echo "[1/3] Cloning ResFinder database..."
    git clone --depth 1 \
        https://git.sr.ht/~genomicepidemiology/resfinder_db \
        "$RESFINDER_DB"
    echo "      Done."
fi

# ── 2. ICEberg IME protein sequences ─────────────────────────────────────────
# ICEberg2 database: integrative and conjugative elements + IMEs
# Protein sequences for tBLASTn (same as published Acinetobacter pipeline)
ICEBERG_FASTA="${DB_DIR}/ICEberg_IME.fasta"
if [ -f "$ICEBERG_FASTA" ]; then
    echo "[SKIP] ICEberg IME FASTA already exists"
else
    echo "[2/3] Downloading ICEberg IME protein sequences..."
    curl -L --retry 3 --retry-delay 5 \
        "https://db-mml.sjtu.edu.cn/ICEberg2/download/IME_aa_all.fas" \
        -o "$ICEBERG_FASTA"
    # Make BLAST database
    makeblastdb -in "$ICEBERG_FASTA" -dbtype prot \
        -out "${DB_DIR}/ICEberg_IME" -title "ICEberg2_IME"
    echo "      Done."
fi

# ── 3. BacMet2 experimentally confirmed HMRG sequences ───────────────────────
BACMET_FASTA="${DB_DIR}/BacMet2_EXP.fasta"
if [ -f "$BACMET_FASTA" ]; then
    echo "[SKIP] BacMet2 EXP FASTA already exists"
else
    echo "[3/3] Downloading BacMet2 experimental resistance gene sequences..."
    curl -L --retry 3 --retry-delay 5 \
        "http://bacmet.biomedicine.gu.se/download/BacMet2_EXP.fasta.gz" \
        -o "${BACMET_FASTA}.gz"
    gunzip "${BACMET_FASTA}.gz"
    # Make BLAST database
    makeblastdb -in "$BACMET_FASTA" -dbtype prot \
        -out "${DB_DIR}/BacMet2_EXP" -title "BacMet2_EXP"
    echo "      Done."
fi

echo ""
echo "================================================"
echo "All databases ready. Verify paths in Snakefile:"
echo "  BACMET_DB  = ${BACMET_FASTA}"
echo "  ICEBERG_DB = ${ICEBERG_FASTA}"
echo "  ResFinder  = ${RESFINDER_DB}"
