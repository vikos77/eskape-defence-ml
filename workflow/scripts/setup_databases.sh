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
    echo "[1/2] Cloning ResFinder database..."
    git clone --depth 1 \
        https://bitbucket.org/genomicepidemiology/resfinder_db \
        "$RESFINDER_DB"
    echo "      Done."
fi

# ── 3. BacMet2 experimentally confirmed HMRG sequences ───────────────────────
# NOTE: bacmet.biomedicine.gu.se returns 502 intermittently.
# If download fails, obtain BacMet2_EXP_database.fasta from the published
# Acinetobacter pipeline HPC storage, or retry when server recovers.
BACMET_FASTA="${DB_DIR}/BacMet2_EXP.fasta"
if [ -f "$BACMET_FASTA" ] && [ "$(wc -c < "$BACMET_FASTA")" -gt 10000 ]; then
    echo "[SKIP] BacMet2 EXP FASTA already exists"
else
    echo "[2/2] Downloading BacMet2 experimental resistance gene sequences..."
    curl -Lk --retry 3 --retry-delay 5 \
        "https://bacmet.biomedicine.gu.se/downloads/BacMet2_EXP_database.fasta" \
        -o "$BACMET_FASTA"
    # Verify it's a real FASTA (not an error page)
    if grep -q "^>" "$BACMET_FASTA" 2>/dev/null; then
        makeblastdb -in "$BACMET_FASTA" -dbtype prot \
            -out "${DB_DIR}/BacMet2_EXP" -title "BacMet2_EXP"
        echo "      Done."
    else
        echo "      WARNING: BacMet server returned an error page."
        echo "      HMRG annotation will be skipped until this is resolved."
        echo "      Retry: bash workflow/scripts/setup_databases.sh"
        rm -f "$BACMET_FASTA"
    fi
fi

echo ""
echo "================================================"
echo "All databases ready. Verify paths in Snakefile:"
echo "  BACMET_DB  = ${BACMET_FASTA}"
echo "  ResFinder  = ${RESFINDER_DB}"
