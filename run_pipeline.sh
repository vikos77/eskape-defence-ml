#!/usr/bin/env bash
# run_pipeline.sh — wrapper for running Snakemake on Apple Silicon (osx-arm64)
#
# CONDA_SUBDIR=osx-64 forces conda to resolve x86_64 packages, which run
# via Rosetta 2 on M-series Macs. Most bioinformatics tools are not yet
# packaged for osx-arm64 natively.
#
# Usage:
#   conda activate eskape-ml
#   bash run_pipeline.sh              # full pipeline, 8 cores
#   bash run_pipeline.sh -n           # dry-run
#   bash run_pipeline.sh --cores 4    # custom core count

set -euo pipefail

export CONDA_SUBDIR=osx-64

CORES="${SNAKEMAKE_CORES:-8}"
EXTRA_ARGS="$@"

cd "$(dirname "$0")"

echo "Platform override: CONDA_SUBDIR=osx-64 (Rosetta2 compat)"
echo "Cores: ${CORES}"
echo ""

snakemake \
    --use-conda \
    --cores "${CORES}" \
    --rerun-incomplete \
    --snakefile workflow/Snakefile \
    ${EXTRA_ARGS}
