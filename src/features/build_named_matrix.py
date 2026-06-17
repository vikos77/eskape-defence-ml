#!/usr/bin/env python3
"""
build_named_matrix.py

Reconstructs the named-236 feature matrix from the original 367-feature matrix.

Background: feature_matrix_3335_named.parquet has existed on disk since
2026-06-16 and is read directly by 9+ downstream scripts (run_q1_named.py,
run_q2_named.py, run_q4_shap_named.py, run_mcnemar_named.py,
run_s6_spearman_named.py, run_subanalysis_named.py, build_nb09.py,
build_nb10.py, c3_extended_holdout.py) and by NB06/07/08/09/10. No script in
the repo ever generated it -- the filter was applied ad hoc and never
committed as code. This script closes that gap: it is the single source of
truth for the 367 -> 236 column drop, and is idempotent against the existing
on-disk file.

Filter ("Feature matrix purge" rationale):
  - PDC      : dp_padloc_PDC-S* / dp_padloc_PDC-M*  (81 cols) -- PADLOC internal
               cluster IDs, no published mechanism class.
  - DS-N     : dp_df_DS-*                            (30 cols) -- DefensePredictor
               ML-predicted candidates, mechanism uncharacterised.
  - All_UG   : dp_df_UG*                             (4 cols)  -- DefenseFinder's
               own "All Uncharacterised Groups" label.
  - catch-all: *_other / *_unknown / *_unsubtyped / *_merge (16 cols) -- explicit
               "not classifiable" designations in PADLOC/DefenseFinder.

Total: 131 dp_ columns + 131 mirrored dc_ columns = 262 columns removed.
236 named dp_ columns + 236 named dc_ columns retained. 569 total columns.

CHANGELOG (2026-06-17): v1 of this script dropped only dp_ (presence/absence)
columns, leaving the mirrored dc_ (count) columns for PDC/DS-N/All_UG/catch-all
systems in place (700 total columns) -- e.g. dc_padloc_PDC-M01 stayed in the
matrix even though dp_padloc_PDC-M01 was removed. Verified harmless (no
downstream script reads dc_ columns for any of the 131 removed systems --
the only dc_ usage anywhere in the pipeline is dc_RM_*, and RM is retained),
but inconsistent with the intent of retaining exactly "236 named, citable
defence systems": 367 dc_ columns -- including 131 for systems with no
citable name -- were still present. v2 drops both dp_ and its dc_ mirror for
every removed system, so the named matrix contains exactly the 236 retained
systems in both representations.

Usage:
    python src/features/build_named_matrix.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(".")
PROC = ROOT / "data" / "processed"

SRC_PATH      = PROC / "feature_matrix_3335.parquet"
OUT_PATH      = PROC / "feature_matrix_3335_named.parquet"

PDC_RE       = re.compile(r"PDC-?[SM]\d")
DS_N_RE      = re.compile(r"_DS-\d")
ALL_UG_RE    = re.compile(r"_UG\d")
CATCH_ALL_RE = re.compile(r"(_other|_unknown|_unsubtyped|_merge)$")


def classify(col: str) -> str | None:
    """Return the removal category for a dp_ or dc_ column, or None if it should be kept."""
    if not (col.startswith("dp_") or col.startswith("dc_")):
        return None
    if PDC_RE.search(col):
        return "PDC"
    if DS_N_RE.search(col):
        return "DS-N"
    if ALL_UG_RE.search(col):
        return "All_UG"
    if CATCH_ALL_RE.search(col):
        return "catch-all"
    return None


def main() -> None:
    print("=" * 70)
    print("Building named-236 feature matrix from 367-feature matrix")
    print("=" * 70)

    fm = pd.read_parquet(SRC_PATH)
    print(f"  Source matrix: {fm.shape}")

    to_drop: dict[str, list[str]] = {"PDC": [], "DS-N": [], "All_UG": [], "catch-all": []}
    for col in fm.columns:
        cat = classify(col)
        if cat is not None:
            to_drop[cat].append(col)

    for cat, cols in to_drop.items():
        print(f"  {cat}: {len(cols)} columns (dp_ + dc_)")

    all_dropped = [c for cols in to_drop.values() for c in cols]
    n_dropped = len(all_dropped)
    print(f"  Total dropped: {n_dropped}")
    assert n_dropped == 262, f"Expected 262 dropped columns, got {n_dropped}"

    fm_named = fm.drop(columns=all_dropped)
    print(f"  Named matrix: {fm_named.shape}")
    assert fm_named.shape[1] == 569, f"Expected 569 columns, got {fm_named.shape[1]}"

    n_dp_named = sum(1 for c in fm_named.columns if c.startswith("dp_"))
    n_dc_named = sum(1 for c in fm_named.columns if c.startswith("dc_"))
    print(f"  Named dp_ columns: {n_dp_named}, named dc_ columns: {n_dc_named}")
    assert n_dp_named == 236, f"Expected 236 named dp_ columns, got {n_dp_named}"
    assert n_dc_named == 236, f"Expected 236 named dc_ columns, got {n_dc_named}"

    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        same_cols = list(existing.columns) == list(fm_named.columns)
        same_index = existing.index.equals(fm_named.index)
        same_values = same_cols and same_index and all(
            existing[c].equals(fm_named[c]) for c in existing.columns
        )
        print(f"  Matches existing on-disk file: {same_values}")
        if same_values:
            print("  Reconstruction confirmed identical to existing file. No write needed.")
            return
        if "--update" not in sys.argv:
            print("  MISMATCH against existing feature_matrix_3335_named.parquet.")
            print("  Run with --update to intentionally overwrite (e.g. after a filter")
            print("  definition change), or investigate before proceeding otherwise.")
            sys.exit(1)
        print("  Mismatch is expected (--update passed). Overwriting.")

    fm_named.to_parquet(OUT_PATH)
    print(f"  Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
