#!/usr/bin/env python
"""
Stage 1 – Select holdout genome accessions for C3 external validation.

Queries NCBI for complete genomes per ESKAPE species, filters the 3,335-genome
training set, and selects up to 60 candidates per species (to be down-selected to
30 after MLST-based ST filtering in Stage 1b).

Candidate selection maximises country/year diversity as the best available NCBI
proxy for phylogenetic independence.  ST-based filtering happens in Stage 1b once
MLST has been run on the candidates.

Outputs
-------
config/holdout_candidates.yaml   – 60 candidates per species (or however many are
                                   available if fewer than 60 novel-accession
                                   complete genomes exist on NCBI)
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent.parent
PROC = ROOT / "data" / "processed"
CONF = ROOT / "config"

# ── ESKAPE taxids ─────────────────────────────────────────────────────────────
# E. cloacae complex: multiple taxids (same as species.yaml)
SPECIES_TAXIDS = {
    "abaumannii":  ["470"],
    "ecloaceae":   ["158836", "550", "61645", "208224", "299767", "1812935"],
    "efaecium":    ["1352"],
    "kpneumoniae": ["573"],
    "paeruginosa": ["287"],
    "saureus":     ["1280"],
}

N_CANDIDATES = 60   # over-sample; Stage 1b down-selects to 30 by ST novelty
NCBI_LIMIT   = 5000  # max records to query per taxid


def query_ncbi_complete(taxid: str) -> list[dict]:
    """Return list of assembly records (complete genomes) for one taxid."""
    cmd = [
        "datasets", "summary", "genome", "taxon", taxid,
        "--assembly-level", "complete",
        "--as-json-lines",
        "--limit", str(NCBI_LIMIT),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARN: datasets returned non-zero for taxid {taxid}: {result.stderr[:200]}",
              file=sys.stderr)
        return []

    records = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def parse_record(rec: dict) -> dict | None:
    """Extract the fields we care about from an assembly record."""
    accession = rec.get("accession", "").strip()
    if not accession.startswith("GCF_"):
        return None

    ai = rec.get("assembly_info", {})
    bs = ai.get("biosample", {})
    attrs = {a["name"].lower(): a.get("value", "") for a in bs.get("attributes", []) if "name" in a}

    country  = (bs.get("geo_loc_name") or attrs.get("geo_loc_name", "unknown")).strip()
    col_date = (bs.get("collection_date") or attrs.get("collection_date", "unknown")).strip()
    try:
        year = int(str(col_date)[:4])
    except (ValueError, TypeError):
        year = 0

    return {
        "accession": accession,
        "country":   country or "unknown",
        "year":      year,
    }


def select_diverse(candidates: list[dict], n: int) -> list[dict]:
    """
    Greedy diversity selection: maximise unique (country, year-decade) combinations.
    Picks `n` records, preferring novel (country, decade) pairs over duplicates.
    """
    selected = []
    seen_pairs: set[tuple] = set()

    # First pass: one per (country, decade)
    for rec in candidates:
        decade = (rec["year"] // 10) * 10
        pair   = (rec["country"][:3].upper(), decade)  # normalise country prefix
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected.append(rec)
        if len(selected) == n:
            return selected

    # Second pass: fill remaining slots from the rest
    selected_acc = {r["accession"] for r in selected}
    for rec in candidates:
        if rec["accession"] not in selected_acc:
            selected.append(rec)
            selected_acc.add(rec["accession"])
        if len(selected) == n:
            return selected

    return selected


def main() -> None:
    print("=" * 60)
    print("Stage 1 – Holdout accession selection")
    print("=" * 60)

    # Load training accession set
    fm = pd.read_parquet(PROC / "feature_matrix_3335.parquet")
    training_acc = set(fm.index.astype(str).str.strip())
    print(f"\nTraining set: {len(training_acc)} accessions loaded")

    holdout_candidates: dict[str, list[str]] = {}
    summary_rows = []

    for species, taxids in SPECIES_TAXIDS.items():
        print(f"\n── {species} (taxids: {taxids})")

        # Gather records from all taxids for this species
        all_records: list[dict] = []
        for tid in taxids:
            recs = query_ncbi_complete(tid)
            print(f"   taxid {tid}: {len(recs)} complete genome records from NCBI")
            all_records.extend(recs)

        # Parse and deduplicate by accession
        parsed: dict[str, dict] = {}
        for rec in all_records:
            p = parse_record(rec)
            if p and p["accession"] not in parsed:
                parsed[p["accession"]] = p

        # Hard exclusion: remove all training accessions
        novel = [p for acc, p in parsed.items() if acc not in training_acc]
        in_training = len(parsed) - len(novel)
        print(f"   NCBI total complete: {len(parsed)} unique accessions")
        print(f"   In training set (excluded): {in_training}")
        print(f"   Novel candidates: {len(novel)}")

        if len(novel) == 0:
            print(f"   ERROR: no novel complete genomes found for {species}")
            holdout_candidates[species] = []
            continue

        # Diversity selection
        selected = select_diverse(novel, N_CANDIDATES)
        holdout_candidates[species] = [r["accession"] for r in selected]

        print(f"   Selected {len(selected)} candidates "
              f"(target={N_CANDIDATES}; will down-select to 30 after MLST)")

        # Verify zero overlap with training
        overlap = training_acc & set(holdout_candidates[species])
        assert len(overlap) == 0, f"ACCESSION LEAK in {species}: {overlap}"
        print(f"   Overlap with training set: 0 (verified)")

        summary_rows.append({
            "species":     species,
            "ncbi_total":  len(parsed),
            "excluded":    in_training,
            "novel":       len(novel),
            "selected":    len(selected),
        })

    # Write output
    out_path = CONF / "holdout_candidates.yaml"
    with open(out_path, "w") as f:
        yaml.dump(holdout_candidates, f, default_flow_style=False)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\nSummary:")
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    print(f"\nTotal candidates: {sum(len(v) for v in holdout_candidates.values())}")
    print("\nNext: download genomes → run MLST → Stage 1b (ST-based down-selection)")


if __name__ == "__main__":
    main()
