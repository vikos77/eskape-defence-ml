#!/usr/bin/env python
"""
Stage 4 – Build holdout feature matrix from DF + PADLOC outputs.

Reads
-----
config/holdout_accessions.yaml                           – 30 per species
data/interim/holdout/{sp}/defensefinder/{acc_fn}/
    defense_finder_systems.tsv                           – DF outputs
data/interim/holdout/{sp}/padloc/{acc_fn}_padloc.csv    – PADLOC outputs
config/system_name_map.csv                               – canonical name mapping
data/processed/feature_matrix_3335.parquet               – to reconstruct FEAT_COLS

Writes
------
data/processed/holdout_feature_matrix.parquet            – (n_holdout × 359) matrix
                                                           index = GCF_* accession
                                                           columns = FEAT_COLS + species
results/holdout_feature_matrix_qc.json                   – per-genome missingness + coverage
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT    = Path(__file__).parent.parent.parent
CONF    = ROOT / "config"
PROC    = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim" / "holdout"
RES     = ROOT / "results"


def acc_to_fname(acc: str) -> str:
    return acc.replace(".", "_")


def load_canonical_maps(snm_path: Path) -> tuple[dict, dict]:
    """
    Load system_name_map.csv → two lookup dicts:
      df_sub_to_canonical   : df_subtype → canonical_name
      padloc_to_canonical   : padloc_system → canonical_name
    Excludes rows where source == 'exclude'.
    """
    snm = pd.read_csv(snm_path)
    valid = snm[snm["source"] != "exclude"].copy()

    df_map = (
        valid[valid["df_subtype"].notna() & (valid["df_subtype"] != "")]
        .set_index("df_subtype")["canonical_name"]
        .to_dict()
    )
    padloc_map = (
        valid[valid["padloc_system"].notna() & (valid["padloc_system"] != "")]
        .set_index("padloc_system")["canonical_name"]
        .to_dict()
    )
    return df_map, padloc_map


def parse_df_systems(tsv_path: Path, df_map: dict) -> set[str]:
    """
    Return canonical names detected by DefenseFinder for one genome.
    Handles both naming conventions:
      - defense_finder_systems.tsv  (training-pipeline style)
      - {acc_fn}_defense_finder_systems.tsv  (holdout-pipeline style)
    """
    # Prefer the prefixed file (non-zero content) if it exists
    acc_fn = tsv_path.parent.name  # e.g. GCF_000746645_1
    prefixed = tsv_path.parent / f"{acc_fn}_defense_finder_systems.tsv"
    if prefixed.exists() and prefixed.stat().st_size > 0:
        tsv_path = prefixed
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception:
        return set()

    # Keep only defence (exclude anti-defence rows if 'activity' col present)
    if "activity" in df.columns:
        df = df[df["activity"].str.strip().str.lower() == "defense"]

    col = "subtype" if "subtype" in df.columns else None
    if col is None:
        return set()

    canonicals = set()
    for sub in df[col].dropna().astype(str).str.strip():
        if sub in df_map:
            canonicals.add(df_map[sub])
    return canonicals


def parse_padloc_csv(csv_path: Path, padloc_map: dict) -> set[str]:
    """Return canonical names detected by PADLOC for one genome."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()

    col = "system" if "system" in df.columns else None
    if col is None:
        return set()

    canonicals = set()
    for sys_name in df[col].dropna().astype(str).str.strip():
        if sys_name in padloc_map:
            canonicals.add(padloc_map[sys_name])
    return canonicals


def reconstruct_feat_cols(fm: pd.DataFrame) -> list[str]:
    """Reproduce the FEAT_COLS computation from the training pipeline."""
    dp_cols    = sorted([c for c in fm.columns if c.startswith("dp_")])
    sp_prev    = fm.groupby("species")[dp_cols].mean()
    spec_score = sp_prev.std() / 0.5
    markers    = spec_score[spec_score >= 0.70].index.tolist()
    return [c for c in dp_cols if c not in markers]


def main() -> None:
    print("=" * 60)
    print("Stage 4 – Build holdout feature matrix")
    print("=" * 60)

    acc_path = CONF / "holdout_accessions.yaml"
    if not acc_path.exists():
        sys.exit("ERROR: config/holdout_accessions.yaml not found.")

    with open(acc_path) as f:
        holdout: dict[str, list[str]] = yaml.safe_load(f)

    # Load canonical name maps
    snm_path = CONF / "system_name_map.csv"
    df_map, padloc_map = load_canonical_maps(snm_path)
    print(f"Canonical maps: {len(df_map)} DF entries, {len(padloc_map)} PADLOC entries")

    # Reconstruct FEAT_COLS from training feature matrix
    fm = pd.read_parquet(PROC / "feature_matrix_3335.parquet")
    FEAT_COLS = reconstruct_feat_cols(fm)
    print(f"FEAT_COLS: {len(FEAT_COLS)} features")

    # Verify FEAT_COLS matches expected
    if len(FEAT_COLS) != 359:
        print(f"WARN: expected 359 FEAT_COLS, got {len(FEAT_COLS)}", file=sys.stderr)

    rows      = []
    row_index = []
    qc_records = []

    for species, accessions in holdout.items():
        sp_interim = INTERIM / species

        for acc in accessions:
            acc_fn = acc_to_fname(acc)

            # Parse DF
            df_tsv  = sp_interim / "defensefinder" / acc_fn / "defense_finder_systems.tsv"
            df_canonicals = parse_df_systems(df_tsv, df_map)

            # Parse PADLOC
            padloc_csv = sp_interim / "padloc" / f"{acc_fn}_padloc.csv"
            padloc_canonicals = parse_padloc_csv(padloc_csv, padloc_map)

            # Union canonical names
            all_canonicals = df_canonicals | padloc_canonicals

            # Build feature row
            row = {f: 0 for f in FEAT_COLS}
            matched = 0
            for canonical in all_canonicals:
                col = f"dp_{canonical}"
                if col in row:
                    row[col] = 1
                    matched += 1

            rows.append(row)
            row_index.append(acc)

            qc_records.append({
                "accession":     acc,
                "species":       species,
                "df_detected":   len(df_canonicals),
                "padloc_detected": len(padloc_canonicals),
                "union_detected":  len(all_canonicals),
                "feat_matched":  matched,
                "df_tsv_exists": df_tsv.exists(),
                "padloc_csv_exists": padloc_csv.exists(),
            })

    # Build DataFrame
    holdout_fm = pd.DataFrame(rows, index=row_index, columns=FEAT_COLS)

    # Attach species label
    species_map = {
        acc: sp
        for sp, accs in holdout.items()
        for acc in accs
    }
    holdout_fm["species"] = [species_map.get(idx, "unknown") for idx in holdout_fm.index]

    print(f"\nHoldout feature matrix: {holdout_fm.shape}")
    print(f"Species distribution:")
    print(holdout_fm["species"].value_counts().to_string())

    # QC: zero-defence genomes
    n_zero = (holdout_fm[FEAT_COLS].sum(axis=1) == 0).sum()
    print(f"\nGenomes with all-zero defence features: {n_zero}")
    if n_zero > 0:
        zero_acc = holdout_fm.index[holdout_fm[FEAT_COLS].sum(axis=1) == 0].tolist()
        for a in zero_acc:
            sp = species_map.get(a, "?")
            print(f"  {a} ({sp}) — DF and/or PADLOC output may be empty/failed")

    # Hard assertion: no training accessions in holdout
    training_acc = set(fm.index.astype(str))
    overlap = training_acc & set(holdout_fm.index)
    assert len(overlap) == 0, f"ACCESSION LEAK: {overlap}"
    print(f"\nAccession overlap with training: 0 (verified)")

    # Save
    out_path = PROC / "holdout_feature_matrix.parquet"
    holdout_fm.to_parquet(out_path)
    print(f"Saved: {out_path}")

    qc_path = RES / "holdout_feature_matrix_qc.json"
    with open(qc_path, "w") as f:
        json.dump(qc_records, f, indent=2)
    print(f"Saved: {qc_path}")

    print("\nNext: Stage 5 – replace NB08 Section 1 with holdout evaluation cells")


if __name__ == "__main__":
    main()
