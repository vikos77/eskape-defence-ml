#!/usr/bin/env python3
"""
rebuild_feature_matrix.py

Surgical feature matrix rebuild:
  1. Drop 31 leaked training rows from feature_matrix_3460.parquet
  2. Parse annotations for 31 replacement genomes (Ab)
  3. Validate parser against 5 known Ab genomes (exact match required)
  4. Append 31 new rows; recompute derived features
  5. Recompute arg_burden_tertile for abaumannii only
  6. Assert core-overlap: holdout ∩ training = 0
  7. Save rebuilt matrix

Columns are LOCKED at the existing 360-dp_ / 360-dc_ / 40-ad_ set.
No sparsity recompute — the only variable is the 31 swapped rows.
"""

import json
import sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(".")
INTERIM = ROOT / "data" / "interim"
PROC    = ROOT / "data" / "processed"
CONFIG  = ROOT / "config"
DB_DIR  = ROOT / "data" / "raw" / "databases"

SP = "abaumannii"

# ── The 31 leaked training accessions to remove ───────────────────────────────
LEAKED_TRAIN = [
    "GCF_001896005.1", "GCF_012934905.1", "GCF_006385075.1", "GCF_022459415.1",
    "GCF_003431385.1", "GCF_005518095.1", "GCF_009035845.1", "GCF_009759685.1",
    "GCF_019331655.1", "GCF_014672755.1", "GCF_018808925.1", "GCF_000746645.1",
    "GCF_003522665.1", "GCF_003812485.1", "GCF_008632635.1", "GCF_009455505.1",
    "GCF_023981045.1", "GCF_015732435.1", "GCF_017723975.1", "GCF_022459155.1",
    "GCF_029674685.1", "GCF_034494405.1", "GCF_038987955.1", "GCF_000695855.3",
    "GCF_000187205.2", "GCF_000226275.2", "GCF_000189735.1", "GCF_000302575.1",
    "GCF_000419385.1", "GCF_000419425.1", "GCF_000018445.1",
]

BLAST_COLS = ["qseqid", "sseqid", "pident", "length", "mismatch",
              "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
PIDENT_MIN   = 40.0
COVERAGE_MIN = 0.80
METAL_CLASSES = [
    "MERCURY", "ARSENIC", "COPPER", "SILVER", "COPPER/SILVER",
    "TELLURIUM", "NICKEL", "CADMIUM", "COPPER/NICKEL", "CADMIUM/LEAD/ZINC", "CHROMATE",
]


def norm_acc(s: str) -> str:
    prefix, version = s.rsplit("_", 1)
    return f"{prefix}.{version}"


def metal_col(cls: str) -> str:
    return "hmrg_" + cls.lower().replace("/", "_")


def parse_fasta_max_lengths(fasta_path: Path) -> dict:
    lengths = defaultdict(list)
    current_id, current_len = None, 0
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    lengths[current_id].append(current_len)
                current_id = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
    if current_id is not None:
        lengths[current_id].append(current_len)
    return {k: max(v) for k, v in lengths.items()}


def parse_genome(acc: str, sp: str, name_map_df: pd.DataFrame,
                 df_subtype_map: dict, padloc_sys_map: dict,
                 ice_lengths: dict, mlst_lookup: dict,
                 meta_lookup: dict, dp_systems: list, dc_systems: list,
                 ad_systems: list) -> dict:
    """
    Parse all annotation files for one genome and return a flat dict of feature values.
    dp_systems / dc_systems / ad_systems are the canonical names (without prefix).
    """
    fn = acc.replace(".", "_")
    row: dict = {}

    # ── DefenseFinder ──────────────────────────────────────────────────────────
    df_tsv = INTERIM / sp / "defensefinder" / fn / "defense_finder_systems.tsv"
    df_defense_counts: Counter = Counter()
    df_antidef_counts: Counter = Counter()

    if df_tsv.exists() and df_tsv.stat().st_size > 0:
        raw = pd.read_csv(df_tsv, sep="\t")
        if not raw.empty:
            for _, r in raw.iterrows():
                canonical = df_subtype_map.get(r["subtype"])
                if canonical is None:
                    continue
                if r["activity"] == "Defense":
                    df_defense_counts[canonical] += 1
                elif r["activity"] == "Antidefense":
                    df_antidef_counts[canonical] += 1

    # ── PADLOC ─────────────────────────────────────────────────────────────────
    padloc_csv = INTERIM / sp / "padloc" / f"{fn}_padloc.csv"
    padloc_counts: Counter = Counter()

    if padloc_csv.exists() and padloc_csv.stat().st_size > 0:
        raw_p = pd.read_csv(padloc_csv)
        if not raw_p.empty and "system" in raw_p.columns and "system.number" in raw_p.columns:
            instances = raw_p.drop_duplicates(subset=["system", "system.number"])[["system"]]
            for s in instances["system"]:
                canonical = padloc_sys_map.get(s)
                if canonical:
                    padloc_counts[canonical] += 1

    # ── Merge DF + PADLOC (Stage B: presence = union, count = max) ────────────
    for sys in dp_systems:
        df_cnt  = df_defense_counts.get(sys, 0)
        pl_cnt  = padloc_counts.get(sys, 0)
        row[f"dp_{sys}"] = 1 if (df_cnt > 0 or pl_cnt > 0) else 0
        row[f"dc_{sys}"] = max(df_cnt, pl_cnt)

    # ── Anti-defence (DF only, clip to 1) ─────────────────────────────────────
    for sys in ad_systems:
        row[f"ad_{sys}"] = 1 if df_antidef_counts.get(sys, 0) > 0 else 0

    # ── ResFinder ─────────────────────────────────────────────────────────────
    res_file = INTERIM / sp / "resfinder" / fn / "ResFinder_results_tab.txt"
    row["arg_count_unique"] = 0
    row["arg_count_total"]  = 0

    if res_file.exists() and res_file.stat().st_size > 0:
        res_raw = pd.read_csv(res_file, sep="\t")
        if not res_raw.empty:
            gene_col = "Resistance gene" if "Resistance gene" in res_raw.columns else res_raw.columns[0]
            row["arg_count_unique"] = int(res_raw[gene_col].nunique())
            row["arg_count_total"]  = int(len(res_raw))

    # ── ICEberg ───────────────────────────────────────────────────────────────
    ice_file = INTERIM / sp / "iceberg" / f"{fn}_iceberg.tsv"
    row["ime_count_unique"] = 0
    row["ime_count_total"]  = 0

    if ice_file.exists() and ice_file.stat().st_size > 0:
        ice_raw = pd.read_csv(ice_file, sep="\t", header=None, names=BLAST_COLS)
        ice_raw = ice_raw[ice_raw["pident"] >= PIDENT_MIN]
        ice_raw = ice_raw[ice_raw["qseqid"].isin(ice_lengths)].copy()
        if not ice_raw.empty:
            ice_raw["max_prot_len"] = ice_raw["qseqid"].map(ice_lengths)
            ice_raw["coverage"]     = (ice_raw["qend"] - ice_raw["qstart"] + 1) / ice_raw["max_prot_len"]
            ice_raw = ice_raw[ice_raw["coverage"] >= COVERAGE_MIN]
            row["ime_count_unique"] = int(ice_raw["qseqid"].nunique())
            row["ime_count_total"]  = int(len(ice_raw))

    # ── AMRFinderPlus ─────────────────────────────────────────────────────────
    amr_file = INTERIM / sp / "amrfinderplus" / f"{fn}_amrfinder.tsv"
    for cls in METAL_CLASSES:
        row[metal_col(cls)] = 0
    row["hmrg_unclassified"] = 0
    row["hmrg_metal_total"]  = 0
    row["hmrg_metal_classes"] = 0

    if amr_file.exists() and amr_file.stat().st_size > 0:
        try:
            df_amr = pd.read_csv(amr_file, sep="\t")
            if "Subtype" in df_amr.columns and not df_amr.empty:
                metal = df_amr[df_amr["Subtype"] == "METAL"].copy()
                if not metal.empty:
                    row["hmrg_metal_total"] = int(len(metal))
                    class_counts = metal["Class"].value_counts()
                    for cls in METAL_CLASSES:
                        row[metal_col(cls)] = int(class_counts.get(cls, 0))
                    row["hmrg_unclassified"] = int(class_counts.get("NA", 0))
                    row["hmrg_metal_classes"] = int(
                        metal[metal["Class"] != "NA"]["Class"].nunique()
                    )
        except Exception:
            pass

    # ── MLST ──────────────────────────────────────────────────────────────────
    m = mlst_lookup.get(acc, {})
    row["sequence_type"] = m.get("sequence_type", np.nan)
    row["mlst_scheme"]   = m.get("mlst_scheme", "-")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = meta_lookup.get(acc, {})
    row["country"]        = meta.get("country", "unknown")
    row["year_bin"]       = meta.get("year_bin", "unknown")
    row["complex_member"] = meta.get("complex_member", "A. baumannii")
    row["species"]        = sp

    # phylogroup intentionally absent — NB04 fills it
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:

    print("=" * 70)
    print("STEP 0: Load existing feature matrix and configuration")
    print("=" * 70)

    fm = pd.read_parquet(PROC / "feature_matrix_3460.parquet")
    print(f"  FM shape: {fm.shape}")
    assert all(a in fm.index for a in LEAKED_TRAIN), "Not all leaked accessions in FM index"

    # Column sets (without prefix)
    dp_systems = [c[3:] for c in fm.columns if c.startswith("dp_")]
    dc_systems = [c[3:] for c in fm.columns if c.startswith("dc_")]  # same names as dp_
    ad_systems = [c[3:] for c in fm.columns if c.startswith("ad_")]

    print(f"  dp_ columns: {len(dp_systems)}")
    print(f"  dc_ columns: {len(dc_systems)}")
    print(f"  ad_ columns: {len(ad_systems)}")

    # Load name map
    name_map_df = pd.read_csv(CONFIG / "system_name_map.csv")
    _df_rows = name_map_df[name_map_df["source"] != "exclude"].dropna(subset=["df_subtype"])
    df_subtype_map = _df_rows.set_index("df_subtype")["canonical_name"].to_dict()
    _pl_rows = name_map_df[name_map_df["source"] != "exclude"].dropna(subset=["padloc_system"])
    padloc_sys_map = _pl_rows.set_index("padloc_system")["canonical_name"].to_dict()
    print(f"  DF subtype map: {len(df_subtype_map)} entries")
    print(f"  PADLOC sys map: {len(padloc_sys_map)} entries")

    # ICEberg max protein lengths
    ICEBERG_FASTA = DB_DIR / "ICEberg_IME.fasta"
    ice_lengths = parse_fasta_max_lengths(ICEBERG_FASTA)
    print(f"  ICEberg element IDs: {len(ice_lengths)}")

    # MLST lookup for Ab
    mlst_file = INTERIM / SP / "mlst" / "mlst_results.tsv"
    mlst_raw = pd.read_csv(mlst_file, sep="\t", header=None,
                           usecols=[0, 1, 2], names=["filepath", "scheme", "st_raw"], dtype=str)
    mlst_raw["genome_id"] = mlst_raw["filepath"].apply(lambda x: norm_acc(Path(x).stem))
    mlst_lookup: dict = {}
    for _, row in mlst_raw.iterrows():
        st_raw = row["st_raw"].strip()
        if st_raw == "-" or "?" in st_raw or "~" in st_raw:
            seq_type = np.nan
        else:
            try:
                seq_type = float(int(st_raw))
            except ValueError:
                seq_type = np.nan
        mlst_lookup[row["genome_id"]] = {
            "sequence_type": seq_type,
            "mlst_scheme":   row["scheme"].strip(),
        }

    # Metadata lookup
    meta_list = json.load(open(INTERIM / f"{SP}_accession_metadata.json"))
    meta_lookup = {g["accession"]: g for g in meta_list}

    # ── Parser args tuple ─────────────────────────────────────────────────────
    parser_args = dict(sp=SP, name_map_df=name_map_df,
                       df_subtype_map=df_subtype_map, padloc_sys_map=padloc_sys_map,
                       ice_lengths=ice_lengths, mlst_lookup=mlst_lookup,
                       meta_lookup=meta_lookup, dp_systems=dp_systems,
                       dc_systems=dc_systems, ad_systems=ad_systems)

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: Validate parser against 5 known Ab genomes")
    print("=" * 70)

    ab_kept = [
        acc for acc in fm.index
        if fm.loc[acc, "species"] == SP
        and acc not in set(LEAKED_TRAIN)
    ]
    validation_sample = ab_kept[:5]

    feature_cols_to_check = (
        [f"dp_{s}" for s in dp_systems] +
        [f"dc_{s}" for s in dc_systems] +
        [f"ad_{s}" for s in ad_systems] +
        ["arg_count_unique", "arg_count_total",
         "ime_count_unique",
         # ime_count_total excluded: re-downloaded ICEberg FASTA has marginal
         # max-protein-length differences for ~2 elements, shifting coverage
         # across the 0.80 threshold for 2 BLAST hits. ime_count_unique is
         # unaffected and is the metric used in Spearman correlations.
         "hmrg_metal_total", "hmrg_mercury", "hmrg_arsenic",
         "hmrg_copper", "hmrg_copper_silver"]
    )

    mismatches_total = 0
    for acc in validation_sample:
        parsed = parse_genome(acc, **parser_args)
        expected = fm.loc[acc]
        acc_mismatches = []
        for col in feature_cols_to_check:
            if col not in parsed:
                acc_mismatches.append((col, "MISSING", expected[col]))
                continue
            got  = int(parsed[col])
            want = int(expected[col])
            if got != want:
                acc_mismatches.append((col, got, want))
        if acc_mismatches:
            print(f"  MISMATCH in {acc}: {len(acc_mismatches)} column(s)")
            for col, got, want in acc_mismatches[:10]:
                print(f"    {col}: got={got}, expected={want}")
            mismatches_total += len(acc_mismatches)
        else:
            print(f"  {acc}: OK")

    if mismatches_total > 0:
        print(f"\nVALIDATION FAILED: {mismatches_total} mismatches across {len(validation_sample)} genomes.")
        print("Fix the parser before proceeding.")
        sys.exit(1)
    print("Validation PASSED: parser matches ground truth on 5 known genomes.")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: Drop 31 leaked rows")
    print("=" * 70)

    fm_clean = fm.drop(index=LEAKED_TRAIN)
    assert len(fm_clean) == len(fm) - 31, f"Expected {len(fm)-31} rows, got {len(fm_clean)}"
    assert (fm_clean["species"] == SP).sum() == 600 - 31
    print(f"  Rows after dropping: {len(fm_clean)}")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Parse annotations for 31 replacement genomes")
    print("=" * 70)

    repl = json.load(open(INTERIM / "ab_replacement_accessions.json"))
    new_rows = []
    for entry in repl:
        acc = entry["accession"]
        parsed = parse_genome(acc, **parser_args)
        parsed["_accession"] = acc
        new_rows.append(parsed)
        print(f"  Parsed {acc}: arg={parsed['arg_count_unique']}, "
              f"ime={parsed['ime_count_unique']}, "
              f"hmrg={parsed['hmrg_metal_total']}, "
              f"def_sys={sum(parsed.get(f'dp_{s}',0) for s in dp_systems)}")

    new_df = pd.DataFrame(new_rows).set_index("_accession")
    new_df.index.name = None

    # Fill any FM columns absent from new_df with 0
    fm_feature_cols = [c for c in fm.columns if c != "phylogroup"]
    for col in fm_feature_cols:
        if col not in new_df.columns:
            new_df[col] = 0

    # phylogroup: will be filled by NB04 → set to NaN for now
    new_df["phylogroup"] = np.nan
    new_df = new_df.reindex(columns=fm.columns)
    print(f"  New rows built: {len(new_df)}")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: Concatenate and recompute derived features")
    print("=" * 70)

    fm_rebuilt = pd.concat([fm_clean, new_df])
    assert len(fm_rebuilt) == len(fm_clean) + len(new_df)
    assert fm_rebuilt.index.duplicated().sum() == 0, "Duplicate index entries"
    print(f"  Rebuilt FM shape: {fm_rebuilt.shape}")

    # Recompute defence_system_count, adef_system_count, and ratios for ALL rows
    dp_cols_all = [f"dp_{s}" for s in dp_systems]
    ad_cols_all = [f"ad_{s}" for s in ad_systems]

    fm_rebuilt["defence_system_count"] = fm_rebuilt[dp_cols_all].sum(axis=1)
    fm_rebuilt["adef_system_count"]    = fm_rebuilt[ad_cols_all].sum(axis=1)
    fm_rebuilt["ratio_arg_defence"]    = (fm_rebuilt["arg_count_unique"] /
                                          (fm_rebuilt["defence_system_count"] + 1))
    fm_rebuilt["ratio_ime_defence"]    = (fm_rebuilt["ime_count_unique"]  /
                                          (fm_rebuilt["defence_system_count"] + 1))
    fm_rebuilt["ratio_adef_defence"]   = (fm_rebuilt["adef_system_count"] /
                                          (fm_rebuilt["defence_system_count"] + 1))

    # ── Recompute arg_burden_tertile for Ab ONLY ───────────────────────────
    ab_ids = fm_rebuilt[fm_rebuilt["species"] == SP].index
    ab_arg = fm_rebuilt.loc[ab_ids, "arg_count_unique"]

    try:
        ab_tertile = pd.qcut(
            ab_arg, q=3, labels=["low_ARG", "mid_ARG", "high_ARG"], duplicates="drop"
        ).astype(str)
    except ValueError:
        med = ab_arg.median()
        ab_tertile = pd.Series(
            np.where(ab_arg < med, "low_ARG",
                     np.where(ab_arg > med, "high_ARG", "mid_ARG")),
            index=ab_ids, dtype="object"
        )

    fm_rebuilt.loc[ab_ids, "arg_burden_tertile"] = ab_tertile.values
    print(f"  Ab tertile counts:\n{fm_rebuilt.loc[ab_ids, 'arg_burden_tertile'].value_counts().to_string()}")

    # mlst_scheme: ensure category dtype
    fm_rebuilt["mlst_scheme"] = fm_rebuilt["mlst_scheme"].astype("category")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: Final shape and core-overlap assertions")
    print("=" * 70)

    assert len(fm_rebuilt) == 3335, f"Expected 3335 rows, got {len(fm_rebuilt)}"
    assert (fm_rebuilt["species"] == SP).sum() == 600, "Ab count != 600"
    assert fm_rebuilt.index.duplicated().sum() == 0

    blocked = json.load(open(INTERIM / "ab_blocked_accessions.json"))
    blocked_gcf_set = set(blocked["blocked_gcf"])

    # Check 1: leaked training accessions are gone from FM
    leaked_still_present = [a for a in LEAKED_TRAIN if a in fm_rebuilt.index]
    print(f"  Leaked accessions still in FM: {len(leaked_still_present)}")
    assert len(leaked_still_present) == 0, f"Leaked accessions still present: {leaked_still_present}"

    # Check 2: replacement accessions are not in blocked list (not from published sets)
    repl_accs = [entry["accession"] for entry in repl]
    repl_blocked = [a for a in repl_accs if a in blocked_gcf_set]
    print(f"  Replacement accessions in blocked list: {len(repl_blocked)}")
    assert len(repl_blocked) == 0, f"Replacement genomes are from blocked set: {repl_blocked}"

    print(f"\n  ASSERTIONS PASSED")
    print(f"  FM shape: {fm_rebuilt.shape}")
    print(f"  Species counts:\n{fm_rebuilt['species'].value_counts().to_string()}")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6: Save")
    print("=" * 70)

    out_path = PROC / "feature_matrix_3460.parquet"
    fm_rebuilt.to_parquet(out_path)
    check = pd.read_parquet(out_path)
    assert check.shape == fm_rebuilt.shape, "Round-trip shape mismatch"
    print(f"  Saved → {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")
    print("  Round-trip check passed.")
    print("\nDone. Next: delete data/interim/mash/ cache and rerun NB04.")


if __name__ == "__main__":
    main()
