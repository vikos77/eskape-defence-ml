#!/usr/bin/env python3
"""
c3_extended_holdout_367.py

Extended C3 holdout validation on the full-367 RF model (rf_q1_367.pkl).
canonical_name -> dp_{canonical} mapping in config/system_name_map.csv
covers every annotated system category (PADLOC cluster IDs, DefensePredictor
ML candidates, DefenseFinder uncharacterised groups, catch-alls), so passing
the 367-feature FEAT_COLS list works without changing how rows are built.
Binomial null p0 = 0.711 (full-367 AB per-class recall from run_q1_367.py).

Adds 10 IC2 complete genomes (S3) to the 33 S2 A. baumannii holdout for a
43-genome Ab recall test.

S3 sources:
  - 9 of 10 genomes: chromosome accessions found in S5/S6 (Excel) — same pipeline
    as S2 genomes in NB08
  - 1 genome: CP002522 (TCDC-AB0715, GCF_000189735_1) — annotation files on disk;
    GCA_000189735.2 in S3 sheet but pipeline ran as GCF_000189735.1

Output: prints recall, binomial test, and prediction breakdown for all 43 Ab genomes.
"""

import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest

ROOT    = Path(".")
INTERIM = ROOT / "data" / "interim"
PROC    = ROOT / "data" / "processed"
CONFIG  = ROOT / "config"
RES     = ROOT / "results"
SUPP    = ROOT  # Supplementary_Data_S1.xlsx is at project root

SP = "abaumannii"
S3_DISK_ACC = "GCF_000189735.1"   # TCDC-AB0715: in LEAKED_TRAIN as v1; disk has GCF_000189735_1

# ── Blast / filter constants (match NB08 / rebuild_feature_matrix) ──────────
BLAST_COLS   = ["qseqid","sseqid","pident","length","mismatch","gapopen",
                "qstart","qend","sstart","send","evalue","bitscore"]
PIDENT_MIN   = 40.0
COVERAGE_MIN = 0.80


def norm_acc(s: str) -> str:
    prefix, version = s.rsplit("_", 1)
    return f"{prefix}.{version}"


def parse_fasta_max_lengths(fasta_path: Path) -> dict:
    lengths: dict = defaultdict(list)
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


def build_row_from_excel(gid: str, df_subtypes_per_genome: dict,
                          padloc_systems_per_genome: dict,
                          df_sub_to_canonical: dict, padloc_to_canonical: dict,
                          feat_cols: list) -> dict:
    """Build a feature row from S5/S6 Excel data (same logic as NB08 Section 1.6)."""
    row = {f: 0 for f in feat_cols}
    df_subs       = df_subtypes_per_genome.get(gid, set())
    padloc_syss   = padloc_systems_per_genome.get(gid, set())
    df_canonicals = {df_sub_to_canonical[s] for s in df_subs if s in df_sub_to_canonical}
    pl_canonicals = {padloc_to_canonical[s] for s in padloc_syss if s in padloc_to_canonical}
    for canonical in (df_canonicals | pl_canonicals):
        col = f"dp_{canonical}"
        if col in row:
            row[col] = 1
    return row


def build_row_from_disk(acc: str, df_sub_to_canonical: dict,
                         padloc_to_canonical: dict, ice_lengths: dict,
                         feat_cols: list) -> dict:
    """Build a feature row from on-disk annotation files."""
    fn  = acc.replace(".", "_")
    row = {f: 0 for f in feat_cols}

    # DefenseFinder
    df_tsv = INTERIM / SP / "defensefinder" / fn / "defense_finder_systems.tsv"
    df_canonicals: set = set()
    if df_tsv.exists() and df_tsv.stat().st_size > 0:
        raw = pd.read_csv(df_tsv, sep="\t")
        if not raw.empty:
            for _, r in raw[raw["activity"] == "Defense"].iterrows():
                canon = df_sub_to_canonical.get(r["subtype"])
                if canon:
                    df_canonicals.add(canon)

    # PADLOC
    padloc_csv = INTERIM / SP / "padloc" / f"{fn}_padloc.csv"
    pl_canonicals: set = set()
    if padloc_csv.exists() and padloc_csv.stat().st_size > 0:
        raw_p = pd.read_csv(padloc_csv)
        if not raw_p.empty and "system" in raw_p.columns and "system.number" in raw_p.columns:
            instances = raw_p.drop_duplicates(subset=["system", "system.number"])[["system"]]
            for s in instances["system"]:
                canon = padloc_to_canonical.get(s)
                if canon:
                    pl_canonicals.add(canon)

    for canonical in (df_canonicals | pl_canonicals):
        col = f"dp_{canonical}"
        if col in row:
            row[col] = 1

    return row


def main() -> None:
    print("=" * 65)
    print("C3 EXTENDED — 43-genome Ab holdout (33 S2 + 10 S3 IC2)")
    print("=" * 65)

    # ── Load Excel and build S5/S6 lookups ───────────────────────────────────
    xl  = pd.ExcelFile(SUPP / "Supplementary_Data_S1.xlsx")

    # S5 — DefenseFinder
    s5_raw = xl.parse("S5_DefenseFinder_Complete", header=1)
    s5_raw.columns = ["sr_no","genome_id","sys_id","type","subtype","activity",
                       "sys_beg","sys_end","proteins","genes_count","profiles"]
    s5 = s5_raw[s5_raw["genome_id"].notna() & (s5_raw["genome_id"] != "Genome_ID")].copy()
    s5 = s5[s5["activity"].str.strip() == "Defense"].copy()
    df_subtypes_per_genome = (
        s5.groupby("genome_id")["subtype"]
        .apply(lambda x: set(x.astype(str).str.strip()))
        .to_dict()
    )

    # S6 — PADLOC
    s6_raw = xl.parse("S6_PADLOC_Complete", header=1)
    s6_raw.columns = ["sr_no","genome_id","system","target_name","hmm_accession",
                       "hmm_name","protein_name","full_seq_evalue","domain_iev",
                       "target_cov","hmm_cov","start","end","strand","target_desc",
                       "relative_pos","contig_end","all_domains","best_hits"]
    s6 = s6_raw[s6_raw["genome_id"].notna() & (s6_raw["genome_id"] != "Genome_ID")].copy()
    padloc_systems_per_genome = (
        s6.groupby("genome_id")["system"]
        .apply(lambda x: set(x.astype(str).str.strip()))
        .to_dict()
    )
    print(f"S5 genome_ids: {len(df_subtypes_per_genome)}")
    print(f"S6 genome_ids: {len(padloc_systems_per_genome)}")

    # ── System name map ───────────────────────────────────────────────────────
    snm = pd.read_csv(CONFIG / "system_name_map.csv")
    snm_valid = snm[snm["source"] != "exclude"].copy()
    df_sub_to_canonical = (
        snm_valid[snm_valid["df_subtype"].notna() & (snm_valid["df_subtype"] != "")]
        .set_index("df_subtype")["canonical_name"].to_dict()
    )
    padloc_to_canonical = (
        snm_valid[snm_valid["padloc_system"].notna() & (snm_valid["padloc_system"] != "")]
        .set_index("padloc_system")["canonical_name"].to_dict()
    )

    # ── FEAT_COLS (identical to training) ─────────────────────────────────────
    # Load directly from saved file to guarantee exact match with rf_q1_367.pkl
    FEAT_COLS = (RES / "q1_367_feat_cols.txt").read_text().strip().splitlines()
    print(f"FEAT_COLS: {len(FEAT_COLS)}")

    # ── Load trained RF model ──────────────────────────────────────────────────
    rf_q1 = joblib.load(RES / "models" / "rf_q1_367.pkl")
    print(f"Model classes: {rf_q1.classes_}\n")

    # ── Build S2 feature matrix (33 true Ab) ─────────────────────────────────
    s2_raw = xl.parse("S2_Complete_Genome_Metadata", header=2)
    s2_raw.columns = ["sr_no","refseq_assembly","accession","taxon","strain",
                       "size_bp","gc_content","assembly_level","year","source",
                       "location","biosample","refseq_gene","genbank_gene",
                       "refseq_protein","genbank_protein"]
    s2 = s2_raw[
        s2_raw["accession"].notna() & (s2_raw["accession"] != "Accession Number")
    ].copy()
    s2["taxon"] = s2["taxon"].astype(str).str.strip()
    s2 = s2.reset_index(drop=True)

    s2_ab = s2[s2["taxon"] == "Acinetobacter baumannii"].copy()
    print(f"S2 true Ab genomes: {len(s2_ab)}")

    s2_rows = []
    for gid in s2_ab["accession"].tolist():
        r = build_row_from_excel(gid, df_subtypes_per_genome, padloc_systems_per_genome,
                                  df_sub_to_canonical, padloc_to_canonical, FEAT_COLS)
        s2_rows.append(r)
    X_s2 = pd.DataFrame(s2_rows, index=s2_ab["accession"].tolist())[FEAT_COLS]
    print(f"S2 Ab feature matrix: {X_s2.shape}")

    # ── Build S3 feature matrix (10 IC2 complete genomes) ─────────────────────
    s3_raw = xl.parse("S3_IC2_Complete_Genomes", header=2)
    s3_raw.columns = ["sr_no","refseq_assembly","accession","taxon","strain",
                       "size_bp","gc_content","assembly_level","year","source",
                       "location","biosample","refseq_gene","genbank_gene",
                       "refseq_protein","genbank_protein"]
    s3 = s3_raw[
        s3_raw["accession"].notna() & (s3_raw["accession"] != "Accession Number")
    ].copy().reset_index(drop=True)
    print(f"\nS3 IC2 complete genomes: {len(s3)}")

    s3_ids   = s3["accession"].astype(str).str.strip().tolist()
    s3_rows  = []
    s3_index = []

    for gid in s3_ids:
        if gid in df_subtypes_per_genome or gid in padloc_systems_per_genome:
            # Found in S5 or S6 — use Excel source
            r = build_row_from_excel(gid, df_subtypes_per_genome, padloc_systems_per_genome,
                                      df_sub_to_canonical, padloc_to_canonical, FEAT_COLS)
            source = "Excel (S5/S6)"
        else:
            # Fallback: parse from disk (TCDC-AB0715 → GCF_000189735_1)
            r = build_row_from_disk(S3_DISK_ACC, df_sub_to_canonical,
                                     padloc_to_canonical, {}, FEAT_COLS)
            source = f"disk ({S3_DISK_ACC})"

        n_sys = sum(v for v in r.values())
        print(f"  {gid}: {n_sys} defence systems  [{source}]")
        s3_rows.append(r)
        s3_index.append(gid)

    X_s3 = pd.DataFrame(s3_rows, index=s3_index)[FEAT_COLS]
    print(f"S3 IC2 feature matrix: {X_s3.shape}")

    # ── Combined 43-genome Ab holdout ─────────────────────────────────────────
    X_combined = pd.concat([X_s2, X_s3])
    assert len(X_combined) == len(s2_ab) + len(s3), "Row count mismatch"
    print(f"\nCombined holdout (S2 Ab + S3 IC2): {len(X_combined)} genomes")

    # ── Predict ────────────────────────────────────────────────────────────────
    y_pred  = rf_q1.predict(X_combined.to_numpy(dtype=float))
    y_proba = rf_q1.predict_proba(X_combined.to_numpy(dtype=float))
    ab_idx  = list(rf_q1.classes_).index("abaumannii")
    prob_ab = y_proba[:, ab_idx]

    pred_df = pd.DataFrame({
        "genome_id":     X_combined.index,
        "cohort":        ["S2"] * len(X_s2) + ["S3_IC2"] * len(X_s3),
        "predicted":     y_pred,
        "prob_abaumannii": prob_ab,
    }).set_index("genome_id")

    # ── Results ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS — all 43 Ab genomes")
    print("=" * 65)

    n_correct = (pred_df["predicted"] == "abaumannii").sum()
    recall    = n_correct / len(pred_df)
    print(f"Correct (predicted abaumannii): {n_correct}/{len(pred_df)}")
    print(f"Recall: {recall:.3f}")
    print()
    print("Prediction distribution:")
    print(pred_df["predicted"].value_counts().to_string())

    # Breakdown by cohort
    print("\nBy cohort:")
    for cohort, grp in pred_df.groupby("cohort"):
        n_c  = (grp["predicted"] == "abaumannii").sum()
        rec  = n_c / len(grp)
        print(f"  {cohort}: {n_c}/{len(grp)}  (recall = {rec:.3f})")

    # Binomial test vs training CV recall (AB per-class recall = 0.711, full-367 RF grouped CV)
    binom = binomtest(int(n_correct), int(len(pred_df)), p=0.711, alternative="greater")
    print(f"\nBinomial test (p0=0.711, alternative='greater'):")
    print(f"  p-value = {binom.pvalue:.4f}  {'SIGNIFICANT' if binom.pvalue < 0.05 else 'n.s.'}")

    # Misclassified genomes
    missed = pred_df[pred_df["predicted"] != "abaumannii"]
    if len(missed) > 0:
        print(f"\nMisclassified genomes ({len(missed)}):")
        for gid, row in missed.iterrows():
            print(f"  {gid} ({row['cohort']}): predicted as {row['predicted']}"
                  f" (p_ab={row['prob_abaumannii']:.3f})")
    else:
        print("\nAll 43 Ab genomes correctly classified.")

    # Zero-defence check
    n_zero = (X_combined.sum(axis=1) == 0).sum()
    print(f"\nZero-defence genomes in holdout: {n_zero}")
    if n_zero > 0:
        zero_ids = X_combined.index[X_combined.sum(axis=1) == 0].tolist()
        for gid in zero_ids:
            print(f"  {gid}: predicted={pred_df.loc[gid,'predicted']} "
                  f"(class prior only, not feature-driven)")

    # SspBCDE check for IC2 proxy
    if "dp_SspBCDE" in FEAT_COLS:
        n_ssp = X_combined["dp_SspBCDE"].sum()
        n_ssp_s3 = X_s3["dp_SspBCDE"].sum() if "dp_SspBCDE" in X_s3.columns else "N/A"
        print(f"\nSspBCDE (IC2 proxy) in combined holdout: {n_ssp}/43")
        print(f"  S2 Ab:  {X_s2['dp_SspBCDE'].sum()}/33")
        print(f"  S3 IC2: {n_ssp_s3}/10  [expected: all/most IC2 carry SspBCDE]")


if __name__ == "__main__":
    main()
