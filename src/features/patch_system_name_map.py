"""
Patch config/system_name_map.csv to add systems discovered in the 4x dataset
expansion (3,335 genomes) that were absent from the original 878-genome vocabulary.

Changes:
  - UPDATE HEC-06: padloc_only → both (DF also detects it in 4x data)
  - UPDATE PD-T4-9: df_only → both (PADLOC also detects it in 4x data)
  - ADD 23 DF-only new systems
  - ADD 5 cross-tool systems (both tools, new in 4x)
  - ADD 10 anti-defence systems
  - ADD 17 PADLOC-only systems
"""

import pandas as pd
from pathlib import Path
from collections import Counter

INTERIM = Path("data/interim")

# ── Rescan for current genome counts ─────────────────────────────────────────
df_def_cnt: Counter = Counter()
df_anti_cnt: Counter = Counter()
df_type_map: dict = {}

for tsv in INTERIM.glob("*/defensefinder/*/defense_finder_systems.tsv"):
    try:
        d = pd.read_csv(tsv, sep="\t")
        if d.empty:
            continue
        for _, row in d.iterrows():
            sub, typ = str(row["subtype"]), str(row["type"])
            df_type_map[sub] = typ
            if row["activity"] == "Antidefense":
                df_anti_cnt[sub] += 1
            else:
                df_def_cnt[sub] += 1
    except Exception:
        pass

padloc_cnt: Counter = Counter()
for csv in INTERIM.glob("*/padloc/*_padloc.csv"):
    try:
        d = pd.read_csv(csv)
        if d.empty:
            continue
        for s in d.drop_duplicates(subset=["system.number", "system"])["system"]:
            padloc_cnt[str(s)] += 1
    except Exception:
        pass

# ── Load map ──────────────────────────────────────────────────────────────────
nm = pd.read_csv("config/system_name_map.csv")
old_len = len(nm)

# ── UPDATE existing rows ──────────────────────────────────────────────────────
mask = nm["canonical_name"] == "TEST_WRITE"   # undo test write if present
nm.loc[mask, "canonical_name"] = "padloc_HEC-06"

mask = nm["canonical_name"] == "padloc_HEC-06"
nm.loc[mask, "canonical_name"]      = "HEC-06"
nm.loc[mask, "df_subtype"]          = "HEC-06"
nm.loc[mask, "source"]              = "both"
nm.loc[mask, "df_genome_count"]     = df_def_cnt["HEC-06"]
nm.loc[mask, "padloc_genome_count"] = padloc_cnt["HEC-06"]
nm.loc[mask, "notes"]               = "exact name match; updated padloc_only→both after 4x expansion"

mask = nm["canonical_name"] == "df_PD-T4-9"
nm.loc[mask, "canonical_name"]      = "PD-T4-9"
nm.loc[mask, "padloc_system"]       = "PD-T4-9"
nm.loc[mask, "source"]              = "both"
nm.loc[mask, "df_genome_count"]     = df_def_cnt["PD-T4-9"]
nm.loc[mask, "padloc_genome_count"] = padloc_cnt["PD-T4-9"]
nm.loc[mask, "notes"]               = "exact name match; updated df_only→both after 4x expansion"

# ── Helper functions ──────────────────────────────────────────────────────────
def df_row(subtype: str, canonical: str) -> dict:
    return dict(
        canonical_name=canonical, df_type=df_type_map.get(subtype, ""),
        df_subtype=subtype, padloc_system="", source="df_only",
        df_genome_count=df_def_cnt[subtype], padloc_genome_count=0,
        notes="new system in 4x dataset",
    )

def padloc_row(system: str, canonical: str) -> dict:
    return dict(
        canonical_name=canonical, df_type="", df_subtype="", padloc_system=system,
        source="padloc_only", df_genome_count=0,
        padloc_genome_count=padloc_cnt[system],
        notes="new system in 4x dataset",
    )

def both_row(df_sub: str, padloc_sys: str, canonical: str, notes: str) -> dict:
    return dict(
        canonical_name=canonical, df_type=df_type_map.get(df_sub, ""),
        df_subtype=df_sub, padloc_system=padloc_sys, source="both",
        df_genome_count=df_def_cnt[df_sub],
        padloc_genome_count=padloc_cnt[padloc_sys],
        notes=notes,
    )

def anti_row(subtype: str) -> dict:
    return dict(
        canonical_name=f"adf_{subtype}", df_type=df_type_map.get(subtype, ""),
        df_subtype=subtype, padloc_system="", source="df_antidefense",
        df_genome_count=df_anti_cnt[subtype], padloc_genome_count=0,
        notes="new anti-defence system in 4x dataset",
    )

# ── NEW ROWS ──────────────────────────────────────────────────────────────────
new_rows = []

# DF-only (23 systems)
for sub in [
    "DS-8", "Gao_Ape", "gcuWGS21", "DS-11", "VP1796", "DS-27", "UG5_small",
    "CAS_Class1-Type-I", "DS-43", "DS-44", "Retron_XII", "Retron_VII_2",
    "Gao_Her_SIR", "VP1826", "gcu167", "pAgo_S1A", "AbiO", "gcu142",
    "UG8", "DRT6", "DS-41", "UG29", "Viperin",
]:
    new_rows.append(df_row(sub, f"df_{sub}"))

# Both tools — exact same name
new_rows.append(both_row("AbiV", "AbiV", "AbiV",
    "exact name match; both tools detect; new in 4x dataset"))
new_rows.append(both_row("HEC-03", "HEC-03", "HEC-03",
    "exact name match; both tools detect; new in 4x dataset"))
new_rows.append(both_row("PD-T4-10", "PD-T4-10", "PD-T4-10",
    "exact name match; both tools detect; new in 4x dataset"))

# Both tools — naming convention differs
new_rows.append(both_row("Wadjet_II", "wadjet_type_II", "Wadjet_II",
    "Wadjet_II(DF)=wadjet_type_II(PADLOC); naming convention differs; new in 4x dataset"))
new_rows.append(both_row("CAS_Class2-Subtype-II-C", "cas_type_II-C", "CAS_II-C",
    "CAS_Class2-Subtype-II-C(DF)=cas_type_II-C(PADLOC); naming convention differs; new in 4x dataset"))

# Anti-defence (10 systems)
for sub in ["acrif1", "dar_ddr_hdf_ulx", "gnarl1", "acric3", "acric4",
            "acrie6", "acrie7", "gad1", "ddra", "hdf"]:
    new_rows.append(anti_row(sub))

# PADLOC-only (17 systems)
for sys in [
    "PDC-M02", "argonaute_solo", "PDC-M29", "PDC-M35", "PDC-M54",
    "PDC-M36", "PDC-M71", "retron_V", "PT_PbeABCD", "thoeris_other",
    "retron_IX", "PDC-S63", "PDC-M12", "PDC-M69", "PDC-S44",
    "PDC-M61", "PDC-S68",
]:
    new_rows.append(padloc_row(sys, f"padloc_{sys}"))

# ── Append and save ───────────────────────────────────────────────────────────
new_df = pd.DataFrame(new_rows, columns=nm.columns)
nm_updated = pd.concat([nm, new_df], ignore_index=True)
nm_updated.to_csv("config/system_name_map.csv", index=False)

print(f"Rows: {old_len} → {len(nm_updated)} (+{len(nm_updated) - old_len})")
print("Row updates: HEC-06 padloc_only→both; PD-T4-9 df_only→both")
print(f"New rows added: {len(new_rows)}")
print("  DF-only: 23  |  Both: 5  |  Anti-defence: 10  |  PADLOC-only: 17")
