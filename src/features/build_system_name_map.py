"""
Build config/system_name_map.csv  -  the canonical system name harmonisation table.

Pass 1: automated exact and case-insensitive matches between DefenseFinder subtypes
        and PADLOC system names.
Pass 2: manual curations for cross-tool naming discrepancies (CBASS, BREX, Cas,
        Retron, SspBCDE<->PT_SspABCD, Gao_Qat<->qatABCD, etc.) and explicit
        exclusions (DMS_other).

Output: config/system_name_map.csv with columns:
    canonical_name       -  name used in the feature matrix
    df_type              -  DefenseFinder 'type' field
    df_subtype           -  DefenseFinder 'subtype' field (empty if PADLOC-only)
    padloc_system        -  PADLOC 'system' field (empty if DF-only)
    source               -  'both' | 'df_only' | 'padloc_only' | 'df_antidefense' | 'exclude'
    df_genome_count      -  number of genomes DF calls this system
    padloc_genome_count  -  number of genomes PADLOC calls this system
    notes                -  curation rationale
"""

from collections import Counter
from pathlib import Path
import pandas as pd

# ── 1. Collect vocabulary and genome counts ───────────────────────────────────
df_def_cnt: Counter = Counter()
df_anti_cnt: Counter = Counter()
df_type_m: dict = {}

for tsv in Path("data/interim").glob("*/defensefinder/*/defense_finder_systems.tsv"):
    try:
        d = pd.read_csv(tsv, sep="\t")
        if d.empty:
            continue
        for _, row in d.iterrows():
            sub = str(row["subtype"])
            df_type_m[sub] = str(row["type"])
            if row["activity"] == "Antidefense":
                df_anti_cnt[sub] += 1
            else:
                df_def_cnt[sub] += 1
    except Exception:
        pass

padloc_cnt: Counter = Counter()
for csv in Path("data/interim").glob("*/padloc/*_padloc.csv"):
    try:
        d = pd.read_csv(csv)
        if d.empty:
            continue
        for s in d.drop_duplicates(subset=["system.number", "system"])["system"]:
            padloc_cnt[str(s)] += 1
    except Exception:
        pass

# ── 2. Automated pass ─────────────────────────────────────────────────────────
rows: list[dict] = []
df_set = set(df_def_cnt.keys())
padloc_set = set(padloc_cnt.keys())
used_df: set = set()
used_padloc: set = set()

# Exact name matches
exact = df_set & padloc_set
for s in sorted(exact):
    rows.append({
        "canonical_name": s,
        "df_type": df_type_m.get(s, ""),
        "df_subtype": s,
        "padloc_system": s,
        "source": "both",
        "df_genome_count": df_def_cnt[s],
        "padloc_genome_count": padloc_cnt[s],
        "notes": "exact_name_match",
    })
    used_df.add(s)
    used_padloc.add(s)

# Case-insensitive matches (not already matched)
remaining_df = {s: s.lower() for s in df_set - used_df}
remaining_padloc = {s: s.lower() for s in padloc_set - used_padloc}
df_by_lower = {v: k for k, v in remaining_df.items()}
padloc_by_lower = {v: k for k, v in remaining_padloc.items()}
for lower in sorted(set(df_by_lower) & set(padloc_by_lower)):
    df_name = df_by_lower[lower]
    p_name = padloc_by_lower[lower]
    rows.append({
        "canonical_name": df_name,  # DF name = canonical (matches published paper)
        "df_type": df_type_m.get(df_name, ""),
        "df_subtype": df_name,
        "padloc_system": p_name,
        "source": "both",
        "df_genome_count": df_def_cnt[df_name],
        "padloc_genome_count": padloc_cnt[p_name],
        "notes": "case_insensitive_match",
    })
    used_df.add(df_name)
    used_padloc.add(p_name)

# Remaining DF-only (automated placeholder)
for s in sorted(df_set - used_df):
    rows.append({
        "canonical_name": f"df_{s}",
        "df_type": df_type_m.get(s, ""),
        "df_subtype": s,
        "padloc_system": "",
        "source": "df_only",
        "df_genome_count": df_def_cnt[s],
        "padloc_genome_count": 0,
        "notes": "automated_df_only",
    })

# Remaining PADLOC-only (automated placeholder)
for s in sorted(padloc_set - used_padloc):
    rows.append({
        "canonical_name": f"padloc_{s}",
        "df_type": "",
        "df_subtype": "",
        "padloc_system": s,
        "source": "padloc_only",
        "df_genome_count": 0,
        "padloc_genome_count": padloc_cnt[s],
        "notes": "automated_padloc_only",
    })

# Anti-defence (DF only)
for s in sorted(df_anti_cnt.keys()):
    rows.append({
        "canonical_name": f"adf_{s}",
        "df_type": df_type_m.get(s, ""),
        "df_subtype": s,
        "padloc_system": "",
        "source": "df_antidefense",
        "df_genome_count": df_anti_cnt[s],
        "padloc_genome_count": 0,
        "notes": "antidefense_system",
    })

base = pd.DataFrame(rows)

# ── 3. Manual curations ───────────────────────────────────────────────────────
# Format: (canonical, df_subtype, padloc_system, source, notes)
manual: list[tuple] = [
    # KEY PUBLISHED-PAPER SYSTEMS
    ("SspBCDE",         "SspBCDE",                  "PT_SspABCD",        "both",        "SspBCDE(DF)=PT_SspABCD(PADLOC)  -  key FACILITATE system in published paper"),
    ("Gao_Qat",         "Gao_Qat",                  "qatABCD",           "both",        "Gao_Qat(DF)=qatABCD(PADLOC)  -  key FACILITATE system in published paper"),
    # CBASS
    ("CBASS_I",         "CBASS_I",                  "cbass_type_I",      "both",        "naming convention differs between tools"),
    ("CBASS_II",        "CBASS_II",                 "cbass_type_II",     "both",        "naming convention differs between tools"),
    ("CBASS_III",       "CBASS_III",                "cbass_type_III",    "both",        "naming convention differs between tools"),
    ("cbass_other",     "",                         "cbass_other",       "padloc_only", "PADLOC catch-all CBASS"),
    ("cbass_IIs",       "",                         "cbass_type_IIs",    "padloc_only", "PADLOC-only CBASS subtype"),
    # BREX
    ("BREX_I",          "BREX_I",                   "brex_type_I",       "both",        "naming convention differs between tools"),
    ("BREX_II",         "BREX_II",                  "brex_type_II",      "both",        "naming convention differs between tools"),
    ("BREX_III",        "BREX_III",                 "brex_type_III",     "both",        "naming convention differs between tools"),
    ("BREX_V",          "",                         "brex_type_V",       "padloc_only", "PADLOC-only BREX subtype"),
    ("BREX_unsubtyped", "BREX",                     "",                  "df_only",     "DF unsubtyped BREX call"),
    # CAS/CRISPR
    ("CAS_I-C",         "CAS_Class1-Subtype-I-C",   "cas_type_I-C",      "both",        "naming convention differs between tools"),
    ("CAS_I-E",         "CAS_Class1-Subtype-I-E",   "cas_type_I-E",      "both",        "naming convention differs between tools"),
    ("CAS_I-F",         "CAS_Class1-Subtype-I-F",   "cas_type_I-F1",     "both",        "DF uses I-F; PADLOC uses I-F1  -  same CRISPR-Cas subtype"),
    ("CAS_III-A",       "CAS_Class1-Subtype-III-A", "cas_type_III-A",    "both",        "naming convention differs between tools"),
    ("CAS_IV-A",        "CAS_Class1-Subtype-IV-A",  "cas_type_IV-A",     "both",        "naming convention differs between tools"),
    ("CAS_II-A",        "CAS_Class2-Subtype-II-A",  "cas_type_II-A",     "both",        "naming convention differs between tools"),
    ("CAS_IV-E",        "CAS_Class1-Subtype-IV-E",  "",                  "df_only",     "DF-only CAS subtype"),
    ("cas_type_other",  "",                         "cas_type_other",    "padloc_only", "PADLOC-only Cas catch-all"),
    # RETRON
    ("Retron_I-A",      "Retron_I_A",               "retron_I-A",        "both",        "naming convention differs between tools"),
    ("Retron_I-B",      "Retron_I_B",               "retron_I-B",        "both",        "naming convention differs between tools"),
    ("Retron_I-C",      "Retron_I_C",               "retron_I-C",        "both",        "naming convention differs between tools"),
    ("Retron_II-A",     "Retron_II",                "retron_II-A",       "both",        "naming convention differs between tools"),
    ("Retron_III-A",    "Retron_III",               "retron_III-A",      "both",        "naming convention differs between tools"),
    ("Retron_VIII",     "",                         "retron_VIII",       "padloc_only", "PADLOC-only retron subtype"),
    ("Retron_XII",      "",                         "retron_XII",        "padloc_only", "PADLOC-only retron subtype"),
    # THOERIS
    ("Thoeris_I",       "Thoeris_I",                "thoeris_type_I",    "both",        "naming convention differs between tools"),
    ("Thoeris_II",      "Thoeris_II",               "thoeris_type_II",   "both",        "naming convention differs between tools"),
    # DRUANTIA
    ("Druantia_I",      "Druantia_I",               "druantia_type_I",   "both",        "naming convention differs between tools"),
    ("Druantia_II",     "Druantia_II",              "druantia_type_II",  "both",        "naming convention differs between tools"),
    ("Druantia_III",    "Druantia_III",             "druantia_type_III", "both",        "naming convention differs between tools"),
    ("druantia_other",  "",                         "druantia_other",    "padloc_only", "PADLOC-only"),
    # GAO FAMILY (Gao et al. 2020; Gao_Qat handled above)
    ("Gao_Hhe",         "Gao_Hhe",                  "hhe",               "both",        "Gao_Hhe(DF)=hhe(PADLOC); Gao et al. 2020"),
    ("Gao_Mza",         "Gao_Mza",                  "mza",               "both",        "Gao_Mza(DF)=mza(PADLOC)"),
    ("Gao_Ppl",         "Gao_Ppl",                  "ppl",               "both",        "Gao_Ppl(DF)=ppl(PADLOC)"),
    ("Gao_Tmn",         "Gao_Tmn",                  "tmn",               "both",        "Gao_Tmn(DF)=tmn(PADLOC)"),
    ("Gao_Upx",         "Gao_Upx",                  "upx",               "both",        "Gao_Upx(DF)=upx(PADLOC)"),
    ("Gao_TerY",        "Gao_TerY",                 "TerY-P",            "both",        "Gao_TerY(DF)=TerY-P(PADLOC)"),
    ("Gao_Iet_df",      "Gao_Iet",                  "",                  "df_only",     "UNCERTAIN: Gao_Iet(DF=10) vs ietAS(PADLOC=89)  -  large count discrepancy, kept separate pending manual check"),
    ("padloc_ietAS",    "",                         "ietAS",             "padloc_only", "UNCERTAIN: ietAS(PADLOC=89) vs Gao_Iet(DF=10)  -  large count discrepancy, kept separate pending manual check"),
    ("mza_other",       "",                         "mza_other",         "padloc_only", "PADLOC-only mza variant"),
    ("TerY-P_other",    "",                         "TerY-P_other",      "padloc_only", "PADLOC-only TerY-P variant"),
    ("qatABCD_other",   "",                         "qatABCD_other",     "padloc_only", "PADLOC-only qat variant"),
    # SOFIC DISCREPANCY (DF=6 vs PADLOC=410  -  68x difference, kept separate)
    ("df_SoFic",        "SoFic",                    "",                  "df_only",     "DISCREPANCY: DF=6 vs PADLOC=410; detection threshold differs substantially; kept separate"),
    ("padloc_SoFic",    "",                         "SoFic",             "padloc_only", "DISCREPANCY: DF=6 vs PADLOC=410; detection threshold differs substantially; kept separate"),
    # PHOSPHOROTHIOATION
    ("Dnd_ABCDE",       "Dnd_ABCDE",                "PT_DndABCDE",       "both",        "naming convention differs between tools"),
    ("Dnd_ABCDEFGH",    "Dnd_ABCDEFGH",             "PT_DndFGH",         "both",        "naming convention differs between tools"),
    ("PT_SspFGH",       "",                         "PT_SspFGH",         "padloc_only", "PADLOC-only"),
    ("PT_SspE",         "",                         "PT_SspE",           "padloc_only", "PADLOC-only"),
    # PARIS
    ("PARIS_I",         "PARIS_I",                  "Paris",             "both",        "PARIS_I(DF)=Paris(PADLOC)"),
    ("PARIS_II",        "PARIS_II",                 "Paris_fused",       "both",        "PARIS_II(DF)=Paris_fused(PADLOC)"),
    ("PARIS_II_merge",  "PARIS_II_merge",           "",                  "df_only",     "DF-only PARIS variant"),
    # DISARM
    ("DISARM_I",        "DISARM_1",                 "disarm_type_I",     "both",        "DISARM_1(DF)=disarm_type_I(PADLOC)"),
    ("DISARM_II",       "",                         "disarm_type_II",    "padloc_only", "PADLOC-only DISARM subtype"),
    # ZORYA
    ("Zorya_I",         "Zorya_TypeI",              "zorya_type_I",      "both",        "naming convention differs between tools"),
    ("Zorya_II",        "Zorya_TypeII",             "zorya_type_II",     "both",        "naming convention differs between tools"),
    # ARGONAUTE
    ("pAgo_I",          "pAgo_S1B",                 "argonaute_type_I",  "both",        "naming convention differs between tools"),
    ("pAgo_III",        "pAgo_S2B",                 "argonaute_type_III","both",        "naming convention differs between tools"),
    ("argonaute_II",    "",                         "argonaute_type_II", "padloc_only", "PADLOC-only"),
    ("argonaute_other", "",                         "argonaute_other",   "padloc_only", "PADLOC-only catch-all"),
    # PYCSAR
    ("Pycsar",          "Pycsar",                   "pycsar_effector",   "both",        "naming convention differs between tools"),
    ("pycsar_other",    "",                         "pycsar_other",      "padloc_only", "PADLOC-only"),
    ("pycsar_unknown",  "",                         "pycsar_unknown",    "padloc_only", "PADLOC-only"),
    # WADJET
    ("Wadjet_I",        "Wadjet_I",                 "wadjet_type_I",     "both",        "naming convention differs between tools"),
    ("wadjet_other",    "",                         "wadjet_other",      "padloc_only", "PADLOC-only"),
    # SEPTU
    ("Septu",           "Septu",                    "septu_type_I",      "both",        "naming convention differs between tools"),
    ("septu_other",     "",                         "septu_other",       "padloc_only", "PADLOC-only"),
    # HACHIMAN
    ("Hachiman",        "Hachiman",                 "hachiman_type_I",   "both",        "naming convention differs between tools"),
    ("hachiman_other",  "",                         "hachiman_other",    "padloc_only", "PADLOC-only"),
    # DRT
    ("DRT_2",           "DRT_2",                    "DRT_class_I",       "both",        "DRT_2(DF)=DRT_class_I(PADLOC)"),
    ("DRT_3",           "DRT_3",                    "DRT_class_II",      "both",        "DRT_3(DF)=DRT_class_II(PADLOC)"),
    ("DRT_4",           "DRT_4",                    "DRT_class_III",     "both",        "DRT_4(DF)=DRT_class_III(PADLOC)"),
    ("DRT_type_III",    "",                         "DRT_type_III",      "padloc_only", "PADLOC-only DRT subtype"),
    ("DRT_other",       "",                         "DRT_other",         "padloc_only", "PADLOC-only DRT catch-all"),
    ("DRT8",            "DRT8",                     "",                  "df_only",     "DF-only DRT subtype"),
    ("DRT9",            "DRT9",                     "",                  "df_only",     "DF-only DRT subtype"),
    # GAO SYSTEMS (distinct from Gao family handled above)
    ("GAO_19",          "",                         "GAO_19",            "padloc_only", "PADLOC-only GAO system (Gao et al. 2020)"),
    ("GAO_20",          "",                         "GAO_20",            "padloc_only", "PADLOC-only GAO system"),
    ("GAO_29",          "",                         "GAO_29",            "padloc_only", "PADLOC-only GAO system"),
    # RST SYSTEMS
    ("Rst_3HP",         "Rst_3HP",                  "3HP",               "both",        "Rst_3HP(DF)=3HP(PADLOC)"),
    ("Rst_TIR-NLR",     "Rst_TIR-NLR",              "TIR-NLR",           "both",        "naming convention differs between tools"),
    ("Rst_DUF4238",     "Rst_DUF4238",              "DUF4238",           "both",        "naming convention differs between tools"),
    ("Rst_Helicase",    "Rst_HelicaseDUF2290",       "Helicase-DUF2290",  "both",        "naming convention differs between tools"),
    ("Rst_Hydrolase",   "Rst_Hydrolase-Tm",          "Hydrolase-TM",      "both",        "naming convention differs between tools"),
    # PIF
    ("Pif",             "Pif",                      "PifA",              "both",        "Pif(DF)=PifA(PADLOC)"),
    # LAMASSU
    ("Lamassu",         "Lamassu-Cap4_nuclease",     "Lamassu_Family",    "both",        "DF has subtypes; PADLOC uses family level; merged to family level"),
    # MOKOSH (DF Type_I_A/C; PADLOC TypeI)
    ("Mokosh_TypeI_A",  "Mokosh_Type_I_A",           "Mokosh_TypeI",      "both",        "Mokosh_Type_I_A(DF)=Mokosh_TypeI(PADLOC)"),
    # VSPR
    ("VSPR",            "",                         "VSPR",              "padloc_only", "PADLOC-only; no DF equivalent found"),
    # AVAST
    ("AVAST_II",        "",                         "AVAST_type_II",     "padloc_only", "PADLOC-only"),
    ("AVAST_III",       "",                         "AVAST_type_III",    "padloc_only", "PADLOC-only"),
    ("AVAST_IV",        "",                         "AVAST_type_IV",     "padloc_only", "PADLOC-only"),
    ("AVAST_V",         "",                         "AVAST_type_V",      "padloc_only", "PADLOC-only"),
    # MISC
    ("dXTPase",         "",                         "dXTPase",           "padloc_only", "PADLOC-only"),
    ("RADAR",           "",                         "RADAR",             "padloc_only", "PADLOC-only"),
    # EXCLUSIONS
    ("EXCLUDE_DMS_other", "",                       "DMS_other",         "exclude",     "Uninformative PADLOC catch-all; 886/900 genomes; excluded from feature matrix"),
]

# Apply manual curations
df = base.copy()
for canonical, df_sub, padloc_sys_name, source, notes in manual:
    mask_df = (df["df_subtype"] == df_sub) if df_sub else pd.Series([False] * len(df), index=df.index)
    mask_p  = (df["padloc_system"] == padloc_sys_name) if padloc_sys_name else pd.Series([False] * len(df), index=df.index)
    df = df.drop(index=df[mask_df | mask_p].index).reset_index(drop=True)

new_rows = [
    {
        "canonical_name":       canonical,
        "df_type":              df_type_m.get(df_sub, "") if df_sub else "",
        "df_subtype":           df_sub,
        "padloc_system":        padloc_sys_name,
        "source":               source,
        "df_genome_count":      df_def_cnt.get(df_sub, 0),
        "padloc_genome_count":  padloc_cnt.get(padloc_sys_name, 0),
        "notes":                notes,
    }
    for canonical, df_sub, padloc_sys_name, source, notes in manual
]

df_final = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Fix SEFIR: both tools have it  -  automated pass put it as padloc_only due to case issue
sefir_mask = df_final["canonical_name"].isin(["SEFIR", "padloc_SEFIR"])
df_final = df_final.drop(index=df_final[sefir_mask].index).reset_index(drop=True)
df_final = pd.concat([df_final, pd.DataFrame([{
    "canonical_name": "SEFIR",
    "df_type": df_type_m.get("SEFIR", ""),
    "df_subtype": "SEFIR",
    "padloc_system": "SEFIR",
    "source": "both",
    "df_genome_count": df_def_cnt.get("SEFIR", 0),
    "padloc_genome_count": padloc_cnt.get("SEFIR", 0),
    "notes": "exact_name_match",
}])], ignore_index=True)

df_final = df_final.sort_values(["source", "canonical_name"]).reset_index(drop=True)

# ── 4. Report and save ────────────────────────────────────────────────────────
print("Final table shape:", df_final.shape)
print(df_final["source"].value_counts().to_string())
print()

for s in ["SspBCDE", "Gao_Qat", "RM_Type_I", "RM_Type_II", "RM_Type_III", "RM_Type_IV", "Gabija"]:
    row = df_final[df_final["canonical_name"] == s]
    if not row.empty:
        r = row.iloc[0]
        print(f"{s}: df={r.df_genome_count}, padloc={r.padloc_genome_count}, source={r.source}")
    else:
        print(f"MISSING: {s}")

out = Path("config/system_name_map.csv")
df_final.to_csv(out, index=False)
print(f"\nWritten: {out} ({len(df_final)} rows)")


if __name__ == "__main__":
    pass
