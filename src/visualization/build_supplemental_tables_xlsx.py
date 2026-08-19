"""
Build Supplemental_Tables.xlsx for journal submission.

One sheet per supplementary table (S1–S16). Each sheet has:
  Row 1  — table title (bold, merged across all columns)
  Row 2  — legend text (italic, merged across all columns)
  Row 3  — column headers (bold, light grey fill)
  Row 4+ — data

Run from project root:
    conda run -n eskape-ml python src/visualization/build_supplemental_tables_xlsx.py
"""

from pathlib import Path
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

TABLES_DIR = Path("results/tables")
OUT_FILE   = Path("results/Supplemental_Tables.xlsx")

GREY_FILL  = PatternFill("solid", fgColor="D9D9D9")
BLUE_FILL  = PatternFill("solid", fgColor="DCE6F1")

# ── Table metadata ─────────────────────────────────────────────────────────────
# Each entry: (csv_filename, sheet_name, title, legend)
TABLES = [
    (
        "S1_feature_matrix_layout.csv",
        "Table S1",
        "Table S1. Feature matrix layout: defence system feature dimensions and "
        "genome counts per species.",
        "Binary defence system features were derived from the union of DefenseFinder "
        "v2.0.1 and PADLOC v2.0.0 predictions across 3,335 genomes. Eight near-exclusive "
        "species markers (spec_score ≥ 0.70) were removed before supervised modelling, "
        "leaving 359 FEAT_COLS used in Q1–Q4.",
    ),
    (
        "S2_q1_classifier_comparison.csv",
        "Table S2",
        "Table S2. Q1 classifier comparison: balanced accuracy, macro-F1, and 95% CI "
        "for all four classifiers under phylogenetically grouped five-fold cross-validation.",
        "All classifiers trained on 3,335 genomes, 359 features, 309 phylogroups "
        "(GroupedStratifiedKFold, 5 folds). Balanced accuracy (BA) and macro-F1 are "
        "fold-averaged. 95% CIs are from phylogroup cluster bootstrap (2,000 iterations). "
        "Macro-F1 and its CI were computed for the primary Random Forest model only. "
        "This table is the source for main text Table 1.",
    ),
    (
        "S3_q1_mcnemar_tests.csv",
        "Table S3",
        "Table S3. Q1 McNemar pairwise tests: between-model significance on per-genome "
        "out-of-fold predictions.",
        "McNemar's test comparing each classifier's per-genome out-of-fold predictions "
        "against Random Forest (n=3,335 genomes). b = genomes correct for RF but wrong "
        "for comparator; c = genomes correct for comparator but wrong for RF. No comparison "
        "reached significance at α = 0.05.",
    ),
    (
        "S4_q1_fold_balanced_accuracies.csv",
        "Table S4",
        "Table S4. Q1 fold-level balanced accuracies for all four classifiers under "
        "phylogenetically grouped five-fold cross-validation.",
        "Each row is one cross-validation fold. Mean row is the fold average, which "
        "equals the mean BA reported in main text Table 1. Fold assignments use "
        "GroupedStratifiedKFold (random_state=42).",
    ),
    (
        "S5_q1_filter_sensitivity.csv",
        "Table S5",
        "Table S5. Q1 filter sensitivity: effect of near-exclusive species marker "
        "threshold on balanced accuracy.",
        "Spec_score = prevalence in focal species ÷ max prevalence across all other "
        "species. The primary analysis used a threshold of 0.70, removing 8 markers "
        "(359 features retained). Raising the threshold to 0.50 removes 15 markers "
        "(352 features retained); balanced accuracy drops from 0.900 to 0.813, "
        "confirming the primary threshold is the least restrictive setting that "
        "eliminates unambiguous taxonomic markers.",
    ),
    (
        "S6_q2_rm_spearman_withinpg.csv",
        "Table S6",
        "Table S6. Q2 RM subtype within-phylogroup Spearman correlations between "
        "RM system presence and ARG burden per species.",
        "ρ (species-level) is the raw Spearman correlation across all Q2-eligible "
        "genomes of that species. ρ (within-phylogroup) is the mean of per-phylogroup "
        "Spearman correlations (singletons excluded). p (within-phylogroup) is from a "
        "one-sample t-test of per-phylogroup correlations against zero. Survives "
        "robustness gate: within-phylogroup p < 0.05 and sign consistent with "
        "species-level direction. BH correction applied across all features tested "
        "per species in the full Q2 run.",
    ),
    (
        "S7_ab_ic2_exclusion.csv",
        "Table S7",
        "Table S7. Q2 A. baumannii IC2-exclusion sub-analysis: AUROC and mean balanced "
        "accuracy after removing IC2 lineage genomes (SspBCDE-positive).",
        "283 of 600 AB genomes (SspBCDE-positive) were excluded. Q2 RF re-trained on "
        "317 non-IC2 AB genomes (Q2-eligible n=243, 34 phylogroups, 59 features after "
        "prevalence filtering). Tertile boundaries recomputed within the non-IC2 subset. "
        "Methodology in Supplemental Methods S-M7.",
    ),
    (
        "S8_ec_hormaechei_restriction.csv",
        "Table S8",
        "Table S8. Q2 E. cloacae complex E. hormaechei restriction sub-analysis: AUROC "
        "and mean balanced accuracy when analysis is restricted to E. hormaechei sensu stricto.",
        "241 of 507 EC genomes identified as E. hormaechei via NCBI Assembly metadata. "
        "Q2 RF trained on Q2-eligible hormaechei genomes (n=181, 33 phylogroups, 65 "
        "features after prevalence filtering). Methodology in Supplemental Methods S-M7.",
    ),
    (
        "S9_q3_silhouette_k_sweep.csv",
        "Table S9",
        "Table S9. Q3 silhouette K-sweep: Euclidean and Jaccard silhouette scores for "
        "K=2–12 on the full dataset, with selected Jaccard scores for K=2–8.",
        "K-means applied to 359-feature binary defence matrix (full dataset, n=3,335). "
        "Jaccard silhouette computed using pairwise Jaccard distances (memory-efficient "
        "subsample of 1,000 genomes per K for K > 8). Dereplicated row (n=309) shows "
        "best-K result only. Best K by Euclidean silhouette is K=4 (score=0.060); "
        "best K by Jaccard silhouette is also K=4 (score=0.098). Gap statistic optimal "
        "K=1 for both full and dereplicated datasets.",
    ),
    (
        "S10_geographic_representation.csv",
        "Table S10",
        "Table S10. Geographic representation: country-of-origin distribution per "
        "ESKAPE species (genome counts and percentages).",
        "Country of origin derived from NCBI BioSample geo_loc_name attribute. "
        "Genomes with missing or unresolvable country metadata are counted under "
        "Unknown. Counts are for all 3,335 training genomes (post-MLST exclusion).",
    ),
    (
        "S11_q3_clustering_summary.csv",
        "Table S11",
        "Table S11. Q3 clustering summary: full and dereplicated dataset metrics "
        "at best K and K=6.",
        "Full dataset: n=3,335 genomes. Dereplicated: one genome per phylogroup "
        "(n=309). ARI: Adjusted Rand Index. ΔARI (full minus dereplicated at their "
        "respective optimal K) = 0.261, exceeding the pre-specified 0.10 clonal "
        "inflation threshold. This table is the source for main text Table 4.",
    ),
    (
        "S12_q3b_multiblock_ari.csv",
        "Table S12",
        "Table S12. Q3b multi-block ARI: per-block and combined-block species "
        "recovery (ARI vs species labels, K=6) for all fourteen conditions after "
        "near-exclusive species marker filtering, with bootstrap 95% CIs.",
        "Clustering performed on 309 dereplicated genomes, K fixed at 6. The same "
        "spec_score >= 0.70 filter was applied to all five feature blocks before "
        "clustering. Conditions are grouped as: individual blocks (Defence, IS, HMRG, "
        "ARG, Anti-defence); pairwise defence combinations (Defence+IS, Defence+HMRG, "
        "Defence+ARG, Defence+Anti-defence); and multi-block combinations "
        "(IS+HMRG+ARG; Defence+IS+HMRG+ARG; Defence+IS+Anti-defence; "
        "IS+HMRG+ARG+Anti-defence; Defence+IS+HMRG+ARG+Anti-defence). Anti-defence "
        "features were binarised from AntiDefenseFinder counts (41 system types); "
        "4 near-exclusive species markers removed, 37 retained. 95% CIs from "
        "bootstrap resampling of 309 genomes (2,000 iterations, K-means refit per "
        "resample). Permutation p from 1,000 species-label shufflings. "
        "Methodology in Supplemental Methods S-M6.",
    ),
    (
        "S13_q2_full_driver_table.csv",
        "Table S13",
        "Table S13. Q2 within-phylogroup robust driver table: per-species defence "
        "system features tested for ARG-burden association, with effect sizes and "
        "robustness gate outcomes.",
        "Includes all features passing the per-species prevalence filter (≥5% of "
        "Q2-eligible genomes). ρ (within-phylogroup) = mean of per-phylogroup Spearman "
        "correlations. Survives robustness gate: within-phylogroup p < 0.05 and sign "
        "consistent with species-level direction. Species shown: EC, EF, KP, PA (Q2 "
        "significant); AB and SA shown for completeness. Category: named = curated "
        "system name; uncharacterised = dp_df_ or dp_padloc_ prefix system with no "
        "canonical name.",
    ),
    (
        "S14_q2_covariate_adjustment.csv",
        "Table S14",
        "Table S14. Q2 IS-element covariate adjustment: partial Spearman results "
        "for all drivers tested in the four Q2-significant species (EC, EF, KP, PA).",
        "Both the defence feature and arg_count_unique were residualised on total "
        "defence count (excluding driver under test) and is_count_total using OLS "
        "within each phylogroup before computing partial Spearman ρ. IS-burden-adjusted-"
        "robust: p_partial < 0.05 with sign of ρ_partial matching ρ_within_pg. "
        "Methodology in Supplemental Methods S-M8.",
    ),
    (
        "S15_dominant_pg_exclusion.csv",
        "Table S15",
        "Table S15. Dominant phylogroup exclusion sub-analysis: Q2 AUROC before and "
        "after removing the single largest phylogroup per species (KP, EF, PA, SA).",
        "EC and AB were exempt (EC largest phylogroup = 6.3% of cohort; AB addressed "
        "by IC2-exclusion subanalysis). Dominant phylogroup identified as the single "
        "largest by genome count before Q2 tertile assignment. AUROC 95% CIs from "
        "fold-level bootstrap (5 folds, 2,000 iterations). Methodology in Supplemental "
        "Methods S-M9.",
    ),
    (
        "S16_holdout_set.csv",
        "Table S16",
        "Table S16. External holdout set: per-genome accession, species, sequence "
        "type, selection basis, RF-predicted species, and classification result (n=180).",
        "30 genomes per ESKAPE species selected from complete NCBI RefSeq assemblies "
        "with zero accession overlap with the 3,335-genome training set. Selection "
        "priority: novel ST > unknown ST > shared ST (distinct GCF accession). "
        "Predicted species from the Q1 RF model trained on all 3,335 training genomes. "
        "Holdout balanced accuracy = 0.944 [0.911–0.972]. Methodology in Supplemental "
        "Methods S-M10.",
    ),
]


def set_col_widths(ws, df: pd.DataFrame, min_w: int = 10, max_w: int = 50):
    """Auto-size columns based on content, capped at max_w."""
    for i, col in enumerate(df.columns, start=1):
        col_vals = [str(col)] + [str(v) for v in df[col].fillna("")]
        width = min(max(len(v) for v in col_vals) + 2, max_w)
        width = max(width, min_w)
        ws.column_dimensions[get_column_letter(i)].width = width


def write_table(wb, csv_path: Path, sheet_name: str, title: str, legend: str):
    df = pd.read_csv(csv_path)
    n_cols = len(df.columns)
    last_col = get_column_letter(n_cols)

    ws = wb.create_sheet(title=sheet_name)

    # Row 1 — title
    ws.append([title])
    ws.merge_cells(f"A1:{last_col}1")
    title_cell = ws["A1"]
    title_cell.font = Font(bold=True, size=11)
    title_cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 30

    # Row 2 — legend
    ws.append([legend])
    ws.merge_cells(f"A2:{last_col}2")
    legend_cell = ws["A2"]
    legend_cell.font = Font(italic=True, size=10)
    legend_cell.alignment = Alignment(wrap_text=True, vertical="top")
    # Height proportional to legend length
    ws.row_dimensions[2].height = max(45, min(len(legend) // 3, 120))

    # Row 3 — blank separator
    ws.append([""])

    # Row 4 — column headers
    ws.append(list(df.columns))
    header_row = 4
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=header_row, column=col_idx)
        cell.font = Font(bold=True, size=10)
        cell.fill = BLUE_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[header_row].height = 20

    # Rows 5+ — data
    for row in df.itertuples(index=False):
        ws.append(list(row))

    # Auto-size columns
    set_col_widths(ws, df)

    # Freeze panes below header row
    ws.freeze_panes = ws.cell(row=5, column=1)

    return ws


def main():
    wb = openpyxl.Workbook()
    # Remove default blank sheet
    wb.remove(wb.active)

    for csv_name, sheet_name, title, legend in TABLES:
        csv_path = TABLES_DIR / csv_name
        if not csv_path.exists():
            print(f"  MISSING: {csv_path}")
            continue
        write_table(wb, csv_path, sheet_name, title, legend)
        print(f"  wrote {sheet_name} ({csv_name})")

    wb.save(OUT_FILE)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
