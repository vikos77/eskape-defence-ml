"""
Q1 positive controls: MLST scheme-call and Mash-distance k-NN as established-
method comparators against the RF defence-repertoire species classifier.

Context: docs/manuscript/revision_roadmap_2026-09-07.md, comment R1-4
("major limitation... lack of a suitable positive control... comparison
with an established approach such as MLST").

Two comparators, both evaluated on the SAME 180-genome external holdout
(Table S16) the RF holdout number (BA=0.944) already uses, for a genuinely
paired three-way comparison:

  1. MLST scheme-call: the PubMLST scheme `mlst` auto-detects for a genome
     IS a cross-species call (the same field Methods §1 used to exclude 125
     discordant training accessions). Map scheme name -> implied species,
     compare to true species.
  2. Mash-distance k-NN: majority vote over the k nearest training genomes
     by whole-genome Mash distance (k=21, s=1000, same as Methods §4).
     k selected by the SAME StratifiedGroupKFold(5, shuffle=True,
     random_state=42)-by-phylogroup CV structure as run_q1_367.py, so the
     comparator gets its hyperparameter from the same evaluation discipline
     RF did, not a hand-picked value.

A third number is reported for context only, NOT as part of the paired
comparison: MLST scheme-call concordance across the full pre-QC training
candidate pool (3,491 genomes, before the 125 discordance exclusions).
This is the "how good is MLST in the wild, before you've used it to clean
anything" number -- it is where the 2 real MLST failure modes actually show
up (96 KP genomes calling as E. coli, 28+1 E. cloacae calling as
Cronobacter/Salmonella, 2 E. cloacae with no scheme match at all). The
holdout comparison below happens to show zero such failures, which is a
real, honestly-reportable result, not evidence that this contextual number
is wrong -- it is a smaller, unluckier-or-luckier sample (360 genomes,
60/species) than the full training pool (3,491).

CI methods deliberately mirror the existing RF numbers rather than using a
"better" method, so the comparison table isn't comparing different
yardsticks:
  - Training-CV BA: t-interval over 5 fold-level BAs (matches
    run_q1_367.py's ci95_ba exactly, including that method's known
    limitation -- CV folds are not i.i.d., see peer_review_panel_report.md,
    parked as out of scope for this comparator).
  - Holdout BA: stratified-by-species resample, 2000 iterations (matches
    export_fig_data_for_r.py's holdout_ci exactly).

Outputs: results/q1_comparators_367.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample

REPO_ROOT = Path(__file__).resolve().parents[2]
N_FOLDS = 5
RANDOM_STATE = 42
K_GRID = [1, 3, 5, 11, 25]
SPECIES_ORDER = ["abaumannii", "ecloaceae", "efaecium",
                  "kpneumoniae", "paeruginosa", "saureus"]

# Verified against data/interim/{species}/mlst/mlst_results.tsv and
# data/interim/holdout/{species}/mlst/mlst_results.tsv column 2 (see
# session notes 2026-09-07). Both A. baumannii PubMLST schemes observed
# (Pasteur "abaumannii_2" in training, Oxford "abaumannii" in holdout) --
# both are concordant A. baumannii calls, not a discordance.
SCHEME_TO_SPECIES = {
    "abaumannii": "abaumannii",
    "abaumannii_2": "abaumannii",
    "saureus": "saureus",
    "paeruginosa": "paeruginosa",
    "efaecium": "efaecium",
    "ecloacae": "ecloaceae",
    "klebsiella": "kpneumoniae",
}
FULL_NAME_TO_CODE = {
    "Acinetobacter baumannii": "abaumannii",
    "Enterobacter cloacae complex": "ecloaceae",
    "Enterococcus faecium": "efaecium",
    "Klebsiella pneumoniae": "kpneumoniae",
    "Pseudomonas aeruginosa": "paeruginosa",
    "Staphylococcus aureus": "saureus",
}


def to_accession(path_str):
    stem = Path(path_str).stem
    base, ver = stem.rsplit("_", 1)
    return f"{base}.{ver}"


# ── 1. MLST: naive concordance on the full pre-QC training candidate pool ──

def mlst_naive_training_concordance():
    """Also separates two distinct MLST failure modes that the manuscript
    must not conflate (user's + advisor's point, 2026-09-08): scheme-level
    SPECIES identification (what we're benchmarking RF against, and what
    is near-perfect) vs. locus-level ST ASSIGNMENT within a correctly-
    identified species (MLST's actual designed task -- fine strain
    resolution -- which genuinely fails on a material fraction of
    genomes with novel allele combinations)."""
    per_species = {}
    for sp in SPECIES_ORDER:
        path = REPO_ROOT / f"data/interim/{sp}/mlst/mlst_results.tsv"
        rows = [l.split("\t") for l in path.read_text().splitlines() if l.strip()]
        schemes = [r[1] for r in rows]
        sts = [r[2] for r in rows]
        n = len(schemes)
        implied = [SCHEME_TO_SPECIES.get(s, "OTHER" if s != "-" else "UNTYPEABLE")
                   for s in schemes]
        n_concordant = sum(1 for x in implied if x == sp)
        n_untypeable = sum(1 for x in implied if x == "UNTYPEABLE")
        n_discordant = n - n_concordant - n_untypeable
        discordant_schemes = sorted({schemes[i] for i, x in enumerate(implied)
                                      if x not in (sp, "UNTYPEABLE")})
        n_st_unassigned = sum(1 for x, st in zip(implied, sts) if x == sp and st == "-")
        per_species[sp] = {
            "n_candidates": n,
            "n_concordant": n_concordant,
            "n_discordant": n_discordant,
            "n_untypeable": n_untypeable,
            "discordant_schemes_seen": discordant_schemes,
            "n_st_unassigned_among_concordant": n_st_unassigned,
            "st_unassigned_rate_among_concordant": n_st_unassigned / n_concordant,
        }
    total = sum(v["n_candidates"] for v in per_species.values())
    total_concordant = sum(v["n_concordant"] for v in per_species.values())
    total_st_unassigned = sum(v["n_st_unassigned_among_concordant"] for v in per_species.values())
    return {
        "per_species": per_species,
        "total_candidates": total,
        "total_concordant": total_concordant,
        "total_discordant": sum(v["n_discordant"] for v in per_species.values()),
        "total_untypeable": sum(v["n_untypeable"] for v in per_species.values()),
        "overall_concordance_rate": total_concordant / total,
        "total_st_unassigned_among_concordant": total_st_unassigned,
        "st_unassigned_rate_among_concordant": total_st_unassigned / total_concordant,
        "note": (
            "st_unassigned = species scheme matched correctly, but no catalogued "
            "ST for this allele combination (novel combination). This is MLST's "
            "actual designed task (strain-level resolution) failing, distinct "
            "from species-level scheme detection (near-100% throughout)."
        ),
    }


# ── 2. MLST: classifier on the 180-genome holdout (paired w/ RF) ──────────

def load_holdout_mlst_calls():
    """Parse holdout mlst_results.tsv per species, tolerating the
    kpneumoniae file's interleaved blastn/log-warning pollution (only lines
    starting with the expected fna path are real result rows). Returns
    accession -> (scheme, ST)."""
    calls = {}
    for sp in SPECIES_ORDER:
        path = REPO_ROOT / f"data/interim/holdout/{sp}/mlst/mlst_results.tsv"
        marker = f"holdout_genomes/{sp}/"  # kpneumoniae file has non-tsv log lines mixed in
        for line in path.read_text().splitlines():
            if marker not in line or "\t" not in line:
                continue
            fields = line.split("\t")
            acc = to_accession(fields[0])
            calls[acc] = (fields[1], fields[2])
    return calls


def mlst_holdout_classifier(s16):
    calls = load_holdout_mlst_calls()
    missing = set(s16["Accession"]) - set(calls)
    assert not missing, f"no MLST call for holdout accessions: {missing}"

    y_true = s16["species_code"].to_numpy()
    schemes = np.array([calls[a][0] for a in s16["Accession"]])
    sts = np.array([calls[a][1] for a in s16["Accession"]])
    y_pred = np.array([SCHEME_TO_SPECIES.get(s, "OTHER" if s != "-" else "UNTYPEABLE")
                        for s in schemes])

    n_untypeable = int((y_pred == "UNTYPEABLE").sum())
    n_discordant = int(((y_pred != "UNTYPEABLE") & (y_pred != y_true)).sum())
    concordant_mask = y_pred == y_true
    n_st_unassigned = int(((sts == "-") & concordant_mask).sum())

    acc = float((y_pred == y_true).mean())
    recalls = {sp: float(recall_score(y_true == sp, y_pred == sp, zero_division=0))
               for sp in SPECIES_ORDER}

    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    for _ in range(2000):
        idx = resample(np.arange(len(y_true)), stratify=y_true,
                        random_state=rng.integers(1_000_000))
        boot.append(float((y_pred[idx] == y_true[idx]).mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "accuracy": acc,
        "ci95": [float(lo), float(hi)],
        "per_species_recall": recalls,
        "n_discordant_calls": n_discordant,
        "n_untypeable_calls": n_untypeable,
        "confusion_note": (
            "0 discordant, 0 untypeable across all 180 holdout genomes"
            if n_discordant == 0 and n_untypeable == 0
            else f"{n_discordant} discordant, {n_untypeable} untypeable"
        ),
        "n_st_unassigned_among_concordant": n_st_unassigned,
        "st_unassigned_rate_among_concordant": n_st_unassigned / int(concordant_mask.sum()),
        "st_unassigned_note": (
            "Species scheme call correct but no catalogued ST (novel allele "
            "combination) -- elevated here vs. the training pool because the "
            "holdout was deliberately enriched for novel STs (Methods §8)."
        ),
    }


# ── 3. Mash-kNN: k selection via training-cohort grouped CV ───────────────

def knn_predict(dist_block, train_species, k):
    """dist_block: (n_query, n_train) distances. Vectorised k-NN majority
    vote with a deterministic tie-break (smaller mean neighbor distance)."""
    preds = []
    nn_idx = np.argsort(dist_block, axis=1)[:, :k]
    for row_i, neigh in enumerate(nn_idx):
        labels = train_species[neigh]
        dists = dist_block[row_i, neigh]
        uniq, counts = np.unique(labels, return_counts=True)
        top_count = counts.max()
        tied = uniq[counts == top_count]
        if len(tied) == 1:
            preds.append(tied[0])
        else:
            mean_d = {t: dists[labels == t].mean() for t in tied}
            preds.append(min(mean_d, key=mean_d.get))
    return np.array(preds)


def mash_knn_training_cv(fm, dm):
    y = fm["species"].to_numpy()
    groups = fm["phylogroup"].to_numpy(dtype=str)
    dm_vals = dm.loc[fm.index, fm.index].to_numpy()  # align to fm row order

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    dummy_X = np.zeros((len(fm), 1))

    per_k_fold_bas = {}
    for k in K_GRID:
        fold_bas = []
        for tr, te in cv.split(dummy_X, y, groups=groups):
            block = dm_vals[np.ix_(te, tr)]
            y_pred = knn_predict(block, y[tr], k)
            fold_bas.append(balanced_accuracy_score(y[te], y_pred))
        per_k_fold_bas[k] = fold_bas

    mean_by_k = {k: float(np.mean(v)) for k, v in per_k_fold_bas.items()}
    best_k = max(mean_by_k, key=mean_by_k.get)
    fold_bas = np.array(per_k_fold_bas[best_k])
    mean_ba = float(fold_bas.mean())
    if fold_bas.std() == 0:
        # stats.t.interval degenerates to (-inf*0, inf*0) = (nan, nan) when
        # scale=0 (every fold identical) -- a real result (all 6 species
        # perfectly separated in every fold), not a computation to hide.
        ci = (mean_ba, mean_ba)
    else:
        ci = stats.t.interval(0.95, df=N_FOLDS - 1, loc=mean_ba, scale=stats.sem(fold_bas))

    # per-class recall at best k, pooled across folds
    per_class_recall = {sp: [] for sp in SPECIES_ORDER}
    for tr, te in cv.split(dummy_X, y, groups=groups):
        block = dm_vals[np.ix_(te, tr)]
        y_pred = knn_predict(block, y[tr], best_k)
        recs = recall_score(y[te], y_pred, labels=SPECIES_ORDER, average=None, zero_division=0)
        for sp, r in zip(SPECIES_ORDER, recs):
            per_class_recall[sp].append(r)
    per_class_recall = {sp: float(np.mean(v)) for sp, v in per_class_recall.items()}

    return {
        "note": (
            "REVISED 2026-09-08: an earlier version of this docstring flagged "
            "this as circular (phylogroups derived from Mash distances scoring "
            "a Mash-distance classifier). That concern doesn't hold: phylogroups "
            "are built by WITHIN-species clustering (Methods §4) over genomes "
            "already assigned to species by NCBI taxonomy, so the CV grouping "
            "has no mechanism to inflate SPECIES-level separation, which is "
            "what this classifier is scored on. The all-k tie at exactly 1.000 "
            "reflects the real, expected geometry: inter-species Mash distance "
            "dwarfs intra-species distance for six organisms spanning two phyla. "
            "Mash/ANI-based whole-genome relatedness is the modern standard for "
            "bacterial species delineation (operationalised in GTDB) and is "
            "treated as the PRIMARY R1-4 comparator, not a sanity check."
        ),
        "k_grid_mean_ba": mean_by_k,
        "best_k": best_k,
        "fold_bas": fold_bas.tolist(),
        "mean_ba": mean_ba,
        "ci95_ba": [float(ci[0]), float(ci[1])],
        "per_class_recall": per_class_recall,
    }


def mash_knn_holdout(fm, s16, holdout_dm, best_k):
    y_train = fm["species"].to_numpy()
    block = holdout_dm.loc[s16["Accession"], fm.index].to_numpy()
    y_pred = knn_predict(block, y_train, best_k)
    y_true = s16["species_code"].to_numpy()

    acc = float((y_pred == y_true).mean())
    recalls = {sp: float(recall_score(y_true == sp, y_pred == sp, zero_division=0))
               for sp in SPECIES_ORDER}

    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    for _ in range(2000):
        idx = resample(np.arange(len(y_true)), stratify=y_true,
                        random_state=rng.integers(1_000_000))
        boot.append(float((y_pred[idx] == y_true[idx]).mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    cm = confusion_matrix(y_true, y_pred, labels=SPECIES_ORDER)
    return {
        "k_used": best_k,
        "accuracy": acc,
        "ci95": [float(lo), float(hi)],
        "per_species_recall": recalls,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": SPECIES_ORDER,
    }


def main():
    fm = pd.read_parquet(REPO_ROOT / "data/processed/feature_matrix_3335.parquet")
    dm = pd.read_parquet(REPO_ROOT / "data/interim/mash/distance_matrix.parquet")
    holdout_dm = pd.read_parquet(REPO_ROOT / "data/interim/mash/holdout_distance_matrix.parquet")

    s16 = pd.read_csv(REPO_ROOT / "results/tables/S16_holdout_set.csv")
    s16 = s16[s16["Accession"].notna() & (s16["Accession"] != "")]
    s16["species_code"] = s16["Species"].map(FULL_NAME_TO_CODE)
    assert s16["species_code"].notna().all()
    assert len(s16) == 180

    print("=== 1. MLST naive concordance, full training candidate pool ===")
    mlst_context = mlst_naive_training_concordance()
    print(f"  {mlst_context['total_concordant']}/{mlst_context['total_candidates']} "
          f"concordant ({mlst_context['overall_concordance_rate']:.4f})")

    print("\n=== 2. MLST as classifier, 180-genome holdout ===")
    mlst_holdout = mlst_holdout_classifier(s16)
    print(f"  accuracy = {mlst_holdout['accuracy']:.4f} {mlst_holdout['ci95']}")
    print(f"  {mlst_holdout['confusion_note']}")

    print("\n=== 3. Mash-kNN, training-cohort grouped CV k-sweep ===")
    knn_cv = mash_knn_training_cv(fm, dm)
    print(f"  k grid mean BA: {knn_cv['k_grid_mean_ba']}")
    print(f"  best k = {knn_cv['best_k']}, CV BA = {knn_cv['mean_ba']:.4f} {knn_cv['ci95_ba']}")

    print("\n=== 4. Mash-kNN, 180-genome holdout (best k from CV) ===")
    knn_holdout = mash_knn_holdout(fm, s16, holdout_dm, knn_cv["best_k"])
    print(f"  accuracy = {knn_holdout['accuracy']:.4f} {knn_holdout['ci95']}")

    with open(REPO_ROOT / "results/q1_367_results.json") as f:
        rf_cv = json.load(f)
    with open(REPO_ROOT / "results/fig1b_holdout_for_r.json") as f:
        rf_holdout = json.load(f)

    out = {
        "rf_reference": {
            "cv_ba": rf_cv["mean_ba"],
            "cv_ci95": rf_cv["ci95_ba"],
            "cv_per_class_recall": rf_cv["per_class_recall"],
            "holdout_ba": rf_holdout["holdout_ba"],
            "holdout_ci95": rf_holdout["holdout_ci"],
            "holdout_per_class_recall": rf_holdout["recalls"],
        },
        "mlst_training_pool_context": mlst_context,
        "mlst_holdout_classifier": mlst_holdout,
        "mash_knn_training_cv": knn_cv,
        "mash_knn_holdout": knn_holdout,
    }
    out_path = REPO_ROOT / "results/q1_comparators_367.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n=== SUMMARY: holdout accuracy, n=180, same genomes as RF's Fig 1B ===")
    print(f"  RF (defence repertoire):                 {rf_holdout['holdout_ba']:.4f} {rf_holdout['holdout_ci']}")
    print(f"  Mash-{knn_cv['best_k']}NN (PRIMARY comparator -- whole-genome")
    print(f"    relatedness, the modern species-delineation standard):")
    print(f"                                            {knn_holdout['accuracy']:.4f} {knn_holdout['ci95']}")
    print(f"  MLST scheme-call (secondary -- supervisor-requested,")
    print(f"    species-call is incidental to MLST's designed ST-typing task):")
    print(f"                                            {mlst_holdout['accuracy']:.4f} {mlst_holdout['ci95']}")
    print(f"  MLST ST-unassignable among correctly-typed holdout genomes: "
          f"{mlst_holdout['n_st_unassigned_among_concordant']}/180 "
          f"({100*mlst_holdout['st_unassigned_rate_among_concordant']:.1f}%) "
          f"-- MLST's real designed task failing, distinct from species-call")


if __name__ == "__main__":
    main()
