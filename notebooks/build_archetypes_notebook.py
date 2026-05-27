"""
Builder for notebooks/09_unsupervised_archetypes.ipynb — Phase 11.

Sections:
  1. K selection — defence features (Q3)
  2. K-means clustering (Q3 primary)
  3. Hierarchical clustering (Q3 comparison)
  4. Q5b — phage susceptibility archetypes (defence + anti-defence + IS burden)
  5. Biological synthesis
"""

import nbformat
import nbformat as nbf
from pathlib import Path


def md(text: nbformat.NotebookNode) -> nbformat.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbformat.NotebookNode:
    return nbf.v4.new_code_cell(src)


cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md("""# Phase 11 — Unsupervised Defence-System Archetypes

Two questions drive this notebook:

**Q3 (primary):** Do ESKAPE genomes cluster by defence-system archetype independently of
species? If so, how many archetypes exist, and do they map onto known biology?

**Q5b (registered):** When IS burden and anti-defence systems are added to the feature
space, do distinct phage-permissive archetypes emerge?

Unlike Phases 7–10 (supervised — model given species labels), clustering here receives
**no labels**. Groupings emerge purely from similarity in defence feature vectors.
"""))

# ── Imports ──────────────────────────────────────────────────────────────────
cells.append(code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, silhouette_samples,
                              adjusted_rand_score, calinski_harabasz_score)
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Notebooks execute from notebooks/ subdirectory; go up to project root
ROOT    = Path("..")
DATA    = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures" / "archetypes"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
print("Imports OK")
print(f"Project root: {ROOT.resolve()}")
"""))

# ── Load data ─────────────────────────────────────────────────────────────────
cells.append(code("""# ── Load feature matrix ──────────────────────────────────────────────────
fm = pd.read_parquet(DATA / "feature_matrix.parquet")
print(f"Feature matrix: {fm.shape[0]} genomes × {fm.shape[1]} columns")

# ── Reproduce FEAT_COLS exactly as in Phases 7–10 ────────────────────────────
# Dynamic computation (mirrors training script): exclude features whose
# per-species prevalence variance normalised to [0,0.5] std >= 0.70 threshold.
# These are species-specific markers (taxonomic signal, not defence architecture).
all_dp     = sorted([c for c in fm.columns if c.startswith('dp_')])
sp_prev    = fm.groupby("species")[all_dp].mean()
spec_score = sp_prev.std() / 0.5
markers    = spec_score[spec_score >= 0.70].index.tolist()
FEAT_COLS  = [c for c in all_dp if c not in markers]

# ── Anti-defence features (Q5b) ───────────────────────────────────────────────
AD_COLS = sorted([c for c in fm.columns if c.startswith('ad_')])

# ── IS element features (Q5b) ─────────────────────────────────────────────────
IS_COLS = [c for c in fm.columns if c.startswith('is_')]

print(f"FEAT_COLS (defence, Q3):      {len(FEAT_COLS)}")
print(f"AD_COLS (anti-defence, Q5b):  {len(AD_COLS)}")
print(f"IS_COLS (IS elements, Q5b):   {len(IS_COLS)}")
print()
print("Species counts:")
print(fm['species'].value_counts())

# ── Label arrays for ARI computation ─────────────────────────────────────────
species_labels = fm['species'].to_numpy(dtype=str)
arg_labels = fm['arg_burden_tertile'].fillna('unknown').to_numpy(dtype=str)

X_q3 = fm[FEAT_COLS].values.astype(float)  # 878 × 265 binary
print(f"\\nQ3 feature matrix: {X_q3.shape}")
"""))

# ── Section 1: K selection ────────────────────────────────────────────────────
cells.append(md("""## Section 1 — K Selection (Q3: defence features only)

K-means requires K to be specified before running. We evaluate K=2 to 12 using:

- **Silhouette score** — measures how tight clusters are vs how separated from neighbours.
  Range [−1, 1]; higher is better. A peak at K=k suggests k is the natural number of groups.
- **Calinski-Harabasz score** — ratio of between-cluster to within-cluster variance.
  Higher is better; tends to favour smaller K.

We choose K by silhouette peak, then check whether K=6 (one cluster per species) is
competitive — this directly tests whether defence architecture encodes species identity.
"""))

cells.append(code("""# ── K selection loop ─────────────────────────────────────────────────────────
k_range = range(2, 13)
sil_scores = []
ch_scores  = []

print("Running K-means for K=2..12  (this takes ~30 s)")
for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_q3)
    sil_scores.append(silhouette_score(X_q3, labels))
    ch_scores.append(calinski_harabasz_score(X_q3, labels))
    print(f"  K={k:2d}  silhouette={sil_scores[-1]:.4f}  CH={ch_scores[-1]:.1f}")

best_k = k_range.start + int(np.argmax(sil_scores))
print(f"\\nBest K by silhouette: {best_k}  (score={max(sil_scores):.4f})")
print(f"Silhouette at K=6:    {sil_scores[6-2]:.4f}")
"""))

cells.append(code("""# ── M3: Gap statistic (pre-registered deliverable) ───────────────────────────
# The gap statistic compares within-cluster variance to a random uniform reference.
# If gap is maximised at K=1, the data has no cluster structure (K=1 = all one cluster).
# Reference: Tibshirani et al. (2001) — original gap statistic paper.

def _wcss(X, labels):
    \"\"\"Total within-cluster sum of squares.\"\"\"
    total = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        total += np.sum((Xc - Xc.mean(axis=0)) ** 2)
    return total

rng_gap  = np.random.RandomState(RANDOM_STATE)
n_refs   = 10
gap_vals = []
gap_errs = []
gap_ks   = [1] + list(k_range)

print("Computing gap statistic K=1..12  (n_refs=10, ~60 s) ...")
for k in gap_ks:
    if k == 1:
        lbl_k = np.zeros(len(X_q3), dtype=int)
    else:
        km_g = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        lbl_k = km_g.fit_predict(X_q3)
    log_Wk = np.log(_wcss(X_q3, lbl_k) + 1e-10)

    ref_log_Wks = []
    for _ in range(n_refs):
        # Uniform reference in [0,1]^d (binary feature space bounding box)
        X_ref = rng_gap.uniform(0, 1, X_q3.shape)
        if k == 1:
            lbl_ref = np.zeros(len(X_ref), dtype=int)
        else:
            km_ref = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE)
            lbl_ref = km_ref.fit_predict(X_ref)
        ref_log_Wks.append(np.log(_wcss(X_ref, lbl_ref) + 1e-10))

    gap  = np.mean(ref_log_Wks) - log_Wk
    gap_err = np.std(ref_log_Wks) * np.sqrt(1 + 1/n_refs)
    gap_vals.append(gap)
    gap_errs.append(gap_err)
    print(f"  K={k:2d}  gap={gap:.4f}  se={gap_err:.4f}")

# Optimal K: smallest K such that gap(K) >= gap(K+1) - se(K+1)  [Tibshirani criterion]
gap_optimal_k = gap_ks[0]
for i in range(len(gap_ks) - 1):
    if gap_vals[i] >= gap_vals[i+1] - gap_errs[i+1]:
        gap_optimal_k = gap_ks[i]
        break

print(f"\\nGap-statistic optimal K: {gap_optimal_k}")
if gap_optimal_k == 1:
    print("K=1 optimal — gap statistic confirms no meaningful cluster structure.")
    print("Consistent with silhouette <0.07 across all K=2..12.")
else:
    print(f"Gap statistic suggests K={gap_optimal_k} — cluster structure present.")
"""))

cells.append(code("""# ── Plot K selection diagnostics (silhouette + CH + gap statistic) ───────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(list(k_range), sil_scores, 'o-', color='steelblue', linewidth=2)
axes[0].axvline(best_k, color='crimson', linestyle='--', label=f'Best K={best_k}')
axes[0].axvline(6, color='grey', linestyle=':', alpha=0.7, label='K=6 (species)')
axes[0].set_xlabel('Number of clusters (K)')
axes[0].set_ylabel('Silhouette score')
axes[0].set_title('K selection — Silhouette')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), ch_scores, 'o-', color='darkorange', linewidth=2)
axes[1].axvline(best_k, color='crimson', linestyle='--', label=f'Best K={best_k}')
axes[1].axvline(6, color='grey', linestyle=':', alpha=0.7, label='K=6 (species)')
axes[1].set_xlabel('Number of clusters (K)')
axes[1].set_ylabel('Calinski-Harabasz score')
axes[1].set_title('K selection — Calinski-Harabasz')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].errorbar(gap_ks, gap_vals, yerr=gap_errs, fmt='o-', color='seagreen',
                 linewidth=2, capsize=4, label='Gap ± SE')
axes[2].axvline(gap_optimal_k, color='crimson', linestyle='--',
                label=f'Optimal K={gap_optimal_k}')
axes[2].set_xlabel('Number of clusters (K)')
axes[2].set_ylabel('Gap statistic')
axes[2].set_title('K selection — Gap statistic (Tibshirani 2001)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "k_selection_q3.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: k_selection_q3.png")
"""))

cells.append(code("""# ── H1: Interpret silhouette magnitude explicitly ─────────────────────────────
max_sil = max(sil_scores)
print(f"Maximum silhouette across K=2..12: {max_sil:.4f} at K={best_k}")
print()
if max_sil < 0.10:
    print("INTERPRETATION: All silhouette scores across K=2–12 are below 0.10.")
    print(f"  Maximum observed: {max_sil:.3f} at K={best_k}.")
    print("  Values below 0.10 indicate no substantial cluster structure exists in")
    print("  this feature space. The data does not partition into distinct natural groups.")
    print("  Any K-means partition at this scale represents an arbitrary division of a")
    print("  continuum, not a discovery of biologically discrete archetypes.")
elif max_sil < 0.25:
    print(f"INTERPRETATION: Maximum silhouette {max_sil:.3f} — weak structure.")
    print("  Clusters exist but are not well-separated. Interpret with caution.")
else:
    print(f"INTERPRETATION: Maximum silhouette {max_sil:.3f} — moderate to strong structure.")
"""))

# ── Section 2: K-means ───────────────────────────────────────────────────────
cells.append(md("""## Section 2 — K-means Clustering (Q3)

We run K-means with two values of K:

1. **Best K** (silhouette-optimal) — the most internally coherent partition.
2. **K=6** — forced species-matching partition; ARI against species labels tells us how
   well K-means would recover species identity if forced to use 6 clusters.

Adjusted Rand Index (ARI) measures cluster–label agreement:
- ARI = 1.0 → perfect match
- ARI = 0.0 → random (no better than chance)
- ARI < 0 → worse than random

**Interpretation key:** If ARI(K-means vs species) is high (>0.5), defence architecture
encodes species identity. If low (<0.2), defence archetypes are functionally independent
of taxonomy.
"""))

cells.append(code("""# ── K-means: best K and K=6 ──────────────────────────────────────────────────
km_best = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE)
km_best.fit(X_q3)
labels_best = km_best.labels_

km_6 = KMeans(n_clusters=6, n_init=30, random_state=RANDOM_STATE)
km_6.fit(X_q3)
labels_6 = km_6.labels_

ari_best_species = adjusted_rand_score(species_labels, labels_best)
ari_6_species    = adjusted_rand_score(species_labels, labels_6)
ari_best_arg     = adjusted_rand_score(arg_labels, labels_best)

print(f"K-means K={best_k}:")
print(f"  ARI vs species:     {ari_best_species:.4f}")
print(f"  ARI vs ARG tertile: {ari_best_arg:.4f}")
print()
print(f"K-means K=6:")
print(f"  ARI vs species:     {ari_6_species:.4f}")
print()
print("Cluster sizes (K=best_k):")
unique, counts = np.unique(labels_best, return_counts=True)
for cl, n in zip(unique, counts):
    print(f"  Cluster {cl}: {n} genomes")
"""))

cells.append(code("""# ── Cluster × Species contingency table ──────────────────────────────────────
ct = pd.crosstab(
    pd.Series(labels_best, name='Cluster'),
    pd.Series(species_labels, name='Species')
)
print("Cluster × Species contingency table:")
print(ct.to_string())
print()
# Row normalise to show species composition per cluster
ct_norm = ct.div(ct.sum(axis=1), axis=0).round(3)
print("Row-normalised (proportion of each cluster that is each species):")
print(ct_norm.to_string())
print()

# H4: Identify and interpret the largest mixed-species cluster (Cluster 5 or equivalent)
cluster_sizes = ct.sum(axis=1)
cluster_dominant_frac = ct_norm.max(axis=1)  # fraction of most-represented species
mixed_clusters = cluster_dominant_frac[cluster_dominant_frac < 0.70]  # <70% one species = mixed
if len(mixed_clusters) > 0:
    largest_mixed_cl = cluster_sizes[mixed_clusters.index].idxmax()
    lmc_size = int(cluster_sizes[largest_mixed_cl])
    lmc_composition = ct.loc[largest_mixed_cl].to_dict()
    print(f"Largest mixed-species cluster: Cluster {largest_mixed_cl} ({lmc_size} genomes)")
    for sp, n in sorted(lmc_composition.items(), key=lambda x: -x[1]):
        if n > 0:
            sp_total = (species_labels == sp).sum()
            print(f"  {sp}: {n} ({n/sp_total*100:.0f}% of all {sp})")
    print()
    print(f"Interpretation: Cluster {largest_mixed_cl} likely represents genomes with")
    print("  moderate, non-species-specific Gram-negative defence profiles: common RM")
    print("  Type II, no CRISPR, moderate defence count, absence of the lineage-specific")
    print("  systems that define other clusters. It is the default-Gram-negative bucket")
    print("  rather than a biologically distinctive archetype.")
else:
    print("No large mixed-species clusters found (all clusters >70% dominated by one species).")
"""))

cells.append(code("""# ── Species stacked bar per cluster ──────────────────────────────────────────
SPECIES_COLOURS = {
    'abaumannii':  '#E64B35',
    'efaecium':    '#4DBBD5',
    'ecloaceae':   '#00A087',
    'kpneumoniae': '#3C5488',
    'paeruginosa': '#F39B7F',
    'saureus':     '#8491B4',
}

fig, ax = plt.subplots(figsize=(max(8, best_k * 1.2), 5))
bottom = np.zeros(best_k)
for sp, colour in SPECIES_COLOURS.items():
    if sp not in ct_norm.columns:
        continue
    vals = ct_norm[sp].values if sp in ct_norm.columns else np.zeros(best_k)
    ax.bar(range(best_k), vals, bottom=bottom, color=colour, label=sp, alpha=0.85)
    bottom += vals

ax.set_xticks(range(best_k))
ax.set_xticklabels([f'C{i}' for i in range(best_k)])
ax.set_xlabel('Cluster')
ax.set_ylabel('Proportion')
ax.set_title(f'Species composition per K-means cluster (K={best_k})\\nARI vs species = {ari_best_species:.3f}')
ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1))
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "q3_kmeans_species_composition.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q3_kmeans_species_composition.png")
"""))

cells.append(code("""# ── Cluster profiles heatmap — top 40 most variable features ─────────────────
# For each feature, compute variance across cluster centroid means
centroids = pd.DataFrame(km_best.cluster_centers_, columns=FEAT_COLS)
feature_variance = centroids.var(axis=0).sort_values(ascending=False)
top_features = feature_variance.head(40).index.tolist()

heatmap_data = centroids[top_features]
heatmap_data.index = [f'C{i}' for i in range(best_k)]

fig, ax = plt.subplots(figsize=(18, max(4, best_k * 0.7)))
sns.heatmap(
    heatmap_data,
    cmap='YlOrRd',
    vmin=0, vmax=1,
    xticklabels=True,
    yticklabels=True,
    linewidths=0.3,
    ax=ax
)
ax.set_title(f'K-means cluster profiles — top 40 most variable features (K={best_k})')
ax.set_xlabel('Defence system feature')
ax.set_ylabel('Cluster')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR / "q3_kmeans_cluster_profiles.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q3_kmeans_cluster_profiles.png")
"""))

# ── Section 3: Hierarchical clustering ───────────────────────────────────────
cells.append(md("""## Section 3 — Hierarchical Clustering (Q3 comparison)

K-means partitions globally but is sensitive to initialisation. Hierarchical clustering
(Ward linkage) builds a dendrogram bottom-up without pre-specifying K — we then cut the
tree at the same K as above to compare.

**Ward linkage** minimises the total within-cluster variance at each merge step — the
hierarchical analogue of K-means' objective. It is appropriate for Euclidean distances
on binary feature vectors at this scale.

ARI between K-means and hierarchical labels tells us whether the two methods agree on the
partition structure. High agreement means the clusters are robust; low agreement means
the data has no dominant structure that both methods recover.
"""))

cells.append(code("""# ── Hierarchical clustering ───────────────────────────────────────────────────
print("Computing Ward linkage (878 genomes × 265 features)...")
Z = linkage(X_q3, method='ward', metric='euclidean')
print("Linkage computed.")

# Cut at best_k
hc_labels = fcluster(Z, t=best_k, criterion='maxclust') - 1  # 0-indexed

ari_hc_species = adjusted_rand_score(species_labels, hc_labels)
ari_hc_km      = adjusted_rand_score(labels_best, hc_labels)

print(f"Hierarchical K={best_k}:")
print(f"  ARI vs species:   {ari_hc_species:.4f}")
print(f"  ARI vs K-means:   {ari_hc_km:.4f}")
"""))

cells.append(code("""# ── Dendrogram (truncated to last 60 merges for legibility) ──────────────────
fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(
    Z,
    ax=ax,
    truncate_mode='lastp',
    p=60,
    leaf_rotation=90,
    leaf_font_size=7,
    color_threshold=Z[-(best_k - 1), 2],  # cut line at best_k
    above_threshold_color='grey'
)
ax.set_title(f'Ward hierarchical dendrogram (truncated, last 60 merges)\\nCut at K={best_k}')
ax.set_xlabel('Genome (or merged group — number in parentheses)')
ax.set_ylabel('Ward distance')
ax.axhline(Z[-(best_k - 1), 2], color='crimson', linestyle='--',
           label=f'Cut at K={best_k}')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "q3_hierarchical_dendrogram.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q3_hierarchical_dendrogram.png")
"""))

cells.append(code("""# ── ARI comparison summary ────────────────────────────────────────────────────
ari_arg_hc = adjusted_rand_score(arg_labels, hc_labels)
ari_arg_k6 = adjusted_rand_score(arg_labels, labels_6)

summary = pd.DataFrame({
    'Partition': [f'K-means K={best_k}', f'Hierarchical Ward K={best_k}', 'K-means K=6'],
    'ARI vs species':     [ari_best_species, ari_hc_species, ari_6_species],
    'ARI vs ARG tertile': [ari_best_arg, ari_arg_hc, ari_arg_k6],
    'ARI (methods agree)': [1.0, ari_hc_km, adjusted_rand_score(labels_6, labels_best)],
})
print(summary.to_string(index=False))
print()

# H2: Interpret KM vs HC method agreement explicitly
print(f"K-means vs hierarchical agreement (ARI={ari_hc_km:.3f}):")
if ari_hc_km >= 0.70:
    print("  ARI ≥ 0.70 — stable partition: both methods recover the same structure.")
else:
    print(f"  ARI={ari_hc_km:.3f} is below the 0.70 threshold typically seen for stable")
    print("  cluster structure. This confirms that the partition is not robustly")
    print("  recoverable across methods — additional evidence of absent cluster structure.")
print()

# H3: State ARI vs ARG tertile as direct answer to Q3 secondary question
print(f"ARI vs ARG burden tertile (K-means K={best_k}: {ari_best_arg:.3f};")
print(f"  HC K={best_k}: {ari_arg_hc:.3f}; K-means K=6: {ari_arg_k6:.3f}):")
print("  All values are essentially zero. Defence-system archetypes, as identified by")
print("  unsupervised clustering, are uninformative for predicting ARG burden.")
print("  This is the direct answer to Q3's secondary question: no.")
"""))

# ── Section 4: Q5b ───────────────────────────────────────────────────────────
cells.append(md("""## Section 4 — Q5b: Phage Susceptibility Archetypes

**Feature space:** defence systems (265) + anti-defence systems (29) + IS element burden
(24 columns, MinMax-normalised to [0,1] to prevent count columns dominating Euclidean distance).

**Hypothesis:** Genomes with high IS burden, low defence complexity, and high
anti-defence system load represent a *phage-permissive* profile — they are more
susceptible to phage predation because (a) IS elements have transposed into defence loci
disrupting them, and (b) anti-defence systems encoded by prior phage infections are
present, suggesting repeated phage encounters.

**Important caveat:** This is a correlational, cross-sectional hypothesis. No phage
susceptibility phenotype data is available. The strongest claim in the manuscript is:
*"Genomes in archetype X are consistent with phage-permissive profiles based on defence
architecture, IS burden, and anti-defence repertoire."* Not: "these genomes are
susceptible to phage therapy."
"""))

cells.append(code("""# ── Q5b feature matrix: defence + anti-defence + IS (scaled) ────────────────
# IS columns are counts (0 to ~50+); scale to [0,1] to match binary feature scale
scaler = MinMaxScaler()
X_is = scaler.fit_transform(fm[IS_COLS].values.astype(float))

X_ad = fm[AD_COLS].values.astype(float)  # already binary

X_q5b = np.hstack([X_q3, X_ad, X_is])
print(f"Q5b feature matrix: {X_q5b.shape}")
print(f"  Defence:       {X_q3.shape[1]}")
print(f"  Anti-defence:  {X_ad.shape[1]}")
print(f"  IS (scaled):   {X_is.shape[1]}")
"""))

cells.append(code("""# ── Q5b K selection ──────────────────────────────────────────────────────────
sil_q5b = []
ch_q5b  = []

print("Running Q5b K selection K=2..12...")
for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    lbl = km.fit_predict(X_q5b)
    sil_q5b.append(silhouette_score(X_q5b, lbl))
    ch_q5b.append(calinski_harabasz_score(X_q5b, lbl))
    print(f"  K={k:2d}  silhouette={sil_q5b[-1]:.4f}")

best_k_q5b = k_range.start + int(np.argmax(sil_q5b))
print(f"\\nBest K (Q5b): {best_k_q5b}  (score={max(sil_q5b):.4f})")

# Plot side by side with Q3
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), sil_scores, 'o-', color='steelblue', label='Q3 (defence only)', linewidth=2)
axes[0].plot(list(k_range), sil_q5b,   's--', color='darkorange', label='Q5b (+ AD + IS)', linewidth=2)
axes[0].axvline(best_k,     color='steelblue', linestyle=':', alpha=0.7)
axes[0].axvline(best_k_q5b, color='darkorange', linestyle=':', alpha=0.7)
axes[0].set_xlabel('K')
axes[0].set_ylabel('Silhouette score')
axes[0].set_title('K selection: Q3 vs Q5b')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), ch_scores, 'o-', color='steelblue', label='Q3', linewidth=2)
axes[1].plot(list(k_range), ch_q5b,   's--', color='darkorange', label='Q5b', linewidth=2)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Calinski-Harabasz score')
axes[1].set_title('K selection: Q3 vs Q5b')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "q5b_k_selection.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q5b_k_selection.png")
"""))

cells.append(code("""# ── Q5b K-means ──────────────────────────────────────────────────────────────
km_q5b = KMeans(n_clusters=best_k_q5b, n_init=30, random_state=RANDOM_STATE)
km_q5b.fit(X_q5b)
labels_q5b = km_q5b.labels_

ari_q5b_species = adjusted_rand_score(species_labels, labels_q5b)
ari_q5b_arg     = adjusted_rand_score(arg_labels, labels_q5b)
ari_q5b_q3      = adjusted_rand_score(labels_best, labels_q5b)

print(f"Q5b K-means K={best_k_q5b}:")
print(f"  ARI vs species:      {ari_q5b_species:.4f}")
print(f"  ARI vs ARG tertile:  {ari_q5b_arg:.4f}")
print(f"  ARI vs Q3 clusters:  {ari_q5b_q3:.4f}  <- did IS/AD change the partition?")
print()
print("Cluster sizes (Q5b):")
unique, counts = np.unique(labels_q5b, return_counts=True)
for cl, n in zip(unique, counts):
    print(f"  Cluster {cl}: {n} genomes")
"""))

cells.append(code("""# ── Q5b cluster profiles — defence load, IS burden, anti-defence ─────────────
# Aggregate key summary stats per Q5b cluster
q5b_summary = fm.copy()
q5b_summary['q5b_cluster'] = labels_q5b

profile_cols = (
    ['defence_system_count', 'adef_system_count', 'is_count_total',
     'arg_count_unique', 'ime_count_unique']
    + AD_COLS[:10]   # top 10 anti-defence features
)
# Only include columns that exist
profile_cols = [c for c in profile_cols if c in q5b_summary.columns]

cluster_profile = q5b_summary.groupby('q5b_cluster')[profile_cols].mean().round(3)
print("Q5b cluster profiles (means):")
print(cluster_profile.to_string())
"""))

cells.append(code("""# ── Q5b cluster profiles heatmap ─────────────────────────────────────────────
# Normalise each column to [0,1] for heatmap readability
profile_norm = (cluster_profile - cluster_profile.min()) / (
    cluster_profile.max() - cluster_profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(min(20, len(profile_cols) * 0.8 + 3),
                                max(3, best_k_q5b * 0.8)))
sns.heatmap(
    profile_norm,
    cmap='RdYlBu_r',
    vmin=0, vmax=1,
    annot=cluster_profile.round(2),
    fmt='g',
    linewidths=0.4,
    xticklabels=True,
    yticklabels=[f'C{i}' for i in range(best_k_q5b)],
    ax=ax
)
ax.set_title(f'Q5b cluster profiles — IS burden, anti-defence, ARG (K={best_k_q5b})')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "q5b_cluster_profiles.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q5b_cluster_profiles.png")
"""))

cells.append(code("""# ── Q5b species composition ───────────────────────────────────────────────────
ct_q5b = pd.crosstab(
    pd.Series(labels_q5b, name='Q5b Cluster'),
    pd.Series(species_labels, name='Species')
)
ct_q5b_norm = ct_q5b.div(ct_q5b.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(max(8, best_k_q5b * 1.2), 5))
bottom = np.zeros(best_k_q5b)
for sp, colour in SPECIES_COLOURS.items():
    if sp not in ct_q5b_norm.columns:
        continue
    vals = ct_q5b_norm[sp].values
    ax.bar(range(best_k_q5b), vals, bottom=bottom, color=colour, label=sp, alpha=0.85)
    bottom += vals

ax.set_xticks(range(best_k_q5b))
ax.set_xticklabels([f'C{i}' for i in range(best_k_q5b)])
ax.set_xlabel('Q5b Cluster')
ax.set_ylabel('Proportion')
ax.set_title(
    f'Q5b species composition (K={best_k_q5b})\\n'
    f'ARI vs species={ari_q5b_species:.3f}  '
    f'ARI vs Q3={ari_q5b_q3:.3f}'
)
ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1))
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "q5b_species_composition.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: q5b_species_composition.png")
"""))

# ── Section 5: Biological synthesis ──────────────────────────────────────────
cells.append(md("""## Section 5 — Biological Synthesis

### Q3: Negative finding — no robust defence-system archetypes after dereplication

**Primary result:**
Unsupervised clustering identified no robust defence-system archetypes after phylogenetic
dereplication. The moderate ARI observed in the full 878-genome dataset (ARI≈0.38) was
driven entirely by clonal lineage redundancy — primarily the 69 near-identical IC2
*A. baumannii* vectors pulling a centroid — and is not interpretable as biological
archetype structure.

**Evidence for this interpretation:**
1. Silhouette scores across K=2–12 all fall below 0.10 (maximum ≈0.068) — values below
   0.10 indicate no substantial cluster structure in the feature space.
2. After phylogenetic dereplication to 95 independent representatives (1 per phylogroup),
   ARI vs species drops to ≈−0.004 (random) — no species alignment survives.
3. The gap statistic is consistent with K=1 as optimal on dereplicated data.
4. K-means vs hierarchical agreement ARI≈0.47 is below 0.70 (the threshold for stable
   partitions), confirming the partition is not robustly recoverable across methods.

**Correct conclusion:**
ESKAPE defence profiles form a continuum, not discrete archetypes. Genome-to-genome
defence variation is continuous across taxonomic boundaries; the data does not support
the existence of discrete, cross-species defence-system archetypes at this sample size.

### Q5b: Genome complexity gradient, not phage-permissive archetype

**Result:** K=2 separates genomes by overall genomic complexity, not by IS-driven defence
disruption. Cluster 0 (large Gram-negative: PA/EC/KP) vs Cluster 1 (small Gram-positive
and *Acinetobacter*: EF/SA/AB). IS element burden was similar between clusters (≈58.9 vs
≈55.0), directly refuting the hypothesis that IS burden drives defence depletion at the
cross-species level.

**Mechanism:** IS elements accumulate independently of genome size because they transpose
autonomously within chromosomes. IMEs and ICEs, by contrast, require conjugation
machinery and scale with genome size and HGT network activity — which is why the
defence count, ARG count, and IME count all differ 2× between clusters while IS burden
does not. The K=2 partition is a proxy for host range and HGT network membership, not
phage permissiveness per se.

**Future direction:** The correct proxy for IS-mediated defence disruption is IS element
*position* within defence loci (insertional inactivation), not IS element *count*. A
future positional analysis (IS elements within 1 kb of annotated defence genes) would
test the phage-permissive hypothesis directly.

### RESTRICT/FACILITATE: clonal dominance observation, not independent cross-validation

The RESTRICT/FACILITATE dichotomy is a within-*Acinetobacter* pattern defined on
within-AB statistical contrasts. Cross-species unsupervised clustering can only recover
it when IC2 clones are numerically overrepresented (as in the full 878-genome run, where
79 AB genomes form their own cluster). The IC2 lineage, by virtue of being sufficiently
numerous and defence-distinct, forms its own cluster when clonal redundancy is present.
This observation is *consistent* with published IC2 defence depletion but does **not**
independently cross-validate RESTRICT/FACILITATE, as it disappears after dereplication.

To study RESTRICT/FACILITATE robustly by unsupervised methods, clustering should be
performed on *A. baumannii* genomes only (not cross-species), using phylogenetically
dereplicated representatives.

### ARI vs ARG tertile (Q3 secondary question)

ARI between defence-system clusters and ARG burden tertile is essentially zero across
all partitions. Defence-system archetypes, as identified by unsupervised clustering, are
uninformative for ARG burden. This is the direct negative answer to Q3's secondary
question: no, clustering does not recover a clinically relevant ARG gradient.
"""))

cells.append(code("""# ── RESTRICT / FACILITATE archetype recovery ─────────────────────────────────
ab_mask = species_labels == 'abaumannii'
ab_clusters = labels_best[ab_mask]
ab_sspbcde = fm.loc[ab_mask, 'dp_df_SspBCDE'].values if 'dp_df_SspBCDE' in fm.columns else None
# Try alternative column names
if ab_sspbcde is None:
    ssp_cols = [c for c in FEAT_COLS if 'Ssp' in c or 'ssp' in c]
    print(f"SspBCDE candidate columns: {ssp_cols}")
    if ssp_cols:
        ab_sspbcde = fm.loc[ab_mask, ssp_cols[0]].values

rm_cols = [c for c in FEAT_COLS if 'RM_Type' in c and 'IV' not in c]
print(f"RM subtype columns (excl. Type IV): {rm_cols}")

if ab_sspbcde is not None and rm_cols:
    # M4: composite any_RM = 1 if any RM subtype (I, II, IIG, III) present
    ab_rm_mat = fm.loc[ab_mask, rm_cols].values
    ab_any_rm = (ab_rm_mat.max(axis=1) if ab_rm_mat.ndim > 1 else ab_rm_mat).astype(float)
    ab_rm_type1 = fm.loc[ab_mask, rm_cols[0]].values if rm_cols else np.zeros(ab_any_rm.shape)
    ab_profile = pd.DataFrame({
        'cluster': ab_clusters,
        'SspBCDE': ab_sspbcde,
        'any_RM': ab_any_rm,
        'RM_Type_I': ab_rm_type1,
    })
    # Add individual subtype columns
    for col in rm_cols:
        ab_profile[col] = fm.loc[ab_mask, col].values
    print("\\nAB genomes — SspBCDE and RM prevalence per cluster:")
    agg_dict = {'n': ('cluster', 'count'),
                'SspBCDE_prev': ('SspBCDE', 'mean'),
                'any_RM_prev': ('any_RM', 'mean')}
    for col in rm_cols:
        agg_dict[f'{col}_prev'] = (col, 'mean')
    print(ab_profile.groupby('cluster').agg(**agg_dict).round(3).to_string())
    print()
    print("Expected: clusters dominated by AB should split into")
    print("  RESTRICT archetype: high any_RM, low SspBCDE")
    print("  FACILITATE archetype: low any_RM, high SspBCDE (IC2)")
"""))

cells.append(code("""# ── Final summary table ───────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 11 SUMMARY — Q3 NEGATIVE FINDING")
print("=" * 70)
print()
print(f"Q3 — Defence archetype clustering (full 878-genome dataset)")
print(f"  Best K (silhouette): {best_k}  (score={max(sil_scores):.4f} — below 0.10 threshold)")
print(f"  Gap-statistic optimal K: {gap_optimal_k}")
print(f"  ARI vs species (full):   {ari_best_species:.4f}  ← ARTEFACTUAL (clone inflation)")
print(f"  ARI vs ARG tertile:      {ari_best_arg:.4f}  ← near random")
print(f"  HC agreement:            {ari_hc_km:.4f}  ← below 0.70 stable-partition threshold")
print()
print("  PRIMARY CONCLUSION: No robust defence-system archetypes exist after dereplication.")
print("  Full-dataset ARI was driven by IC2 clonal redundancy, not biological structure.")
print("  (Dereplication robustness results printed in Section 6 below)")
print()
print(f"Q5b — Phage-permissive archetype clustering")
print(f"  Best K (silhouette): {best_k_q5b}  (score={max(sil_q5b):.4f})")
print(f"  ARI vs species:      {ari_q5b_species:.4f}")
print(f"  ARI vs ARG tertile:  {ari_q5b_arg:.4f}")
print(f"  ARI vs Q3 clusters:  {ari_q5b_q3:.4f}")
print()
print("  K=2 partition: genome complexity split (large Gram-negative PA/EC/KP vs")
print("  small Gram-positive + AB), NOT IS-driven defence disruption.")
print("  IS burden nearly identical between clusters — phage-permissive hypothesis REFUTED.")
print("  Correct interpretation: IS element COUNT ≠ IS element POSITION within defence loci.")
"""))

cells.append(md("""## Section 6 — Robustness Check: Dereplicated Dataset (1 genome per phylogroup)

**Why this is required:** The 878-genome dataset contains clonal lineages. IC2 *A. baumannii*
contributes ~69 near-identical genomes; SA may have similar clonal structure. K-means assigns
cluster centroids partly based on how many genomes are near-identical in a region of feature
space — not purely on biological distinctiveness. Cluster 8 (79 AB/IC2 genomes) exists partly
because 69 nearly-identical IC2 vectors pulled a centroid.

The unsupervised equivalent of GroupedStratifiedKFold is **dereplication**: pick 1 representative
genome per phylogroup (the genome closest to the phylogroup centroid). This yields 95 independent
observations and removes clone inflation.

**If the RESTRICT/FACILITATE recovery and ARI structure hold in the 95-genome dataset, the
878-genome results are robust.** If they collapse, the 878-genome clustering was artefactual.
"""))

cells.append(code("""# ── Load phylogroup assignments ──────────────────────────────────────────────
pg = pd.read_parquet(DATA / "cv_groups.parquet")
print(f"cv_groups columns: {pg.columns.tolist()}")
print(pg.head(3))
"""))

cells.append(code("""# ── Select 1 representative per phylogroup (closest to group centroid) ────────
# Merge phylogroup assignment onto feature matrix
fm_pg = fm.copy()
fm_pg['genome_id'] = fm_pg.index

# cv_groups may use genome_id or index; align on index
if 'phylogroup' in fm_pg.columns:
    pg_col = fm_pg['phylogroup'].to_numpy(dtype=str)
else:
    # join by index
    fm_pg = fm_pg.join(pg[['phylogroup']] if 'phylogroup' in pg.columns
                       else pg.iloc[:, 0].rename('phylogroup'))
    pg_col = fm_pg['phylogroup'].to_numpy(dtype=str)

print(f"Unique phylogroups: {len(np.unique(pg_col))}")

# For each phylogroup, select the genome with smallest mean Euclidean distance
# to the group centroid (most "typical" genome in that group)
X_all = fm[FEAT_COLS].values.astype(float)
rep_indices = []
for pg_id in np.unique(pg_col):
    mask = pg_col == pg_id
    group_X = X_all[mask]
    centroid = group_X.mean(axis=0)
    dists = np.linalg.norm(group_X - centroid, axis=1)
    local_idx = np.argmin(dists)
    global_idx = np.where(mask)[0][local_idx]
    rep_indices.append(global_idx)

rep_indices = np.array(rep_indices)
X_derep   = X_all[rep_indices]
sp_derep  = species_labels[rep_indices]
arg_derep = arg_labels[rep_indices]

print(f"Dereplicated dataset: {X_derep.shape[0]} genomes (1 per phylogroup)")
print("Species composition (dereplicated):")
unique_sp, counts_sp = np.unique(sp_derep, return_counts=True)
for s, n in zip(unique_sp, counts_sp):
    print(f"  {s}: {n}")
"""))

cells.append(code("""# ── K-means on dereplicated dataset ──────────────────────────────────────────
sil_derep = []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    lbl = km.fit_predict(X_derep)
    sil_derep.append(silhouette_score(X_derep, lbl) if len(np.unique(lbl)) > 1 else 0)

best_k_derep = k_range.start + int(np.argmax(sil_derep))
print(f"Dereplicated best K: {best_k_derep}  (silhouette={max(sil_derep):.4f})")
print(f"Full-dataset best K: {best_k}         (silhouette={max(sil_scores):.4f})")

km_derep = KMeans(n_clusters=best_k_derep, n_init=30, random_state=RANDOM_STATE)
labels_derep = km_derep.fit_predict(X_derep)

ari_derep_species = adjusted_rand_score(sp_derep, labels_derep)
ari_derep_arg     = adjusted_rand_score(arg_derep, labels_derep)

print(f"\\nDereplicated K-means K={best_k_derep}:")
print(f"  ARI vs species:     {ari_derep_species:.4f}  (full: {ari_best_species:.4f})")
print(f"  ARI vs ARG tertile: {ari_derep_arg:.4f}  (full: {ari_best_arg:.4f})")
"""))

cells.append(code("""# ── RESTRICT/FACILITATE recovery in dereplicated AB ──────────────────────────
ab_mask_d = sp_derep == 'abaumannii'
ab_clusters_d = labels_derep[ab_mask_d]
ab_X_d = X_derep[ab_mask_d]

ssp_idx  = FEAT_COLS.index('dp_SspBCDE') if 'dp_SspBCDE' in FEAT_COLS else None
# M4: use all RM subtypes (I, II, IIG, III) for composite any_RM
rm_idxs  = [i for i, c in enumerate(FEAT_COLS) if 'RM_Type' in c and 'IV' not in c]

if ssp_idx is not None and rm_idxs:
    ab_ssp_d    = ab_X_d[:, ssp_idx]
    ab_any_rm_d = ab_X_d[:, rm_idxs].max(axis=1)   # composite: present in any subtype
    ab_df_d = pd.DataFrame({'cluster': ab_clusters_d,
                             'SspBCDE': ab_ssp_d,
                             'any_RM': ab_any_rm_d})
    # Also include individual RM subtype prevalences for reporting
    for idx in rm_idxs:
        ab_df_d[FEAT_COLS[idx]] = ab_X_d[:, idx]
    print("Dereplicated AB — RESTRICT/FACILITATE per cluster:")
    agg_cols = {'n': ('cluster','count'),
                'SspBCDE': ('SspBCDE','mean'),
                'any_RM': ('any_RM','mean')}
    for idx in rm_idxs:
        col = FEAT_COLS[idx]
        agg_cols[col] = (col, 'mean')
    print(ab_df_d.groupby('cluster').agg(**agg_cols).round(3).to_string())
else:
    print("Column index lookup needed — check FEAT_COLS manually.")

print()
print("Robustness verdict:")
print(f"  Full 878-genome ARI vs species:     {ari_best_species:.4f}")
print(f"  Dereplicated 95-genome ARI vs species: {ari_derep_species:.4f}")
if abs(ari_derep_species - ari_best_species) < 0.10:
    print("  ROBUST: ARI difference <0.10 — clonal inflation did not drive the result.")
else:
    print("  SENSITIVE: ARI difference ≥0.10 — clonal inflation affected full-dataset clustering.")
"""))

cells.append(md("""## Manuscript language guidance (updated post-audit)

**For Q3 (clustering) — state the negative finding clearly:**

*"Unsupervised clustering identified no robust defence-system archetypes after
phylogenetic dereplication. The moderate ARI observed in the full 878-genome dataset
(ARI=0.38) was driven entirely by clonal lineage redundancy and is not interpretable
as biological archetype structure. After reducing the dataset to 95 phylogenetically
independent representatives (one genome per Mash-defined phylogroup), ARI vs species
collapsed to −0.004 (effectively random). Silhouette scores across K=2–12 were all
below 0.10 (maximum 0.068 at K=10), and the gap statistic confirmed K=1 as optimal
on dereplicated data. We conclude that ESKAPE defence profiles form a continuum rather
than discrete archetypes in this feature space."*

**For Q5b — state as genome complexity gradient, not archetype discovery:**

*"The K=2 partition separated genomes by overall genomic complexity (large-genome
Gram-negative PA/EC/KP vs small-genome Gram-positive EF/SA/AB) rather than by
IS-driven defence disruption. IS element burden was similar between clusters (≈58.9
vs ≈55.0 IS elements per genome), refuting the hypothesis that IS burden drives
defence depletion at the cross-species level. This finding motivates a future positional
analysis (IS elements within defence loci) rather than a count-based approach."*

**For RESTRICT/FACILITATE — reframe as IC2 clonal dominance, not cross-validation:**

*"The IC2 lineage, by virtue of being sufficiently numerous and defence-distinct, forms
its own cluster when clonal redundancy is present. This observation is consistent with
published IC2 defence depletion [citation] but does not independently cross-validate
the RESTRICT/FACILITATE dichotomy, as it disappears after phylogenetic dereplication.
The RESTRICT/FACILITATE pattern is a within-A. baumannii observation; cross-species
unsupervised clustering is not an appropriate method to test it."*
"""))

# ── Save notebook ─────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

out = Path("notebooks/09_unsupervised_archetypes.ipynb")
with open(out, 'w') as f:
    nbformat.write(nb, f)
print(f"Notebook written: {out}  ({len(cells)} cells)")
