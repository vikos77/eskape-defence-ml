"""
Bootstrap CI utilities for phylogenetically-structured genomic data.

Two strategies:
- cluster_bootstrap_ci: resample phylogroups (correct for grouped CV data, n_groups >= 15)
- genome_bootstrap_ci:  resample individual (y_true, y_pred) pairs (use when n_groups < 15)

Both return (lower_95, upper_95).
"""

import numpy as np


MIN_CLUSTERS_FOR_CLUSTER_BOOTSTRAP = 15


def cluster_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_boot=2000, seed=42):
    """
    Cluster bootstrap 95% CI: resample phylogroups with replacement, take all
    genomes in each drawn group. Consistent with GroupedStratifiedKFold's
    effective sample size = number of phylogroups.

    Parameters
    ----------
    y_true  : array-like, true labels
    y_pred  : array-like, predicted labels or scores
    groups  : array-like, phylogroup label per genome (same index as y_true/y_pred)
    metric_fn : callable(y_true, y_pred) -> float
    n_boot  : int, number of bootstrap iterations
    seed    : int, random seed

    Returns
    -------
    (lo, hi) : float tuple, 2.5th and 97.5th percentiles of bootstrap distribution
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < MIN_CLUSTERS_FOR_CLUSTER_BOOTSTRAP:
        raise ValueError(
            f"Only {n_groups} phylogroups — cluster bootstrap unreliable below "
            f"{MIN_CLUSTERS_FOR_CLUSTER_BOOTSTRAP}. Use genome_bootstrap_ci with caveat."
        )

    rng = np.random.RandomState(seed)
    boot_scores = []

    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        boot_scores.append(metric_fn(y_true[idx], y_pred[idx]))

    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return float(lo), float(hi)


def genome_bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, seed=42):
    """
    Genome-level bootstrap 95% CI: resample (y_true, y_pred) pairs with replacement.
    Anticonservative for clonal-structured data; use only when n_phylogroups < 15.

    Parameters
    ----------
    y_true    : array-like
    y_pred    : array-like
    metric_fn : callable(y_true, y_pred) -> float
    n_boot    : int
    seed      : int

    Returns
    -------
    (lo, hi) : float tuple
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)  # one index array; both arrays use same idx
        boot_scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_auto(y_true, y_pred, groups, metric_fn, n_boot=2000, seed=42):
    """
    Select bootstrap strategy based on phylogroup count.
    Returns (lo, hi, method) where method is 'cluster' or 'genome'.
    """
    n_groups = len(np.unique(groups))
    if n_groups >= MIN_CLUSTERS_FOR_CLUSTER_BOOTSTRAP:
        lo, hi = cluster_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_boot, seed)
        return lo, hi, "cluster"
    else:
        lo, hi = genome_bootstrap_ci(y_true, y_pred, metric_fn, n_boot, seed)
        return lo, hi, "genome"
