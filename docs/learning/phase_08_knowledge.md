# Phase 8 — Random Forest: Concepts, Code, and Results

**Date:** 2026-05-21
**Notebook:** `notebooks/06_random_forest.ipynb`
**Key results file:** `results/q1_rf_results.parquet`, `results/q1_rf_feature_importance.csv`

---

## 1. What is a Random Forest and why does it exist?

### The problem with a single decision tree

A decision tree splits data by asking binary questions ("does this genome have dp_SspBCDE = 1?") at each node, building a flowchart down to leaf predictions.

Three problems:
- **High variance:** Change 5 training genomes → completely different tree structure
- **Instability on sparse binary data:** Seizes on whichever feature gives any split gain near the root
- **Unreliable feature importance:** Features that appear near the root dominate, partly by chance

### Bagging (Bootstrap Aggregating — Breiman 1996)

Draw B bootstrap samples from training data — same size, **with replacement** (~63% unique, 37% left out).
Train one tree per sample. Aggregate by majority vote.

Why it helps: Trees trained on different subsets have partially uncorrelated errors. Averaging uncorrelated noisy estimators reduces variance as 1/B.

**Out-of-Bag (OOB) set:** Each genome is excluded from ~37% of trees. Evaluate it on those trees → free internal accuracy estimate. No extra CV needed.

### Random feature subsampling (Breiman 2001) — the "Random" in Random Forest

If one feature is very strong (e.g., dp_SspBCDE for AB), every tree splits on it near the root regardless of bootstrap sample → trees are correlated → averaging correlated errors gives little variance reduction.

**Fix:** At each split, randomly sample only m features (default: m = √p ≈ 16 for our 265 features). Best split chosen only among those m candidates.

Forces trees to use secondary features → less correlation between trees → better variance reduction.

**Random Forest = bagging + random feature subsampling at each split.**

---

## 2. Feature importance — three methods

### Gini importance (Mean Decrease in Impurity, MDI)

At each split, impurity (Gini coefficient) decreases. Sum impurity decrease attributed to each feature across all nodes and trees. Normalise.

**Bias:** Over-rates features with many possible thresholds (count features vs binary features). For mostly-binary feature matrices, bias is small but present.

### Permutation importance

Evaluate model on held-out set. Shuffle one feature at random (breaking its real relationship). Drop in accuracy = importance.

**Advantages over Gini:**
1. Computed on held-out data — cannot reward training-set memorisation
2. Not biased by cardinality
3. Model-agnostic

**Limitation:** If features are correlated, shuffling one alone doesn't matter (model uses correlated surrogate) → underestimates correlated features.

### SHAP (SHapley Additive exPlanations — Lundberg & Lee 2017)

For each genome and each feature: how much did this feature push the prediction away from the dataset average?

Additivity guarantee: SHAP contributions sum exactly to (predicted probability − baseline).

TreeSHAP (Lundberg et al. 2020): exact, fast, exploits tree structure. For multiclass RF in SHAP ≥0.46, returns 3D array: `(n_samples, n_features, n_classes)`. Index as `shap_values[:, :, class_idx]` for per-class plots.

---

## 3. Hyperparameters explained

| Parameter | What it controls | Best value (this project) |
|-----------|-----------------|--------------------------|
| n_estimators | Number of trees. More = lower variance, diminishing returns after ~200 | 100 |
| max_depth | Max tree depth. None = fully grown = memorises training | 20 |
| min_samples_leaf | Min genomes per leaf. Larger = smoother, less overfit | 1 |
| max_features | Features sampled per split. Lower = more diverse trees | sqrt (≈16) |
| class_weight | Weight minority classes upward | balanced |

**max_depth=20 beat max_depth=None.** Fully-grown trees with 720 training genomes overfitted. Capping depth forces generalisation. This is visible in the OOB vs grouped-CV gap (0.933 vs 0.878 = 0.055 gap attributable to phylogenetic leakage in OOB).

---

## 4. Key numerical results — this project

### Q1: Species classification (grouped CV, filtered features)

| Model | BA | 95% CI |
|-------|----|--------|
| LR (Phase 7) | 0.837 | [0.813–0.859] |
| RF default | 0.876 | [0.850–0.893] |
| RF best (tuned) | **0.878** | [0.859–0.898] |
| OOB (RF best, all data) | 0.932 | — |
| Null baseline | 0.130 | — |

**Decision:** Delta = +0.041 (> 0.02 threshold). CI barely overlaps (0.0002 units). Conservative pre-registered criterion: "No meaningful improvement" — report both, prefer LR for parsimony. Manuscript will note consistent +0.04 with near-zero CI overlap.

### Per-class performance (Q1, RF best)

| Species | Recall | Precision | Key note |
|---------|--------|-----------|----------|
| S. aureus | 0.993 | 1.000 | Easiest — SA defence profile is highly distinctive |
| E. faecium | 0.953 | 0.745 | High recall but many false positives from AB |
| P. aeruginosa | 0.893 | 0.876 | Strong |
| E. cloacae | 0.849 | 0.810 | Strong |
| K. pneumoniae | 0.856 | 0.950 | Strong |
| **A. baumannii** | **0.700** | 0.938 | **Worst — IC2 defence-depauperate clones confused with EF** |

**Why AB→EF confusion:** IC2 AB clones have sparse defence repertoires (published RESTRICT phenotype). E. faecium also has low defence density (small 2.8 Mb Gram-positive genome). In 265-feature space, sparse-defence AB looks like sparse-defence EF. This is a biological signal, not a modelling failure.

### Feature importance comparison (top features)

| Feature | Gini rank | Perm rank | SHAP rank | Interpretation |
|---------|-----------|-----------|-----------|---------------|
| dp_df_Mok_Hok_Sok | 5 | 5 | **1** | Consistent — trustworthy |
| dp_padloc_PDC-S13 | 3 | 2 | **2** | Consistent — trustworthy |
| dp_df_FS_Sma | 1 | 12 | 3 | Gini inflated — Perm/SHAP more reliable |
| dp_df_Abi2 | 2 | 1 | 4 | Consistent — trustworthy |
| dp_df_MazEF | 16 | **264** | 6 | Correlated feature — not independently important |
| dp_CAS_I-F | 21 | 3 | — | Gini missed — actually very important |
| dp_SspBCDE | 6 | >10 | **10** | Key published AB signal — ML independently recovers it |

**Rule of thumb:** Trust features consistent across all three methods. Extreme Gini–Permutation disagreements signal cardinality bias (Gini inflates) or feature correlation (Permutation deflates correlated features).

**dp_df_MazEF interpretation:** SHAP rank 6 but Permutation rank 264. MazEF (toxin-antitoxin) travels with other defence systems (correlated). Shuffling it alone doesn't hurt the model because the model switches to correlated surrogates. SHAP averages over all feature orderings and detects the conditional contribution.

**dp_SspBCDE at SHAP rank 10:** Independently recovered by ML. The published Fisher's exact test identified this as a key AB/IC2 marker. RF SHAP cross-validates that finding without using any species labels for feature selection.

### Q2: ARG burden prediction

| Species | RF BA | LR BA | Delta | Interpretation |
|---------|-------|-------|-------|----------------|
| E. cloacae | 0.508 | 0.752 | **-0.244** | LR wins strongly — n too small for RF |
| K. pneumoniae | 0.546 | 0.719 | **-0.173** | LR wins strongly |
| E. faecium | 0.681 | 0.512 | +0.169 | RF wins — non-linear interactions |
| P. aeruginosa | 0.621 | 0.645 | -0.024 | Similar |
| S. aureus | 0.475 | 0.470 | ≈ same | Near chance — chromosomal resistance |
| A. baumannii | 0.500 | 0.473 | ≈ same | RESTRICT phenotype — low ARG = high defence |

**Key lesson:** RF is NOT universally better than LR. With ~130–150 genomes per species and a binary target, RF's flexibility becomes overfitting. LR's L2 regularisation wins at small n. This is one of the most important practical lessons in applied ML: model complexity must match sample size.

---

## 5. Technical bugs fixed this session

### Bug 1: PyArrow-backed arrays in parquet loading
**Problem:** `pd.read_parquet()` with PyArrow backend returns `ArrowStringArray` for string columns. `sklearn` GridSearchCV/CV splitters require plain numpy arrays for indexing.
**Fix:** `fm["species"].to_numpy(dtype=str)` and `fm["phylogroup"].to_numpy(dtype=str)`.
**Error message:** `TypeError: only integer scalar arrays can be converted to a scalar index`

### Bug 2: SHAP 0.51.0 API change
**Problem:** SHAP ≥0.46 returns multiclass SHAP values as a 3D array `(n_samples, n_features, n_classes)` instead of the old list of 2D arrays `[array_class_0, array_class_1, ...]`.
**Fix:** Index as `shap_values[:, :, cls_idx]` per class. Global importance: `np.abs(shap_values).mean(axis=0).mean(axis=1)`.

### Bug 3: Nested parallelism in GridSearchCV
**Problem:** `n_jobs=-1` in both `GridSearchCV` and `RandomForestClassifier` causes joblib nested parallelism errors on macOS.
**Fix:** Set `n_jobs=1` in the RF base during grid search; keep `n_jobs=-1` in GridSearchCV only.

### Bug 4: nbconvert --output path doubling
**Problem:** `nbconvert --output notebooks/06.ipynb` writes to `notebooks/notebooks/06.ipynb`.
**Fix:** Use `--inplace` flag instead.

---

## 6. Comprehension check questions (with answers)

**Q: OOB score was 0.933 but grouped-CV was 0.878. Why the gap?**
A: OOB holds out individual genomes randomly. Closely related genomes (same phylogroup) appear in both training (other trees) and OOB evaluation. The model partially "sees" their relatives. Grouped CV removes entire phylogroups from the test fold — the model has never seen any member of that clade. The gap (0.055) is the cost of correct phylogenetic independence.

**Q: best max_depth was 20, not None. What does that tell you?**
A: Unlimited trees (None) overfit the training set — leaves containing single genomes, training accuracy 1.0. Capping at depth 20 forces the model to generalise beyond individual genome patterns. The fact that max_depth=20 was selected confirms that overfitting risk was real at this sample size.

**Q: RF is dramatically worse than LR for Q2 EC and KP. Why?**
A: Per-species Q2 has only ~130–150 genomes and a binary target. RF has many parameters (effectively n_estimators × tree_depth parameters) relative to sample size. LR with L2 regularisation penalises complexity directly. When n is small, regularised linear models typically beat flexible non-parametric ones. EF is the exception — RF wins there, possibly because non-linear interactions between defence systems matter for EF's ARG burden in ways they don't for EC/KP.
