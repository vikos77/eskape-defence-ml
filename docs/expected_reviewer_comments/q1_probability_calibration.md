# Expected reviewer comment: probability calibration of boosted classifiers

## Anticipated comment

> "Balanced accuracy and F1 summarise label-level predictions. Can you show that the
> predicted class probabilities are also trustworthy, particularly for any downstream
> risk-scoring use?"

## Pre-emptive response

Calibration was assessed for XGBoost and Random Forest on fold 0 (n=832 genomes) using
one-vs-rest reliability diagrams and Brier scores across three representative species
(A. baumannii, K. pneumoniae, S. aureus).

**Results:**

| Model    | Mean Brier score (3 classes) |
|----------|------------------------------|
| XGBoost  | 0.0543                       |
| RF       | 0.0645                       |

Null baseline (predicting 1/6 for all six classes): ~0.139.

Both models score well below the null baseline. XGBoost is marginally better calibrated
than RF by Brier score, consistent with the known tendency of RF to produce
overconfident probability estimates near 0 and 1 (Niculescu-Mizil & Caruana, 2005).
Reliability diagrams showed both models tracking close to the diagonal across the three
classes examined, with no systematic overconfidence or underconfidence pattern.

**Why this is not reported as a primary result:**
Q1 is a classification task where the label is the output of interest, not the
probability. Calibration becomes the primary concern for Q2 (ARG burden risk scoring)
if predicted probabilities are used clinically. For Q2 we report AUROC as the
effect-size metric, which is rank-based and does not depend on calibration.
If a reviewer requests formal calibration analysis for Q2 predictions, it can be
produced from the saved per-genome probability arrays in `results/`.

## Analysis location

The full calibration code (reliability diagram generation, Brier score computation)
was removed from NB07 as a primary section but is recoverable from git history:
commit prior to `NB07 S10: remove calibration section; move to reviewer comments`.
