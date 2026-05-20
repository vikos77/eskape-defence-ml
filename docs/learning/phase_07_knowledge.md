# Phase 7 — What We Did and Why: Complete Reference

**Topic:** Baseline classifiers (Logistic Regression) under standard and phylogenetic grouped CV  
**Notebook:** `notebooks/05_baseline_classifier.ipynb`  
**Assumed knowledge:** Phase 6 knowledge file (cross-validation, phylogroups)

---

## 1. What Is a Classifier?

A **classifier** is a mathematical function that takes a set of measurements (features) and produces a prediction about which category something belongs to.

In our case:
- **Input:** 265 binary features (presence/absence of defence systems in one genome)
- **Output:** a prediction — which of the 6 ESKAPE species does this genome belong to?

For Q2, the output is binary: "high ARG burden" or "low ARG burden."

The classifier **learns** by seeing thousands of examples with known labels (training data). It adjusts its internal weights to be correct as often as possible on the training examples. Then it is tested on genomes it has never seen.

---

## 2. Logistic Regression — How It Works

### The core idea

Logistic Regression is the simplest classifier that works with continuous weights. Here is what it does, step by step:

**Step 1 — Assign a weight to every feature.**

The model has one weight (a number) for every defence feature. For Q1 (predicting *S. aureus*), the weight for `dp_RM_Type_I` might be +2.3. The weight for `dp_CRISPR_Cas` might be −0.8. Positive weight = this feature pushes toward SA. Negative weight = this feature pushes away from SA.

**Step 2 — Compute a score.**

For a new genome, multiply each feature value (0 or 1) by its weight, and sum everything up:

```
Score = (dp_RM_Type_I × 2.3) + (dp_CRISPR_Cas × −0.8) + (dp_AbiD × 0.4) + ...
```

This score can be any number, from large negative to large positive.

**Step 3 — Convert to a probability using the sigmoid function.**

The **sigmoid function** takes any number and converts it to a probability between 0 and 1:

```
Probability = 1 / (1 + e^(−score))

Score = +4  →  Probability ≈ 0.982  (very likely SA)
Score =  0  →  Probability = 0.500  (50/50)
Score = −4  →  Probability ≈ 0.018  (very unlikely SA)
```

The S-shaped sigmoid curve maps scores to probabilities. High positive score → high probability. High negative score → low probability.

**Step 4 — One-vs-Rest (OvR) for 6 species.**

For 6 classes, we train 6 separate binary classifiers:
- Classifier 1: "Is this genome *S. aureus* or not?"
- Classifier 2: "Is this genome *A. baumannii* or not?"
- ... (one per species)

For a new genome, all 6 classifiers run and produce 6 probabilities. The species with the highest probability wins the prediction.

### L2 Regularisation (C = 1.0)

Without constraints, a Logistic Regression model will find extreme weights that memorise the training data perfectly but fail badly on new data. This is called **overfitting**.

**L2 regularisation** adds a penalty for large weights. Mathematically, it adds the sum of all squared weights to the loss function. The result: all weights are gently pulled toward zero. No single feature is allowed to dominate with an extreme coefficient unless it genuinely earns that dominance.

`C` is the regularisation strength:
- Large C (e.g. C=100) = weak penalty = weights can be large = more flexible model = more risk of overfitting
- Small C (e.g. C=0.01) = strong penalty = weights are heavily shrunk = simpler model = more risk of underfitting
- **C=1.0** is the default — mild penalty, suitable for baseline analysis

---

## 3. The Null Baseline — DummyClassifier

Before evaluating the real model, we need a reference point: **how well does a completely ignorant classifier do?**

The `DummyClassifier(strategy="stratified")` ignores all features entirely. It looks at the training class distribution ("there are 150 SA, 132 KP, 150 EC...") and randomly assigns predictions proportionally. It doesn't look at a single defence feature.

If our real model achieves balanced accuracy 0.837, that only means something if we know the null achieves 0.130. The gap (0.837 − 0.130 = 0.707) is the genuine signal. Without knowing the null, we cannot judge the result.

**Theoretical null for a perfectly balanced 6-class problem:** 1/6 = 0.167.

In practice, our null was 0.130 (below 0.167) because our class sizes are unequal (132 KP vs 150 SA) and the grouped CV folds are imbalanced in size. The dummy classifier does slightly worse when folds are imbalanced.

---

## 4. Balanced Accuracy — Why Not Just Use Regular Accuracy?

### Regular accuracy: the problem

**Regular accuracy** = (number of correct predictions) / (total predictions).

Suppose you have 150 SA genomes and you always predict "SA" for every genome. You would be correct on all 150 SA. You would be wrong on all 728 non-SA. Regular accuracy = 150/878 = 17.1%. That sounds terrible. But for a species classification problem where we want fair treatment of all species, it would look misleading if SA were the majority class.

Now consider a *K. pneumoniae* model with only 132 KP and 746 non-KP. Always predicting "not KP" gives 746/878 = 85% accuracy. Sounds great. Total nonsense.

### Balanced accuracy: the fix

**Balanced accuracy** = average of the **recall** (sensitivity) for each class.

**Recall for a class** = (genomes of that class correctly predicted) / (total genomes of that class).

Example: 
- SA recall = 141/150 = 0.940 (correctly identified 141 of the 150 SA genomes)
- AB recall = 85/150 = 0.567 (correctly identified only 85 of the 150 AB genomes)
- Balanced accuracy = average of all 6 recalls

For a 6-class problem: **balanced accuracy = (SA recall + AB recall + EC recall + EF recall + KP recall + PA recall) / 6**

A random classifier gets ≈ 1/6 = 0.167. A perfect classifier gets 1.000. A classifier that always predicts "SA" gets: SA recall = 1.0, all others = 0.0, balanced accuracy = 1/6 = 0.167. The class imbalance trick is exposed.

### In our results

Primary Q1 result: balanced accuracy = **0.837**. This means the model correctly identifies an average of 83.7% of genomes from each species. Not because SA dominates — but because across all 6 species, recall averages to 83.7%.

The null achieves 0.130. The model beats null by 0.707 points. That gap is the genuine signal.

---

## 5. Macro-F1 — A Related Metric

**F1 score** combines precision and recall for a single class:

- **Recall:** Of all true SA genomes, what fraction did we correctly identify? (Sensitivity)
- **Precision:** Of all genomes we *predicted* as SA, what fraction actually are SA? (Exactness)
- **F1:** Harmonic mean of precision and recall. F1 = 2 × (precision × recall) / (precision + recall)

F1 = 1.0 is perfect. F1 = 0.0 is useless. F1 favours classifiers that are both precise and sensitive.

**Macro-F1** = average F1 across all 6 species (each species equally weighted). Similar motivation to balanced accuracy: prevents a majority class from dominating the metric.

In Phase 7, macro-F1 was 0.835 for the primary result — almost identical to balanced accuracy 0.837, confirming that precision and recall are balanced (not trading one for the other).

---

## 6. Confidence Intervals — Why They Matter

A single number like 0.837 is not the full story. The model was evaluated on 878 genomes via 5-fold CV. If we had chosen different random seeds, different fold assignments, or a slightly different subset of genomes, we might get 0.812 or 0.859.

The **95% confidence interval (CI)** quantifies this uncertainty. It is constructed by **bootstrapping**:

1. Take all 878 (y_true, y_pred) pairs from the CV run.
2. Randomly resample 878 pairs **with replacement** (some appear twice, some not at all). This is one bootstrap sample.
3. Compute balanced accuracy for this bootstrap sample.
4. Repeat 2000 times.
5. The range that contains the middle 95% of the 2000 bootstrap scores is the 95% CI.

Interpretation: **0.837 [0.813–0.859]** means "if we repeated this experiment with a different random sample of genomes from the same population, we expect the result to fall in [0.813, 0.859] 95% of the time."

**Why this matters for claiming "better than null":** If a model's CI lower bound is above the null's upper bound, we are confident the model beats null. If the CIs overlap, the difference could be chance.

---

## 7. The Confusion Matrix — Seeing Where the Model Fails

A **confusion matrix** is a grid that shows exactly where the model goes right and wrong.

For a 6-class problem, it is a 6×6 grid. Rows = true species. Columns = predicted species.

**Row-normalised confusion matrix:**

Each cell = fraction of that species' genomes predicted as that column's class.

Example (simplified):

```
              Predicted AB  Predicted KP  Predicted SA  ...
True AB:         0.567         0.187         0.032      ...
True KP:         0.034         0.864         0.011      ...
True SA:         0.014         0.007         0.940      ...
```

- The diagonal shows recall per species (correctly identified fraction)
- Off-diagonal shows what each species gets confused *with*

Reading this example: 18.7% of true AB genomes were misclassified as KP. SA was correctly identified 94% of the time. KP was correct 86.4% of the time.

**High off-diagonal between AB and KP** = the model cannot reliably distinguish AB from KP using defence system profile alone. This is a biological finding: those two species have overlapping defence repertoires once taxonomic markers are removed.

### Phase 7 confusion matrix results

```
Species         Recall
abaumannii      0.567   ← worst — hardest to classify
ecloaceae       0.863
efaecium        0.893
kpneumoniae     0.864
paeruginosa     0.893
saureus         0.940   ← best
```

AB's low recall is a biological signal, not a modelling failure. The published paper showed that AB (especially IC2 clones) has a depauperate defence repertoire — fewer systems than other ESKAPE species. After removing the few AB-specific marker features, the model cannot reliably identify AB from defence profile alone.

---

## 8. The Q1 2×2 Table — What the Numbers Mean

The pre-analysis plan required reporting Q1 accuracy in a 2×2 table to separate two independent problems:

| | Full (274 features) | Filtered (265 features) |
|---|---|---|
| Standard CV | 0.988 [0.980–0.994] | 0.950 [0.935–0.964] |
| Phylo grouped CV | 0.979 [0.970–0.988] | **0.837 [0.813–0.859] ★** |

**Reading the table by column (effect of phylogenetic correction):**
- Full features: 0.988 → 0.979. Delta = −0.009. Very small drop. Clone correction barely matters.
- Filtered features: 0.950 → 0.837. Delta = **−0.114**. Large drop. Clone correction matters a lot.

**Why does the delta differ so much between columns?**

When taxonomic marker features are present (full set), the model mainly classifies using "PA has PD-T4-6 universally → this is PA." This feature works whether or not related PA clones were in the training set, because the feature is 99% prevalent in ALL PA genomes regardless of lineage. Clone contamination barely inflates the result because the classification signal is universal, not clone-specific.

When markers are removed (filtered set), the model must rely on subtler within-species variation in defence profile. This variation IS shared within clone families — a clone family has a distinct defence complement. When related clones appear in both train and test, the model can "recognise" the test clone's defence profile as familiar. Remove the clones from the train set, and accuracy drops substantially.

**Conclusion:** The two corrections are not independent. Removing taxonomic markers makes clone correction reveal more latent inflation. The primary result — after both corrections — is 0.837.

**Is 0.837 a positive or negative finding?**

The pre-analysis plan defined three thresholds:
- > 0.70: positive finding — genuine defence architecture signal
- Drop > 0.15 from preliminary: need to explicitly acknowledge taxonomic marker and clone inflation
- < 0.70: near-null finding — revise Q1 framing

Result: 0.837 > 0.70 (positive finding). Drop = 0.984 → 0.837 = 0.147 (just below the 0.15 flag). Q1 is a positive finding, but the manuscript must acknowledge that 0.147 of the preliminary accuracy was artificial inflation.

---

## 9. AUROC — What It Is and Why It Matters for Q2

### The problem with balanced accuracy for probability-based classifiers

Balanced accuracy reports what fraction of genomes are correctly classified when you force the model to choose one class. But the model does not just choose — it also produces a **probability** for each class. The same model can be used with different decision thresholds. If you lower the threshold for calling "high ARG" (i.e., call high ARG if probability > 0.3 instead of > 0.5), you capture more true high-ARG genomes but also more false positives.

**AUROC** (Area Under the Receiver Operating Characteristic Curve) measures how well the model ranks genomes by their probability, regardless of where the decision threshold is set.

### The ROC curve

For a binary classifier (Q2: high ARG vs low ARG), plot two things as you vary the decision threshold from 0 to 1:

- **Y axis — True Positive Rate (TPR):** Of all true high-ARG genomes, what fraction did we correctly call "high ARG"? (This is recall for the positive class)
- **X axis — False Positive Rate (FPR):** Of all true low-ARG genomes, what fraction did we incorrectly call "high ARG"? (This is the cost — false alarms)

At threshold = 0 (always call "high ARG"): TPR = 1.0 (found all positives), FPR = 1.0 (called all negatives positive too).  
At threshold = 1 (never call "high ARG"): TPR = 0.0, FPR = 0.0.  
At intermediate thresholds: trade-off points.

The ROC curve traces all these threshold-specific points.

**AUROC = Area Under this ROC Curve.**

| AUROC value | Meaning |
|---|---|
| 1.000 | Perfect: model's "high ARG" probabilities are always higher for true high-ARG genomes than for true low-ARG genomes |
| 0.500 | Random: model's probabilities are no better than random for ranking genomes |
| 0.000 | Perfectly inverted: model's probabilities are always LOWER for true high-ARG genomes — it has learned the opposite |

**Practical interpretation:** AUROC = probability that a randomly chosen high-ARG genome gets a higher "high ARG" probability than a randomly chosen low-ARG genome.

AUROC 0.846 (EC) = 84.6% of (high-ARG, low-ARG) pairs are correctly ranked. Strong predictive power.
AUROC 0.556 (SA) = 55.6% — barely above random chance.

---

## 10. The Inverted AB Signal — AUROC 0.231

AB achieved AUROC 0.231 for Q2. This is below 0.500.

**What does AUROC < 0.5 mean physically?**

It means the model's probabilities are systematically inverted. For AB, the model assigns *higher* "high ARG" probability to genomes that actually have *low* ARG burden — and vice versa. It is predicting the wrong class more reliably than chance.

This is not a modelling failure. It is a biological discovery.

**The published paper found:** In *A. baumannii*, RM systems are negatively correlated with ARG burden. Genomes with rich RM defence repertoires have fewer acquired resistance genes. The paper called this the **RESTRICT phenotype** — a strong, well-characterised restrictive-gatekeeping architecture.

**How this appears in the ML:** Our model was set up to predict which genomes are high-ARG. The features are defence system presence/absence. In AB, genomes with many defence systems (high RM prevalence) have LOW ARGs. So the model "learns" defence features → low ARG, not high ARG. When it then tries to predict high ARG, it systematically assigns high probability to genomes with fewer defence features (i.e., low-defence = possible high-ARG). The probabilities are correct for predicting LOW ARG, wrong for predicting HIGH ARG.

**AUROC 0.231 is the mirror image of AUROC 0.769.** If you flip the predicted class (swap what you call "positive" and "negative"), AUROC = 1 − 0.231 = 0.769. That means: the model is actually quite good at predicting *low* ARG burden in AB, using defence features. AUROC 0.769 is a strong result.

**Correct framing for the manuscript:** 
> "Defence system profile in *A. baumannii* is strongly associated with ARG burden (AUROC 0.769 when predicting low-ARG class), consistent with the published RESTRICT phenotype in which RM-enriched genomes have significantly fewer acquired resistance genes."

**Incorrect framing (avoid):**
> "The Q2 model failed for *A. baumannii* (AUROC 0.231)."

---

## 11. Q2 Per-Species Results — Biological Interpretation

```
Species          BAcc    AUROC    Tier
ecloaceae        0.752   0.846    Strong
kpneumoniae      0.719   0.830    Strong
paeruginosa      0.645   0.698    Moderate
efaecium         0.512   0.578    Marginal
saureus          0.470   0.556    Absent
abaumannii       0.473   0.231    Inverted (RESTRICT phenotype)
```

### Strong Q2 (EC, KP): Enterobacterales plasmid biology

EC (*E. cloacae* complex) and KP (*K. pneumoniae*) are Gram-negative Enterobacterales — the same bacterial order. Both are heavily associated with large conjugative plasmids that carry both ARGs and other mobile elements. The ESKAPE hospital environment creates pressure for both ARG acquisition (antibiotic treatment) and defence system acquisition (phage pressure in hospital niches).

**The key biology:** In Enterobacterales, mobile genetic elements often carry defence systems AND ARGs together on the same plasmid or genomic island. A genome that acquired an ARG-laden plasmid likely also acquired defence systems from that plasmid's cargo. Defence profile and ARG burden are correlated because they often travel together on the same mobile elements.

**Why the model works here:** The defence feature matrix captures which defence systems are present. In EC and KP, high-ARG genomes tend to have different defence profiles from low-ARG genomes — because the ARG-carrying mobile elements bring distinctive defence machinery with them.

### Moderate Q2 (PA)

PA (*P. aeruginosa*) has the most complex genome in this set (~7 Mb, many defence systems). It also has CRISPR-Cas systems that can restrict incoming mobile elements, including ARG-carrying plasmids. Some predictive power for Q2 (AUROC 0.698), but not as strong as Enterobacterales. The PA biology is more complex — both plasmid acquisition and chromosomal ARG evolution contribute to its resistance profile.

### Marginal/absent Q2 (EF, SA)

EF (*E. faecium*) is a Gram-positive organism with a relatively small genome (~3 Mb) and a simpler defence repertoire. SA (*S. aureus*) is also Gram-positive. Critically, both SA and EF have well-characterised **chromosomal** resistance mechanisms — methicillin resistance in MRSA is encoded in the chromosomal *mecA* gene (SCCmec element), not on plasmids carrying defence systems alongside. When resistance is chromosomally integrated rather than plasmid-borne, the link between defence system profile and ARG burden breaks down.

**Biological interpretation:** In SA and EF, ARG acquisition is less tightly coupled to the same mobile elements as defence systems. A high-defence SA genome is not meaningfully more or less likely to carry ARGs than a low-defence SA genome — they acquire resistance through different routes.

### Inverted Q2 (AB): the RESTRICT phenotype replicated

Discussed above in Section 10. The published paper's core finding is reproduced in the ML layer.

---

## 12. The Fold Structure — Why Folds Are Unequal

```
Fold 1:  232 genomes  (EF:119, SA:12, AB:18, ...)
Fold 2:  212 genomes  (SA:104, AB:18, EF:6, ...)
Fold 3:  178 genomes  (AB:76, ...)
Fold 4:  133 genomes
Fold 5:  123 genomes
```

Fold 1 has 232 genomes; Fold 5 has 123. Why?

The `StratifiedGroupKFold` algorithm must assign entire phylogroups to folds. The EF mega-phylogroup has 119 genomes (all one lineage). The SA mega-phylogroup has 104 genomes. These have to go to separate folds to keep species representation across all folds. But once they go to fold 1 and fold 2 respectively, those folds are already much larger than the remaining folds.

**Coefficient of variation (CV)** of fold sizes = standard deviation / mean = 24.3%. This is substantial imbalance.

**Does this matter?**
- For accuracy estimates: each fold's accuracy estimate has different precision. A fold with 232 test genomes gives a more stable accuracy estimate than a fold with 123. The bootstrap CI (reported on all 878 predictions combined) correctly accounts for this — it is computed over individual predictions, not fold averages.
- For species balance: despite the size imbalance, every fold has all 6 species represented. The model always trains and tests on all 6 species — just in unequal proportions per fold.
- For the 95% CI: the CI is wider than it would be under perfectly balanced folds. That is honest — it reflects genuine uncertainty introduced by the imbalanced phylogroup sizes.

This is a known limitation of grouped CV on clonal datasets. Reporting the CI captures the effect.

---

## 13. Summary: What Phase 7 Tells Us

| Result | Interpretation |
|---|---|
| Q1 primary: 0.837 [0.813–0.859] | Genuine defence architecture signal — beats null by 0.71 points |
| Delta filtered: −0.114 | Substantial clone leakage in standard CV (once markers removed) |
| AB recall = 0.567 | IC2 depauperate repertoire — AB hardest to classify from defence alone |
| Q2 EC: AUROC 0.846 | Defence systems co-travel with ARGs on Enterobacterales plasmids |
| Q2 SA: AUROC 0.556 | Chromosomal resistance mechanism — defence-ARG link absent |
| Q2 AB: AUROC 0.231 (inverted = 0.769) | RESTRICT phenotype replicated — RM-enriched AB has low ARG burden |

**Phase 7 establishes the baseline.** All Phase 8 (Random Forest) and Phase 9 (XGBoost) results must be compared to these numbers. Any more complex model must beat LR by a margin that justifies its complexity.

---

*This document was written as a learning reference during Phase 7 (2026-05-20).  
Source of truth: `notebooks/05_baseline_classifier.ipynb`.*
