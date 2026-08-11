# Tuning — v3 (pseudo-absence ratio sensitivity analysis)

**Notebook:** [`tune_v3.ipynb`](tune_v3.ipynb)

## What a sensitivity analysis is and why it was done
Since the first notebook (`model/logistic_regression/baseline_comparison.ipynb`), the
dataset was built with a fixed ratio of **2 pseudo-absences per observed fire** (2:1),
justified by general literature (Barbet-Massin et al. 2012) but never empirically tested
against alternatives in this project. A sensitivity analysis tests reasonable variants
of that design decision (here: the sampling ratio) and measures whether the model's
result changes — instead of assuming the original choice was optimal.

Three ratios were tested: **1:1** (balanced classes), **2:1** (the one used in v1/v2),
and **3:1** (more pseudo-absences, more imbalanced classes). The limit on which ratios
are feasible depends on how many eligible pseudo-absences remain after the 3 km
exclusion buffer: **78,878** available, i.e. up to 1:38 — 1:1, 2:1, and 3:1 use very
little of that margin, so there's no data-availability problem.

## How the best ratio was selected
For each ratio, **exactly the same `tuning/v2` pipeline** was repeated: `GridSearchCV`
(10-fold spatial-block CV, `scoring='average_precision'`) restricted to `year<=2019`
(no temporal leakage), with the same hyperparameter grids from v1/v2, followed by full
evaluation (10-fold spatial block CV + temporal hold-out `>=2020`, with confusion
matrix). The winning ratio is the one that achieves the best **average spatial PR-AUC on
Random Forest** — the selection metric used throughout the project, and the model with
the most consistent performance in v1 and v2.

**Consistency check:** this notebook's 2:1 ratio reproduced, figure for figure, the
results already reported in [`v2/README.md`](../v2/README.md) (same `random_state=42`
seed, same sample size) — confirming v3's pipeline is methodologically identical to
v2's, and that the only factor being changed is the ratio.

## Results by ratio (LR and RF, tuning `year<=2019`, full-year evaluation)

| Ratio | Model | Rows (presences) | AUC spatial | PR-AUC spatial | F1 spatial | AUC temporal | PR-AUC temporal | F1 temporal |
|---|---|---|---|---|---|---|---|---|
| **1:1** | Logistic Regression | 4,154 (2,077) | 0.846 ± 0.032 | **0.823 ± 0.060** | 0.754 ± 0.079 | 0.810 | **0.747** | 0.663 |
| **1:1** | Random Forest | 4,154 (2,077) | 0.878 ± 0.027 | **0.866 ± 0.036** | 0.790 ± 0.047 | 0.817 | **0.704** | 0.684 |
| 2:1 | Logistic Regression | 6,231 (2,077) | 0.832 ± 0.064 | 0.703 ± 0.105 | 0.613 ± 0.123 | 0.808 | 0.617 | 0.557 |
| 2:1 | Random Forest | 6,231 (2,077) | 0.877 ± 0.041 | 0.779 ± 0.067 | 0.680 ± 0.074 | 0.819 | 0.569 | 0.560 |
| 3:1 | Logistic Regression | 8,308 (2,077) | 0.831 ± 0.051 | 0.618 ± 0.113 | 0.504 ± 0.167 | 0.806 | 0.542 | 0.486 |
| 3:1 | Random Forest | 8,308 (2,077) | 0.873 ± 0.032 | 0.702 ± 0.067 | 0.603 ± 0.098 | 0.817 | 0.492 | 0.493 |

Full results (with confusion matrices) in
[`tuning_v3_sensitivity_metrics.csv`](tuning_v3_sensitivity_metrics.csv). The winning
ratio's data is isolated in [`tuning_v3_final_metrics.csv`](tuning_v3_final_metrics.csv).

## Best hyperparameters by ratio

| Ratio | Logistic Regression | Random Forest |
|---|---|---|
| 1:1 | `C=0.01`, `penalty='l2'` | `n_estimators=200`, `max_features='sqrt'`, `min_samples_leaf=3`, `max_depth=None` |
| 2:1 | `C=0.01`, `penalty='l1'` | `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=5`, `max_depth=None` |
| 3:1 | `C=0.01`, `penalty='l1'` | `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=5`, `max_depth=None` |

## Optimal ratio: **1:1**

Both **Random Forest** and **Logistic Regression** agree that **1:1** is the best ratio,
with a large and consistent difference across every metric:

- RF: spatial PR-AUC goes from **0.779** (2:1, the ratio used so far) to **0.866** (1:1)
  — a **+0.087** improvement. Temporal PR-AUC goes from 0.569 to 0.704 (**+0.135**).
- LR: spatial PR-AUC goes from 0.703 (2:1) to **0.823** (1:1) — **+0.120**. Temporal
  PR-AUC goes from 0.617 to **0.747** (**+0.130**).
- AUC-ROC barely changes between ratios (0.83–0.88 in every case) because AUC-ROC is
  largely insensitive to class imbalance — which is exactly why PR-AUC is the right
  metric for this comparison, as used throughout the project.
- F1 also improves noticeably with 1:1 (RF: 0.680→0.790; LR: 0.613→0.754), because with
  balanced classes the default decision threshold (0.5) stops being biased toward
  predicting "not burned".

**Why does this happen?** With 2:1 and 3:1, the model sees more "not burned" examples
than "burned" ones, and even though `class_weight` wasn't used here (unlike the baseline
notebook), class imbalance still hurts PR-AUC because there are more potential false
positives to discriminate among. With 1:1 the model has an equal amount of positive and
negative signal to learn the decision boundary, which in this dataset (few presences,
~2,077) outweighs the theoretical advantage of "more variety of pseudo-absences" that
higher ratios offer.

## Improvement over v1 and v2

- **v1** tuned hyperparameters but never questioned the sampling ratio (2:1, fixed since
  the first notebook), and also had temporal leakage in the tuning.
- **v2** fixed the temporal leakage, but still used the 2:1 ratio without comparing it
  against alternatives.
- **v3** keeps v2's fix and adds a new search dimension (the pseudo-absence ratio),
  finding that **the ratio matters far more than hyperparameter tuning itself**:
  switching from 2:1 to 1:1 improves RF's spatial PR-AUC by +0.087, while the temporal
  leakage fix (v1→v2) only changed RF's spatial PR-AUC by -0.004. This suggests that, for
  this dataset, the sampling design decision has more impact on performance than the
  fine-grained choice of hyperparameters.

## Recommendation
Adopt ratio **1:1** as the new "official" dataset for future training (replacing the 2:1
used in `data/model_dataset/model_dataset.csv`), documenting this change in
`model/logistic_regression/`'s README and regenerating that CSV if this recommendation is
adopted — it wasn't overwritten automatically here, to avoid breaking the reproducibility
of v1/v2, which explicitly depend on the frozen 2:1 dataset. (In practice, the eventual
final model in [`tuning/v4/`](../v4/README.md) goes further still — it doesn't just
switch to 1:1, it also replaces the 3 km exclusion buffer with a 7–15 km annulus.)
