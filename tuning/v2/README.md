# Tuning — v2 (temporal-leakage fix)

**Notebook:** [`tune_v2.ipynb`](tune_v2.ipynb)

## What this version fixes
In `v1` (see [`../v1/README.md`](../v1/README.md)), the `GridSearchCV` that chose the LR
and RF hyperparameters saw the **full** dataset (2001-2024), including years `>=2020`.
Those years are exactly the set used afterwards for the **temporal hold-out** — meaning
that although the final model was never *trained* on 2020-2024 data before reporting the
temporal number, those years **did influence which hyperparameters were chosen**,
because they appeared in the training folds during the spatial-block search. That's a
subtle temporal leak into the model-selection process, though not into the final model's
training itself.

`v2` fixes this by restricting the `GridSearchCV` (LR and RF) to rows with `year<=2019`
only. The hyperparameter grid is **identical** to `v1` to isolate the effect of this one
change. The final evaluation (spatial block CV + temporal split) still uses the full
dataset, same as `v1`, because that's where performance is **reported**, not where the
model is **selected**.

## Search grids (identical to v1)

| Model | Hyperparameter | Candidate values |
|---|---|---|
| Logistic Regression | `C` | `0.01, 0.1, 1.0, 10.0` |
| | `penalty` | `l1, l2` |
| | `solver` | `liblinear` |
| Random Forest | `n_estimators` | `200, 300, 500` |
| | `max_features` | `sqrt, 0.5` |
| | `min_samples_leaf` | `3, 5, 10, 20` |
| | `max_depth` | `None, 10, 20` |

- LR: 8 candidates × 10 folds = **80 fits** (data: `year<=2019` only, 5,153 rows).
- RF: 72 candidates × 10 folds = **720 fits** (data: `year<=2019` only, 5,153 rows).
- Selection metric: **PR-AUC** (`scoring='average_precision'`).

## Best hyperparameters found (without temporal leakage)

| Model | Hyperparameters | Changed vs. v1? |
|---|---|---|
| Logistic Regression | `C=0.01`, `penalty='l1'`, `solver='liblinear'` | Yes — `C` went from `0.1` to `0.01` (more regularisation) |
| Random Forest | `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=5`, `max_depth=None` | Yes — `n_estimators` 500→300, `max_features` 0.5→`sqrt`, `min_samples_leaf` 3→5 |

## Final results (tuned models, leakage-free)

| Model | AUC (spatial) | PR-AUC (spatial) | F1 (spatial) | AUC (temporal) | PR-AUC (temporal) | F1 (temporal) |
|---|---|---|---|---|---|---|
| Logistic Regression (v2) | 0.832 ± 0.064 | 0.703 ± 0.105 | 0.613 ± 0.123 | 0.808 | 0.617 | 0.557 |
| Random Forest (v2) | 0.877 ± 0.041 | 0.779 ± 0.067 | 0.680 ± 0.074 | 0.819 | 0.569 | 0.560 |

**Confusion matrices:**

| Model | Validation | TN | FP | FN | TP |
|---|---|---|---|---|---|
| LR (v2) | Spatial (sum of 10 folds) | 3656 | 498 | 891 | 1186 |
| LR (v2) | Temporal (≥2020) | 630 | 199 | 76 | 173 |
| RF (v2) | Spatial (sum of 10 folds) | 3736 | 418 | 762 | 1315 |
| RF (v2) | Temporal (≥2020) | 628 | 201 | 74 | 175 |

Results also saved to [`../../outputs/metrics/tuning_v2_metrics.csv`](../../outputs/metrics/tuning_v2_metrics.csv).

## v1 (leaked) vs. v2 (fixed) comparison

| Model | Metric | v1 | v2 | Difference |
|---|---|---|---|---|
| LR | AUC spatial | 0.835 ± 0.060 | 0.832 ± 0.064 | -0.003 |
| LR | PR-AUC spatial | 0.705 ± 0.102 | 0.703 ± 0.105 | -0.002 |
| LR | AUC temporal | 0.807 | 0.808 | +0.001 |
| LR | PR-AUC temporal | 0.618 | 0.617 | -0.001 |
| RF | AUC spatial | 0.877 ± 0.040 | 0.877 ± 0.041 | ~0.000 |
| RF | PR-AUC spatial | 0.783 ± 0.066 | 0.779 ± 0.067 | -0.004 |
| RF | AUC temporal | 0.816 | 0.819 | +0.003 |
| RF | PR-AUC temporal | 0.558 | 0.569 | +0.011 |

**Conclusion:** the temporal leak in `v1`'s hyperparameter search did change **which
hyperparameters** were chosen (especially for RF: fewer trees, a more conservative
`max_features`, larger leaves — a slightly more regularised model when it only sees data
≤2019), but the effect on the **final reported metrics** is minimal (differences of
0.001 to 0.011 across all metrics, within the noise of a single temporal fold). This is
the expected conclusion: the leak affected model *selection*, not its final *training*,
so the practical impact is small. Even so, `v2` is the methodologically correct version
and should be preferred as the official reference going forward.

Additional note: v2's hyperparameters match almost exactly what was documented in the
repository before it was discovered that v1's `GridSearchCV` cell was entirely missing —
it's likely the original (lost) notebook already restricted tuning to `year<=2019`, and
`v2` is simply reconstructing that original, correct methodology.
