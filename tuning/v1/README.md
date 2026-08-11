# Tuning — v1

**Notebook:** [`tune_rf.ipynb`](tune_rf.ipynb)

## What this folder is
This is where the **hyperparameter** search lives (a model's internal "settings", like
how many trees a Random Forest has or how much regularisation logistic regression uses).
Each tuning attempt is a new version (`v1`, `v2`, ...) so the history of what was tried
and what it produced isn't lost.

`v1` is the first (and so far only, at this point) tuning round. It uses `GridSearchCV`
with spatial-block cross-validation (10 folds) to choose the best hyperparameters for
**both** models (Logistic Regression and Random Forest), then evaluates both tuned
versions with the usual protocol (spatial block CV + temporal split). That's why this
notebook also contains **the final comparison table** for the two models.

## Why `GridSearchCV`?
It's an exhaustive search: every combination of a list of candidate values is tried
(e.g. `n_estimators = [100, 300, 500]`) and the combination with the best average
`PR-AUC` in cross-validation is chosen. The temporal test set (≥2020) is never used
during this search — that set stays "untouched" for the final report, so there's no
information leakage (*data leakage*).

**Important technical detail:** with `GroupKFold`, the CV object and the groups must be
passed like this:
```python
GridSearchCV(estimator, param_grid, cv=GroupKFold(n_splits=10), scoring='average_precision')
grid.fit(X, y, groups=g_tune)   # groups goes in .fit(), not in cv=
```
Passing `cv=gkf.split(...)` (the generator) instead of the object breaks with a
`PicklingError`.

## Search grids used

| Model | Hyperparameter | Candidate values |
|---|---|---|
| Logistic Regression | `C` | `0.01, 0.1, 1.0, 10.0` |
| | `penalty` | `l1, l2` |
| | `solver` | `liblinear` |
| Random Forest | `n_estimators` | `200, 300, 500` |
| | `max_features` | `sqrt, 0.5` |
| | `min_samples_leaf` | `3, 5, 10, 20` |
| | `max_depth` | `None, 10, 20` |

- LR: 4×2×1 = **8 candidates** × 10 folds = **80 fits**.
- RF: 3×2×4×3 = **72 candidates** × 10 folds = **720 fits** (9 folds train / 1 validates per fit).
- Selection metric: **PR-AUC** (`scoring='average_precision'`) averaged across the 10 folds per candidate; the candidate with the best average PR-AUC wins.

## Best hyperparameters found

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `C=0.1`, `penalty='l1'`, `solver='liblinear'` |
| Random Forest | `n_estimators=500`, `max_features=0.5`, `min_samples_leaf=3`, `max_depth=None` |

## Final results (tuned models)

| Model | AUC (spatial) | PR-AUC (spatial) | AUC (temporal) | PR-AUC (temporal) |
|---|---|---|---|---|
| Logistic Regression (tuned) | 0.835 ± 0.060 | 0.705 ± 0.102 | 0.807 | 0.618 |
| Random Forest (tuned) | 0.877 ± 0.040 | 0.783 ± 0.066 | 0.816 | 0.558 |

**Interpretation (in simple terms):**
- Random Forest wins on **spatial** validation (better at predicting new places) and is
  more stable (lower standard deviation across blocks).
- On **temporal** validation both models are essentially tied — fire between years isn't
  strongly determined by these variables, consistent with ENSO (`oni`) having a weak
  correlation with fire.

**Central finding:** the tuned logistic regression's L1 regularisation set the
coefficient of **`oni`** (the ENSO index) to **exactly zero** — the linear model found no
monotonic (increasing or decreasing) relationship between ENSO and fire worth keeping.
In the Random Forest, however, `oni` does NOT end up last: it sits in the middle of the
importance table (rank 6 of 9), above `dist_roads`, `dist_coca`, and `dist_mosaic`. This
suggests the relationship between ENSO and fire is real but **non-linear** — for
example, both extreme El Niño and La Niña events could increase risk, something a linear
coefficient simply can't capture but a decision tree can (because it can split the `oni`
range into several independent segments).

Separately, `dist_mosaic` does keep a small but non-zero coefficient in the LR
(`-0.110`), and remains the **least important** variable in the Random Forest (last by
impurity, 0.034). This no longer supports the original hypothesis that L1 was
"switching it off" — it looks more like its information overlaps with other correlated
human-pressure variables (`dist_roads`, `dist_coca`), and the tree prefers splitting on
those first. The SHAP interpretation (see
[`outputs/shap/README.md`](../../outputs/shap/README.md)) should focus on **`oni`** as
the clearest case of a non-linear relationship, without ruling out `dist_mosaic`.

## Note on information leakage
The hyperparameters were chosen using the same spatial blocks used to report spatial
performance → the spatial number is slightly optimistic. **Temporal** performance
(≥2020) was never used during tuning, so it's the unbiased estimate. The effect of
tuning was checked to be small (spatial PR-AUC +0.012–0.015, temporal ±0.003) — a nested
CV would remove this bias entirely but costs ~10× more compute for a minimal gain.
