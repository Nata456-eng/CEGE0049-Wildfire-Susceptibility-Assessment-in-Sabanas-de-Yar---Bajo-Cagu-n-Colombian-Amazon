# Random Forest

**Notebook:** [`train_rf.ipynb`](train_rf.ipynb)
**Dataset:** [`data/model_dataset/model_dataset.csv`](../../data/model_dataset/model_dataset.csv) (same frozen dataset as the logistic regression)

## What this notebook does
Trains a Random Forest with reasonable hyperparameters (not yet optimised) and evaluates
it with the same protocol as the logistic regression, so the comparison is fair:
1. **Spatial block CV** (5 folds, ~0.25° ≈ 27 km blocks) — generalisation to new places.
2. **Temporal split** (train ≤2019, test ≥2020) — generalisation to new years.
3. **Feature importance** by impurity (fast to compute, but biased toward continuous
   variables — for a fairer measure, see SHAP in [`outputs/shap/`](../../outputs/shap/README.md)).

## Model configuration
```python
RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=5,
                        class_weight='balanced', random_state=42, n_jobs=-1)
```
Trees don't need feature scaling (unlike logistic regression).

## Results (before hyperparameter tuning)

| Validation | AUC-ROC | PR-AUC | F1 |
|---|---|---|---|
| Spatial block CV | 0.875 ± 0.022 | 0.769 ± 0.067 | 0.709 ± 0.060 |
| Temporal hold-out (≥2020) | 0.817 | 0.563 | 0.549 |

### Feature importance (impurity-based — biased, see note above)
| Predictor | Importance |
|---|---|
| ndvi | 0.222 |
| wind_ms | 0.201 |
| vpd_kPa | 0.162 |
| temp_C | 0.109 |
| dist_parks | 0.082 |
| oni | 0.075 |
| dist_roads | 0.068 |
| dist_coca | 0.046 |
| dist_mosaic | 0.035 |

**Note:** this impurity-based importance ranks `dist_mosaic` last — but in the EDA (see
[`eda/`](../../eda/)) it has the strongest bivariate relationship with fire. This
suggests the relationship is NON-linear (a threshold effect: "close to the agricultural
frontier = high risk"), which impurity doesn't measure well. SHAP confirms this — see
[`outputs/shap/README.md`](../../outputs/shap/README.md) for the actual SHAP-based
ranking (computed on the later, final v4 model).

## The tuned version
Optimal hyperparameters are searched for in
[`tuning/v1/tune_rf.ipynb`](../../tuning/v1/tune_rf.ipynb) with `GridSearchCV`, which
also has the final table comparing tuned LR vs. tuned RF.

> **Note:** this baseline (and its v1 tuning) uses the frozen 2:1 dataset above. The
> final deployed model ([`tuning/v4/`](../../tuning/v4/README.md),
> [`outputs/probability_map/`](../../outputs/probability_map/README.md)) instead uses a
> 1:1 ring-sampled (7–15 km) dataset, found to perform substantially better in
> [`tuning/v3/`](../../tuning/v3/README.md)'s sensitivity analysis.
