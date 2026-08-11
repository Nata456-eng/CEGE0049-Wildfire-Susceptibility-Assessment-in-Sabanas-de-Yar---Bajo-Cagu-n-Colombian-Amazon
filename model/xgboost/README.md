# XGBoost

**Notebooks:** [`train_xgboost.ipynb`](train_xgboost.ipynb) (ratio 1:1),
[`xgboost_v2.ipynb`](xgboost_v2.ipynb) (ratio 2:1)
**Dataset:** rebuilt from `data/model_dataset/pixel_year_full.csv` with the same logic
and 3 km exclusion buffer as [`tuning/v3/`](../../tuning/v3/README.md) — so results are
comparable, figure for figure, against the `1:1` and `2:1` LR/RF rows already reported
there.

## What these notebooks do
This is the third model, added following the same protocol used for LR and RF:
`RandomizedSearchCV` (150 of 300 candidate configurations sampled) with `GroupKFold(10)`
spatial-block CV, `scoring='average_precision'`, restricted to `year<=2019` (same
temporal-leakage fix as [`tuning/v2/`](../../tuning/v2/README.md)), then full evaluation
(spatial block CV + temporal hold-out ≥2020) on the complete dataset.

Two versions exist because — like LR and RF — XGBoost was tested at both pseudo-absence
ratios from the [`tuning/v3/`](../../tuning/v3/README.md) sensitivity analysis:
`train_xgboost.ipynb` uses ratio **1:1**, `xgboost_v2.ipynb` uses ratio **2:1**.

## Results — ratio 1:1 (identical dataset to LR/RF in `tuning/v3`)

Best hyperparameters (`RandomizedSearchCV`, 150/300 configurations, tuning `year<=2019`):
`n_estimators=100`, `max_depth=30`, `colsample_bytree=0.444` (≈ `max_features=4`),
`min_child_weight=30`. Tuning PR-AUC (spatial CV, `year<=2019`): **0.867**.

| Model | AUC spatial | PR-AUC spatial | F1 spatial | AUC temporal | PR-AUC temporal | F1 temporal |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.823 | 0.754 | 0.810 | 0.747 | 0.663 |
| Random Forest | 0.878 | **0.866** | 0.790 | 0.817 | 0.704 | 0.684 |
| **XGBoost** | 0.871 | 0.864 | 0.789 | 0.811 | **0.717** | **0.693** |

(full table in [`xgboost_vs_lr_rf_comparison.csv`](xgboost_vs_lr_rf_comparison.csv))

Random Forest still wins on spatial PR-AUC, but **XGBoost has the best temporal
PR-AUC and F1** of the three models — it generalises to new years slightly better than
RF here, while RF remains the strongest at generalising to new places.

## Results — ratio 2:1 (identical dataset to LR/RF in `tuning/v3`)

Best hyperparameters: same as above (`n_estimators=100`, `max_depth=30`,
`colsample_bytree=0.444`, `min_child_weight=30`). Tuning PR-AUC (spatial CV,
`year<=2019`): **0.777**.

| Model | AUC spatial | PR-AUC spatial | F1 spatial | AUC temporal | PR-AUC temporal | F1 temporal |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.83 | 0.70 | 0.61 | 0.81 | **0.62** | 0.56 |
| Random Forest | **0.88** | **0.78** | 0.68 | **0.82** | 0.57 | 0.56 |
| XGBoost | 0.86 | 0.74 | **0.69** | 0.79 | 0.54 | 0.55 |

(full table in [`xgboost_vs_lr_rf_comparison_2to1.csv`](xgboost_vs_lr_rf_comparison_2to1.csv))

Same pattern seen throughout [`tuning/v3/`](../../tuning/v3/README.md): every model does
markedly worse at ratio 2:1 than at ratio 1:1, confirming the sampling ratio matters more
than which model is used.

## Where this fits in the project
This is the **first** XGBoost round, run right after `tuning/v3` established that 1:1 is
the best pseudo-absence ratio — it exists to confirm XGBoost doesn't beat Random Forest
under that same 3 km-buffer dataset design. A **later, separate** round happens in
[`tuning/v4/`](../../tuning/v4/README.md), which reruns all three models (LR, RF,
XGBoost) together on a new 7–15 km annulus-sampled dataset (a stronger correction for
spatial autocorrelation) and is the comparison that actually selected the final deployed
model (Random Forest) used in [`outputs/probability_map/`](../../outputs/probability_map/README.md).
