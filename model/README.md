# `model/` — Machine Learning models

Each model lives in **its own folder**. All three read exactly the same dataset
([`data/model_dataset/model_dataset.csv`](../data/model_dataset/model_dataset.csv)) and
are evaluated with the same metrics, so the comparison between them is fair.

## Why compare several models?

This is a classic methodological question: *why use a "black box" Machine Learning
model instead of something simple and transparent?* The answer: start with the simplest
possible model (logistic regression) as a **baseline**, and only justify using a more
complex model (Random Forest, XGBoost) if it actually improves performance on the SAME
data with the SAME validation.

## Subfolders

### `logistic_regression/` — the baseline model
- [`baseline_comparison.ipynb`](logistic_regression/baseline_comparison.ipynb): this
  notebook does TWO things in order:
  1. **Builds** the frozen dataset (`data/model_dataset/model_dataset.csv`) from Google
     Earth Engine — this only runs once; if the file already exists it's loaded directly
     and the slow step is skipped.
  2. **Trains and evaluates** a simple logistic regression on that dataset.
- See [`logistic_regression/README.md`](logistic_regression/README.md) for full detail.

### `random_forest/` — the first "real" ML model
- [`train_rf.ipynb`](random_forest/train_rf.ipynb): trains a Random Forest with
  reasonable hyperparameters (not the "best" ones — that happens in
  [`tuning/`](../tuning/)).
- See [`random_forest/README.md`](random_forest/README.md).

### `xgboost/` — the third model, gradient boosting
- [`train_xgboost.ipynb`](xgboost/train_xgboost.ipynb) and
  [`xgboost_v2.ipynb`](xgboost/xgboost_v2.ipynb): trains XGBoost following the same
  protocol (same 9 predictors, same spatial and temporal validation) at pseudo-absence
  ratios 1:1 and 2:1 respectively, and compares it against the already-tuned LR and RF.
- See [`xgboost/README.md`](xgboost/README.md).

A later, separate round in [`tuning/v4/`](../tuning/v4/README.md) reruns all three
models together (LR, RF, XGBoost) on a new 7–15 km ring-sampled dataset and selects the
final deployed model.

## Metrics you'll see in every notebook
- **AUC-ROC**: how well the model distinguishes burned from non-burned pixels (0.5 =
  random, 1.0 = perfect). Primary metric.
- **PR-AUC** (Precision-Recall AUC): like AUC-ROC but stricter when the event (fire) is
  rare — here it's the MOST important metric because only 2.5% of pixel-years burn (in
  the unbalanced raw data).
- **F1**: balance between precision and recall, depends on a threshold (0.5 by default)
  — reported but secondary.
- **Spatial block CV**: the model is validated by splitting the map into blocks (~27 km)
  to measure how well it generalises to NEW places.
- **Temporal split**: trained on years ≤2019 and tested on years ≥2020, to measure how
  well it generalises to NEW years.

Final results are summarised in [`tuning/v1/`](../tuning/v1/README.md) (first tuned LR
vs. RF comparison) and [`tuning/v4/`](../tuning/v4/README.md) (final three-model
comparison that selected Random Forest).
