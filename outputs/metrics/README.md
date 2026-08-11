# `outputs/metrics/` — Exported metric tables

## Available files
- [`tuning_v2_metrics.csv`](tuning_v2_metrics.csv) — final metrics for the LR and RF tuned in
  [`tuning/v2/tune_v2.ipynb`](../../tuning/v2/tune_v2.ipynb) (hyperparameter search
  restricted to `year<=2019`, no temporal leakage). One row per model, with
  `best_params`, AUC/PR-AUC/F1 (spatial mean ± std, and temporal value), and confusion
  matrices (`cm_spatial_*`, `cm_temporal_*`) broken down into `tn/fp/fn/tp` columns. See
  the table already transcribed in [`tuning/v2/README.md`](../../tuning/v2/README.md).

For `v1` metrics (with the temporal leakage already documented and fixed in `v2`), and
for the untuned base models, metrics are still only printed inside each notebook — you
can see them by running:
- [`model/logistic_regression/baseline_comparison.ipynb`](../../model/logistic_regression/baseline_comparison.ipynb)
- [`model/random_forest/train_rf.ipynb`](../../model/random_forest/train_rf.ipynb)
- [`tuning/v1/tune_rf.ipynb`](../../tuning/v1/tune_rf.ipynb) (final comparison table, tuned models, with the temporal leak in tuning later fixed by v2)

The `v1` summary table is already transcribed in [`tuning/v1/README.md`](../../tuning/v1/README.md).

## Metrics that now live elsewhere
Since this README was first written, XGBoost was added and a further round of tuning
(`v3`, `v4`) happened — those rounds each save their own metric CSVs directly in their
own folders instead of copying them here:
- [`tuning/v3/tuning_v3_sensitivity_metrics.csv`](../../tuning/v3/tuning_v3_sensitivity_metrics.csv) / [`tuning_v3_final_metrics.csv`](../../tuning/v3/tuning_v3_final_metrics.csv)
- [`tuning/v4/v4_buffer_7_15km/v4_model_comparison_metrics.csv`](../../tuning/v4/v4_buffer_7_15km/v4_model_comparison_metrics.csv) — the final three-model (LR/RF/XGBoost) comparison that selected Random Forest.
- [`model/xgboost/xgboost_vs_lr_rf_comparison.csv`](../../model/xgboost/xgboost_vs_lr_rf_comparison.csv) / [`xgboost_vs_lr_rf_comparison_2to1.csv`](../../model/xgboost/xgboost_vs_lr_rf_comparison_2to1.csv) — the first XGBoost round.
- [`tuning/tuning_comparison_full.csv`](../../tuning/tuning_comparison_full.csv) — a consolidated table spanning every version (baseline through v4); see the caveat about its v4 row in [`tuning/v4/README.md`](../../tuning/v4/README.md).

They weren't moved into this folder to avoid touching files from before this
reorganisation without review — this README just documents where to find them.
