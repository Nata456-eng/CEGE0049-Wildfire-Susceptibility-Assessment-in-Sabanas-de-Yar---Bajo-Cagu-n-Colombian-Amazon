# `outputs/metrics/` — tablas de métricas exportadas

## Archivos disponibles
- [`tuning_v2_metrics.csv`](tuning_v2_metrics.csv) — métricas finales de LR y RF
  afinados en [`tuning/v2/tune_v2.ipynb`](../../tuning/v2/tune_v2.ipynb) (búsqueda de
  hiperparámetros restringida a `year<=2019`, sin fuga temporal). Una fila por modelo,
  con `best_params`, AUC/PR-AUC/F1 (media ± std espacial, y valor temporal), y las
  matrices de confusión (`cm_spatial_*`, `cm_temporal_*`) desglosadas en columnas
  `tn/fp/fn/tp`. Ver la tabla ya transcrita en
  [`tuning/v2/README.md`](../../tuning/v2/README.md).

Para las métricas de `v1` (con la fuga temporal ya documentada y corregida en `v2`), y
para los modelos base (sin afinar), las métricas todavía solo se imprimen dentro de
cada notebook — puedes verlas ejecutando:
- [`model/logistic_regression/baseline_comparison.ipynb`](../../model/logistic_regression/baseline_comparison.ipynb)
- [`model/random_forest/train_rf.ipynb`](../../model/random_forest/train_rf.ipynb)
- [`tuning/v1/tune_rf.ipynb`](../../tuning/v1/tune_rf.ipynb) (tabla comparativa final, modelos afinados, con fuga temporal en el tuning)

La tabla resumen de `v1` ya está transcrita en [`tuning/v1/README.md`](../../tuning/v1/README.md).

## Próximo paso sugerido
Cuando se agregue XGBoost, conviene que los notebooks base y de tuning restantes
también guarden sus métricas aquí como archivos `.csv` (siguiendo el mismo patrón que
`tuning_v2_metrics.csv`), para poder comparar todos los modelos sin tener que releer la
salida impresa de cada notebook. Esto no se implementó todavía para los notebooks base
para no modificar su lógica sin que el estudiante lo revise primero.
