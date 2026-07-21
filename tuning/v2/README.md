# Tuning — v2 (corrección de fuga temporal)

**Notebook:** [`tune_v2.ipynb`](tune_v2.ipynb)

## Qué corrige esta versión
En `v1` (ver [`../v1/README.md`](../v1/README.md)), el `GridSearchCV` que eligió los
hiperparámetros de LR y RF vio el dataset **completo** (2001-2024), incluyendo los años
`>=2020`. Esos años son justamente el conjunto que se usa después para el **temporal
hold-out** — es decir, aunque el modelo final nunca se *entrenaba* con datos de
2020-2024 antes de reportar el número temporal, esos años **sí influyeron en qué
hiperparámetros se eligieron**, porque aparecían en los folds de entrenamiento durante
la búsqueda por bloques espaciales. Eso es una fuga temporal sutil hacia el proceso de
selección de modelo, aunque no hacia el entrenamiento final del modelo.

`v2` corrige esto restringiendo el `GridSearchCV` (LR y RF) a filas con `year<=2019`
únicamente. La grilla de hiperparámetros es **idéntica** a la de `v1` para aislar el
efecto de este único cambio. La evaluación final (spatial block CV + temporal split)
sigue usando el dataset completo, igual que en `v1`, porque ahí es donde se **reporta**
el desempeño, no donde se **elige** el modelo.

## Grillas de búsqueda (idénticas a v1)

| Modelo | Hiperparámetro | Valores candidatos |
|---|---|---|
| Logistic Regression | `C` | `0.01, 0.1, 1.0, 10.0` |
| | `penalty` | `l1, l2` |
| | `solver` | `liblinear` |
| Random Forest | `n_estimators` | `200, 300, 500` |
| | `max_features` | `sqrt, 0.5` |
| | `min_samples_leaf` | `3, 5, 10, 20` |
| | `max_depth` | `None, 10, 20` |

- LR: 8 candidatos × 10 folds = **80 fits** (datos: `year<=2019` únicamente, 5,153 filas).
- RF: 72 candidatos × 10 folds = **720 fits** (datos: `year<=2019` únicamente, 5,153 filas).
- Métrica de selección: **PR-AUC** (`scoring='average_precision'`).

## Mejores hiperparámetros encontrados (sin fuga temporal)

| Modelo | Hiperparámetros | ¿Cambió respecto a v1? |
|---|---|---|
| Logistic Regression | `C=0.01`, `penalty='l1'`, `solver='liblinear'` | Sí — `C` pasó de `0.1` a `0.01` (más regularización) |
| Random Forest | `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=5`, `max_depth=None` | Sí — `n_estimators` 500→300, `max_features` 0.5→`sqrt`, `min_samples_leaf` 3→5 |

## Resultados finales (modelos afinados sin fuga)

| Modelo | AUC (espacial) | PR-AUC (espacial) | F1 (espacial) | AUC (temporal) | PR-AUC (temporal) | F1 (temporal) |
|---|---|---|---|---|---|---|
| Logistic Regression (v2) | 0.832 ± 0.064 | 0.703 ± 0.105 | 0.613 ± 0.123 | 0.808 | 0.617 | 0.557 |
| Random Forest (v2) | 0.877 ± 0.041 | 0.779 ± 0.067 | 0.680 ± 0.074 | 0.819 | 0.569 | 0.560 |

**Matrices de confusión:**

| Modelo | Validación | TN | FP | FN | TP |
|---|---|---|---|---|---|
| LR (v2) | Espacial (suma 10 folds) | 3656 | 498 | 891 | 1186 |
| LR (v2) | Temporal (≥2020) | 630 | 199 | 76 | 173 |
| RF (v2) | Espacial (suma 10 folds) | 3736 | 418 | 762 | 1315 |
| RF (v2) | Temporal (≥2020) | 628 | 201 | 74 | 175 |

Resultados guardados también en [`../../outputs/metrics/tuning_v2_metrics.csv`](../../outputs/metrics/tuning_v2_metrics.csv).

## Comparación v1 (con fuga) vs v2 (corregido)

| Modelo | Métrica | v1 | v2 | Diferencia |
|---|---|---|---|---|
| LR | AUC espacial | 0.835 ± 0.060 | 0.832 ± 0.064 | -0.003 |
| LR | PR-AUC espacial | 0.705 ± 0.102 | 0.703 ± 0.105 | -0.002 |
| LR | AUC temporal | 0.807 | 0.808 | +0.001 |
| LR | PR-AUC temporal | 0.618 | 0.617 | -0.001 |
| RF | AUC espacial | 0.877 ± 0.040 | 0.877 ± 0.041 | ~0.000 |
| RF | PR-AUC espacial | 0.783 ± 0.066 | 0.779 ± 0.067 | -0.004 |
| RF | AUC temporal | 0.816 | 0.819 | +0.003 |
| RF | PR-AUC temporal | 0.558 | 0.569 | +0.011 |

**Conclusión:** la fuga temporal en la búsqueda de hiperparámetros de `v1` sí cambió
**qué hiperparámetros** se eligieron (sobre todo en RF: menos árboles, `max_features`
más conservador, hojas más grandes — un modelo ligeramente más regularizado cuando solo
ve datos ≤2019), pero el efecto sobre las **métricas finales reportadas** es mínimo
(diferencias de 0.001 a 0.011 en todas las métricas, dentro del ruido de un solo fold
temporal). Esto es la conclusión esperable: la fuga afectaba la *selección* de modelo,
no su *entrenamiento* final, así que el impacto práctico es pequeño. Aun así, `v2` es la
versión metodológicamente correcta y debe preferirse como referencia oficial hacia
adelante.

Nota adicional: los hiperparámetros de `v2` coinciden casi exactamente con los que
estaban documentados en el repositorio antes de que se detectara que la celda de
`GridSearchCV` de `v1` faltaba por completo — es probable que el notebook original
(perdido) ya restringiera el tuning a `year<=2019`, y que `v2` simplemente esté
reconstruyendo esa metodología original correcta.
