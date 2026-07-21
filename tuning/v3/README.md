# Tuning — v3 (análisis de sensibilidad del ratio de pseudo-ausencias)

**Notebook:** [`tune_v3.ipynb`](tune_v3.ipynb)

## Qué es un análisis de sensibilidad y por qué se hizo
Desde el primer notebook (`model/logistic_regression/baseline_comparison.ipynb`), el
dataset se construyó con un ratio fijo de **2 pseudo-ausencias por cada incendio
observado** (2:1), justificado por literatura general (Barbet-Massin et al. 2012) pero
nunca probado empíricamente contra otras alternativas en este proyecto. Un análisis de
sensibilidad prueba variantes razonables de esa decisión de diseño (aquí: el ratio de
muestreo) y mide si el resultado del modelo cambia — en vez de asumir que la elección
original era la óptima.

Se probaron 3 ratios: **1:1** (clases balanceadas), **2:1** (el usado en v1/v2) y
**3:1** (más pseudo-ausencias, clases más desbalanceadas). El límite de qué ratios son
factibles depende de cuántas pseudo-ausencias elegibles hay después del buffer de
exclusión de 3 km: **78,878** disponibles, es decir hasta 1:38 — 1:1, 2:1 y 3:1 usan
muy poco de ese margen, así que no hay problema de disponibilidad de datos.

## Cómo se seleccionó el mejor ratio
Para cada ratio se repitió **exactamente el mismo pipeline de `tuning/v2`**:
`GridSearchCV` (10-fold spatial-block CV, `scoring='average_precision'`) restringido a
`year<=2019` (sin fuga temporal), con las mismas grillas de hiperparámetros de v1/v2, y
luego evaluación completa (spatial block CV de 10 folds + hold-out temporal `>=2020`,
con matriz de confusión). El ratio ganador es el que logra el mejor **PR-AUC espacial
promedio en Random Forest** — la métrica de selección usada en todo el proyecto, y el
modelo con mejor desempeño consistente en v1 y v2.

**Chequeo de consistencia:** el ratio 2:1 de este notebook reprodujo, cifra por cifra,
los resultados ya reportados en [`v2/README.md`](../v2/README.md) (misma semilla
`random_state=42`, mismo tamaño de muestra) — confirma que el pipeline de v3 es
metodológicamente idéntico al de v2, y que el único factor que cambia es el ratio.

## Resultados por ratio (LR y RF, tuning `year<=2019`, evaluación año completo)

| Ratio | Modelo | Filas (presencias) | AUC espacial | PR-AUC espacial | F1 espacial | AUC temporal | PR-AUC temporal | F1 temporal |
|---|---|---|---|---|---|---|---|---|
| **1:1** | Logistic Regression | 4,154 (2,077) | 0.846 ± 0.032 | **0.823 ± 0.060** | 0.754 ± 0.079 | 0.810 | **0.747** | 0.663 |
| **1:1** | Random Forest | 4,154 (2,077) | 0.878 ± 0.027 | **0.866 ± 0.036** | 0.790 ± 0.047 | 0.817 | **0.704** | 0.684 |
| 2:1 | Logistic Regression | 6,231 (2,077) | 0.832 ± 0.064 | 0.703 ± 0.105 | 0.613 ± 0.123 | 0.808 | 0.617 | 0.557 |
| 2:1 | Random Forest | 6,231 (2,077) | 0.877 ± 0.041 | 0.779 ± 0.067 | 0.680 ± 0.074 | 0.819 | 0.569 | 0.560 |
| 3:1 | Logistic Regression | 8,308 (2,077) | 0.831 ± 0.051 | 0.618 ± 0.113 | 0.504 ± 0.167 | 0.806 | 0.542 | 0.486 |
| 3:1 | Random Forest | 8,308 (2,077) | 0.873 ± 0.032 | 0.702 ± 0.067 | 0.603 ± 0.098 | 0.817 | 0.492 | 0.493 |

Resultados completos (con matrices de confusión) en
[`tuning_v3_sensitivity_metrics.csv`](tuning_v3_sensitivity_metrics.csv). Los datos del
ratio ganador están aislados en [`tuning_v3_final_metrics.csv`](tuning_v3_final_metrics.csv).

## Mejores hiperparámetros por ratio

| Ratio | Logistic Regression | Random Forest |
|---|---|---|
| 1:1 | `C=0.01`, `penalty='l2'` | `n_estimators=200`, `max_features='sqrt'`, `min_samples_leaf=3`, `max_depth=None` |
| 2:1 | `C=0.01`, `penalty='l1'` | `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=5`, `max_depth=None` |
| 3:1 | `C=0.01`, `penalty='l1'` | `n_estimators=300`, `max_features='sqrt'`, `min_samples_leaf=5`, `max_depth=None` |

## Ratio óptimo: **1:1**

Tanto **Random Forest** como **Logistic Regression** coinciden en que **1:1** es el
mejor ratio, con una diferencia grande y consistente en todas las métricas:

- RF: PR-AUC espacial pasa de **0.779** (2:1, el usado hasta ahora) a **0.866** (1:1)
  — una mejora de **+0.087**. PR-AUC temporal pasa de 0.569 a 0.704 (**+0.135**).
- LR: PR-AUC espacial pasa de 0.703 (2:1) a **0.823** (1:1) — **+0.120**. PR-AUC
  temporal pasa de 0.617 a **0.747** (**+0.130**).
- AUC-ROC apenas cambia entre ratios (0.83–0.88 en todos los casos) porque AUC-ROC es
  poco sensible al desbalance de clases — por eso PR-AUC es la métrica correcta para
  esta comparación, tal como se usó durante todo el proyecto.
- F1 también mejora notablemente con 1:1 (RF: 0.680→0.790; LR: 0.613→0.754), porque con
  clases balanceadas el umbral de decisión por defecto (0.5) deja de estar sesgado
  hacia predecir "no quemado".

**¿Por qué pasa esto?** Con 2:1 y 3:1, el modelo ve más ejemplos de "no quemado" que de
"quemado", y aunque `class_weight` no se usó aquí (a diferencia del notebook base), el
desbalance de clases igual perjudica el PR-AUC porque hay más falsos positivos
potenciales entre los que discriminar. Con 1:1 el modelo tiene la misma cantidad de
señal positiva y negativa para aprender el límite de decisión, lo que en este dataset
(pocas presencias, ~2,077) pesa más que la ventaja teórica de "más variedad de
pseudo-ausencias" que ofrecen los ratios más altos.

## Mejora respecto a v1 y v2

- **v1** afinó hiperparámetros pero nunca cuestionó el ratio de muestreo (2:1, fijo
  desde el primer notebook), y además tenía fuga temporal en el tuning.
- **v2** corrigió la fuga temporal, pero seguía usando el ratio 2:1 sin haberlo
  comparado contra alternativas.
- **v3** mantiene la corrección de v2 y agrega una nueva dimensión de búsqueda (el
  ratio de pseudo-ausencias), encontrando que **el ratio importa mucho más que el
  tuning de hiperparámetros en sí**: cambiar de 2:1 a 1:1 mejora el PR-AUC espacial de
  RF en +0.087, mientras que la corrección de fuga temporal (v1→v2) solo cambió el
  PR-AUC espacial de RF en -0.004. Esto sugiere que, para este dataset, la decisión de
  diseño de muestreo tiene más impacto en el desempeño que la elección fina de
  hiperparámetros.

## Recomendación
Adoptar el ratio **1:1** como el nuevo dataset "oficial" para entrenamientos futuros
(reemplazando el 2:1 usado en `data/model_dataset/model_dataset.csv`), documentando este
cambio en el README de `model/logistic_regression/` y regenerando dicho CSV si se decide
avanzar con esta recomendación — no se sobrescribió automáticamente aquí para no romper
la reproducibilidad de v1/v2, que dependen explícitamente del dataset 2:1 congelado.
