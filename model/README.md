# `model/` — Modelos de Machine Learning

Aquí vive **un modelo por carpeta**. Los tres deben leer exactamente el mismo dataset
([`data/model_dataset/model_dataset.csv`](../data/model_dataset/model_dataset.csv)) y
evaluarse con las mismas métricas, para que la comparación entre ellos sea justa.

## ¿Por qué comparar varios modelos?
Es una pregunta metodológica clásica: *¿por qué usar Machine Learning "caja negra" en
vez de un modelo simple y transparente?* La respuesta: se empieza con el modelo más
simple posible (regresión logística) como **línea base**, y solo se justifica usar un
modelo más complejo (Random Forest, XGBoost) si realmente mejora el desempeño con los
MISMOS datos y la MISMA validación.

## Subcarpetas

### `logistic_regression/` — el modelo base (baseline)
- [`baseline_comparison.ipynb`](logistic_regression/baseline_comparison.ipynb): este notebook hace DOS cosas en orden:
  1. **Construye** el dataset congelado (`data/model_dataset/model_dataset.csv`) desde
     Google Earth Engine — esto solo se ejecuta una vez; si el archivo ya existe, se
     carga directamente y se salta el paso lento.
  2. **Entrena y evalúa** una Regresión Logística simple sobre ese dataset.
- Ver [README_logistic_regression.md](logistic_regression/README.md) para el detalle completo.

### `random_forest/` — el primer modelo de "verdadero" ML
- [`train_rf.ipynb`](random_forest/train_rf.ipynb): entrena un Random Forest con
  hiperparámetros razonables (no los "mejores" — eso se hace en [`tuning/`](../tuning/)).
- Ver [README.md](random_forest/README.md).

### `xgboost/` — PENDIENTE
Tercer modelo a implementar, siguiendo el mismo protocolo (mismos 9 predictores, misma
validación espacial y temporal). Ver [README.md](xgboost/README.md).

## Métricas que vas a ver en todos los notebooks
- **AUC-ROC**: qué tan bien el modelo distingue pixeles quemados de no-quemados (0.5 =
  azar, 1.0 = perfecto). Métrica principal.
- **PR-AUC** (Precision-Recall AUC): como el AUC-ROC pero más estricta cuando el evento
  (incendio) es raro — aquí es la métrica MÁS importante porque solo 2.5% de los
  pixeles-año se queman.
- **F1**: balance entre precisión y sensibilidad, depende de un umbral (0.5 por
  defecto) — se reporta pero es secundaria.
- **Spatial block CV**: el modelo se valida dividiendo el mapa en bloques (~27 km) para
  medir qué tan bien generaliza a LUGARES nuevos.
- **Temporal split**: se entrena con años ≤2019 y se prueba en años ≥2020, para medir
  qué tan bien generaliza a AÑOS nuevos.

Los resultados finales están resumidos en [`tuning/v1/`](../tuning/v1/README.md),
donde se comparan los modelos ya afinados (tuned).
