# `data/` — Todos los datos del proyecto

Esta carpeta centraliza **todos** los datos usados en la tesis. Antes estaban repartidos
en varias carpetas (`datasets/`, `log_regression/datasets/`); ahora todo vive aquí, en
tres subcarpetas con un significado claro.

## `raw/` — Datos crudos (fuente original)
Archivos tal como se descargaron/recibieron, **antes** de cualquier limpieza:
- `coca_cultivation.csv` + `coca_cultivation/` — datos de cultivos de coca (UNODC-SIMCI), shapefile limpio.
- `national_natural_parks/` — shapefile de Parques Nacionales Naturales (RUNAP).
- `roads/` — shapefile de la OMS.

Estos archivos se usaron para crear los "assets" (capas) que viven en Google Earth Engine
(proyecto `col-amazon-fire-susceptibility`).  

## `processed/` — Datos procesados (tablas intermedias)
Tablas ya calculadas a partir de Google Earth Engine, listas para graficar o analizar:
- `burned_df.csv` — área quemada anual (target).
- `climate_df.csv` — variables climáticas de temporada seca por año.
- `coca_df.csv` — hectáreas de coca por año.
- `lulc_area_df.csv` — área por clase de cobertura del suelo (LULC) por año.
- `pixel_year_df.csv` — tabla pixel-año usada en análisis exploratorios.

Estos archivos los generan los notebooks de `eda/` (si no existen, se recalculan desde
Earth Engine y se guardan aquí automáticamente — así no hay que repetir cálculos lentos).

## `model_dataset/` — El dataset congelado para Machine Learning
Este es **el dataset oficial** que usan TODOS los modelos (regresión logística, Random
Forest, y en el futuro XGBoost). Se llama "congelado" (frozen) porque una vez creado,
no se debe modificar — así garantizamos que todos los modelos se comparan sobre
exactamente los mismos datos.

- `pixel_year_full.csv` — tabla completa antes de balancear clases (83,184 filas, 2.5% incendios).
- `model_dataset.csv` — **el dataset final** usado para entrenar/evaluar (6,231 filas,
  balanceado 1:2 incendio:no-incendio). Contiene las 9 variables predictoras + `burned`
  (la variable objetivo) + `lon`,`lat`,`year` (metadatos, NO son predictores).

Este dataset lo construye [`model/logistic_regression/baseline_comparison.ipynb`](../model/logistic_regression/baseline_comparison.ipynb)
la primera vez que se ejecuta; después simplemente se carga desde aquí.
