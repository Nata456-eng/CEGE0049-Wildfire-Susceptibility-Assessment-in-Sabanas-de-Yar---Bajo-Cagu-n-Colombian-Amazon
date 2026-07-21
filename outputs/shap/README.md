# `outputs/shap/` — PENDIENTE (interpretabilidad del modelo)

Aquí van los gráficos de **SHAP** (SHapley Additive exPlanations) y de **importancia
por permutación**, calculados sobre el Random Forest afinado (el modelo ganador hasta
ahora).

## ¿Qué es SHAP, en simple?
La importancia por impureza (la que ya calculamos en
[`model/random_forest/`](../../model/random_forest/)) tiene un sesgo: favorece a las
variables continuas (como `ndvi` o `vpd_kPa`) sobre las variables de distancia binarias
en su comportamiento (como `dist_mosaic`). SHAP mide, para cada predicción individual,
cuánto "empujó" cada variable la probabilidad de incendio hacia arriba o hacia abajo —
es más justo y es el estándar actual en la literatura de Machine Learning interpretable.

## Qué se espera encontrar (hipótesis a confirmar)
En la regresión logística afinada, la regularización L1 puso el coeficiente de **`oni`**
(el índice ENSO) en cero — el modelo lineal no encontró ninguna relación monótona con el
fuego (ver [`tuning/v1/README.md`](../../tuning/v1/README.md)). Sin embargo, en el
Random Forest afinado, `oni` no quedó al final de la tabla de importancia — quedó en la
mitad (rango 6 de 9). La hipótesis es que SHAP muestre que `oni` sí es importante, pero
de forma **no lineal** (por ejemplo, ambos extremos de El Niño y La Niña aumentando el
riesgo, no una tendencia creciente o decreciente simple).

Un segundo caso a revisar: `dist_mosaic` (distancia a la frontera agropecuaria) conserva
un coeficiente pequeño pero distinto de cero en la LR, y es la variable con **menor**
importancia por impureza en el Random Forest. Aquí la hipótesis es distinta: que su
información se solape con otras variables humanas correlacionadas (`dist_roads`,
`dist_coca`), y SHAP ayude a distinguir si de verdad aporta poco o si el árbol
simplemente prefiere dividir usando esas otras variables primero.

## Qué archivos van aquí (cuando se calculen)
- `shap_summary_plot.png` — importancia global de cada predictor.
- `shap_dependence_oni.png`, `shap_dependence_dist_mosaic.png` (y similares) — cómo
  cambia la contribución de cada variable según su valor (para confirmar/descartar la
  no linealidad).
- `permutation_importance.csv` — importancia por permutación, para comparar con SHAP.
