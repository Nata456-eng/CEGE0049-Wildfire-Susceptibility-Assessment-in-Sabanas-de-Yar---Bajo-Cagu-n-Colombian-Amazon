# Tuning — v1

**Notebook:** [`tune_rf.ipynb`](tune_rf.ipynb)

## Qué es esta carpeta
Aquí vive la búsqueda de **hiperparámetros** (los "ajustes" internos de un modelo, como
cuántos árboles tiene un Random Forest o cuánta regularización usa la regresión
logística). Cada intento de afinación es una versión nueva (`v1`, `v2`, ...) para no
perder el historial de qué se probó y qué resultado dio.

`v1` es la primera (y hasta ahora única) ronda de afinación. Usa `GridSearchCV` con
validación cruzada por bloques espaciales (10 folds) para elegir los mejores
hiperparámetros de **ambos** modelos (Logistic Regression y Random Forest), y luego
evalúa las dos versiones afinadas con el mismo protocolo de siempre (spatial block CV +
temporal split). Por eso este notebook también contiene **la tabla comparativa final**
de los dos modelos.

## ¿Por qué un `GridSearchCV`?
Es una búsqueda exhaustiva: se prueban todas las combinaciones de una lista de valores
candidatos (ej. `n_estimators = [100, 300, 500]`) y se elige la combinación con mejor
`PR-AUC` promedio en validación cruzada. Nunca se usa el conjunto de prueba temporal
(≥2020) durante esta búsqueda — ese conjunto queda "intacto" para el reporte final, así
no hay fuga de información (*data leakage*).

**Detalle técnico importante:** con `GroupKFold` hay que pasar el objeto de
validación cruzada y los grupos así:
```python
GridSearchCV(estimator, param_grid, cv=GroupKFold(n_splits=10), scoring='average_precision')
grid.fit(X, y, groups=g_tune)   # groups va en .fit(), no en cv=
```
Pasar `cv=gkf.split(...)` (el generador) en vez del objeto rompe con `PicklingError`.

## Grillas de búsqueda usadas

| Modelo | Hiperparámetro | Valores candidatos |
|---|---|---|
| Logistic Regression | `C` | `0.01, 0.1, 1.0, 10.0` |
| | `penalty` | `l1, l2` |
| | `solver` | `liblinear` |
| Random Forest | `n_estimators` | `200, 300, 500` |
| | `max_features` | `sqrt, 0.5` |
| | `min_samples_leaf` | `3, 5, 10, 20` |
| | `max_depth` | `None, 10, 20` |

- LR: 4×2×1 = **8 candidatos** × 10 folds = **80 fits**.
- RF: 3×2×4×3 = **72 candidatos** × 10 folds = **720 fits** (9 folds entrenan / 1 valida por fit).
- Métrica de selección: **PR-AUC** (`scoring='average_precision'`) promediada entre los 10 folds por candidato; gana el candidato con mejor PR-AUC promedio.

## Mejores hiperparámetros encontrados

| Modelo | Hiperparámetros |
|---|---|
| Logistic Regression | `C=0.1`, `penalty='l1'`, `solver='liblinear'` |
| Random Forest | `n_estimators=500`, `max_features=0.5`, `min_samples_leaf=3`, `max_depth=None` |

## Resultados finales (modelos afinados)

| Modelo | AUC (espacial) | PR-AUC (espacial) | AUC (temporal) | PR-AUC (temporal) |
|---|---|---|---|---|
| Logistic Regression (tuned) | 0.835 ± 0.060 | 0.705 ± 0.102 | 0.807 | 0.618 |
| Random Forest (tuned) | 0.877 ± 0.040 | 0.783 ± 0.066 | 0.816 | 0.558 |

**Interpretación (explicada simple):**
- Random Forest gana en la validación **espacial** (mejor prediciendo lugares nuevos) y
  es más estable (menor desviación estándar entre bloques).
- En la validación **temporal** ambos modelos quedan prácticamente empatados — el fuego
  entre años no está fuertemente determinado por estas variables, lo cual es
  consistente con que ENSO (`oni`) tenga una correlación débil con el fuego.

**Hallazgo central:** la regularización L1 de la regresión logística afinada puso el
coeficiente de **`oni`** (el índice ENSO) en **exactamente cero** — el modelo lineal no
encontró ninguna relación monótona (creciente o decreciente) entre ENSO y el fuego que
valiera la pena conservar. Sin embargo, en el Random Forest, `oni` NO queda al final:
queda en la mitad de la tabla de importancia (rango 6 de 9), por encima de `dist_roads`,
`dist_coca` y `dist_mosaic`. Esto sugiere que la relación entre ENSO y el fuego es real
pero **no lineal** — por ejemplo, tanto los eventos El Niño como La Niña extremos
podrían aumentar el riesgo, algo que un coeficiente lineal simplemente no puede
capturar, pero que un árbol de decisión sí (porque puede partir el rango de `oni` en
varios segmentos independientes).

Por separado, `dist_mosaic` sí conserva un coeficiente pequeño pero distinto de cero en
la LR (`-0.110`), y sigue siendo la variable **menos importante** en el Random Forest
(última en impureza, 0.034). Esto ya no respalda la hipótesis original de que L1 la
"apagaba" — más bien parece que su información se solapa con otras variables humanas
correlacionadas (`dist_roads`, `dist_coca`), y el árbol prefiere dividir usando esas
otras variables primero. La interpretación con SHAP (pendiente, ver
[`outputs/shap/README.md`](../../outputs/shap/README.md)) debería enfocarse en **`oni`**
como el caso más claro de relación no lineal, sin descartar `dist_mosaic`.

## Nota sobre fuga de información (leakage)
Los hiperparámetros se eligieron con los mismos bloques espaciales usados para reportar
el desempeño espacial → el número espacial es ligeramente optimista. El desempeño
**temporal** (≥2020) nunca se usó durante la afinación, así que es la estimación no
sesgada. Se comprobó que el efecto de la afinación es pequeño (PR-AUC espacial
+0.012–0.015, temporal ±0.003) — hacer una validación anidada (nested CV) eliminaría
este sesgo por completo pero cuesta ~10 veces más cómputo por una ganancia mínima.
