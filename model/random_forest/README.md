# Random Forest

**Notebook:** [`train_rf.ipynb`](train_rf.ipynb)
**Dataset:** [`data/model_dataset/model_dataset.csv`](../../data/model_dataset/model_dataset.csv) (same frozen dataset as the logistic regression)

## Qué hace este notebook
Entrena un Random Forest con hiperparámetros razonables (no optimizados todavía) y lo
evalúa con el mismo protocolo que la regresión logística, para que la comparación sea
justa:
1. **Spatial block CV** (5 folds, bloques de 0.25° ≈ 27 km) — generalización a lugares nuevos.
2. **Temporal split** (train ≤2019, test ≥2020) — generalización a años nuevos.
3. **Feature importance** por impureza (rápida de calcular, pero sesgada hacia variables
   continuas — para una medida más justa hay que usar SHAP, ver [`outputs/shap/`](../../outputs/shap/), pendiente).

## Configuración del modelo
```python
RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=5,
                        class_weight='balanced', random_state=42, n_jobs=-1)
```
Los árboles no necesitan escalar las variables (a diferencia de la regresión logística).

## Resultados (antes de afinar hiperparámetros)

| Validación | AUC-ROC | PR-AUC | F1 |
|---|---|---|---|
| Spatial block CV | 0.875 ± 0.022 | 0.769 ± 0.067 | 0.709 ± 0.060 |
| Temporal hold-out (≥2020) | 0.817 | 0.563 | 0.549 |

### Importancia de variables (impureza — sesgada, ver nota arriba)
| Predictor | Importancia |
|---|---|
| ndvi | 0.222 |
| wind_ms | 0.201 |
| vpd_kPa | 0.162 |
| temp_C | 0.109 |
| dist_parks | 0.082 |
| oni | 0.075 |
| dist_roads | 0.068 |
| dist_coca | 0.046 |
| dist_mosaic | 0.035 |

**Ojo:** esta importancia por impureza pone a `dist_mosaic` de último — pero en el EDA
(ver [`eda/`](../../eda/)) es la variable con la relación bivariada más fuerte con el
fuego. Esto sugiere que su relación es NO lineal (un umbral: "cerca de la frontera
agropecuaria = riesgo alto"), algo que la impureza no mide bien. SHAP (pendiente)
debería confirmar esto.

## La versión afinada (tuned)
Los hiperparámetros óptimos se buscan en [`tuning/v1/tune_rf.ipynb`](../../tuning/v1/tune_rf.ipynb)
con `GridSearchCV`. Ahí también está la tabla final comparando LR vs. RF, ambos afinados.
