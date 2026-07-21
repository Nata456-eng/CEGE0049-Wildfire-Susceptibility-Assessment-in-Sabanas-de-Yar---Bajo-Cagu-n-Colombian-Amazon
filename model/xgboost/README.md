# XGBoost — PENDIENTE

Este es el tercer modelo del proyecto (todavía no implementado). Cuando se construya,
debe seguir **exactamente el mismo protocolo** que los otros dos modelos, para que la
comparación sea justa:

## Checklist para cuando se implemente
- [ ] Cargar el mismo dataset congelado: [`data/model_dataset/model_dataset.csv`](../../data/model_dataset/model_dataset.csv)
- [ ] Usar los mismos 9 predictores:
      `dist_roads, dist_parks, dist_coca, dist_mosaic, temp_C, vpd_kPa, ndvi, wind_ms, oni`
- [ ] Evaluar con **spatial block CV** (`GroupKFold`, bloques de 0.25°) — igual que LR y RF.
- [ ] Evaluar con **temporal split** (train ≤2019, test ≥2020) — igual que LR y RF.
- [ ] Reportar las mismas 3 métricas: AUC-ROC, PR-AUC, F1.
- [ ] Afinar hiperparámetros con `GridSearchCV` (ver convención en [`tuning/`](../../tuning/)).
- [ ] Añadir la fila de resultados a la tabla comparativa en [`tuning/v1/README.md`](../../tuning/v1/README.md).

## Por qué XGBoost
Es el modelo de árboles "boosted" más citado en la literatura de susceptibilidad a
incendios (ver revisión de Mihajlovski & Zhiyanski 2025, Fire 8(10):380). Completar
esta tercera comparación (Logística vs. Random Forest vs. XGBoost) cierra la tabla de
modelos del Objetivo 1 de la tesis.
