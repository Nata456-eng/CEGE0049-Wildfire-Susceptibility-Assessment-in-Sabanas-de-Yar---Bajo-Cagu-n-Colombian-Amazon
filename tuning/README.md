# `tuning/` — Búsqueda de hiperparámetros

Cada subcarpeta (`v1/`, `v2/`, ...) es una ronda de afinación de hiperparámetros con
`GridSearchCV`. Se numeran por versión (en vez de sobrescribir) para conservar el
historial de qué se probó y con qué resultado — útil para la tesis y para el defense.

| Versión | Qué se afinó | Resultado |
|---|---|---|
| [`v1/`](v1/) | Logistic Regression + Random Forest | Ver [`v1/README.md`](v1/README.md) — tabla comparativa final de ambos modelos afinados. |
| [`v2/`](v2/) | Logistic Regression + Random Forest, con la corrección de fuga temporal (`GridSearchCV` solo ve `year<=2019`) | Ver [`v2/README.md`](v2/README.md) — misma grilla que v1, pero con tuning correctamente aislado del hold-out `>=2020`. |
| [`v3/`](v3/) | Análisis de sensibilidad del ratio de pseudo-ausencias (1:1, 2:1, 3:1), con el mismo pipeline de v2 | Ver [`v3/README.md`](v3/README.md) — el ratio 1:1 supera claramente a 2:1 (usado hasta v2) y a 3:1. |
| [`v4/`](v4/) | Grilla ampliada de Random Forest (20 combinaciones `n_estimators`×`max_depth`) sobre el ratio 2:1, seleccionando por F1 temporal | Ver [`v4/README.md`](v4/README.md) — el tuning amplio mejora poco el desempeño temporal de 2:1; sigue muy por debajo del ratio 1:1 de v3. |

Cuando se afine XGBoost (pendiente), su búsqueda de hiperparámetros debería ir en una
nueva carpeta `v5/` siguiendo el mismo patrón: notebook + `README.md` con la grilla
usada, los mejores parámetros encontrados, y la interpretación de los resultados.
