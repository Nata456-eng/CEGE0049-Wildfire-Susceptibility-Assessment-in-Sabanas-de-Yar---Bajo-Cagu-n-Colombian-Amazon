# `outputs/` — Todo lo que se "ve": gráficos, mapas y métricas

Todos los archivos generados por los notebooks (gráficos, mapas, métricas, resultados
de SHAP) se guardan aquí, organizados por tipo:

- **`figures/climate/`** — gráficos de clima y su relación con el fuego (ej. área
  quemada vs. ENSO, serie de tiempo de área quemada).
- **`figures/social/`** — gráficos sociales/infraestructura (ej. cobertura del suelo y
  coca vs. área quemada).
- **`maps/`** — mapas espaciales (quintiles de fuego por período, snapshot del último
  año, mapas con capas de ríos/caminos/parques/coca).
- **`diagnostics/`** — gráficos de diagnóstico estadístico (ej. matriz de correlación
  de Spearman entre predictores). Ver [`diagnostics/README.md`](diagnostics/README.md).
- **`metrics/`** — PENDIENTE: tablas de métricas de los modelos exportadas a CSV. Ver
  [`metrics/README.md`](metrics/README.md).
- **`shap/`** — PENDIENTE: gráficos de interpretabilidad (SHAP, importancia por
  permutación) del Random Forest. Ver [`shap/README.md`](shap/README.md).

Cada notebook que produce un gráfico o mapa lo guarda automáticamente en la subcarpeta
correspondiente (no hace falta moverlos a mano). Si vuelves a ejecutar un notebook, el
archivo existente se sobreescribe con la versión más reciente.
