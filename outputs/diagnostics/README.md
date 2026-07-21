# `outputs/diagnostics/` — Diagnósticos estadísticos

Gráficos generados por [`eda/diagnosis.ipynb`](../../eda/diagnosis.ipynb):

- `spearman.png` — matriz de correlación de Spearman entre los 10 predictores
  candidatos iniciales. Sirvió para detectar el bloque de humedad redundante
  (`rh_pct`, `precip_mm`, `vpd_kPa` muy correlacionados entre sí), lo que llevó a
  eliminar `rh_pct` y `precip_mm` del set final de predictores (ver
  [`model/logistic_regression/README.md`](../../model/logistic_regression/README.md), sección 4).
