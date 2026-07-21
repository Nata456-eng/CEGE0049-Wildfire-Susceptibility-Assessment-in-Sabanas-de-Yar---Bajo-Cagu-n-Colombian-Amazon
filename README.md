# Wildfire Susceptibility Assessment — Sabanas del Yarí–Bajo Caguán (Colombian Amazon)

Proyecto de tesis (MSc Geospatial Sciences, UCL) que construye un mapa de
**susceptibilidad espacial a incendios** para un núcleo de deforestación en la
Amazonía colombiana, usando Google Earth Engine + Machine Learning.

**¿Nunca has trabajado con Machine Learning?** Empieza por [`WORKFLOW.md`](WORKFLOW.md) —
ahí se explica todo el flujo del proyecto en lenguaje simple: qué datos hay, cómo se
entrenan los modelos, dónde ver los resultados y cómo interpretarlos.

## Estructura del repositorio

```
├── data/               # Todos los datos (crudos, procesados, y el dataset de ML)
│   ├── raw/            # Datos fuente sin procesar (shapefiles, CSV originales)
│   ├── processed/      # Tablas intermedias calculadas desde Earth Engine
│   └── model_dataset/  # El dataset "congelado" que usan todos los modelos
│
├── eda/                # Análisis exploratorio de datos (EDA) — entender antes de modelar
│   ├── climatic/       # Fuego vs. clima y ENSO
│   ├── social/         # Fuego vs. coca, caminos, parques, cobertura del suelo
│   ├── nucleus_extraction/  # Definición del área de estudio + mapas de quintiles
│   └── diagnosis.ipynb # Correlación, colinealidad (VIF), autocorrelación espacial
│
├── model/              # Modelos de Machine Learning, uno por carpeta
│   ├── logistic_regression/  # Modelo base (baseline), transparente
│   ├── random_forest/        # Primer modelo de ML "real"
│   └── xgboost/              # Pendiente
│
├── tuning/             # Búsqueda de hiperparámetros (por versión: v1, v2, ...)
│   └── v1/             # GridSearchCV para LR + RF, y la tabla comparativa final
│
├── outputs/            # Todo lo generado: gráficos, mapas, métricas, SHAP
│   ├── figures/        # Gráficos (climate/, social/)
│   ├── maps/           # Mapas espaciales
│   ├── diagnostics/    # Gráficos de diagnóstico estadístico
│   ├── metrics/        # (pendiente) tablas de métricas exportadas
│   └── shap/           # (pendiente) interpretabilidad del modelo (SHAP)
│
├── utils/              # Compatibilidad: re-exportan los módulos compartidos de la raíz
├── col_amazon_fire_utils.py  # Módulo compartido: conexión a Earth Engine, geometría del núcleo, extractores
└── plot_helpers.py           # Funciones de graficado compartidas
```

Cada carpeta principal (`data/`, `eda/`, `model/`, `tuning/`, `outputs/`) tiene su
propio `README.md` con más detalle.

## Cómo ejecutar los notebooks

1. Abre una terminal **en la raíz de este repositorio** y activa el entorno conda:
   ```powershell
   conda activate fire_thesis
   ```
2. Instala las dependencias si hace falta (dentro del entorno `fire_thesis`):
   ```powershell
   pip install pandas numpy matplotlib seaborn earthengine-api geemap geopandas shapely scikit-learn statsmodels scipy esda libpysal jenkspy contextily
   ```
3. Autentica Google Earth Engine (una sola vez por máquina):
   ```powershell
   earthengine authenticate
   ```
4. Abre cualquier notebook en VS Code y ejecuta las celdas en orden, de arriba hacia
   abajo. Cada notebook es un kernel independiente — si cierras y vuelves a abrir uno,
   hay que volver a ejecutar todas sus celdas desde el principio.

## Convención de rutas (importante)

Cada notebook vive en una carpeta distinta y necesita "subir" un número diferente de
niveles para llegar a la raíz del repo (donde están `col_amazon_fire_utils.py` y la
carpeta `data/`). Por eso al inicio de cada notebook vas a ver algo como:

```python
import os, sys
sys.path.insert(0, os.path.abspath('../..'))   # sube 2 niveles hasta la raíz
```

El número de `..` depende de qué tan profundo esté el notebook (revisa el árbol de
carpetas arriba). Si mueves un notebook de carpeta, **hay que actualizar este número**.

## Notas y solución de problemas

- Si un notebook falla al importar `col_amazon_fire_utils`, revisa el `sys.path.insert`
  de la primera celda — probablemente necesita más o menos `'..'` según su profundidad.
- Las llamadas a Google Earth Engine pueden ser lentas; casi todos los notebooks
  guardan resultados intermedios en `data/processed/` o `data/model_dataset/` para no
  recalcular cada vez. Borra esos CSV solo si quieres forzar un recálculo completo.
- El entorno `fire_thesis` necesita `earthengine-api`, `geemap` y `geopandas` para las
  partes espaciales.
