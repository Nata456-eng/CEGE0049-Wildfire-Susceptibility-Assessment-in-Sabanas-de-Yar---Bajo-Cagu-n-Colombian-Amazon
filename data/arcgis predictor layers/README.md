# `data/arcgis predictor layers/` — capas individuales para abrir en ArcGIS

Las 9 variables predictoras del modelo final (Random Forest v4) + un producto MODIS de
fuego, cada una como un GeoTIFF de una sola banda, para no tener que seleccionar bandas
dentro de un raster multibanda al abrirlas en ArcGIS.

## Archivos

| Archivo | Variable | Unidad / rango típico |
|---|---|---|
| `01_dist_roads.tif` | Distancia a carreteras | metros |
| `02_dist_parks.tif` | Distancia a parques nacionales | metros |
| `03_dist_coca.tif` | Distancia a cultivos de coca (histórico, hasta 2023) | metros |
| `04_dist_mosaic.tif` | Distancia a mosaico agropecuario (frontera agrícola, MapBiomas 2024) | metros |
| `05_temp_C.tif` | Temperatura, promedio temporada seca 2001–2024 | °C |
| `06_vpd_kPa.tif` | Déficit de presión de vapor, promedio temporada seca 2001–2024 | kPa |
| `07_ndvi.tif` | NDVI, promedio temporada seca 2001–2024 | 0–1 |
| `08_wind_ms.tif` | Velocidad del viento, promedio temporada seca 2001–2024 | m/s |
| `09_oni.tif` | Índice ONI (ENSO) | constante 0 (neutral) |
| `10_modis_fire_frequency_2001_2024.tif` | Nº de años (de 24) en que MODIS detectó quema en ese pixel | 0–24 |
| `11_modis_fire_frequency_2001_2024_polygons.shp` | Versión vectorial (polígonos) de la capa 10 | ver abajo |

Las 9 primeras son bandas separadas de `probability map/predictor_stack_v4.tif` (mismo
archivo que alimenta el mapa de susceptibilidad) — ver
[`probability map/README.md`](../../probability%20map/README.md) para el detalle completo
de cómo se construyó ese stack. Las capas 10 y 11 son nuevas (ver abajo).

## Grid común — las 10 capas se superponen pixel a pixel

Todas comparten exactamente la misma grilla: 500 m, EPSG:4326, mismo extent y mismo
`transform` (819 × 1144 pixeles). Verificado explícitamente al generar la capa MODIS, no
solo asumido — así que puedes cargarlas todas en ArcGIS y compararlas celda por celda sin
tener que reproyectar/remuestrear nada.

**Nodata = -9999** en las 10 capas, ~51.6% de la bounding box (el núcleo de 5 municipios
no es un rectángulo, así que la máscara fuera del polígono es la mayoría del bounding
box). El archivo fuente (`predictor_stack_v4.tif`) no trae el nodata declarado en sus
metadatos GDAL (ver el sidenote técnico en el README del mapa de probabilidad) — al
separar las bandas aquí sí se declaró `nodata=-9999` explícitamente en cada archivo, para
que ArcGIS lo pinte como transparente en vez de como un valor real.

## La capa MODIS (`10_modis_fire_frequency_2001_2024.tif`) — decisión tomada

No es un snapshot de un año: es el **conteo de años, 2001–2024, en que MODIS/061/MCD64A1
detectó una quema en ese pixel** (0 a 24). Se eligió esta representación —no una imagen de
un año puntual— por consistencia con las otras 9 capas: los predictores climáticos ya son
un promedio 2001–2024 (una línea base climatológica, no un año específico —
`WORKFLOW.md`: "susceptibilidad, no pronóstico"), y 2001–2024 es exactamente el rango de
años de `data/model_dataset/pixel_year_full.csv`, la tabla con la que se entrenó el
modelo. La capa de frecuencia es la contraparte, del lado del fuego, de esa misma línea
base: "cuántas veces se quemó realmente este pixel en los años que el modelo usó para
aprender" — comparable directamente contra las 9 capas de predictores en vez de depender
de qué año arbitrario se hubiera elegido.

Usa la misma definición de "quemado" que `col_amazon_fire_utils.get_burned_df` (año
quemado = `BurnDate > 0` en algún día de ese año calendario), solo que aquí se mantiene
por pixel en vez de reducirse a un solo número para todo el núcleo.

**Si necesitas otra representación** (un año específico, una máscara binaria
"alguna-vez-quemado", o "años desde el último incendio"), es un cambio pequeño en
`02_export_modis_fire_frequency.py` — pide que se regenere con esa definición.

**Lectura rápida del resultado:** solo ~13% de los pixeles válidos del núcleo se quemaron
al menos una vez en 24 años (frecuencia > 0); la mayoría de esos pixeles se quemó 1–2
veces, con una cola larga de pixeles de alta recurrencia. Coherente con lo ya visto en
`outputs/maps/burned_area_quintiles_basemap.png`: el fuego está fuertemente concentrado
en el cuadrante noroeste del núcleo, no distribuido uniformemente.

## La capa vectorial (`11_modis_fire_frequency_2001_2024_polygons.shp`)

Es la capa 10 convertida a polígonos: cada polígono es un parche de pixeles contiguos
(500 m) que comparten el mismo número de años quemados 2001–2024. Solo se guardan los
parches que se quemaron al menos una vez — el resto del núcleo (nunca quemado) no se
exporta como polígono (sería un polígono gigante sin valor analítico). No es una
geometría "promedio" en sentido literal (no se puede promediar la forma de 24 años de
incendios distintos); "promedio 2001–2024" se traduce aquí como el campo `avg_freq` de
cada polígono: la fracción de esos 24 años en que ese parche específico ardió.

Campos (nombres truncados a 10 caracteres por el límite de Shapefile):

- `yrs_burned` (entero, 1–24): años quemados de 2001 a 2024.
- `avg_freq` (float, 0–1): `yrs_burned / 24` — el "promedio" pedido, por polígono.
- `area_ha` (float): área geodésica en hectáreas (calculada con `pyproj.Geod`, no
  distorsionada por la proyección geográfica EPSG:4326 del archivo).

Resultado actual: 18,533 polígonos, ~1,474,309 ha con al menos una quema en el período
(la mayoría se quemó 1–3 veces; una cola larga de parches de alta recurrencia llega hasta
16 años de 24). Distribución completa en la salida de
`03_polygonize_modis_fire_frequency.py`.

## Caveats heredados (aplican igual aquí que en el mapa de probabilidad)

- **Extrapolación en `dist_roads`/`dist_coca` más allá de 10 km.** El modelo se entrenó
  con esas dos variables limitadas a 10 km (optimización de muestreo); en estas capas el
  radio se amplió a 100 km para cubrir todo el núcleo sin huecos, así que valores grandes
  en `01_dist_roads.tif` / `03_dist_coca.tif` representan extrapolación, no interpolación.
- **`dist_mosaic` usa uso de suelo 2024 y `dist_coca` usa coca acumulada hasta 2023**,
  mientras el clima es un promedio 2001–2024 — capas humanas "actuales", clima "típico",
  no un año calendario coherente.
- **500 m / EPSG:4326** es la resolución nativa de todo el stack (limitada por MODIS y por
  el reanálisis climático ERA5, mucho más grueso que 500 m) — no reinterpolar a una
  resolución más fina esperando ganar detalle real.

## Cómo se generaron / cómo regenerar

```powershell
cd "data\arcgis predictor layers"
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 01_split_predictor_bands.py
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 02_export_modis_fire_frequency.py
```

- `01_split_predictor_bands.py` — local, instantáneo, lee `predictor_stack_v4.tif` y
  reparte sus 9 bandas. Solo necesita volver a correrse si el stack cambia.
- `02_export_modis_fire_frequency.py` — llama a Google Earth Engine (proyecto
  `col-amazon-fire-susceptibility`, requiere autenticación ya configurada), tarda
  ~1 minuto.
