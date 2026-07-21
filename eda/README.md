# `eda/` — Análisis Exploratorio de Datos (EDA)

**EDA = Exploratory Data Analysis.** Antes de construir cualquier modelo, hay que
*entender* los datos: ¿cómo varía el fuego en el tiempo? ¿está relacionado con el clima?
¿con la coca, los caminos, los parques? Estos notebooks responden esas preguntas con
gráficos y estadísticas simples, SIN todavía entrenar ningún modelo predictivo.

## Contenido

| Carpeta/archivo | Qué hace | Hallazgo clave |
|---|---|---|
| `climatic/Climatic_EDA.ipynb` | Compara área quemada anual vs. clima (temperatura, humedad, ENSO/El Niño). | 2023 fue el año más seco pero con POCO fuego → el clima solo no explica los incendios → justifica usar Machine Learning con variables humanas. |
| `social/social_eda.ipynb` | Mapas y series de tiempo de cobertura del suelo (LULC), cultivos de coca, caminos, parques, ríos. | No existe una clase "pastizal puro" en la zona — se mezcla con "mosaico agropecuario", que es el mejor proxy de frontera agrícola. |
| `nucleus_extraction/Data_distribution_thesis.ipynb` | Define el área de estudio (5 municipios GAUL) y genera mapas de quintiles de fuego por período. | — |
| `diagnosis.ipynb` | Chequeos estadísticos de las variables candidatas: correlación (Spearman), colinealidad (VIF), autocorrelación espacial (Moran's I). | Humedad relativa y VPD están muy correlacionadas (redundantes) → se descartó `rh_pct`. El fuego está espacialmente agrupado (Moran's I=0.33, p=0.001) → hay que validar el modelo con bloques espaciales, no al azar. |

## Cómo ejecutar estos notebooks
1. Abre el notebook en VS Code.
2. Activa el entorno conda `fire_thesis`.
3. Ejecuta las celdas en orden (de arriba hacia abajo). La primera celda siempre hace
   `sys.path.insert(...)` para poder importar `col_amazon_fire_utils.py`, que vive en la
   raíz del repositorio — no lo muevas de ahí.
4. La primera vez, estos notebooks consultan Google Earth Engine (requiere
   `earthengine authenticate` una sola vez por máquina) y guardan los resultados en
   [`data/processed/`](../data/processed/) para no tener que recalcular cada vez.
5. Las figuras/mapas se guardan automáticamente en [`outputs/figures/`](../outputs/figures/)
   y [`outputs/maps/`](../outputs/maps/).

## Para quien nunca ha hecho ML
El EDA es como "conocer a tus datos antes de la primera cita": aquí no se predice nada,
solo se describe lo que ya pasó (correlaciones, tendencias, mapas). Esto es fundamental
porque un modelo de Machine Learning solo puede ser tan bueno como el entendimiento que
tengamos de los datos que le damos.
