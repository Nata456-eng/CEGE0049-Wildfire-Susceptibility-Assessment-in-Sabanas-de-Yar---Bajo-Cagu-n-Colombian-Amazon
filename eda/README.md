# `eda/` — Exploratory Data Analysis (EDA)

**EDA = Exploratory Data Analysis.** Before building any model, the data needs to be
*understood*: how does fire vary over time? Is it related to climate? To coca, roads,
parks? These notebooks answer those questions with simple charts and statistics,
WITHOUT training any predictive model yet.

## Contents

| Folder/file | What it does | Key finding |
|---|---|---|
| `climatic/Climatic_EDA.ipynb` | Compares annual burned area vs. climate (temperature, humidity, ENSO/El Niño). | 2023 was the driest year but had LITTLE fire → climate alone doesn't explain the fires → justifies using Machine Learning with human variables. |
| `social/social_eda.ipynb` | Maps and time series of land cover (LULC), coca crops, roads, parks, rivers. | There's no "pure pasture" class in the area — it's mixed with "agropastoral mosaic," which is the best proxy for the agricultural frontier. |
| `nucleus_extraction/Data_distribution_thesis.ipynb` | Defines the study area (5 GAUL municipalities) and generates fire quintile maps by period. | — |
| `diagnosis.ipynb` | Statistical checks on candidate variables: correlation (Spearman), collinearity (VIF), spatial autocorrelation (Moran's I). | Relative humidity and VPD are strongly correlated (redundant) → `rh_pct` was dropped. Fire is spatially clustered (Moran's I=0.33, p=0.001) → the model must be validated with spatial blocks, not random splits. |

## How to run these notebooks
1. Open the notebook in VS Code.
2. Activate the `fire_thesis` conda environment.
3. Run the cells in order (top to bottom). The first cell always does
   `sys.path.insert(...)` to import `col_amazon_fire_utils.py`, which lives at the
   repository root — don't move it from there.
4. The first time, these notebooks query Google Earth Engine (requires
   `earthengine authenticate` once per machine) and save results to
   [`data/processed/`](../data/processed/) so they don't need to recompute every time.
5. Figures/maps are saved automatically to [`outputs/figures/`](../outputs/figures/)
   and [`outputs/maps/`](../outputs/maps/).

## For anyone who has never done ML
EDA is like "getting to know your data before the first date": nothing is predicted
here, only what already happened is described (correlations, trends, maps). This is
fundamental because a Machine Learning model can only be as good as the understanding we
have of the data we feed it.
