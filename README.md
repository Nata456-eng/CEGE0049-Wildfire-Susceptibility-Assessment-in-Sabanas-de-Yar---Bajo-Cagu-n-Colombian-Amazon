# Wildfire Susceptibility Assessment — Sabanas del Yarí–Bajo Caguán (Colombian Amazon)

MSc Geospatial Sciences (UCL) thesis project that builds a **spatial wildfire
susceptibility map** for a deforestation nucleus in the Colombian Amazon, using Google
Earth Engine + Machine Learning.

**New to Machine Learning?** Start with [`WORKFLOW.md`](WORKFLOW.md) — it explains the
whole project flow in plain language: what data exists, how the models are trained,
where to find the results, and how to interpret them.

## Repository structure

```text
├── data/               # All data (raw, processed, and the ML dataset)
│   ├── raw/            # Unprocessed source data (shapefiles, original CSVs)
│   ├── processed/      # Intermediate tables computed from Earth Engine
│   └── model_dataset/  # The "frozen" dataset every model reads from
│
├── eda/                # Exploratory Data Analysis (EDA) — understand before modelling
│   ├── climatic/       # Fire vs. climate and ENSO
│   ├── social/         # Fire vs. coca, roads, parks, land cover
│   ├── nucleus_extraction/  # Study-area definition + fire quintile maps
│   └── diagnosis.ipynb # Correlation, collinearity (VIF), spatial autocorrelation
│
├── model/              # Machine Learning models, one per folder
│   ├── logistic_regression/  # Transparent baseline model
│   ├── random_forest/        # First "real" ML model
│   └── xgboost/               # Third model, gradient boosting
│
├── tuning/             # Hyperparameter search (versioned: v1 → v4)
│   ├── v1/             # First GridSearchCV pass (LR + RF), final comparison table
│   ├── v2/             # Same grids, temporal-leakage fix in the tuning CV
│   ├── v3/             # Pseudo-absence ratio sensitivity analysis (1:1 wins)
│   └── v4/             # 7–15 km annulus sampling, ratio 1:1, LR/RF/XGBoost — selects the final RF model
│
├── outputs/            # Everything generated: charts, maps, metrics, SHAP, the final map
│   ├── figures/        # Charts (climate/, social/)
│   ├── maps/           # Spatial maps (fire quintiles, latest-year snapshot)
│   ├── diagnostics/    # Statistical diagnostic charts
│   ├── metrics/        # Exported model metric tables
│   ├── shap/           # Model interpretability (SHAP, permutation importance)
│   └── probability_map/  # Final deployed model: pixel-level susceptibility map + ArcGIS-ready layers
│
├── utils/              # Compatibility shims that re-export the shared modules at repo root
├── col_amazon_fire_utils.py  # Shared module: Earth Engine connection, nucleus geometry, extractors
└── plot_helpers.py           # Shared plotting helpers
```

Every top-level folder (`data/`, `eda/`, `model/`, `tuning/`, `outputs/`) has its own
`README.md` with more detail.

## How to run the notebooks

1. Open a terminal **at the root of this repository** and activate the conda environment:
   ```powershell
   conda activate fire_thesis
   ```
2. Install dependencies if needed (inside the `fire_thesis` environment):
   ```powershell
   pip install pandas numpy matplotlib seaborn earthengine-api geemap geopandas shapely scikit-learn statsmodels scipy esda libpysal jenkspy contextily requests rasterio geedim
   ```
3. Authenticate Google Earth Engine (once per machine):
   ```powershell
   earthengine authenticate
   ```
4. Open any notebook in VS Code and run its cells in order, top to bottom. Each notebook
   is an independent kernel — if you close and reopen one, you need to re-run all of its
   cells from the start.

## Path convention (important)

Each notebook lives in a different folder and needs to go up a different number of
levels to reach the repo root (where `col_amazon_fire_utils.py` and the `data/` folder
live). That's why the first cell of every notebook has something like:

```python
import os, sys
sys.path.insert(0, os.path.abspath('../..'))   # goes up 2 levels to the root
```

The number of `..` depends on how deep the notebook sits (check the folder tree above).
If you move a notebook to a different folder, **you must update this number** — and any
hardcoded output paths in that same cell.

## Notes and troubleshooting

- If a notebook fails to import `col_amazon_fire_utils`, check the `sys.path.insert` in
  its first cell — it probably needs more or fewer `'..'` for its depth.
- Google Earth Engine calls can be slow; most notebooks cache intermediate results in
  `data/processed/` or `data/model_dataset/` so they don't recompute every run. Only
  delete those CSVs if you want to force a full recalculation.
- The `fire_thesis` environment needs `earthengine-api`, `geemap`, and `geopandas` for
  the spatial parts.
