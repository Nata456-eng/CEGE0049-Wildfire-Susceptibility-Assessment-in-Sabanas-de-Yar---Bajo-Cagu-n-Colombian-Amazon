"""
Validation, Phase 2, Step 1 -- Generate a fire-susceptibility probability raster from
the EVALUATION model (Random Forest v4, trained ONLY on data through 2019), predicting
over the study nucleus using climate conditions from 2001-2019 only.

WHY THIS RASTER IS DIFFERENT FROM `fire_susceptibility_probability_v4.tif`:
`outputs/probability_map/fire_susceptibility_probability_v4.tif` comes from the
DEPLOYED/MAPPING model -- refit on the FULL dataset (all years 2001-2024, train+test
combined) -- and its climatological-normal predictor stack averages climate over
2001-2024. Using that raster (or that model) against the SINCHI 2020-2024 burn scars
would NOT be a genuine out-of-sample test, because that model already saw 2020-2024
during training, and its climate predictors already reflect those years.

This script instead reproduces the EVALUATION model from `tuning/v4` -- the one whose
reported AUC/PR-AUC/F1 "performance metrics" describe true out-of-sample skill (temporal
hold-out: train <=2019, test >=2020, see `tuning/v4/README.md`). Concretely:

1. Rebuild the SAME 7-15 km annulus, ratio 1:1 dataset used everywhere in this project
   (`pixel_year_full.csv`, seed 42) -- built across ALL years first, exactly like
   `tuning/v4/v4_comparison_buffer_7_15km.ipynb` and
   `outputs/probability_map/predict_susceptibility_map.ipynb` do (the temporal split is
   applied AFTER the dataset is assembled, not before -- matching `tuning/v4`'s own
   `train_mask = model_df[YEAR_COL] <= TRAIN_END_YEAR` cell exactly).
2. Fit the Random Forest with `tuning/v4`'s best hyperparameters, but ONLY on the rows
   with year <= 2019 -- the exact `X_train`/`y_train` split `tuning/v4` evaluated on its
   2020-2024 holdout. This is the "evaluation model": it has never seen 2020-2024.
3. Build the climate predictor stack (`temp_C`, `vpd_kPa`, `ndvi`, `wind_ms`) as a
   per-pixel mean of ONLY the 2001-2019 dry seasons (Dec[Y-1]-Feb[Y]) -- NOT 2001-2024.
   This matters just as much as excluding 2020-2024 rows from training: if the
   prediction covariates baked in 2020-2024 climate conditions, the resulting map would
   partially reflect information the model was never supposed to see, contaminating the
   out-of-sample comparison even though the model object itself was "clean".
4. The 4 static anthropogenic layers (`dist_roads`, `dist_parks`, `dist_coca`,
   `dist_mosaic`) are kept as the same "current" snapshot used everywhere else in this
   project -- these were never time-varying features in the training data itself (see
   `outputs/probability_map/README.md`'s assumptions section), so reusing the current
   snapshot here does not leak 2020-2024 information that the model didn't already
   implicitly rely on.
5. `oni` is set to 0 (ENSO-neutral), matching the deployed map's convention -- unrelated
   to the leakage concern above.
6. Predicts over every pixel of the nucleus at 500 m / EPSG:4326 (matching the grid the
   model was trained on), then reprojects to EPSG:32618 at an explicit 500 m resolution
   with NEAREST-NEIGHBOR resampling -- consistent with how this project always treats a
   continuous-but-to-be-classified susceptibility surface (see
   `carbon_estimation/04_resample_susceptibility_for_arcgis.py`): every output pixel
   keeps EXACTLY the value of the one native pixel it falls in, no interpolation.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 01_generate_susceptibility_model2019.py
"""

import ast
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import col_amazon_fire_utils as utils

DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

PIXEL_YEAR_PATH = REPO_ROOT / "data" / "model_dataset" / "pixel_year_full.csv"
V4_HP_PATH = REPO_ROOT / "tuning" / "v4" / "v4_buffer_7_15km" / "v4_best_hyperparameters.csv"

MODEL_OUT = HERE / "rf_model2019.joblib"
STACK_TIF = DATA_DIR / "predictor_stack_model2019.tif"
PROB_500M_TIF = DATA_DIR / "susceptibility_model2019_500m.tif"
PROB_UTM_OUT = HERE / "susceptibility_model2019_500m_UTM.tif"

pred_cols = ["dist_roads", "dist_parks", "dist_coca", "dist_mosaic",
             "temp_C", "vpd_kPa", "ndvi", "wind_ms", "oni"]

TRAIN_END_YEAR = 2019   # matches tuning/v4's temporal split exactly
CLIMATE_START_YEAR = 2001
CLIMATE_END_YEAR = 2019  # inclusive -- climatological normal restricted to training years only

NODATA = -9999.0
DST_CRS = CRS.from_epsg(32618)  # WGS84 / UTM zone 18N -- matches every other UTM layer
TARGET_RES = 500.0  # metres -- forced explicitly for the final UTM raster

print("=" * 70)
print("SETUP -- Earth Engine + nucleus geometry")
print("=" * 70)
ee = utils.initialize_ee()
nucleus_geom = utils.get_nucleus_geometry()
print("EE initialized, nucleus geometry ready.")

# ---------------------------------------------------------------------------
# Step 1a -- Rebuild the v4 (7-15 km annulus, ratio 1:1) dataset across ALL years
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 1a -- Rebuilding the v4 annulus dataset (all years, same recipe as tuning/v4)")
print("=" * 70)

pixel_year = pd.read_csv(PIXEL_YEAR_PATH)

BLOCK = 0.25
INNER_BUFFER = 7000
OUTER_BUFFER = 15000
inner_deg = INNER_BUFFER / 111000
outer_deg = OUTER_BUFFER / 111000
RATIO = 1
SEED = 42

presences = pixel_year[pixel_year["burned"] == 1].copy()
absences_all = pixel_year[pixel_year["burned"] == 0].copy()

kept_absences = []
for y in sorted(pixel_year["year"].unique()):
    pres_y = presences[presences["year"] == y][["lon", "lat"]].values
    abs_y = absences_all[absences_all["year"] == y]
    if len(pres_y) == 0:
        kept_absences.append(abs_y)
        continue
    tree = cKDTree(pres_y)
    dists, _ = tree.query(abs_y[["lon", "lat"]].values, k=1)
    in_ring = (dists >= inner_deg) & (dists <= outer_deg)
    kept_absences.append(abs_y[in_ring])
absences_ring = pd.concat(kept_absences, ignore_index=True)

n_needed = int(round(RATIO * len(presences)))
print(f"Annulus {INNER_BUFFER/1000:.0f}-{OUTER_BUFFER/1000:.0f} km | "
      f"eligible: {len(absences_ring):,} | needed: {n_needed:,}")

# NOTE: an explicit per-year loop is used here (instead of
# `groupby("year").apply(...)`) because in this environment's pandas version
# `groupby(...).apply(lambda g: g.sample(...))` silently DROPS the "year"
# grouping column from the returned rows, which would later break the
# temporal train/test split entirely (all rows would look like they have no
# absences for any single year once "year" disappears). The loop below is
# semantically identical to the intended recipe (same per-year sample size,
# same random_state) but keeps every original column intact.
sample_parts = []
for y in sorted(absences_ring["year"].unique()):
    g = absences_ring[absences_ring["year"] == y]
    n = min(len(g), int((presences["year"] == y).sum() * RATIO))
    if n > 0:
        sample_parts.append(g.sample(n=n, random_state=SEED))
absences_sample = pd.concat(sample_parts, ignore_index=True)

model_df = pd.concat([presences, absences_sample], ignore_index=True) \
             .sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Full annulus dataset (all years): {model_df.shape}, "
      f"{int(model_df['burned'].sum())} presences")

# ---------------------------------------------------------------------------
# Step 1b -- Temporal split: keep ONLY year <= 2019 for training (the evaluation model)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print(f"STEP 1b -- Temporal split: training on year <= {TRAIN_END_YEAR} ONLY")
print("=" * 70)

train_mask = model_df["year"] <= TRAIN_END_YEAR
X_train = model_df.loc[train_mask, pred_cols]
y_train = model_df.loc[train_mask, "burned"].astype(int).values

n_excluded = int((~train_mask).sum())
print(f"Training rows (year <= {TRAIN_END_YEAR}): {train_mask.sum():,}")
print(f"Excluded rows (year >= 2020, held out, NEVER used to fit this model): {n_excluded:,}")
assert model_df.loc[train_mask, "year"].max() <= TRAIN_END_YEAR, \
    "Training data leaked a year > 2019 -- check the temporal split."
print(f"Verified: max year in training data = {model_df.loc[train_mask, 'year'].max()} "
      f"(<= {TRAIN_END_YEAR}, no leakage)")

# ---------------------------------------------------------------------------
# Step 1c -- Fit the Random Forest with tuning/v4's hyperparameters, on X_train only
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 1c -- Fitting the evaluation-model Random Forest (train <= 2019 only)")
print("=" * 70)


def _parse_param(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if float(v).is_integer() else float(v)
    s = str(v)
    if s == "None":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


RF_PARAMS = {"n_estimators": 500, "max_features": 0.5, "min_samples_leaf": 5, "max_depth": None}
v4_hp = pd.read_csv(V4_HP_PATH)
rf_row = v4_hp[v4_hp["model"] == "Random Forest"]
if len(rf_row):
    RF_PARAMS = {c: _parse_param(rf_row.iloc[0][c]) for c in v4_hp.columns
                 if c != "model" and pd.notna(rf_row.iloc[0][c])}
print("RF hyperparameters (tuning/v4):", RF_PARAMS)

rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_PARAMS)
rf.fit(X_train, y_train)
print(f"Evaluation-model RF trained on {X_train.shape[0]:,} rows "
      f"(year <= {TRAIN_END_YEAR} only).")

joblib.dump(rf, MODEL_OUT)
print(f"Saved: {MODEL_OUT}")

# ---------------------------------------------------------------------------
# Step 2 -- Build the 9-band predictor image (climate restricted to 2001-2019)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print(f"STEP 2 -- Building predictor stack (climate normal: {CLIMATE_START_YEAR}-{CLIMATE_END_YEAR})")
print("=" * 70)

# --- 2a: static anthropogenic distance layers (unchanged "current" snapshot -- see
# module docstring point 4 for why this doesn't leak 2020-2024 information) ---
MAP_SEARCH_RADIUS = 100000

mb_lulc = ee.Image("projects/mapbiomas-colombia/assets/LULC/COLECCION3/INTEGRACION/COLOMBIA-1")
from_codes = [3, 6, 5, 4, 11, 12, 13, 15, 18, 35, 21, 33, 9, 23, 24, 29, 30, 68]
to_groups = [1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 6, 7, 7, 7, 7, 7, 7]

base = "projects/col-amazon-fire-susceptibility/assets/"
layers = {
    "parks": ee.FeatureCollection(base + "national_natural_parks"),
    "roads": ee.FeatureCollection(base + "osm_roads_v2"),
    "coca": ee.FeatureCollection(base + "coca_cultivation_v2"),
}

buf = nucleus_geom.buffer(MAP_SEARCH_RADIUS)

dist_roads = layers["roads"].filterBounds(buf).distance(MAP_SEARCH_RADIUS).rename("dist_roads")
dist_parks = layers["parks"].filterBounds(buf).distance(MAP_SEARCH_RADIUS).rename("dist_parks")


def coca_ever(f):
    total = ee.List([ee.Number(f.get(f"coca_{y}")) for y in range(2001, 2024)]).reduce(ee.Reducer.sum())
    return f.set("coca_total", total)


coca_any = layers["coca"].filterBounds(buf).map(coca_ever).filter(ee.Filter.gt("coca_total", 0))
dist_coca = coca_any.distance(MAP_SEARCH_RADIUS).rename("dist_coca")

mosaic_mask = mb_lulc.select("classification_2024").clip(buf).remap(from_codes, to_groups, 0).eq(5)
dist_mosaic = (mosaic_mask.selfMask().fastDistanceTransform().sqrt()
               .multiply(ee.Image.pixelArea().sqrt()).rename("dist_mosaic"))

dist_stack = dist_roads.addBands(dist_parks).addBands(dist_coca).addBands(dist_mosaic)
print("Static distance predictors ready:", dist_stack.bandNames().getInfo())


# --- 2b: climate stack, per-pixel mean of ONLY 2001-2019 dry seasons ---
def climate_year(year):
    """Dry-season (Dec[y-1]-Feb[y]) climate image for ONE year."""
    start = ee.Date.fromYMD(year - 1, 12, 1)
    end = ee.Date.fromYMD(year, 3, 1)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(start, end)

    def derive(img):
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        wind = u.hypot(v).rename("wind_ms")
        t = img.select("temperature_2m").subtract(273.15)
        td = img.select("dewpoint_temperature_2m").subtract(273.15)
        es = t.expression("0.6108*exp(17.27*T/(T+237.3))", {"T": t})
        e = td.expression("0.6108*exp(17.27*Td/(Td+237.3))", {"Td": td})
        return img.addBands([wind, es.subtract(e).rename("vpd_kPa")])

    era5 = era5.map(derive)
    temp = era5.select("temperature_2m").mean().subtract(273.15).rename("temp_C")
    vpd = era5.select("vpd_kPa").mean()
    wind = era5.select("wind_ms").mean()
    ndvi = (ee.ImageCollection("MODIS/061/MOD13A1").filterDate(start, end)
            .select("NDVI").mean().multiply(0.0001).rename("ndvi"))
    return temp.addBands([vpd, ndvi, wind])


yearly_climate = ee.ImageCollection(
    [climate_year(y) for y in range(CLIMATE_START_YEAR, CLIMATE_END_YEAR + 1)]
)
climate_normal = yearly_climate.mean()  # per-pixel 2001-2019 dry-season mean ONLY

oni_neutral = ee.Image.constant(0).rename("oni").toFloat()

print(f"Climate normal built from {CLIMATE_END_YEAR - CLIMATE_START_YEAR + 1} dry seasons "
      f"({CLIMATE_START_YEAR}-{CLIMATE_END_YEAR}) -- 2020-{2024} EXCLUDED.")
print("Climatological-normal climate bands:", climate_normal.bandNames().getInfo())

# --- 2c: assemble the 9-band stack, band order must match pred_cols ---
predictor_image = (dist_stack
                    .addBands(climate_normal)
                    .addBands(oni_neutral)
                    .select(pred_cols)
                    .toFloat()
                    .clip(nucleus_geom)
                    .unmask(-9999, False))

band_names = predictor_image.bandNames().getInfo()
assert band_names == pred_cols, f"Band order mismatch: {band_names} != {pred_cols}"
print("Predictor stack ready, band order verified:", band_names)

# ---------------------------------------------------------------------------
# Step 3 -- Export the predictor stack, 500 m / EPSG:4326
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3 -- Exporting predictor stack (500 m, EPSG:4326)")
print("=" * 70)

import geemap  # noqa: E402  (imported here, same pattern as predict_susceptibility_map.ipynb)

geemap.ee_export_image(
    predictor_image,
    filename=str(STACK_TIF),
    scale=500,
    crs="EPSG:4326",
    region=nucleus_geom,
    file_per_band=False,
)
print(f"Saved: {STACK_TIF}")

# ---------------------------------------------------------------------------
# Step 4 -- Predict susceptibility probability for every pixel
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 4 -- Predicting susceptibility probability (evaluation model, <=2019)")
print("=" * 70)

with rasterio.open(STACK_TIF) as src:
    profile = src.profile.copy()
    stack = src.read()
    assert stack.shape[0] == len(pred_cols), \
        f"Expected {len(pred_cols)} bands, got {stack.shape[0]}"

n_bands, H, W = stack.shape
flat = stack.reshape(n_bands, -1).T

valid = ~np.any((flat == NODATA) | np.isnan(flat), axis=1)
print(f"Valid pixels: {valid.sum():,} / {flat.shape[0]:,} ({100 * valid.mean():.1f}%)")

proba = np.full(flat.shape[0], NODATA, dtype=np.float32)
X_pred = pd.DataFrame(flat[valid], columns=pred_cols)  # keep feature names, matches X_train
proba[valid] = rf.predict_proba(X_pred)[:, 1].astype(np.float32)
proba_grid = proba.reshape(H, W)

profile.update(count=1, dtype="float32", nodata=NODATA)
with rasterio.open(PROB_500M_TIF, "w", **profile) as dst:
    dst.write(proba_grid, 1)
    dst.set_band_description(1, "fire_susceptibility_probability_model2019")

valid_vals = proba_grid[proba_grid != NODATA]
print(f"Saved: {PROB_500M_TIF}")
print(f"Probability range: [{valid_vals.min():.3f}, {valid_vals.max():.3f}] | "
      f"mean: {valid_vals.mean():.3f}")

# ---------------------------------------------------------------------------
# Step 5 -- Reproject to EPSG:32618, explicit 500 m, NEAREST NEIGHBOR
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5 -- Reprojecting to EPSG:32618, 500 m, nearest neighbor")
print("=" * 70)

with rasterio.open(PROB_500M_TIF) as src:
    src_transform = src.transform
    src_crs = src.crs
    src_data = src.read(1)
    src_nodata = src.nodata
    src_bounds = src.bounds

dst_transform, dst_width, dst_height = calculate_default_transform(
    src_crs, DST_CRS, src.width, src.height, *src_bounds,
    resolution=(TARGET_RES, TARGET_RES),
)

dst_data = np.full((dst_height, dst_width), NODATA, dtype="float32")
reproject(
    source=src_data,
    destination=dst_data,
    src_transform=src_transform,
    src_crs=src_crs,
    src_nodata=src_nodata,
    dst_transform=dst_transform,
    dst_crs=DST_CRS,
    dst_nodata=NODATA,
    resampling=Resampling.nearest,  # preserve exact model output values, no interpolation
)

utm_profile = {
    "driver": "GTiff",
    "height": dst_height,
    "width": dst_width,
    "count": 1,
    "dtype": "float32",
    "crs": DST_CRS,
    "transform": dst_transform,
    "nodata": NODATA,
    "compress": "deflate",
}
with rasterio.open(PROB_UTM_OUT, "w", **utm_profile) as dst:
    dst.write(dst_data, 1)

print(f"Saved: {PROB_UTM_OUT}")
print(f"  shape={dst_data.shape}, CRS={DST_CRS}, resolution={TARGET_RES} m")
print()
print("Done. This raster (susceptibility_model2019_500m_UTM.tif) represents the")
print("EVALUATION model's (trained <=2019 only) susceptibility prediction, using ONLY")
print(f"{CLIMATE_START_YEAR}-{CLIMATE_END_YEAR} climate conditions -- ready to compare")
print("against the SINCHI 2020-2024 burn scars from Phase 1 as a genuine out-of-sample check.")
