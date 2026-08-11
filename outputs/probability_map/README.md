# `outputs/probability_map/` — Pixel-level fire susceptibility map (Random Forest v4)

**Notebook:** [`predict_susceptibility_map.ipynb`](predict_susceptibility_map.ipynb)

Applies the winning model — Random Forest, v4 (7–15 km annulus pseudo-absence sampling,
ratio 1:1), the version validated in [`tuning/v4`](../../tuning/v4/README.md) and interpreted in
[`outputs/shap/shap_v2_rf_v4.ipynb`](../shap/shap_v2_rf_v4.ipynb) — to **every
pixel** of the study nucleus, instead of just the sampled pixel-year table used for
training/evaluation. This answers item 4 ("2026 map") of the roadmap in
[`WORKFLOW.md`](../../WORKFLOW.md).

## Step by step — how this map was built

1. **Picked which "year" the pixel grid represents.** Of the 9 predictors, 4 are already
   static/current in the training pipeline (`dist_roads`, `dist_parks`, `dist_coca`,
   `dist_mosaic`), but 5 vary by year (`temp_C`, `vpd_kPa`, `ndvi`, `wind_ms`, `oni`). Asked
   the user to choose between (a) the most recent year (2024) as a literal snapshot, (b) a
   2001–2024 **climatological normal** as a year-independent baseline, or (c) both. Chose
   **(b)**, consistent with `WORKFLOW.md`'s own framing ("susceptibility, not forecast" —
   a fixed snapshot, not a forecast for any specific year).

2. **Reused the project's existing "final model" recipe instead of inventing a new one.**
   `outputs/shap/shap_v2_rf_v4.ipynb` already establishes how this project turns the
   tuned v4 hyperparameters into a deployable model: rebuild the 7–15 km annulus, 1:1
   ratio dataset from `data/model_dataset/pixel_year_full.csv` (seed 42), load
   `tuning/v4/v4_buffer_7_15km/v4_best_hyperparameters.csv`, and fit
   `RandomForestClassifier` on the **full** dataset (all years, train+test combined — not
   just the pre-2020 training split used during evaluation). This notebook copies that
   recipe verbatim so the deployed model is *identical* to the one already validated and
   SHAP-interpreted, not a re-derived variant. The fitted model is persisted to
   `rf_v4_final.joblib`.

3. **Built a 9-band predictor image in Google Earth Engine covering the whole study
   nucleus** (the dissolved geometry of the 5 municipalities, ~112,290 km²):
   - The 4 static anthropogenic distance layers, using the same GEE assets and logic as
     `model/logistic_regression/baseline_comparison.ipynb` (2024 MapBiomas classification
     for `dist_mosaic`, cumulative coca-ever-grown through 2023 for `dist_coca`, current
     roads/parks assets) — see the search-radius sidenote below for the one deliberate
     change from that notebook.
   - The 4 climate bands as a **per-pixel** mean of the 2001–2024 dry-season (Dec(y-1)–
     Feb(y)) images — i.e. the spatial gradient in temperature/VPD/NDVI/wind across the
     nucleus is preserved; it's not collapsed to one global number.
   - `oni` set to a constant **0** (ENSO-neutral) everywhere.
   - Bands assembled in the exact order the model expects (`dist_roads`, `dist_parks`,
     `dist_coca`, `dist_mosaic`, `temp_C`, `vpd_kPa`, `ndvi`, `wind_ms`, `oni`), with an
     assertion in the notebook that fails loudly if that order ever drifts.

4. **Exported the 9-band image as a local GeoTIFF** (`predictor_stack_v4.tif`) via
   `geemap.ee_export_image`, at 500 m resolution / EPSG:4326 (matching the scale used to
   build `pixel_year_full.csv`), directly downloaded (no Google Drive round-trip needed —
   the whole stack is only ~16 MB).

5. **Ran the fitted Random Forest on every pixel** of the exported stack (`rf.predict_proba`),
   masking out pixels with a nodata value in any band, and wrote the result back out as
   `fire_susceptibility_probability_v4.tif` — a single-band float32 GeoTIFF, probability of
   burning in [0, 1], nodata = -9999.

6. **Sanity-checked the result before calling it done** — this is where two real bugs
   turned up and got fixed (see the technical sidenotes below): a masking bug that
   silently corrupted ~52% of the raster with zeros instead of nodata, and a training-time
   search-radius cap that left large nodata holes *inside* the study area. Verification
   steps used: per-band nodata-fraction checks (should equal the polygon-vs-bounding-box
   area ratio, ~51.6%, identically across all 9 bands), a coarse ASCII rendering of the
   valid-pixel mask (to confirm it traces the actual 5-municipality outline, not some
   tiling artifact), and checking that the output probability distribution has a real
   spread (no suspiciously-tied quartiles, which was the first symptom that something was
   wrong).

7. **Generated a quick-look PNG** (`quicklook_probability_map.png`) for a visual sanity
   check before opening the GeoTIFF in ArcGIS.

## Assumptions made (flag these if your context changes)

- **Climatological-normal conditions, not a forecast.** Confirmed with the user (see step
  1) — if you instead want a specific year's actual conditions (e.g. "what would 2024's
  dry season look like under today's human pressure"), the climate-stack cell needs to
  call `climate_year(2024)` instead of averaging 2001–2024, and `oni` should use that
  year's real ONI value instead of 0.
- **Refit on the full dataset, not just the pre-2020 training split.** Matches the
  existing SHAP notebook's convention: once a model is selected and validated, the
  deployed/interpreted version is refit on all available data. The temporal/spatial
  validation metrics reported in `tuning/v4` still describe out-of-sample performance —
  they were computed on the held-out split before this final refit.
- **"Current" human-pressure layers, not year-matched to the climate.** `dist_mosaic` uses
  the 2024 MapBiomas classification and `dist_coca` uses cumulative coca-ever-grown
  through 2023 regardless of which climate year/normal is used — this was already true of
  the training pipeline (these two layers were never computed per-year), so the map
  mixes "current-as-of-2023/2024 land use" with "2001–2024-average climate," which is a
  reasonable reading of "susceptibility under today's human footprint and typical dry-season
  weather."
- **500 m resolution, EPSG:4326.** Chosen to match the scale already used to build
  `pixel_year_full.csv`, so the deployment grid isn't finer or coarser than what the model
  was actually trained on. Reproject in ArcGIS if you need a different CRS (e.g. a UTM
  zone for area-accurate distance/area measurements downstream).

## Technical sidenotes — things that bit us, read before re-running

- **`ee.Image.unmask(value)` defaults to `sameFootprint=True`.** This only fills gaps
  *inside* the image's current footprint (e.g. after `.clip()`) — pixels genuinely outside
  that footprint stay masked, and Earth Engine's raw pixel-fetch API silently writes `0`
  for still-masked pixels instead of any nodata sentinel. The first export run looked
  fine at a glance (real-looking min/max/mean) but had ~52% of the raster quietly replaced
  by zeros outside the study polygon, which corrupted downstream predictions there. Fixed
  by calling `.unmask(-9999, False)` (`sameFootprint=False`) so -9999 extends across the
  *entire* export rectangle, not just the clipped footprint. **The tell that something was
  wrong:** the output probability distribution had its 25th/50th/75th percentiles all
  reporting the exact same value — a huge cluster of pixels sharing bit-identical
  predictor values across every band is essentially always a red flag, not real data.
- **The 10 km search radius on `dist_roads`/`dist_coca` was a training-sampling
  optimization, not an ecological cutoff.** `baseline_comparison.ipynb` caps these two
  `.distance()` calls at 10 km purely to keep the ~8,000-point GEE sample cheap; points
  farther than that come back masked and were dropped via `.dropna()` when
  `pixel_year_full.csv` was built. Reusing that same 10 km cap for a full-region raster
  left large nodata holes *inside* the nucleus (any pixel >10 km from both a road and any
  historical coca polygon). Fixed by widening it to 100 km — the same radius
  `dist_parks` already used, which empirically covers the entire nucleus with zero extra
  masking (Colombia's road and coca networks are denser than its park network, so 100 km
  is more than sufficient). **Caveat this leaves behind:** the model was never trained on
  `dist_roads`/`dist_coca` values above 10 km, so predictions in pixels beyond that range
  (deep-forest interior, far from any road or coca field) are a model **extrapolation**,
  not an interpolation. Those areas also come out uniformly low-susceptibility in the
  quicklook map, which is the expected direction of extrapolation for a distance-based
  predictor, but treat that part of the map with more caution than the road/agricultural-
  frontier corridors the model actually learned from.
- **Probabilities are calibrated to the training class balance, not true prevalence.**
  The Random Forest was trained on a 1:1 balanced sample (annulus pseudo-absences), not
  the true ~2.5% real-world fire rate. Read pixel values as **relative susceptibility
  ranking** (exactly what's needed for a Jenks/quantile classification into low→high
  classes), not as a literal "probability this pixel burns in a given year."
- **File-locking during re-runs:** if you open `fire_susceptibility_probability_v4.tif` in
  ArcGIS Pro (or QGIS) while iterating on the notebook, the next `rasterio.open(..., 'w')`
  call will fail with a permission-denied error until you close that layer/project — GIS
  desktop apps keep an open file handle on rasters you've added to a map.
- **Band order is load-bearing.** `predictor_image.select(pred_cols)` plus an `assert`
  right after it exists specifically to catch silent reordering — `rf.predict_proba`
  has no idea what a column "means," it just trusts positional order matching training.
  If you ever add/reorder predictors, that assertion (in section 2c of the notebook) is
  what will catch a mismatch instead of it silently producing garbage predictions.

## Outputs (generated by running the notebook — not committed if large)
- `rf_v4_final.joblib` — the fitted Random Forest model.
- `predictor_stack_v4.tif` — intermediate 9-band covariate GeoTIFF.
- `fire_susceptibility_probability_v4.tif` — **the deliverable**: single-band float32
  GeoTIFF, EPSG:4326, ~500 m resolution, pixel value = P(burned) in [0, 1], nodata = -9999.
  Load this directly into ArcGIS. Valid (non-nodata) coverage is ~48.4% of the bounding
  box, matching the nucleus polygon's actual share of its own bounding box.
- `quicklook_probability_map.png` — quick preview for sanity-checking before ArcGIS.
- `arcgis_layers/` — each of the 9 predictors, plus two MODIS fire products, exported as
  individual single-band files for ArcGIS — see below.

## Environment
Run with the `fire_thesis` conda environment (`earthengine-api`, `geemap`, `rasterio`,
`scikit-learn`). Requires Earth Engine authentication for project
`col-amazon-fire-susceptibility` (same as every other notebook in this repo).

To re-run end to end from the command line (e.g. after editing the notebook):

```powershell
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "outputs\probability_map\predict_susceptibility_map.ipynb"
```

Make sure nothing has the output `.tif` files open (ArcGIS/QGIS) before re-running — see
the file-locking sidenote above.

## Turning this into susceptibility classes
This notebook stops at the continuous probability surface. Classifying it into discrete
levels (Very low → Very high) via Natural Breaks (Jenks) is the next roadmap item — either
in ArcGIS directly (`Symbology → Classify → Jenks`, 5 classes) or in a follow-up notebook.

---

## `arcgis_layers/` — individual layers for ArcGIS

The 9 predictor variables of the final model + two MODIS fire products, each as a
single-band GeoTIFF (or shapefile), so you don't have to pick a band out of a
multi-band raster when opening them in ArcGIS.

### Files

| File | Variable | Unit / typical range |
|---|---|---|
| `01_dist_roads.tif` | Distance to roads | metres |
| `02_dist_parks.tif` | Distance to national parks | metres |
| `03_dist_coca.tif` | Distance to coca crops (historical, through 2023) | metres |
| `04_dist_mosaic.tif` | Distance to agropastoral mosaic (agricultural frontier, MapBiomas 2024) | metres |
| `05_temp_C.tif` | Temperature, dry-season average 2001–2024 | °C |
| `06_vpd_kPa.tif` | Vapour pressure deficit, dry-season average 2001–2024 | kPa |
| `07_ndvi.tif` | NDVI, dry-season average 2001–2024 | 0–1 |
| `08_wind_ms.tif` | Wind speed, dry-season average 2001–2024 | m/s |
| `09_oni.tif` | ONI index (ENSO) | constant 0 (neutral) |
| `10_modis_fire_frequency_2001_2024.tif` | Number of years (of 24) MODIS detected a burn in that pixel | 0–24 |
| `11_modis_fire_frequency_2001_2024_polygons.shp` | Vector (polygon) version of layer 10 | see below |

The first 9 are individual bands split from `../predictor_stack_v4.tif` (the same file
that feeds the susceptibility map) — see the main sections above for the full detail of
how that stack was built. Layers 10 and 11 are new (see below).

### Common grid — all 11 layers overlay pixel-for-pixel

All layers share exactly the same grid: 500 m, EPSG:4326, same extent and same
`transform` (819 × 1144 pixels). Verified explicitly when the MODIS layer was generated,
not just assumed — so you can load them all into ArcGIS and compare them cell by cell
without reprojecting/resampling anything.

**Nodata = -9999** in all 10 raster layers, ~51.6% of the bounding box (the 5-municipality
nucleus isn't a rectangle, so the mask outside the polygon is the majority of the
bounding box). The source file (`predictor_stack_v4.tif`) doesn't carry a declared nodata
value in its GDAL metadata (see the technical sidenote above) — when splitting the bands
here, `nodata=-9999` was declared explicitly in each file so ArcGIS renders it as
transparent instead of as a real value.

### The MODIS layer (`10_modis_fire_frequency_2001_2024.tif`) — decision made

Not a single-year snapshot: it's the **count of years, 2001–2024, in which
MODIS/061/MCD64A1 detected a burn in that pixel** (0 to 24). This representation was
chosen — rather than a single point-in-time image — for consistency with the other 9
layers: the climate predictors are already a 2001–2024 average (a climatological
baseline, not a specific year — `WORKFLOW.md`: "susceptibility, not forecast"), and
2001–2024 is exactly the year range of `data/model_dataset/pixel_year_full.csv`, the
table the model was trained on. The frequency layer is the fire-side counterpart of that
same baseline: "how many times did this pixel actually burn in the years the model
learned from" — directly comparable against the 9 predictor layers instead of depending
on an arbitrarily chosen year.

Uses the same "burned" definition as `col_amazon_fire_utils.get_burned_df` (a burned
year = `BurnDate > 0` on any day of that calendar year), just kept per-pixel here instead
of reduced to one number for the whole nucleus.

**If you need a different representation** (a specific year, a binary
"ever-burned" mask, or "years since the last fire"), it's a small change in
`02_export_modis_fire_frequency.py` — ask for it to be regenerated with that definition.

**Quick read of the result:** only ~13% of the nucleus's valid pixels burned at least
once in 24 years (frequency > 0); most of those pixels burned 1–2 times, with a long tail
of high-recurrence pixels. Consistent with what's already visible in
`../maps/burned_area_quintiles_basemap.png`: fire is strongly concentrated in the
northwest quadrant of the nucleus, not uniformly distributed.

### The vector layer (`11_modis_fire_frequency_2001_2024_polygons.shp`)

Layer 10 converted to polygons: each polygon is a patch of contiguous (500 m) pixels
that share the same number of years burned, 2001–2024. Only patches that burned at
least once are kept — the rest of the nucleus (never burned) isn't exported as a polygon
(it would be one giant polygon with no analytical value). It isn't an "average"
geometry in the literal sense (you can't average the shape of 24 different years of
fires); "average 2001–2024" is expressed here as each polygon's `avg_freq` field: the
fraction of those 24 years in which that specific patch burned.

Fields (names truncated to Shapefile's 10-character limit):

- `yrs_burned` (integer, 1–24): years burned, 2001 to 2024.
- `avg_freq` (float, 0–1): `yrs_burned / 24` — the requested "average," per polygon.
- `area_ha` (float): geodesic area in hectares (computed with `pyproj.Geod`, not
  distorted by the file's geographic EPSG:4326 projection).

Current result: 18,533 polygons, ~1,474,309 ha with at least one burn in the period
(most burned 1–3 times; a long tail of high-recurrence patches reaches up to 16 of 24
years). Full distribution in the output of `03_polygonize_modis_fire_frequency.py`.

### Inherited caveats (apply here the same way as on the probability map)

- **Extrapolation in `dist_roads`/`dist_coca` beyond 10 km.** The model was trained with
  those two variables capped at 10 km (a sampling optimisation); in these layers the
  radius was widened to 100 km to cover the whole nucleus without gaps, so large values in
  `01_dist_roads.tif` / `03_dist_coca.tif` represent extrapolation, not interpolation.
- **`dist_mosaic` uses 2024 land cover and `dist_coca` uses coca accumulated through
  2023**, while climate is a 2001–2024 average — "current" human layers, "typical"
  climate, not one coherent calendar year.
- **500 m / EPSG:4326** is the native resolution of the whole stack (limited by MODIS and
  by the ERA5 climate reanalysis, much coarser than 500 m) — don't resample to a finer
  resolution expecting to gain real detail.

### How they were generated / how to regenerate

```powershell
cd "outputs\probability_map\arcgis_layers"
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 01_split_predictor_bands.py
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 02_export_modis_fire_frequency.py
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 03_polygonize_modis_fire_frequency.py
```

- `01_split_predictor_bands.py` — local, instant, reads `../predictor_stack_v4.tif` and
  splits its 9 bands. Only needs re-running if the stack changes.
- `02_export_modis_fire_frequency.py` — calls Google Earth Engine (project
  `col-amazon-fire-susceptibility`, requires authentication already configured), takes
  ~1 minute.
- `03_polygonize_modis_fire_frequency.py` — local, instant, reads the output of the
  script above and vectorises it. Requires `geopandas`, `pyproj`, `shapely` (already in
  `fire_thesis`).
