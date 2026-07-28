# `probability map/` — Pixel-level fire susceptibility map (Random Forest v4)

**Notebook:** [`predict_susceptibility_map.ipynb`](predict_susceptibility_map.ipynb)

Applies the winning model — Random Forest, v4 (7–15 km annulus pseudo-absence sampling,
ratio 1:1), the version validated in [`tuning/v4`](../tuning/v4/) and interpreted in
[`outputs/shap/shap_v2_rf_v4.ipynb`](../outputs/shap/shap_v2_rf_v4.ipynb) — to **every
pixel** of the study nucleus, instead of just the sampled pixel-year table used for
training/evaluation. This answers item 4 ("Mapa 2026") of the roadmap in
[`WORKFLOW.md`](../WORKFLOW.md).

## Step by step — how this map was built

1. **Picked which "year" the pixel grid represents.** Of the 9 predictors, 4 are already
   static/current in the training pipeline (`dist_roads`, `dist_parks`, `dist_coca`,
   `dist_mosaic`), but 5 vary by year (`temp_C`, `vpd_kPa`, `ndvi`, `wind_ms`, `oni`). Asked
   the user to choose between (a) the most recent year (2024) as a literal snapshot, (b) a
   2001–2024 **climatological normal** as a year-independent baseline, or (c) both. Chose
   **(b)**, consistent with `WORKFLOW.md`'s own framing ("susceptibilidad, no pronóstico" —
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

## Environment
Run with the `fire_thesis` conda environment (`earthengine-api`, `geemap`, `rasterio`,
`scikit-learn`). Requires Earth Engine authentication for project
`col-amazon-fire-susceptibility` (same as every other notebook in this repo).

To re-run end to end from the command line (e.g. after editing the notebook):

```powershell
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "probability map\predict_susceptibility_map.ipynb"
```

Make sure nothing has the output `.tif` files open (ArcGIS/QGIS) before re-running — see
the file-locking sidenote above.

## Turning this into susceptibility classes
This notebook stops at the continuous probability surface. Classifying it into discrete
levels (Muy bajo → Muy alto) via Natural Breaks (Jenks) is the next roadmap item — either
in ArcGIS directly (`Symbology → Classify → Jenks`, 5 classes) or in a follow-up notebook.
