# `validation/` — Out-of-sample validation of the fire susceptibility map

Validates `fire_susceptibility_probability_v4.tif` (the 500 m Random Forest v4
susceptibility surface in
[`outputs/probability_map/`](../outputs/probability_map/README.md)) against an
**independent** source of burned-area evidence: SINCHI's historical burn-scar polygons,
derived from Landsat at 1:100,000 scale. This is a genuine out-of-sample check — the
model was never trained on this dataset (training/evaluation used MODIS MCD64A1 burned
area, not SINCHI burn scars).

## Source data

**Raw input (not modified in place):**
[`Cicatrices_de_quema_por_regiB3n_(HistB3rico)._Escala3A100.000_SINCHI/`](Cicatrices_de_quema_por_regiB3n_(HistB3rico)._Escala3A100.000_SINCHI/)
— shapefile of burn-scar polygons across the Colombian Amazon, SINCHI methodology,
1:100,000 scale (Landsat-derived, ~30 m), **EPSG:4170** (MAGNA-SIRGAS, geographic/degrees).

Relevant columns:
- `periodo` — year (text)
- `mes` — month name, in Spanish (`'Enero'`, `'Febrero'`, ...)
- `categoria` — land-cover type at the time of the burn scar (`'Bosque'`, `'Otras
  coberturas'`, `'Vegetación secundaria o en transición'`)
- `area_hecta` — polygon area in hectares
- `departamen`, `nom_munici` — administrative location
- `geometry` — `Polygon`/`MultiPolygon`

**Folder-name note:** the outer folder name has mangled characters
(`regiB3n`/`HistB3rico`/`Escala3A100.000` instead of `región`/`histórico`/`Escala 1:100.000`)
— an encoding artifact from how it was originally extracted/downloaded, left as-is
rather than renamed, so it's clear this is exactly the file the user provided.

## Phase 1 — Data cleaning and preparation

**Folder:** [`01_data_cleaning/`](01_data_cleaning/)

Goal: turn the raw SINCHI shapefile into a clean, filtered, reprojected GeoDataFrame
ready to be rasterized and compared pixel-for-pixel against the 500 m susceptibility
raster.

### Step 1 — Load and baseline report

Script: [`01_data_cleaning/01_load_and_report.py`](01_data_cleaning/01_load_and_report.py)

Loads the raw shapefile with geopandas and reports, before touching anything:
- initial record count
- source CRS
- temporal range (unique `periodo` values)

This is the verifiable starting point every later cleaning step is compared against.

**Result:**
- Initial record count: **15,212** polygons
- CRS: **EPSG:4170** (MAGNA-SIRGAS, geographic)
- Columns: `objectid`, `categoria`, `fecha_regi`, `path_row`, `area_hecta`, `cantidad_p`,
  `periodo`, `periodo_me`, `mes`, `area_km`, `sigla_car`, `departamen`, `nom_munici`,
  `estado_leg`, `paisaje`, `shape_Leng`, `shape_Area`, `geometry`
- `periodo` (year, text) spans **2017–2026**, 10 distinct values: `2017, 2018, 2019,
  2020, 2021, 2022, 2023, 2024, 2025, 2026`. Note `2026` (the current year) appearing —
  worth checking in a later step whether those records are provisional/partial-year.

### Step 2 — Spatial clip to the study nucleus

Script: [`01_data_cleaning/02_clip_to_nucleus.py`](01_data_cleaning/02_clip_to_nucleus.py)

Clips the raw burn scars to the study nucleus using a **spatial intersection**
(`gpd.clip`), not a department/municipality name filter — the nucleus is smaller than,
and doesn't align with, the administrative boundaries of the departments/municipalities
it overlaps, so an attribute filter would both include area outside the nucleus and
potentially miss real slivers of it.

Reuses the nucleus boundary already exported for the carbon-at-risk overlay
(`carbon_estimation/resampled_ArcGIS_30m/nucleus_boundary_UTM.shp`, EPSG:32618) rather
than re-querying Earth Engine, reprojecting it to the burn-scar layer's own CRS
(EPSG:4170) just for the clip. The burn-scar layer's own CRS is left untouched at this
step — reprojection to a final working CRS happens in a later cleaning step.

**Result:**
- Records before clip: **15,212**
- Records after clip (inside the nucleus): **6,333**
- Dropped (outside the nucleus): **8,879**
- CRS preserved: EPSG:4170

**Output:** [`01_data_cleaning/data/02_clipped_to_nucleus.gpkg`](01_data_cleaning/data/02_clipped_to_nucleus.gpkg)

### Step 3 — Filter to dry-season months (Dec–Jan–Feb)

Script: [`01_data_cleaning/03_filter_dry_season.py`](01_data_cleaning/03_filter_dry_season.py)

Builds a numeric `mes_num` column by mapping the Spanish month names in `mes` to 1–12,
after normalizing case/whitespace (`.str.strip().str.lower()`) so variants like
`' Enero'`, `'ENERO'`, `'enero '` all map correctly instead of silently becoming NaN.
Then keeps only the dry-season months this project's model was actually trained on:
**Dec(Y-1) + Jan(Y) + Feb(Y)** — the same definition used throughout the project (see
`col_amazon_fire_utils.get_climate_df`'s `dry_season_predictors`, and
`model/logistic_regression/README.md`: "24 dry seasons, Dec[Y-1]-Feb[Y]"). Comparing the
susceptibility map (fit on Dec–Jan–Feb fire behavior) against burn scars from other
months wouldn't be a fair out-of-sample test.

**Result:**
- All `mes` values mapped successfully (0 unmapped).
- Records before filter: **6,333**
- Records after filter (Dec, Jan, Feb only): **2,332**
- Dropped (outside Dec–Jan–Feb): **4,001**
- Retained breakdown: Enero (1) = 732, Febrero (2) = 960, Diciembre (12) = 640

**Output:** [`01_data_cleaning/data/03_dry_season_filtered.gpkg`](01_data_cleaning/data/03_dry_season_filtered.gpkg)

### Step 4 — Filter to the out-of-sample validation period (2020–2024)

Script: [`01_data_cleaning/04_filter_validation_period.py`](01_data_cleaning/04_filter_validation_period.py)

Converts `periodo` (stored as text) to a numeric year and keeps only **2020–2024
inclusive**.

**Critical distinction — WHICH model this validates (read before Phase 2):**
this project has two different fitted versions of the winning Random Forest v4 model,
and they are **not interchangeable** for validation purposes:

| | Trained on | Role |
|---|---|---|
| **Evaluation model** (`tuning/v4`) | Train ≤2019, test ≥2020 (temporal hold-out) | The one whose reported AUC/PR-AUC/F1 "performance metrics" describe out-of-sample skill |
| **Deployed/mapping model** (`predict_susceptibility_map.ipynb`) | Refit on the FULL dataset, all years 2001–2024 (train+test combined) | Generated `fire_susceptibility_probability_v4.tif`, the map used for the carbon-at-risk overlay and stakeholder communication |

This validation targets the **evaluation model** — the one whose performance metrics are
actually reported for the project — against an independent burn-scar source (SINCHI)
it never saw. That's exactly why the SINCHI records are restricted to **2020–2024**: it
is the precise window the evaluation model was held out from during training, making
this a genuine out-of-sample check.

**This has a direct consequence for Phase 2:** the raster to compare against burn scars
must be the evaluation model's predictions (re-fit on ≤2019 data only, or the fitted
estimator from `tuning/v4`) — **not** `fire_susceptibility_probability_v4.tif` directly,
since that raster's underlying model was already trained on 2020–2024 and using it here
would silently invalidate the "out-of-sample" claim.

**Result:**
- Records before filter: **2,332**
- Records after filter (2020–2024 inclusive): **1,227**
- Dropped (outside 2020–2024): **1,105**
- Retained breakdown by year: 2020 = 108, 2021 = 165, 2022 = 242, 2023 = 286, 2024 = 426

**Output:** [`01_data_cleaning/data/04_validation_period_2020_2024.gpkg`](01_data_cleaning/data/04_validation_period_2020_2024.gpkg)

### Step 5 — Reproject to EPSG:32618 (final step of Phase 1)

Script: [`01_data_cleaning/05_reproject_to_utm.py`](01_data_cleaning/05_reproject_to_utm.py)

Reprojects the final filtered GeoDataFrame to **EPSG:32618** (WGS84 / UTM zone 18N) —
the same CRS as the susceptibility raster and every other UTM layer in this project
(`carbon_estimation/`, etc.). Deliberately the **last** step in Phase 1, after every
filter (nucleus clip, dry-season filter, validation-period filter) — reprojecting
recomputes every vertex of every polygon, so it's applied once, only to the geometries
that actually survive filtering, not to the ~15,212 raw records.

**Result:**
- Records: **1,227** (unchanged by reprojection — only the CRS changes)
- CRS check: **PASSED** — resulting CRS confirmed as EPSG:32618

**Output:** [`01_data_cleaning/data/05_burn_scars_validation_ready_UTM.gpkg`](01_data_cleaning/data/05_burn_scars_validation_ready_UTM.gpkg)
— **the Phase 1 deliverable**: clean, filtered (nucleus + dry season + 2020–2024
validation period), reprojected to EPSG:32618, ready to rasterize and compare in
Phase 2 against the **evaluation model's** (≤2019-trained) predictions — not directly
against `fire_susceptibility_probability_v4.tif` (see the Step 4 note above).

## Phase 1 summary (record count through each step)

| Step | Filter applied | Records remaining |
|---|---|---|
| 1 | None (raw load) | 15,212 |
| 2 | Spatial clip to nucleus | 6,333 |
| 3 | Dry season only (Dec/Jan/Feb) | 2,332 |
| 4 | Validation period (2020–2024) | 1,227 |
| 5 | Reprojected to EPSG:32618 (no record change) | 1,227 |

## Phase 2 — Susceptibility map from the evaluation model (≤2019)

**Folder:** [`02_model_2019_prediction/`](02_model_2019_prediction/)

Goal: produce a susceptibility raster generated **only** from the evaluation model
(trained on data through 2019, never on 2020–2024) and **only** from 2001–2019 climate
conditions, so it can be compared against the SINCHI 2020–2024 burn scars from Phase 1
as a genuine out-of-sample test. As established in Phase 1 / Step 4 above, this raster
is deliberately **not** `fire_susceptibility_probability_v4.tif` — that raster's
underlying model was refit on the full 2001–2024 dataset (including 2020–2024), which
would invalidate the out-of-sample comparison.

### Step 1 — Generate the evaluation-model susceptibility raster

Script: [`02_model_2019_prediction/01_generate_susceptibility_model2019.py`](02_model_2019_prediction/01_generate_susceptibility_model2019.py)

Two independent things had to be restricted to ≤2019, or the "out-of-sample" claim would
be invalid even if the other one were done correctly:

1. **Training data:** rebuilds the same 7–15 km annulus, 1:1 ratio pseudo-absence
   dataset used throughout this project (`data/model_dataset/pixel_year_full.csv`,
   seed 42) — built across all years first, then filtered to `year <= 2019`, exactly
   reproducing the temporal hold-out split used in `tuning/v4` (whose reported
   AUC/PR-AUC/F1 metrics are the ones actually cited for this project). Fits a Random
   Forest with `tuning/v4`'s best hyperparameters
   (`n_estimators=500, max_features=0.5, min_samples_leaf=5`) on this ≤2019 subset only.
2. **Prediction covariates:** the climate bands (`temp_C`, `vpd_kPa`, `ndvi`, `wind_ms`)
   are built as a per-pixel dry-season (Dec[Y-1]–Feb[Y]) mean over **2001–2019 only**,
   excluding 2020–2024 — otherwise the predicted map would partially reflect climate
   information from the very years being held out, even though the model object itself
   was "clean". The 4 static anthropogenic layers (`dist_roads`, `dist_parks`,
   `dist_coca`, `dist_mosaic`) reuse the same "current" snapshot as the deployed map,
   since these were never time-varying training features to begin with. `oni` is set to
   0 (ENSO-neutral), matching the deployed map's convention.

Predicts probability for every pixel of the nucleus at 500 m / EPSG:4326, then
reprojects to **EPSG:32618** at an explicit 500 m resolution with **nearest-neighbor**
resampling (no interpolation — same convention as every other continuous-but-to-be-
classified susceptibility layer in this project, e.g.
`carbon_estimation/04_resample_susceptibility_for_arcgis.py`).

**Implementation note:** the per-year pseudo-absence sampling was originally attempted
with `groupby("year").apply(lambda g: g.sample(...))`, but in this environment's pandas
version that silently **drops the `year` column** from the returned rows — which broke
the temporal split downstream (every training row appeared to have no year, so the
`year <= 2019` filter kept only the presences, which still carried their own `year`
values from a different code path, and none of the sampled absences). Fixed by
replacing it with an explicit per-year loop that preserves every original column.

**Result:**
- Full annulus dataset (all years): 4,154 rows (2,077 presences, 1:1 balanced)
- Training rows (year ≤ 2019): **3,656** (1,828 presences + 1,828 pseudo-absences)
- Held-out rows (year ≥ 2020, never used to fit this model): 498
- Verified: max year in training data = 2019 (no leakage)
- RF hyperparameters used: `{'max_features': 0.5, 'min_samples_leaf': 5, 'n_estimators': 500}`
- Climate normal built from 19 dry seasons (2001–2019); 2020–2024 explicitly excluded
- Valid pixels predicted: 453,916 / 936,936 (48.4% — the rest fall outside the nucleus mask)
- Probability range: **[0.001, 0.947]**, mean **0.194**
- Final raster: shape (816, 1145), CRS EPSG:32618, resolution 500 m

**Outputs:**
- [`02_model_2019_prediction/rf_model2019.joblib`](02_model_2019_prediction/rf_model2019.joblib) — the fitted evaluation-model estimator (≤2019 training data only)
- [`02_model_2019_prediction/susceptibility_model2019_500m_UTM.tif`](02_model_2019_prediction/susceptibility_model2019_500m_UTM.tif) — **the Phase 2 Step 1 deliverable**: susceptibility probability from the evaluation model, using only 2001–2019 climate conditions, EPSG:32618, 500 m, nearest-neighbor resampled — ready to compare pixel-for-pixel against `01_data_cleaning/data/05_burn_scars_validation_ready_UTM.gpkg`
- `02_model_2019_prediction/data/predictor_stack_model2019.tif`, `02_model_2019_prediction/data/susceptibility_model2019_500m.tif` — intermediate native-grid (500 m / EPSG:4326) files

### Step 2 — Rasterize burn scars to burn-area fraction, on the exact same grid

Script: [`02_model_2019_prediction/02_rasterize_burn_fraction.py`](02_model_2019_prediction/02_rasterize_burn_fraction.py)

Rasterizes the Phase 1 deliverable (`05_burn_scars_validation_ready_UTM.gpkg`) onto
**exactly** the same 500 m grid as `susceptibility_model2019_500m_UTM.tif` — that raster
is loaded purely as a template (its transform, dimensions, CRS and pixel alignment are
inherited directly; no new grid is generated).

Instead of a binary burned/not-burned flag, each output cell holds the **fraction** of
its 500 m footprint actually covered by burn-scar polygons (0–1), to preserve the burn-
intensity gradient rather than collapsing every partially-burned cell to the same value
as a fully-burned one. Computed via sub-pixel oversampling (rasterio/GDAL has no native
"rasterize as area fraction" mode): every 500 m template cell is subdivided into 20×20
sub-cells of 25 m each (matching the ~30 m native scale of the Landsat-derived SINCHI
scars), the polygons are rasterized onto that fine grid as 0/1, and each block of
20×20 sub-cells is averaged back down to one fraction value per 500 m cell. The
template's own valid-pixel mask is then re-applied so cells outside the nucleus are
NoData (`-9999`, not a false `0`), while cells inside the nucleus with no burn-scar
overlap correctly get an explicit `0`.

**Result:**
- Fine (sub-pixel) grid: 16,320 × 22,900 = 373,728,000 cells at 25 m
- Burned sub-cells: 2,605,524 (0.70% of the fine grid)
- Aggregated 500 m grid: (816, 1145) — confirmed identical to the template
- Cells with no burn-scar overlap (fraction = 0, inside nucleus): 423,299
- Cells with some burn-scar overlap (fraction > 0): 27,749
- NoData cells (outside the nucleus): 483,272
- Burn-fraction range (valid cells only): **[0.0000, 1.0000]**, mean **0.0144**
- Post-write verification: shape, transform and CRS confirmed **identical** to
  `susceptibility_model2019_500m_UTM.tif`

**Output:** [`02_model_2019_prediction/cicatrices_burnfraction_500m_UTM.tif`](02_model_2019_prediction/cicatrices_burnfraction_500m_UTM.tif)
— pixel-for-pixel aligned with the Step 1 susceptibility raster, ready for direct
cell-by-cell comparison.

## Phase 3 — Final validation: fraction of observed fire captured per susceptibility class

**Folder:** [`03_final_validation/`](03_final_validation/)

Goal: the final cell-by-cell comparison between the susceptibility raster and the
SINCHI burn scars — quantify what fraction of the total observed 2020–2024 burned area
falls in each Jenks susceptibility class, with a focus on the **high** and **very high**
classes. As established in Phase 1/Phase 2, this is a genuine **out-of-sample** check:
the susceptibility raster comes from the evaluation model (trained ≤2019 only, never saw
2020–2024 fire), and SINCHI's burn scars are independently derived from Landsat —
a different sensor and methodology from the MODIS MCD64A1 data the model's `burned`
target was trained on. This measures real predictive capacity, not in-sample fit.

### Step 1 — Load, align, and define constants

Script: [`03_final_validation/01_load_and_align.py`](03_final_validation/01_load_and_align.py)

Loads `susceptibility_model2019_500m_UTM.tif` (Phase 2, Step 1) and
`cicatrices_burnfraction_500m_UTM.tif` (Phase 2, Step 2) as NoData-respecting masked
arrays. Since both rasters were already built on the identical grid, this step only runs
a quick safety assert (shape, transform, CRS) rather than re-deriving alignment. Defines
the cell area (500 m × 500 m = 25 ha = 0.25 km²) and the Jenks breaks used to classify
susceptibility everywhere else in this project (same breaks as
`carbon_estimation/06_carbon_at_risk_by_susceptibility.py`):
`very low (<0.117)`, `low (0.117–0.25)`, `moderate (0.25–0.408)`, `high (0.408–0.612)`,
`very high (≥0.612)`.

**Because SINCHI's raster stores a burn-area FRACTION (not a binary flag), every later
"burned area" computation in this phase must be `burn_fraction × cell_area`, summed per
class — never a count of "burned" cells × cell area** (see Phase 2, Step 2 discussion
above on why the fractional formulation is the more precise one).

**Result:**
- Alignment assert: **PASSED** — identical shape (816, 1145), transform, and CRS
  (EPSG:32618) between both rasters
- Valid (in-nucleus) pixels: **451,048** in both rasters (identical NoData mask, as expected)
- Susceptibility probability range: [0.0008, 0.9466], mean 0.1939
- Burn-fraction range: [0.0000, 1.0000], mean 0.0144
- Total observed burned area (sum of fraction × cell area, 2020–2024 dry seasons,
  nucleus only): **1,628.35 km²**

### Step 2 — Common valid-data mask

Script: [`03_final_validation/02_common_valid_mask.py`](03_final_validation/02_common_valid_mask.py)

Builds a mask of cells where **both** rasters have real (non-NoData) data, and restricts
every subsequent Phase 3 computation to that intersection — equal valid-pixel *counts*
(confirmed in Step 1) do not by themselves prove the valid cells sit in the exact same
*locations*; a cell valid in one raster but NoData in the other could otherwise silently
have its NoData sentinel (`-9999`) treated as real data, biasing every downstream statistic.

**Result:**
- Valid cells — susceptibility only: 451,048 / 934,320
- Valid cells — burn fraction only: 451,048 / 934,320
- Cells valid in one raster but not the other: **0** — masks are identical
- **Common valid cells: 451,048** (48.28% of the grid) — used for every subsequent
  Phase 3 computation

**Output:** [`03_final_validation/data/common_valid_mask.tif`](03_final_validation/data/common_valid_mask.tif)

### Step 3 — Classify susceptibility into the 5 Jenks classes

Script: [`03_final_validation/03_classify_susceptibility.py`](03_final_validation/03_classify_susceptibility.py)

Classifies every common-valid cell of the continuous susceptibility raster into the
project's five Jenks classes (same breaks as Step 1 / `carbon_estimation/06_carbon_at_risk_by_susceptibility.py`),
via `np.digitize`, restricted to the common valid mask from Step 2. Cells outside the
common mask get an explicit NoData sentinel (`255`), distinct from the 0–4 class codes.

**Result:**

| Class | Cells | % of valid | Area (km²) |
|---|---:|---:|---:|
| very_low (<0.117) | 199,927 | 44.32% | 49,981.75 |
| low (0.117–0.25) | 118,127 | 26.19% | 29,531.75 |
| moderate (0.25–0.408) | 79,402 | 17.60% | 19,850.50 |
| high (0.408–0.612) | 43,529 | 9.65% | 10,882.25 |
| very_high (≥0.612) | 10,063 | 2.23% | 2,515.75 |

Verified: every one of the 451,048 common-valid cells was classified (no cell left
unclassified or double-counted).

**Output:** [`03_final_validation/data/susceptibility_class_common.tif`](03_final_validation/data/susceptibility_class_common.tif)
(uint8 class codes 0–4, NoData=255)

### Step 4 — Observed burned area per cell and total

Script: [`03_final_validation/04_burned_area_observed.py`](03_final_validation/04_burned_area_observed.py)

Computes the observed burned area for every common-valid cell as
`burn_fraction × 25 ha`, then sums it across the common mask to get the total observed
burned area for the 2020–2024 dry seasons within the nucleus.

**Result:**
- Per-cell burned area range: [0.0000, 25.0000] ha, mean 0.3610 ha
- **Total observed burned area: 162,834.75 ha = 1,628.35 km²** — matches the Step 1
  sanity-check total exactly, confirming the common-mask restriction changed nothing
  (the two rasters' valid cells were already identical, per Step 2)

**Output:** [`03_final_validation/data/burned_area_observed_ha.tif`](03_final_validation/data/burned_area_observed_ha.tif)

### Step 5 (FINAL) — Capture fraction by susceptibility class

Script: [`03_final_validation/05_capture_rate_by_class.py`](03_final_validation/05_capture_rate_by_class.py)

For each of the 5 Jenks classes, computes:
- `burned_area_class_ha` = sum of `burn_fraction × 25 ha` over the class's cells
- **`capture_fraction`** (the main validation metric) = `burned_area_class_ha / total_burned_area_ha`
  — what proportion of all observed 2020–2024 fire fell into this class
- `pct_of_total_area` (context column only, **not** an additional metric) =
  `class_area_ha / total_common_area_ha` — lets the capture fraction be read against how
  much of the landscape the class occupies. A class that is a small share of the area
  but captures a large share of the fire is a much stronger validation result than one
  where the two shares are similar (roughly what random chance would predict).

**Result:**

| Class | Cells | Class area (ha) | Burned area (ha) | **Capture fraction** | % of total area |
|---|---:|---:|---:|---:|---:|
| very_low | 199,927 | 4,998,175.00 | 22,237.75 | **0.1366** | 0.4432 |
| low | 118,127 | 2,953,175.00 | 30,205.62 | **0.1855** | 0.2619 |
| moderate | 79,402 | 1,985,050.00 | 36,275.00 | **0.2228** | 0.1760 |
| high | 43,529 | 1,088,225.00 | 25,884.44 | **0.1590** | 0.0965 |
| very_high | 10,063 | 251,575.00 | 48,231.94 | **0.2962** | 0.0223 |

Consistency checks (all passed): capture fractions sum to 1.000000; area percentages
sum to 1.000000; sum of per-class burned area (162,834.75 ha) matches the Step 4 total
exactly; sum of per-class cell counts (451,048) matches the common-valid total exactly.

**Headline result:** the **high** and **very high** classes together cover only
**11.88%** of the landscape area, yet captured **45.52%** of all observed 2020–2024
burned area — measured against an independent, out-of-sample source (SINCHI/Landsat
burn scars) that the evaluation model (trained ≤2019) never saw. The `very_high` class
alone (2.23% of the area) captured nearly 30% of all observed fire — almost 13× what its
area share alone would predict.

**Output:** [`03_final_validation/capture_rate_by_class.csv`](03_final_validation/capture_rate_by_class.csv)
— **the final technical deliverable of the entire validation workflow**.

### Step 6 — Plain-language stakeholder table (IDEAM, UNGRD, communities)

Script: [`03_final_validation/06_stakeholder_summary.py`](03_final_validation/06_stakeholder_summary.py)

Re-expresses the Step 5 results without statistical jargon, for a non-technical
decision-maker audience. Ordered **very high → very low** (most decision-relevant class
first — the natural reading order when scanning top-to-bottom for "where should we
focus"), unlike the low-to-high gradient order used in the technical Step 5 table.

| Risk level | % of territory | % of total fire | Burn intensity (%) |
|---|---:|---:|---:|
| Very high | 2.2 | 29.6 | 19.2 |
| High | 9.7 | 15.9 | 2.4 |
| Moderate | 17.6 | 22.3 | 1.8 |
| Low | 26.2 | 18.5 | 1.0 |
| Very low | 44.3 | 13.7 | 0.4 |

Both percentage columns sum to exactly 100.0% (verified in the script).

**Headline figures for stakeholders:**
- *"Out of every 100 hectares burned, 45.5 were in areas marked as high or very high
  risk, which occupy only 11.9% of the territory."*
- *"Fire was 43 times more intense in very-high-risk areas than in very-low-risk areas."*

**Footnote (also embedded in the CSV):** *"This validation compares the map against
real fires observed independently (Landsat satellite, SINCHI historical data) that
occurred between 2020 and 2024 — years the model never saw during training. It's a fair
test: the model had to anticipate fire that hadn't happened yet when it was built."*

**Output:** [`03_final_validation/validation_stakeholder_summary.csv`](03_final_validation/validation_stakeholder_summary.csv)
— the plain-language table plus the two headline sentences and the footnote, appended
as trailing comment lines in the same CSV.

### Step 7 — Scar-level (per-fire-event) detection rate

Script: [`03_final_validation/07_scar_level_detection_rate.py`](03_final_validation/07_scar_level_detection_rate.py)

Complementary to Step 5's area-weighted capture fraction (which large scars can
dominate). Answers a different, more intuitive question: **out of all the individual
fire scars SINCHI recorded, how many did the model actually flag as high risk?** — every
fire event counts once, regardless of its size.

For each of the 1,227 individual scar polygons, clips the classified raster (Step 3) to
that polygon's footprint (`all_touched=True`, so scars smaller than one 500 m pixel
still register) and computes the class covering the most overlapping pixels (its
"dominant class"), plus whether it touches *any* high/very-high pixel at all. Two
detection rates are reported: **strict** (majority of the scar's footprint was
high/very high) and **lenient** (at least some overlap with high/very high).

**Result** (1,226 / 1,227 scars had at least one overlapping pixel; 1 tiny edge sliver excluded):

| Detection rule | High or very high | Moderate or above |
|---|---:|---:|
| Strict (majority of scar's footprint) | 229 / 1,226 (**18.7%**) | 733 / 1,226 (59.8%) |
| Lenient (any overlap) | 632 / 1,226 (**51.5%**) | 1,000 / 1,226 (81.6%) |

Dominant-class distribution across all scars: very_high 6.1%, high 12.6%, moderate
41.1%, low 28.7%, very_low 11.5%.

**Headline sentence:** *"Out of 1,226 individual fire scars recorded by SINCHI
(2020–2024, out-of-sample), 229 (18.7%) had most of their burned footprint in a zone the
model marked as high or very high risk. Widening to 'moderate risk or above', that rises
to 733 scars (59.8%)."*

**Interpretation note:** this is a stricter test than Step 5's area-weighted capture
fraction, because it gives a tiny 1-hectare scar the same vote as a massive one — and
because "majority of footprint" is a demanding bar (a scar straddling a moderate/high
boundary, mostly on the moderate side, still counts as a "miss" here even though the
model flagged real elevated risk nearby). The lenient version (51.5% touched high/very
high) is a fairer read of "did the model see this coming at all."

**Output:** [`03_final_validation/scar_level_detection_rate.csv`](03_final_validation/scar_level_detection_rate.csv)
— per-scar dominant class, high/moderate-overlap flags and percentages, and scar area.

## Validation workflow — final summary

This out-of-sample validation confirms that the fire susceptibility map's high-risk
classes concentrate real, independently-observed fire far beyond what their area share
alone would predict:

- The evaluation model was trained only on data through 2019 and never saw any
  2020–2024 fire.
- The comparison source (SINCHI burn scars) is Landsat-derived, independent of the
  MODIS MCD64A1 data the model's `burned` target was trained on.
- **high + very_high classes: 11.88% of the area → 45.52% of observed fire captured**
  (area-weighted, Step 5).
- **51.5% of individual fire scars touched a high/very-high zone; 18.7% had the
  majority of their footprint there** (per-event, Step 7) — a stricter, size-independent
  view of the same result.
