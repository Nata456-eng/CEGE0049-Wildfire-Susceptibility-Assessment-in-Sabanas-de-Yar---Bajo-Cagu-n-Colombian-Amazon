# `carbon_estimation/` — Phase 5 of `WORKFLOW.md`: carbon at risk

Estimates carbon stock exposed to fire susceptibility, by combining aboveground biomass
with the susceptibility map in [`outputs/probability_map/`](../outputs/probability_map/README.md).
Numbered scripts, one per pipeline step (more will be added as later steps of Phase 1,
and later phases, are built).

**Naming note:** the folder is `carbon_estimation` (underscore), not `carbon estimation`
(space) as originally requested — every other folder in this repo uses lowercase +
underscores, no spaces (`logistic_regression`, `nucleus_extraction`, `model_dataset`...),
and this project's structure was just cleaned up specifically to enforce that convention.
Flag it if you actually wanted the literal name with a space.

## Phase 1, Step 1 — `01_extract_agb_biomass.py`

Downloads Aboveground Live Woody Biomass Density (AGB) from Global Forest Watch,
clips it to the study nucleus, and writes a local GeoTIFF ready for ArcGIS.

**Source:** [GFW — Aboveground Live Woody Biomass Density](https://data.globalforestwatch.org/datasets/gfw::aboveground-live-woody-biomass-density/about)
(WHRC/Zarin et al. 2016 pantropical biomass map), year-2000 baseline, native ~30 m
resolution (10°/40000 px ≈ 27.8 m, close enough to "30 m" that GFW calls it that), units
**Megagrams of biomass per hectare (Mg/ha)**.

### Why this isn't a GEE `Export.image.toDrive()` call, despite what was originally asked
Checked first, not assumed: this specific 30 m dataset is **not** a pre-ingested Earth
Engine asset. The only WHRC biomass product in the official EE Data Catalog is
`WHRC/biomass/tropical`, a coarser 500 m *national-level* product — a different,
lower-resolution dataset. GFW distributes the 30 m one as 10°×10° GeoTIFF tiles behind a
signed-URL download API instead:
```
https://data-api.globalforestwatch.org/dataset/whrc_aboveground_woody_biomass_stock_2000/v1.4/download/geotiff?grid=10/40000&tile_id={tile_id}&pixel_meaning=Mg_ha-1&x-api-key=...
```
confirmed by hand — the endpoint 307-redirects to a temporary-credentialed S3 URL, not a
public `gs://` URI Earth Engine's `loadGeoTIFF` could read directly. Getting it into GEE
would mean uploading the tiles to a Google Cloud Storage bucket and ingesting them as an
asset first — extra setup, and no actual benefit, since no server-side EE computation is
needed here (the 2 covering tiles are already known). So: download the 2 tiles directly,
clip to the nucleus with `rasterio`, done locally — same deliverable (a 30 m GeoTIFF
clipped to the nucleus, EPSG:4326, ready for ArcGIS) without the GEE round-trip.

### Which tiles, and how that was confirmed
The nucleus spans roughly lon -75.63 to -70.48, lat -0.74 to 2.95 (from
`predictor_stack_v4.tif`'s extent) — straddling the equator, entirely inside the 80°W
tile column. That means tiles **`10N_080W`** and **`00N_080W`**. This wasn't assumed from
the naming convention alone — both tile IDs were tested against the download endpoint and
confirmed to redirect to a real S3 object before being hardcoded into the script.

### What the script does
1. Calls `col_amazon_fire_utils.get_nucleus_geometry()` (the same Earth Engine call every
   other notebook in this repo uses to define "the nucleus") and pulls its GeoJSON via one
   `.getInfo()` — the only Earth Engine usage in this script, just for the boundary, not
   for any raster processing.
2. Downloads the 2 tiles (`10N_080W.tif`, `00N_080W.tif`, ~1.1–1.5 GB each) from the GFW
   data API into `raw_tiles_cache/` (gitignored — re-running skips tiles already
   downloaded).
3. Mosaics them with `rasterio.merge`.
4. Clips to the **exact nucleus polygon** (not just its bounding box) with
   `rasterio.mask`, matching the nodata convention used everywhere else in this project.
5. Resamples to exactly **30 m** (`30 / 111,320` degrees/pixel, the same
   degrees-per-metre approximation already used elsewhere in this repo, e.g.
   `tuning/v4`'s distance calculations) in an **explicit CRS, EPSG:4326** — matching every
   other raster in this project.
6. Writes `agb_density_mg_ha_2000_30m.tif`, DEFLATE-compressed (see file-size note below).

### Nodata caveat — read before interpreting near-zero values
The source tiles are `uint16` with **nodata = 0**, declared in their own GDAL metadata.
That means this dataset **cannot distinguish a genuine zero-biomass pixel** (bare ground,
water, urban, recently cleared land) **from a true data gap** — both read as 0. That's a
property of the source product, not something this script introduces. Valid-pixel
coverage came out to 46.9% of the bounding box, a bit below the ~51.6%
polygon-vs-bounding-box ratio seen on every other raster in this project (500 m
predictor stack, MODIS fire layers) — the gap is most likely real zero/no-data pixels
(rivers, clearings) inside the nucleus being folded into nodata by this convention, not a
clipping bug.

### Result (sanity-checked)
13,635 × 19,063 pixels, AGB range **1.0–397.3 Mg/ha**, mean **248.0 Mg/ha** — a
sensible range for tropical Amazon forest. `quicklook_agb_density.png` shows the same
5-municipality outline as every other map in this project, with visibly lower biomass in
the northwest — the same area already identified as the most fire-affected quadrant in
[`outputs/maps/burned_area_quintiles_basemap.png`](../outputs/maps/burned_area_quintiles_basemap.png)
and in the SHAP/fire-frequency work — a reassuring cross-check, not a coincidence:
degraded, more fire-prone land tends to carry less standing biomass.

### File size — why the deliverable isn't committed to git
At 30 m over the whole ~112,290 km² nucleus, the output is inherently large:
**~1.0 GB uncompressed**, **~364 MB** even with DEFLATE + a floating-point predictor
(still applied — see the `profile` dict in the script). That's far above what this repo
has committed before (previous largest tracked file: `rf_v4_final.joblib`, 25 MB) and
above GitHub's 100 MB hard limit without Git LFS. Per your call, `agb_density_mg_ha_2000_30m.tif`
and `raw_tiles_cache/` are both gitignored — the deliverable is generated locally, ready
to open directly in ArcGIS, but doesn't get pushed. Re-run the script to regenerate it
any time; the raw tile cache means a re-run after the first one only needs to redo the
(fast) merge/clip/reproject step, not the ~2.5 GB tile download.

### Outputs
- `raw_tiles_cache/10N_080W.tif`, `00N_080W.tif` — raw source tiles (gitignored).
- `agb_density_mg_ha_2000_30m.tif` — **the deliverable**: single-band float32 GeoTIFF,
  EPSG:4326, exactly 30 m resolution, pixel value = AGB in Mg/ha, nodata = -9999
  (gitignored, see above).
- `quicklook_agb_density.png` — preview PNG (committed — small, useful even without the
  full-resolution file).

### Environment
`fire_thesis` conda environment (`requests`, `rasterio`, `shapely`, `earthengine-api` —
all already used elsewhere in this repo). Requires Earth Engine authentication (same
project, `col-amazon-fire-susceptibility`) only for the one nucleus-geometry call.

```powershell
cd carbon_estimation
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 01_extract_agb_biomass.py
```

## Phase 1, Step 2 — `02_convert_biomass_to_carbon.py`

Converts the AGB raster from Step 1 (Mg **biomass**/ha) into Aboveground Carbon Density
(Mg **C**/ha), by multiplying every valid pixel by **0.5** — a fixed factor, not a
re-derived model.

**Why 0.5:** dry woody biomass is conventionally assumed to be ~50% carbon by mass (IPCC
2006 Guidelines for National GHG Inventories default carbon fraction; species-specific
values are commonly cited a little lower, ~0.47, but the source AGB map carries no
species information to apply a finer factor, so 0.5 is the standard simplified default
used across most tropical-forest carbon studies in this situation).

**Scope caveat:** this is **aboveground carbon only**. Belowground (root) biomass, dead
wood, litter, and soil organic carbon are not included and would need separate
data/factors to add later if the thesis needs total ecosystem carbon rather than just
the aboveground live-woody pool.

### Result (sanity-checked)
Same grid as the AGB raster (46.9% valid, unchanged — the nodata mask is copied over
exactly, not recomputed). Carbon density range **0.5–198.7 Mg C/ha**, mean **124.0 Mg
C/ha** — exactly half of the Step 1 AGB range/mean, as expected from a flat ×0.5. The
quicklook shows the identical spatial pattern as `quicklook_agb_density.png`, just
rescaled.

### Outputs
- `carbon_density_mg_c_ha_2000_30m.tif` — single-band float32 GeoTIFF, same grid/CRS as
  `agb_density_mg_ha_2000_30m.tif`, pixel value = Mg C/ha, nodata = -9999. **Gitignored**
  — same reasoning as the AGB raster (~407 MB compressed, same order of magnitude, still
  far above what this repo commits).
- `quicklook_carbon_density.png` — preview PNG (committed).

```powershell
cd carbon_estimation
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 02_convert_biomass_to_carbon.py
```

## Phase 1, Step 3 — `03_hansen_correction.py`

The AGB/carbon rasters from Steps 1–2 are a **year-2000 baseline** — they say nothing
about forest cleared since then. This step corrects for that using Hansen Global Forest
Change, zeroing out carbon wherever forest loss was detected between 2001 and 2024, so
the result is "carbon in forest that **still exists**," not "carbon that existed in
2000."

### Step by step

1. **Load the Hansen image and verify it, in Earth Engine, before assuming anything.**
   Dataset: `UMD/hansen/global_forest_change_2024_v1_12` — loaded directly and its band
   list printed (`treecover2000`, `loss`, `gain`, `lossyear`, ...) to confirm the bands
   this script depends on actually exist, rather than assuming the asset ID and band
   names from the task description alone. (Note: Earth Engine flags this specific
   version as deprecated in favour of `..._2025_v1_13` — kept the exact version named in
   the request; swap it for the 2025 release later if you want loss years through 2025.)
2. **Verified `lossyear` actually spans 2001–2024** — queried its min/max inside the
   nucleus directly: **1–24**, all 24 distinct values present. Confirms the "loss
   between 2001–2024" premise the mask logic depends on.
3. **Found and fixed a real bug before it could corrupt the mask:** the task said
   "pixels with `lossyear`== 0 → no loss." But `lossyear` doesn't store a literal `0`
   for "no loss ever" pixels — Earth Engine **masks** those pixels (leaves them as
   nodata) instead. Checked with a histogram of `lossyear` inside the nucleus: only keys
   `1`–`24` ever appear, never `0`. A literal `lossyear.eq(0)` would therefore have
   stayed masked (not evaluated to 1) almost everywhere loss did **not** happen — the
   exact opposite of the intended mask. Fixed with `.unmask(0)` **before** the `.eq(0)`
   comparison, so "no loss ever recorded" pixels become a real `0` first.
4. **Built the mask in Earth Engine** (as required — this step happens server-side, not
   after download): `lossyear.unmask(0).eq(0)` → **1** where no loss was recorded
   2001–2024, **0** where loss was recorded in any of those years.
5. **Clipped to the nucleus and explicitly sentinel-filled the export rectangle**:
   `.unmask(2, False)` — `sameFootprint=False`, the same lesson already documented in
   [`outputs/probability_map/README.md`](../outputs/probability_map/README.md)'s
   technical sidenotes — so the value **2** (deliberately outside the valid `{0, 1}`
   range) fills pixels outside the nucleus / with no Hansen data, distinguishable from a
   real "loss detected" `0`.
6. **Exported the mask at 30 m, EPSG:4326.** Hit Earth Engine's 48 MB single-request
   download limit on the first attempt (`geemap.ee_export_image` — this mask is ~500 MB
   uncompressed at 30 m over the whole nucleus). Switched to `geemap.download_ee_image`,
   which auto-splits the region into tiles under that limit, downloads them (135 tiles
   here) in parallel, and mosaics them back into one GeoTIFF — added the `geedim`
   package as a dependency (required by that function).
7. **Locally, snapped the mask onto the exact carbon grid** with **nearest-neighbor**
   resampling (a categorical 0/1/2 mask must never be interpolated into fractional
   values) — in this case the two grids already coincided almost exactly (both
   `19063 x 13635`), but the snap step guarantees it rather than assuming it.
8. **Applied the mask:**
   - carbon pixel nodata → stays nodata (`-9999`)
   - carbon valid, mask `1` (no loss) → output = carbon value, unchanged
   - carbon valid, mask `0` (loss recorded) → output = **`0.0`** (a real zero: that
     carbon is no longer standing forest — not "no data")
   - carbon valid, mask `2` (no Hansen data there) → output = nodata (can't determine
     loss status, so don't guess)
9. **Wrote the corrected raster.**

### Result (sanity-checked)
121,876,906 valid pixels; **14,341,331 (11.8%)** zeroed out for recorded forest loss
2001–2024. Corrected range **0.0–198.7 Mg C/ha**, mean **109.6 Mg C/ha** — down from
124.0 before correction, consistent with ~11.8% of carbon-bearing area having been
cleared. `quicklook_hansen_mask.png` shows the loss pixels (red) concentrated in the
same northwest quadrant already identified, throughout this whole project, as the most
human-pressured / fire-affected area (`outputs/maps/burned_area_quintiles_basemap.png`,
the SHAP results, the raw AGB map) — another cross-check that lines up, not a
coincidence.

### Outputs
- `hansen_no_loss_mask_2001_2024_30m.tif` — single-band byte GeoTIFF, values {0, 1, 2}
  (loss / no-loss / no-data), same grid as the carbon rasters. Small enough to commit
  (~5.2 MB).
- `carbon_density_existing_forest_2024_30m.tif` — **the deliverable**: single-band
  float32 GeoTIFF, same grid as Steps 1–2, pixel value = Mg C/ha in forest that still
  stood as of the 2024 Hansen update, nodata = -9999. **Gitignored** — same reasoning as
  the earlier full-resolution rasters (~364 MB compressed).
- `quicklook_carbon_existing_forest.png`, `quicklook_hansen_mask.png` — preview PNGs
  (committed).

### Environment
Adds `geedim` to the dependency list (needed by `geemap.download_ee_image` for the
tiled export) — see the updated install command in the root
[`README.md`](../README.md).

```powershell
cd carbon_estimation
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 03_hansen_correction.py
```

## Phase 2, Harmonization — `04_resample_susceptibility_for_arcgis.py`

Resamples the fire-susceptibility probability surface onto the exact 30 m grid of
`carbon_density_existing_forest_2024_30m.tif` (the final Phase 1 deliverable), and
bundles both 30 m rasters together in `resampled_ArcGIS_30m/` so they can be loaded side
by side in ArcGIS. **Only reads**
`outputs/probability_map/fire_susceptibility_probability_v4.tif` — that file is never
modified.

### Step by step
1. Open `carbon_density_existing_forest_2024_30m.tif` and read its transform, CRS,
   width, and height — this is the **snap grid**, copied directly rather than
   independently recomputed.
2. Open `outputs/probability_map/fire_susceptibility_probability_v4.tif` (500 m,
   read-only).
3. Resample with **nearest neighbor** (`rasterio.warp.reproject`,
   `Resampling.nearest`) directly onto the snap grid's transform/CRS/shape — every 30 m
   output pixel gets exactly the value of the one 500 m pixel that contains it, no
   interpolated intermediate values.
4. Write `resampled_ArcGIS_30m/fire_susceptibility_probability_map_v4_resampled_30m.tif`.
5. **Verify** grid alignment against the carbon raster — not assume it: re-open both
   files and assert their `transform`, `shape`, `crs`, and `bounds` are exactly equal.
   Confirmed: **True** on all four checks.
6. Copy `carbon_density_existing_forest_2024_30m.tif` into `resampled_ArcGIS_30m/` too,
   so both final 30 m products for this overlay sit together in one folder.

### ⚠️ Limitation (required reading before using this layer)
> Fire susceptibility was resampled from 500 m to 30 m using nearest neighbor to allow
> overlay with the native-30 m biomass. This resampling does **not** increase the
> effective resolution of susceptibility, which remains at 500 m (limited by MODIS
> MCD64A1); each 30 m pixel inherits the value of the 500 m pixel that contains it.

### Result (sanity-checked)
Same 13,635 × 19,063 grid as the carbon raster (verified, see above). Probability range
**0.006–0.948**, mean **0.204** — identical to the original 500 m raster's distribution,
confirming no values were altered, only re-gridded. `quicklook_susceptibility_30m_nn.png`
shows the same spatial pattern as `outputs/probability_map/quicklook_probability_map.png`
— same shape, same hotspots, just plotted on a finer pixel grid.

### `resampled_ArcGIS_30m/` — outputs
- `fire_susceptibility_probability_map_v4_resampled_30m.tif` — single-band float32
  GeoTIFF, grid-identical to `carbon_density_existing_forest_2024_30m.tif`, pixel value =
  P(burned) inherited from the 500 m model, nodata = -9999. Small enough to commit
  (~10.7 MB — nearest-neighbor upsampling from a coarse grid compresses very well).
- `carbon_density_existing_forest_2024_30m.tif` — copy of the Phase 1, Step 3
  deliverable, placed here for convenience. **Gitignored** (same ~364 MB file as the
  original in the parent folder).
- `quicklook_susceptibility_30m_nn.png` (in the parent `carbon_estimation/` folder,
  alongside the other quicklooks) — preview PNG.

```powershell
cd carbon_estimation
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 04_resample_susceptibility_for_arcgis.py
```

## Phase 1, Step 4 — `05_reproject_to_utm_and_stock.py`

Goal: leave `resampled_ArcGIS_30m/` containing **only** EPSG:32618 (WGS 84 / UTM zone
18N) layers, with carbon converted from density to per-pixel stock, ready for the
carbon-at-risk overlay. This step also found and moved out two files that had been
produced directly in ArcGIS before this reprojection (`carbon_at_risk_30m.tif`,
`susceptibility_high_veryhigh_30m.tif`) — confirmed to be EPSG:4326 (their ArcGIS-default
nodata value, `-3.4e38`, rather than this project's `-9999` convention, was itself a tell
that they came from ArcGIS, not this pipeline).

### Why the order matters (this exact sequence, not reordered)
1. **Reproject to a metric CRS with an explicit 30 m resolution first.** Area math
   ("a pixel = 0.09 ha") is only true in a projected, metres-based CRS. In EPSG:4326 a
   pixel is 0.0002695° on a side — not a fixed ground area, since it shrinks toward the
   poles. Only once the data is in UTM does "30 m × 30 m = 900 m² = 0.09 ha" hold
   exactly, for every pixel.
2. **Convert density → stock only after reprojecting**, using the pixel size actually
   measured on the reprojected output (not hardcoded), so the multiplier is exactly
   right even if reprojection introduced any rounding.

### Step by step
1. **Reproject to EPSG:32618 at exactly 30 m, on one fixed grid.** Susceptibility is
   reprojected first, with `resolution=(30, 30)` passed **explicitly** to
   `calculate_default_transform` — not left to auto-derive from the source pixel count,
   which would land close to but not exactly 30 m. That call's resulting
   transform/width/height becomes the **fixed reference grid**: carbon density is then
   reprojected directly onto that same transform (not given its own independently
   computed one), guaranteeing pixel-for-pixel alignment instead of a fractional-pixel
   offset between the two layers. Resampling method differs by layer:
   - Susceptibility → `Resampling.nearest` (never interpolate a susceptibility value
     into one that doesn't correspond to any real prediction — consistent with how this
     layer was already resampled 500 m → 30 m in Phase 2).
   - Carbon density → `Resampling.bilinear` (a genuinely continuous field).
2. **Verified, not assumed:** re-opened both reprojected files and checked
   `crs == EPSG:32618` and `res == (30.0, 30.0)` for each — confirmed. The actual
   measured resolution (not a hardcoded `30`) is what feeds the area calculation next.
3. **Converted carbon density → stock per pixel.** `pixel_area_ha = (res_x × res_y) /
   10,000` computed from the real reprojected resolution — came out to exactly
   **0.09 ha**, as expected once truly in UTM at 30 m. Multiplied only the valid
   (non-nodata) pixels by that area; nodata pixels are copied through as nodata, never
   multiplied.
4. **Verified the conversion**, not just trusted it: density range was
   **[0.00, 198.08] Mg C/ha** (very close to the pre-reprojection 198.7 — the small
   difference is expected bilinear smoothing at the resample), and stock range came out
   **[0.0000, 17.8276] Mg C/pixel** — matches `198.08 × 0.09 = 17.83` exactly. The script
   asserts this match and would stop (not silently continue) if it didn't.
5. **Grid-alignment re-check across all 3 final UTM layers** (susceptibility, carbon
   density, carbon stock) — same transform, same shape, same CRS: confirmed **True**.
6. **Cleaned up** `resampled_ArcGIS_30m/` — moved every EPSG:4326 artifact (the two
   inputs to this step, plus the two pre-existing ArcGIS files) into `_old/`. Nothing
   deleted, only moved, so the originals are still there if needed.

One re-run was needed: the first attempt failed at the move step with a Windows file
lock (`fire_susceptibility_probability_map_v4_resampled_30m.tif` was open in ArcGIS —
the same class of issue already documented in
[`outputs/probability_map/README.md`](../outputs/probability_map/README.md)'s technical
sidenotes). Steps 1–4 (the actual reprojection and conversion) are cheap to redo, so the
whole script was simply re-run after closing the file, rather than trying to resume
mid-way.

### Result (sanity-checked)
| Layer | CRS | Resolution | Value range | Mean |
|---|---|---|---|---|
| `fire_susceptibility_probability_map_v4_resampled_30m_UTM.tif` | EPSG:32618 | 30.0 × 30.0 m | 0.0057 – 0.9480 | 0.2044 |
| `carbon_density_existing_forest_2024_30m_UTM.tif` | EPSG:32618 | 30.0 × 30.0 m | 0.0000 – 198.0847 Mg C/ha | 109.6735 |
| `carbon_stock_per_pixel_2024_30m_UTM.tif` | EPSG:32618 | 30.0 × 30.0 m | 0.0000 – 17.8276 Mg C/pixel | 9.8706 |

`quicklook_carbon_stock_utm.png` shows the identical spatial pattern already seen in
every earlier version of this map, now in UTM.

### Why both density AND stock are kept
They serve different purposes for the next phase (carbon-at-risk):
- **`carbon_density_existing_forest_2024_30m_UTM.tif` (Mg C/ha)** — for reporting mean
  density and comparing against literature values, which are conventionally reported
  per hectare.
- **`carbon_stock_per_pixel_2024_30m_UTM.tif` (Mg C/pixel)** — for summing: since each
  pixel now holds an absolute quantity (not a rate), pixels can be summed directly
  (e.g. over the whole nucleus, or only over high-susceptibility pixels) to get a
  meaningful total tonnage. Summing density values directly would not have been
  meaningful — see the discussion of this exact distinction in the conversation history
  around this step.

### `resampled_ArcGIS_30m/` — final contents
- `fire_susceptibility_probability_map_v4_resampled_30m_UTM.tif` — ~10.9 MB, committed.
- `carbon_density_existing_forest_2024_30m_UTM.tif` — ~347 MB, **gitignored**.
- `carbon_stock_per_pixel_2024_30m_UTM.tif` — ~347 MB, **gitignored**.
- `_old/` — the 4 pre-UTM files (8 with sidecars: 2 from this pipeline in EPSG:4326, 2
  produced directly in ArcGIS before reprojecting) plus their `.aux.xml`/`.ovr`/`.tfw`/
  `.xml` sidecars. **Entire folder gitignored** — superseded, not needed going forward,
  kept locally only in case something needs cross-checking against the pre-UTM version.

```powershell
cd carbon_estimation
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 05_reproject_to_utm_and_stock.py
```
