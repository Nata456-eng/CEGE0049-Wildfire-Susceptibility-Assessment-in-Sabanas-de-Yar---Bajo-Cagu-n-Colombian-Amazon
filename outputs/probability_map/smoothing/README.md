# `outputs/probability_map/smoothing/` — Smoothed susceptibility map for stakeholders

**Script:** [`01_smooth_susceptibility_for_stakeholders.py`](01_smooth_susceptibility_for_stakeholders.py)

## Why this exists

`fire_susceptibility_probability_v4.tif` is native to a **500 m** Random Forest
prediction grid. That's fine for analysis, but when zoomed into for a presentation or
report it reads as a blocky, "pixelated" mosaic — every 500 m cell is a hard-edged
square, which is distracting and hard for non-technical stakeholders to read at a
glance. This gets worse, not better, when that same 500 m layer is resampled up to
30 m for the carbon-at-risk overlay in `carbon_estimation/`, since nearest-neighbor
resampling copies each 500 m block's value onto many small 30 m cells without changing
it — same blockiness, just at higher apparent resolution.

This folder produces a **visually smoothed version of the same surface**, purely so a
map handed to stakeholders shows the overall spatial pattern (where risk is
concentrated) without the grid artifacts competing for attention.

## This is a presentation product only — not an analytical one

The smoothed layers here are **not** used anywhere else in the project. They do not
feed into `carbon_estimation/`, the Jenks reclassification, the high-risk threshold
masks, or any other threshold-based analysis. Those correctly keep using the original,
unsmoothed continuous surface (`fire_susceptibility_probability_v4.tif` and its
nearest-neighbor 30 m/UTM resample), because:

- Smoothing a value that then gets compared against a hard cutoff (e.g. `>= 0.408`)
  would shift some pixels across class boundaries — the classification would no longer
  correspond to the model's actual output at that pixel.
- Nearest-neighbor resampling is deliberately used for the analytical layer so every
  30 m pixel keeps EXACTLY the 500 m value that covers it (see
  `carbon_estimation/04_resample_susceptibility_for_arcgis.py`) — no interpolation, no
  invented intermediate values. Smoothing does the opposite on purpose (blends
  neighboring values) and would be inconsistent with that design choice if reused
  anywhere near the analysis.

**Rule of thumb:** smoothed layer → maps for people. Original continuous layer →
anything that counts, classifies, or thresholds.

## Method

1. **Gaussian smoothing on the native 500 m grid (EPSG:4326)** — done with **NaN-aware
   normalized convolution**: `scipy.ndimage.gaussian_filter` is run separately on (a) the
   data with NoData zeroed out, and (b) a 1/0 valid-data mask, then the two are divided.
   This is the standard trick to blur a raster that has NoData holes without either (a)
   NoData bleeding fake zeros into its neighbors, or (b) the coverage edge being blurred
   away/shrunk. Pixels that were NoData originally stay NoData in the output — smoothing
   never invents data outside the original valid-data footprint.
   - `sigma = 1.5` native (500 m) pixels ≈ 750 m smoothing radius. Chosen to visibly
     soften the hard 500 m block edges without erasing the real spatial gradient (the
     coarse "where risk is concentrated" pattern across the nucleus is preserved, not
     flattened). Adjust `SIGMA_PIXELS` in the script and re-run if you want more/less
     smoothing.
2. **Reprojected to 30 m / EPSG:32618 with BILINEAR resampling** — appropriate here
   (unlike the analytical layers) because this is explicitly a continuous field being
   prepared for visual display, not for classification. Bilinear gives smooth-looking
   transitions between cells instead of the blocky nearest-neighbor look.
3. **Quicklook PNG** — original vs. smoothed side by side, same `YlOrRd`, `vmin=0,
   vmax=1` color scale used in `outputs/probability_map/quicklook_probability_map.png`,
   for a direct visual before/after comparison.

## Outputs

- `fire_susceptibility_smoothed_500m.tif` — intermediate: smoothed surface, still on
  the native 500 m grid, EPSG:4326, nodata = -9999. Kept mainly for inspection/debugging.
- `fire_susceptibility_smoothed_30m_UTM.tif` — **the deliverable**: smoothed surface,
  30 m resolution, EPSG:32618, nodata = -9999. Use this one for stakeholder-facing maps.
- `quicklook_smoothed_vs_original.png` — before/after comparison preview.

```powershell
cd outputs/probability_map/smoothing
"C:\Users\Natal\.conda\envs\fire_thesis\python.exe" 01_smooth_susceptibility_for_stakeholders.py
```
