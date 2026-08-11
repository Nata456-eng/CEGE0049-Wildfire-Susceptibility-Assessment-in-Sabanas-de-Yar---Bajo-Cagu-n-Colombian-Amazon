"""
Validation, Phase 3, Step 7 -- Scar-level (per-fire-event) detection rate.

Complementary to Step 5's area-weighted "capture fraction" (which answers "out of all
burned AREA, what share fell in each risk class" -- a metric that large scars can
dominate). This step instead answers a stakeholder's more intuitive question: "out of
all the individual fire scars SINCHI recorded, how many did the model actually flag as
high risk?" -- treating every fire event as one vote, regardless of its size.

METHOD:
For each of the 1,227 individual SINCHI burn-scar polygons (Phase 1 deliverable), clip
the classified susceptibility raster (Step 3) to that polygon's footprint
(`rasterio.mask.mask(..., all_touched=True)`, so even scars smaller than one 500 m
pixel still register against the pixel(s) they overlap), then compute:
  - `dominant_class`: the susceptibility class covering the MOST pixels under that scar
    (majority rule -- what risk level was this fire event, overall, sitting in?)
  - `any_high`: whether ANY overlapping pixel was "high" or "very high" (a more lenient
    check -- did the model flag at least part of this fire as high risk?)
  - `pct_high_overlap`: the fraction of the scar's overlapping pixels that were "high"
    or "very high"

Two headline detection rates are reported:
  - STRICT ("majority rule"): % of scars whose dominant (most common) overlapping class
    was "high" or "very high".
  - LENIENT ("any overlap"): % of scars that touched AT LEAST ONE "high"/"very high"
    pixel anywhere in their footprint.
A "moderate or above" variant of each is also reported, since "moderate" is still a
materially elevated risk level relative to "very low"/"low".

Just like every other Phase 3 step, this validates the EVALUATION model (<=2019
training data) against 2020-2024 SINCHI burn scars it never saw -- a genuine
out-of-sample check.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 07_scar_level_detection_rate.py
"""

from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from shapely.geometry import mapping

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = HERE / "data"

SCARS_GPKG = REPO_ROOT / "validation" / "01_data_cleaning" / "data" / "05_burn_scars_validation_ready_UTM.gpkg"
CLASS_TIF = DATA_DIR / "susceptibility_class_common.tif"
RESULT_CSV = HERE / "scar_level_detection_rate.csv"

CLASS_LABELS = {0: "very_low", 1: "low", 2: "moderate", 3: "high", 4: "very_high"}
CLASS_NODATA = 255
HIGH_RISK_CODES = {3, 4}          # high, very_high
MODERATE_PLUS_CODES = {2, 3, 4}   # moderate, high, very_high

print("=" * 70)
print("STEP 1 -- Loading burn scars and the classified susceptibility raster")
print("=" * 70)

scars = gpd.read_file(SCARS_GPKG)
print(f"Loaded {len(scars):,} individual burn-scar polygons")

src = rasterio.open(CLASS_TIF)
print(f"Classified raster: {CLASS_TIF.name}, shape={src.shape}, CRS={src.crs}")
assert scars.crs.to_epsg() == src.crs.to_epsg(), "CRS mismatch between scars and raster"

print()
print("=" * 70)
print("STEP 2 -- Clipping the raster to each scar's footprint and classifying it")
print("=" * 70)

records = []
n_no_overlap = 0
for idx, row in scars.iterrows():
    geom = [mapping(row.geometry)]
    try:
        out_image, _ = rasterio.mask.mask(
            src, geom, crop=True, all_touched=True, nodata=CLASS_NODATA, filled=True
        )
    except ValueError:
        # Shapes do not overlap the raster at all (shouldn't happen -- scars were
        # clipped to the nucleus in Phase 1 -- but guard against edge slivers anyway).
        n_no_overlap += 1
        continue

    band = out_image[0]
    valid = band != CLASS_NODATA
    if valid.sum() == 0:
        n_no_overlap += 1
        continue

    values = band[valid]
    counts = Counter(values.tolist())
    dominant_code = counts.most_common(1)[0][0]
    n_valid_px = int(valid.sum())
    n_high_px = sum(c for code, c in counts.items() if code in HIGH_RISK_CODES)
    n_mod_plus_px = sum(c for code, c in counts.items() if code in MODERATE_PLUS_CODES)

    records.append({
        "scar_index": idx,
        "dominant_class": CLASS_LABELS[dominant_code],
        "dominant_is_high_plus": dominant_code in HIGH_RISK_CODES,
        "dominant_is_moderate_plus": dominant_code in MODERATE_PLUS_CODES,
        "any_high_plus": n_high_px > 0,
        "any_moderate_plus": n_mod_plus_px > 0,
        "pct_high_overlap": n_high_px / n_valid_px,
        "pct_moderate_plus_overlap": n_mod_plus_px / n_valid_px,
        "n_overlapping_pixels": n_valid_px,
        "area_hecta": row.get("area_hecta", np.nan),
    })

src.close()

results = pd.DataFrame(records)
print(f"Scars with at least one overlapping raster pixel: {len(results):,} / {len(scars):,}")
if n_no_overlap:
    print(f"Scars with NO overlapping pixel (excluded, likely tiny edge slivers): {n_no_overlap:,}")

print()
print("=" * 70)
print("STEP 3 -- Detection rates")
print("=" * 70)

n_scars = len(results)

strict_high = int(results["dominant_is_high_plus"].sum())
lenient_high = int(results["any_high_plus"].sum())
strict_mod = int(results["dominant_is_moderate_plus"].sum())
lenient_mod = int(results["any_moderate_plus"].sum())

print(f"Total scars analyzed: {n_scars:,}")
print()
print("STRICT (majority-rule) detection -- dominant class of the scar was:")
print(f"  high or very high:      {strict_high:,} / {n_scars:,} "
      f"({100 * strict_high / n_scars:.1f}%)")
print(f"  moderate or above:      {strict_mod:,} / {n_scars:,} "
      f"({100 * strict_mod / n_scars:.1f}%)")
print()
print("LENIENT (any-overlap) detection -- scar touched at least one pixel of:")
print(f"  high or very high:      {lenient_high:,} / {n_scars:,} "
      f"({100 * lenient_high / n_scars:.1f}%)")
print(f"  moderate or above:      {lenient_mod:,} / {n_scars:,} "
      f"({100 * lenient_mod / n_scars:.1f}%)")

print()
print("Dominant-class distribution across all scars:")
dom_counts = results["dominant_class"].value_counts()
for label in ["very_high", "high", "moderate", "low", "very_low"]:
    n = int(dom_counts.get(label, 0))
    print(f"  {label:>10s}: {n:>5,d} scars ({100 * n / n_scars:.1f}%)")

print()
print("=" * 70)
print("STEP 4 -- Saving per-scar results")
print("=" * 70)

results.to_csv(RESULT_CSV, index=False)
print(f"Saved: {RESULT_CSV}")

print()
print("=" * 70)
print("Headline sentence")
print("=" * 70)
print(f"Out of {n_scars:,} individual fire scars recorded by SINCHI (2020-2024, "
      f"out-of-sample), {strict_high:,} ({100 * strict_high / n_scars:.1f}%) had most "
      f"of their burned footprint in a zone the model marked as high or very high risk. "
      f"Widening to 'moderate risk or above', that rises to {strict_mod:,} scars "
      f"({100 * strict_mod / n_scars:.1f}%).")
