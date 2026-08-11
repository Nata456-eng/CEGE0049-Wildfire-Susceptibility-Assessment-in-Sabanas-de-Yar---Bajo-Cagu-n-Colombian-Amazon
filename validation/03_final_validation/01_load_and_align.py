"""
Validation, Phase 3, Step 1 -- Load the two aligned 500 m rasters and define the
constants (cell area, Jenks susceptibility classes) needed for the final cell-by-cell
capture-rate comparison.

METHODOLOGICAL CONTEXT (must be respected by every later Phase 3 step too):
This is a genuine OUT-OF-SAMPLE validation:
  - The susceptibility raster comes from the EVALUATION model (RF trained on year <=2019
    only, see `validation/02_model_2019_prediction/01_generate_susceptibility_model2019.py`)
    -- it never saw any fire from 2020-2024.
  - The burn-scar raster is SINCHI's independently-derived (Landsat, 1:100,000 scale)
    burn-scar product for the 2020-2024 dry seasons -- an entirely different sensor and
    methodology from MODIS MCD64A1, which is what the model's `burned` target was
    trained on. There is no shared labeling pipeline between the two.
Together, this measures the model's REAL predictive capacity (does it flag areas that
actually burned, in data it never saw, observed by an independent source?), not merely
its in-sample goodness-of-fit.

SINCHI's raster stores the BURN-SCAR AREA FRACTION per cell (0-1), not a binary flag
(see Step 2 of Phase 2, `02_rasterize_burn_fraction.py`). Consequently, every later
"burned area" computation in this phase MUST be:

    burned_area(cell) = burn_fraction(cell) x cell_area

and NEVER a simple count of "burned" cells x cell_area -- the fractional formulation is
strictly more precise, since it accounts for partially-burned cells instead of
collapsing them to the same weight as fully-burned ones.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 01_load_and_align.py
"""

from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent

SUSCEPTIBILITY_TIF = REPO_ROOT / "validation" / "02_model_2019_prediction" / "susceptibility_model2019_500m_UTM.tif"
BURN_FRACTION_TIF = REPO_ROOT / "validation" / "02_model_2019_prediction" / "cicatrices_burnfraction_500m_UTM.tif"

# --- Cell area (both rasters are 500 m x 500 m, EPSG:32618 -- see Phase 2) ---
CELL_SIDE_M = 500.0
CELL_AREA_M2 = CELL_SIDE_M ** 2      # 250,000 m^2
CELL_AREA_HA = CELL_AREA_M2 / 10_000  # 25 ha
CELL_AREA_KM2 = CELL_AREA_M2 / 1_000_000  # 0.25 km^2

# --- Jenks natural breaks used throughout this project to classify susceptibility
# (same breaks as `carbon_estimation/06_carbon_at_risk_by_susceptibility.py`) ---
JENKS_BREAKS = [0.117, 0.25, 0.408, 0.612]
CLASS_LABELS = ["very_low", "low", "moderate", "high", "very_high"]
# np.digitize bin edges: (-inf, 0.117) -> very_low, [0.117, 0.25) -> low,
# [0.25, 0.408) -> moderate, [0.408, 0.612) -> high, [0.612, inf) -> very_high
CLASS_BIN_EDGES = [-np.inf] + JENKS_BREAKS + [np.inf]

print("=" * 70)
print("STEP 1 -- Loading both rasters as masked arrays (NoData respected)")
print("=" * 70)

with rasterio.open(SUSCEPTIBILITY_TIF) as src_susc:
    susc = src_susc.read(1, masked=True)
    susc_transform = src_susc.transform
    susc_crs = src_susc.crs
    susc_shape = src_susc.shape
    susc_nodata = src_susc.nodata

with rasterio.open(BURN_FRACTION_TIF) as src_burn:
    burn = src_burn.read(1, masked=True)
    burn_transform = src_burn.transform
    burn_crs = src_burn.crs
    burn_shape = src_burn.shape
    burn_nodata = src_burn.nodata

print(f"Susceptibility raster: {SUSCEPTIBILITY_TIF.name}")
print(f"  shape={susc_shape}, CRS={susc_crs}, transform={tuple(susc_transform)[:6]}, "
      f"nodata={susc_nodata}")
print(f"Burn-fraction raster:  {BURN_FRACTION_TIF.name}")
print(f"  shape={burn_shape}, CRS={burn_crs}, transform={tuple(burn_transform)[:6]}, "
      f"nodata={burn_nodata}")

print()
print("=" * 70)
print("STEP 2 -- Safety assert: both rasters share the same grid")
print("=" * 70)

assert susc_shape == burn_shape, (
    f"Shape mismatch: susceptibility {susc_shape} != burn fraction {burn_shape}"
)
assert susc_transform == burn_transform, (
    f"Transform mismatch: susceptibility {susc_transform} != burn fraction {burn_transform}"
)
assert susc_crs == burn_crs, (
    f"CRS mismatch: susceptibility {susc_crs} != burn fraction {burn_crs}"
)
print("PASSED: identical shape, transform and CRS -- safe to compare cell-by-cell "
      "without any further resampling/reprojection.")

print()
print("=" * 70)
print("STEP 3 -- Cell area and Jenks susceptibility classes")
print("=" * 70)
print(f"Cell area: {CELL_SIDE_M:.0f} m x {CELL_SIDE_M:.0f} m = "
      f"{CELL_AREA_HA:.0f} ha = {CELL_AREA_KM2:.2f} km^2")
print("Jenks classes (susceptibility probability, 0-1 scale):")
for i, label in enumerate(CLASS_LABELS):
    lo = CLASS_BIN_EDGES[i]
    hi = CLASS_BIN_EDGES[i + 1]
    lo_str = "-inf" if lo == -np.inf else f"{lo:.3f}"
    hi_str = "+inf" if hi == np.inf else f"{hi:.3f}"
    print(f"  {label:>10s}: [{lo_str}, {hi_str})")

print()
print("=" * 70)
print("STEP 4 -- Quick data sanity report")
print("=" * 70)

n_valid_susc = int((~susc.mask).sum()) if np.ma.is_masked(susc) else susc.size
n_valid_burn = int((~burn.mask).sum()) if np.ma.is_masked(burn) else burn.size
print(f"Valid (non-NoData) pixels -- susceptibility: {n_valid_susc:,}, "
      f"burn fraction: {n_valid_burn:,}")
assert n_valid_susc == n_valid_burn, (
    "Valid-pixel counts differ between the two rasters -- their NoData masks should "
    "be identical (both derive from the same nucleus mask)."
)
print("PASSED: both rasters have identical valid-pixel counts (same NoData mask).")

print(f"Susceptibility probability range: [{susc.min():.4f}, {susc.max():.4f}], "
      f"mean={susc.mean():.4f}")
print(f"Burn-fraction range: [{burn.min():.4f}, {burn.max():.4f}], mean={burn.mean():.4f}")

total_burned_area_km2 = float(burn.sum()) * CELL_AREA_KM2
print(f"Total observed burned area (sum of fraction x cell area, 2020-2024 dry seasons, "
      f"nucleus only): {total_burned_area_km2:,.2f} km^2")

print()
print("Ready for Phase 3, Step 2: classify every valid cell into a Jenks susceptibility")
print("class and compute what fraction of the total observed burned area falls in each")
print("class (burned_area = burn_fraction x cell_area, summed per class).")
