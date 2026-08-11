"""
Validation, Phase 3, Step 4 -- Compute the observed burned area per cell as
`burn_fraction x 25 ha`, restricted to the common valid mask (Step 2), and sum it to
get the total observed burned area for the 2020-2024 dry seasons.

Uses hectares here (not km^2) to match the user's requested unit for this step; 25 ha
is the same 500 m x 500 m cell area defined in Phase 3 Step 1 (25 ha = 0.25 km^2).

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 04_burned_area_observed.py
"""

from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = HERE / "data"

BURN_FRACTION_TIF = REPO_ROOT / "validation" / "02_model_2019_prediction" / "cicatrices_burnfraction_500m_UTM.tif"
COMMON_MASK_TIF = DATA_DIR / "common_valid_mask.tif"
BURNED_AREA_OUT = DATA_DIR / "burned_area_observed_ha.tif"

CELL_AREA_HA = 25.0  # 500 m x 500 m, see Phase 3 Step 1
BURNED_AREA_NODATA = -9999.0

print("=" * 70)
print("STEP 1 -- Loading the burn-fraction raster and the common valid mask")
print("=" * 70)

with rasterio.open(BURN_FRACTION_TIF) as src:
    burn_fraction = src.read(1)
    profile = src.profile.copy()

with rasterio.open(COMMON_MASK_TIF) as src:
    common_valid = src.read(1).astype(bool)

print(f"Burn fraction raster: {BURN_FRACTION_TIF.name}, shape={burn_fraction.shape}")
print(f"Common valid mask: {COMMON_MASK_TIF.name}, valid cells={common_valid.sum():,}")

print()
print("=" * 70)
print("STEP 2 -- Computing burned area per cell (burn_fraction x 25 ha)")
print("=" * 70)

burned_area_ha = np.full(burn_fraction.shape, BURNED_AREA_NODATA, dtype="float32")
burned_area_ha[common_valid] = burn_fraction[common_valid] * CELL_AREA_HA

valid_areas = burned_area_ha[common_valid]
print(f"Per-cell burned area range (common-valid cells): "
      f"[{valid_areas.min():.4f}, {valid_areas.max():.4f}] ha, mean={valid_areas.mean():.4f} ha")

print()
print("=" * 70)
print("STEP 3 -- Total observed burned area (sum over the common mask)")
print("=" * 70)

total_burned_area_ha = float(valid_areas.sum())
total_burned_area_km2 = total_burned_area_ha / 100.0

print(f"Total observed burned area (2020-2024 dry seasons, nucleus, common-valid cells):")
print(f"  {total_burned_area_ha:,.2f} ha")
print(f"  {total_burned_area_km2:,.2f} km^2")

print()
print("=" * 70)
print("STEP 4 -- Saving the per-cell burned-area raster")
print("=" * 70)

profile.update(count=1, dtype="float32", nodata=BURNED_AREA_NODATA)
with rasterio.open(BURNED_AREA_OUT, "w", **profile) as dst:
    dst.write(burned_area_ha, 1)
    dst.set_band_description(1, "observed_burned_area_ha_per_cell")

print(f"Saved: {BURNED_AREA_OUT}")
print("\nReady for Phase 3, Step 5: sum this per-cell burned area within each")
print("susceptibility class to get the fraction of total observed fire captured "
      "by each class.")
