"""
Validation, Phase 3, Step 2 -- Build a COMMON valid-data mask: cells where BOTH the
susceptibility raster and the SINCHI burn-fraction raster have real (non-NoData) data.

Even though Phase 1 Step 1 already confirmed both rasters share the same shape,
transform and CRS, and even reported identical valid-pixel COUNTS (451,048 in each),
equal counts do not by themselves prove the valid cells are in the exact same
LOCATIONS. Every later Phase 3 computation (classification, per-class captured-area
fraction) must be restricted to the intersection of both masks -- otherwise a cell that
is valid in one raster but NoData in the other could be silently counted using a
NoData sentinel (-9999) as if it were real data, biasing every downstream statistic.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 02_common_valid_mask.py
"""

from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

SUSCEPTIBILITY_TIF = REPO_ROOT / "validation" / "02_model_2019_prediction" / "susceptibility_model2019_500m_UTM.tif"
BURN_FRACTION_TIF = REPO_ROOT / "validation" / "02_model_2019_prediction" / "cicatrices_burnfraction_500m_UTM.tif"
COMMON_MASK_OUT = DATA_DIR / "common_valid_mask.tif"

print("=" * 70)
print("STEP 1 -- Loading both rasters")
print("=" * 70)

with rasterio.open(SUSCEPTIBILITY_TIF) as src_susc:
    susc = src_susc.read(1)
    susc_nodata = src_susc.nodata
    profile = src_susc.profile.copy()

with rasterio.open(BURN_FRACTION_TIF) as src_burn:
    burn = src_burn.read(1)
    burn_nodata = src_burn.nodata

print(f"Susceptibility: {SUSCEPTIBILITY_TIF.name} (nodata={susc_nodata})")
print(f"Burn fraction:  {BURN_FRACTION_TIF.name} (nodata={burn_nodata})")

print()
print("=" * 70)
print("STEP 2 -- Building the common valid-data mask")
print("=" * 70)

valid_susc = ~np.isclose(susc, susc_nodata)
valid_burn = ~np.isclose(burn, burn_nodata)
common_valid = valid_susc & valid_burn

n_valid_susc = int(valid_susc.sum())
n_valid_burn = int(valid_burn.sum())
n_common = int(common_valid.sum())
n_susc_only = int((valid_susc & ~valid_burn).sum())
n_burn_only = int((~valid_susc & valid_burn).sum())
n_total = susc.size

print(f"Valid cells -- susceptibility only:  {n_valid_susc:,} / {n_total:,}")
print(f"Valid cells -- burn fraction only:    {n_valid_burn:,} / {n_total:,}")
print(f"Cells valid in susceptibility but NOT in burn fraction: {n_susc_only:,}")
print(f"Cells valid in burn fraction but NOT in susceptibility: {n_burn_only:,}")
print(f"COMMON valid cells (both rasters have real data): {n_common:,} "
      f"({100 * n_common / n_total:.2f}% of the grid)")

if n_susc_only == 0 and n_burn_only == 0:
    print("Masks are IDENTICAL -- the common mask equals each individual mask exactly.")
else:
    print(f"Masks differ by {n_susc_only + n_burn_only:,} cell(s) -- restricting all "
          f"downstream Phase 3 computations to the common mask avoids biasing results "
          f"with cells absent from one raster.")

print()
print("=" * 70)
print("STEP 3 -- Saving the common valid mask")
print("=" * 70)

profile.update(count=1, dtype="uint8", nodata=None)
with rasterio.open(COMMON_MASK_OUT, "w", **profile) as dst:
    dst.write(common_valid.astype("uint8"), 1)
    dst.set_band_description(1, "common_valid_mask_susceptibility_and_burn_fraction")

print(f"Saved: {COMMON_MASK_OUT}")
print(f"\n{n_common:,} valid cells will be used for every subsequent Phase 3 computation.")
