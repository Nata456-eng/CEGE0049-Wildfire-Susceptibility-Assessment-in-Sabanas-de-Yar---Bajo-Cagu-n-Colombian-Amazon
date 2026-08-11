"""
Validation, Phase 3, Step 3 -- Classify the continuous susceptibility raster into the
project's five Jenks classes, restricted to the common valid-data mask (Step 2).

Uses the exact same Jenks breaks as everywhere else in this project (see
`carbon_estimation/06_carbon_at_risk_by_susceptibility.py` and Phase 3 Step 1):
very low (<0.117), low (0.117-0.25), moderate (0.25-0.408), high (0.408-0.612),
very high (>=0.612).

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 03_classify_susceptibility.py
"""

from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

SUSCEPTIBILITY_TIF = REPO_ROOT / "validation" / "02_model_2019_prediction" / "susceptibility_model2019_500m_UTM.tif"
COMMON_MASK_TIF = DATA_DIR / "common_valid_mask.tif"
CLASS_OUT = DATA_DIR / "susceptibility_class_common.tif"

CELL_AREA_KM2 = 0.25  # 500 m x 500 m, see Phase 3 Step 1

JENKS_BREAKS = [0.117, 0.25, 0.408, 0.612]
CLASS_LABELS = ["very_low", "low", "moderate", "high", "very_high"]
CLASS_BIN_EDGES = [-np.inf] + JENKS_BREAKS + [np.inf]
CLASS_NODATA = 255  # sentinel for cells outside the common valid mask

print("=" * 70)
print("STEP 1 -- Loading the susceptibility raster and the common valid mask")
print("=" * 70)

with rasterio.open(SUSCEPTIBILITY_TIF) as src:
    susc = src.read(1)
    profile = src.profile.copy()

with rasterio.open(COMMON_MASK_TIF) as src:
    common_valid = src.read(1).astype(bool)

print(f"Susceptibility raster: {SUSCEPTIBILITY_TIF.name}, shape={susc.shape}")
print(f"Common valid mask: {COMMON_MASK_TIF.name}, valid cells={common_valid.sum():,}")

print()
print("=" * 70)
print("STEP 2 -- Classifying into the 5 Jenks classes (common-valid cells only)")
print("=" * 70)

# np.digitize(x, [0.117, 0.25, 0.408, 0.612]) returns 0..4 directly, matching
# CLASS_LABELS order (0=very_low, 1=low, 2=moderate, 3=high, 4=very_high).
class_codes = np.digitize(susc, JENKS_BREAKS).astype("uint8")
class_codes[~common_valid] = CLASS_NODATA

print("Class code mapping:")
for code, label in enumerate(CLASS_LABELS):
    lo = CLASS_BIN_EDGES[code]
    hi = CLASS_BIN_EDGES[code + 1]
    lo_str = "-inf" if lo == -np.inf else f"{lo:.3f}"
    hi_str = "+inf" if hi == np.inf else f"{hi:.3f}"
    print(f"  code {code} = {label:>10s}: [{lo_str}, {hi_str})")
print(f"  code {CLASS_NODATA} = NoData (outside the common valid mask)")

print()
print("=" * 70)
print("STEP 3 -- Cell counts and area per class (common-valid cells only)")
print("=" * 70)

n_common = int(common_valid.sum())
print(f"{'class':>10s} {'cells':>10s} {'% of valid':>11s} {'area (km2)':>11s}")
for code, label in enumerate(CLASS_LABELS):
    n = int((class_codes == code).sum())
    pct = 100 * n / n_common
    area_km2 = n * CELL_AREA_KM2
    print(f"{label:>10s} {n:>10,d} {pct:>10.2f}% {area_km2:>10,.2f}")

total_classified = int(np.isin(class_codes, range(len(CLASS_LABELS))).sum())
assert total_classified == n_common, (
    f"Classified cell count ({total_classified:,}) != common valid cells ({n_common:,})"
)
print(f"\nVerified: every common-valid cell was classified ({total_classified:,} == {n_common:,}).")

print()
print("=" * 70)
print("STEP 4 -- Saving the classified raster")
print("=" * 70)

profile.update(count=1, dtype="uint8", nodata=CLASS_NODATA)
with rasterio.open(CLASS_OUT, "w", **profile) as dst:
    dst.write(class_codes, 1)
    dst.set_band_description(1, "susceptibility_jenks_class_common_valid")

print(f"Saved: {CLASS_OUT}")
print("\nReady for Phase 3, Step 4: compute the observed burned-area fraction "
      "(burn_fraction x cell_area) captured by each susceptibility class.")
