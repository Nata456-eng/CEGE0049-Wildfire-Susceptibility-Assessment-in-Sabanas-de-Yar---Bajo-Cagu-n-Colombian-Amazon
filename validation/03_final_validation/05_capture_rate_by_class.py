"""
Validation, Phase 3, Step 5 (FINAL) -- For each of the 5 Jenks susceptibility classes,
compute the main validation metric of this whole workflow: what fraction of the total
observed 2020-2024 burned area (SINCHI, out-of-sample) fell into that class.

For each class:
  - burned_area_class_ha = sum(burn_fraction x 25 ha) over the class's cells
  - CAPTURE FRACTION (main metric) = burned_area_class_ha / total_burned_area_ha
    -> what proportion of all observed fire fell in this susceptibility class.
  - pct_of_total_area (context column only, NOT a validation metric) =
    class_area_ha / total_common_area_ha
    -> lets the capture fraction be read against how much of the landscape the class
       occupies: a class that is 10% of the area but captures 40% of the fire is a much
       stronger validation result than a class that is 40% of the area capturing 40% of
       the fire (that would be roughly what random chance predicts).

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 05_capture_rate_by_class.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

CLASS_TIF = DATA_DIR / "susceptibility_class_common.tif"
BURNED_AREA_TIF = DATA_DIR / "burned_area_observed_ha.tif"
COMMON_MASK_TIF = DATA_DIR / "common_valid_mask.tif"
RESULT_CSV = HERE / "capture_rate_by_class.csv"

CELL_AREA_HA = 25.0
CLASS_LABELS = ["very_low", "low", "moderate", "high", "very_high"]
CLASS_NODATA = 255

print("=" * 70)
print("STEP 1 -- Loading the classified raster, burned-area raster, and common mask")
print("=" * 70)

with rasterio.open(CLASS_TIF) as src:
    class_codes = src.read(1)

with rasterio.open(BURNED_AREA_TIF) as src:
    burned_area_ha = src.read(1)

with rasterio.open(COMMON_MASK_TIF) as src:
    common_valid = src.read(1).astype(bool)

n_common = int(common_valid.sum())
total_common_area_ha = n_common * CELL_AREA_HA
total_burned_area_ha = float(burned_area_ha[common_valid].sum())

print(f"Common-valid cells: {n_common:,} -> total area: {total_common_area_ha:,.2f} ha")
print(f"Total observed burned area: {total_burned_area_ha:,.2f} ha")

print()
print("=" * 70)
print("STEP 2 -- Computing burned area, capture fraction, and area share per class")
print("=" * 70)

rows = []
for code, label in enumerate(CLASS_LABELS):
    class_mask = (class_codes == code) & common_valid
    n_cells = int(class_mask.sum())
    class_area_ha = n_cells * CELL_AREA_HA
    burned_area_class_ha = float(burned_area_ha[class_mask].sum())
    capture_fraction = burned_area_class_ha / total_burned_area_ha
    pct_of_total_area = class_area_ha / total_common_area_ha

    rows.append({
        "class": label,
        "n_cells": n_cells,
        "class_area_ha": class_area_ha,
        "burned_area_class_ha": burned_area_class_ha,
        "capture_fraction": capture_fraction,
        "pct_of_total_area": pct_of_total_area,
    })

results = pd.DataFrame(rows)

print(f"{'class':>10s} {'n_cells':>10s} {'class_ha':>12s} {'burned_ha':>12s} "
      f"{'capture_frac':>13s} {'pct_of_area':>12s}")
for _, r in results.iterrows():
    print(f"{r['class']:>10s} {int(r['n_cells']):>10,d} {r['class_area_ha']:>12,.2f} "
          f"{r['burned_area_class_ha']:>12,.2f} {r['capture_fraction']:>13.4f} "
          f"{r['pct_of_total_area']:>12.4f}")

print()
print("=" * 70)
print("STEP 3 -- Consistency checks")
print("=" * 70)

sum_capture = results["capture_fraction"].sum()
sum_pct_area = results["pct_of_total_area"].sum()
sum_burned = results["burned_area_class_ha"].sum()
sum_cells = results["n_cells"].sum()

print(f"Sum of capture_fraction across classes: {sum_capture:.6f} (expected ~1.0)")
print(f"Sum of pct_of_total_area across classes: {sum_pct_area:.6f} (expected ~1.0)")
print(f"Sum of burned_area_class_ha: {sum_burned:,.2f} ha "
      f"(expected {total_burned_area_ha:,.2f} ha)")
print(f"Sum of n_cells across classes: {sum_cells:,} (expected {n_common:,})")

assert np.isclose(sum_capture, 1.0, atol=1e-6), "Capture fractions do not sum to 1.0"
assert np.isclose(sum_pct_area, 1.0, atol=1e-6), "Area percentages do not sum to 1.0"
assert sum_cells == n_common, "Class cell counts do not sum to the common-valid total"
print("PASSED: all consistency checks hold.")

print()
print("=" * 70)
print("STEP 4 -- Headline result: high + very high classes")
print("=" * 70)

high_plus = results.loc[results["class"].isin(["high", "very_high"]), "capture_fraction"].sum()
high_plus_area = results.loc[results["class"].isin(["high", "very_high"]), "pct_of_total_area"].sum()
print(f"'high' + 'very_high' classes together: {100 * high_plus_area:.2f}% of the "
      f"landscape area, but captured {100 * high_plus:.2f}% of all observed 2020-2024 "
      f"burned area (out-of-sample, independent SINCHI/Landsat source).")

print()
print("=" * 70)
print("STEP 5 -- Saving results")
print("=" * 70)

results.to_csv(RESULT_CSV, index=False)
print(f"Saved: {RESULT_CSV}")
