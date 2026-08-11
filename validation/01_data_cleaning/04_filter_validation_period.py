"""
Validation, Phase 1, Step 4 -- Filter to the out-of-sample validation period: years
2020-2024 inclusive.

`periodo` is stored as text in the source shapefile -- converted to a numeric year here
so the range filter is a real integer comparison, not a string comparison (which would
sort '2020' > '2024' lexicographically... actually fine for 4-digit years, but still not
a numeric comparison and error-prone if the column ever contains non-numeric junk).

WHY 2020-2024, and WHICH model this validates against (read before using this dataset
in a later phase):

This project has TWO different fitted versions of the winning Random Forest v4 model,
and they are not interchangeable for validation purposes:

1. The EVALUATION model (`tuning/v4`) -- trained ONLY on data through 2019
   (train <= 2019, test >= 2020, the temporal hold-out protocol documented in
   `tuning/v4/README.md`). This is the model whose AUC/PR-AUC/F1 "performance metrics"
   are reported for the project. It never saw 2020-2024 during training.
2. The DEPLOYED/MAPPING model -- refit on the FULL dataset (train+test combined, all
   years 2001-2024) specifically to generate the pixel-level map,
   `fire_susceptibility_probability_v4.tif` (see `outputs/probability_map/README.md`).
   This model HAS seen 2020-2024 during training -- using it against 2020-2024 SINCHI
   burn scars would NOT be a genuine out-of-sample test, since those years already
   informed its fit.

This validation is explicitly meant to check the EVALUATION model (#1) -- the one whose
performance metrics are actually reported -- against an independent burn-scar source it
never saw. That is why the SINCHI records are restricted to 2020-2024: it is exactly the
window that model #1 was held out from during training, making this a genuine
out-of-sample check. It also means a LATER phase of this validation must compare
against predictions from model #1 (re-fit on <=2019 data only, e.g. reloading
`tuning/v4`'s fitted estimator or retraining it with the same pre-2020 split), NOT
directly against `fire_susceptibility_probability_v4.tif` -- that raster comes from
model #2, which was already trained on the very years being validated here, and using it
would silently invalidate the "out-of-sample" claim.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 04_filter_validation_period.py
"""

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DRY_SEASON_IN = DATA_DIR / "03_dry_season_filtered.gpkg"
VALIDATION_PERIOD_OUT = DATA_DIR / "04_validation_period_2020_2024.gpkg"

VALIDATION_YEAR_MIN = 2020
VALIDATION_YEAR_MAX = 2024

print("=" * 70)
print("STEP 4 -- Filtering to the out-of-sample validation period (2020-2024)")
print("=" * 70)

gdf = gpd.read_file(DRY_SEASON_IN)
print(f"Loaded dry-season-filtered burn scars: {len(gdf):,} records, CRS={gdf.crs}")

gdf["periodo_num"] = gdf["periodo"].astype(int)

print()
print("Record count by year, before filtering:")
print(gdf["periodo_num"].value_counts().sort_index())

validation_period = gdf[
    (gdf["periodo_num"] >= VALIDATION_YEAR_MIN) & (gdf["periodo_num"] <= VALIDATION_YEAR_MAX)
].copy()

print()
print(f"Records before validation-period filter: {len(gdf):,}")
print(f"Records after filter ({VALIDATION_YEAR_MIN}-{VALIDATION_YEAR_MAX} inclusive): "
      f"{len(validation_period):,}")
print(f"Dropped (outside {VALIDATION_YEAR_MIN}-{VALIDATION_YEAR_MAX}): "
      f"{len(gdf) - len(validation_period):,}")

print()
print("Breakdown of retained records by year:")
print(validation_period["periodo_num"].value_counts().sort_index())

validation_period.to_file(VALIDATION_PERIOD_OUT, driver="GPKG")
print(f"\nSaved: {VALIDATION_PERIOD_OUT}")
print()
print("REMINDER: this 2020-2024 window is out-of-sample ONLY for the evaluation model")
print("(tuning/v4, trained <=2019). A later phase must validate against THAT model's")
print("predictions, not against fire_susceptibility_probability_v4.tif (which was")
print("refit on all years, including 2020-2024, and would not be a fair test here).")
