"""
Validation, Phase 1, Step 5 -- Reproject the final filtered GeoDataFrame to EPSG:32618
(WGS84 / UTM zone 18N), the same CRS used by the susceptibility raster and every other
UTM layer in this project.

Deliberately the LAST step in Phase 1, after every filter (nucleus clip, dry-season
filter, validation-period filter): reprojecting is not free (recomputes every vertex of
every polygon), so it's applied only once, to the geometries that actually survive
filtering -- not to the ~15,212 raw records, most of which get discarded anyway.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 05_reproject_to_utm.py
"""

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
VALIDATION_PERIOD_IN = DATA_DIR / "04_validation_period_2020_2024.gpkg"
FINAL_OUT = DATA_DIR / "05_burn_scars_validation_ready_UTM.gpkg"

DST_CRS = "EPSG:32618"  # WGS84 / UTM zone 18N -- matches the susceptibility raster and
                        # every other UTM layer in this project (carbon_estimation/, etc.)

print("=" * 70)
print("STEP 5 -- Reprojecting final filtered burn scars to EPSG:32618")
print("=" * 70)

gdf = gpd.read_file(VALIDATION_PERIOD_IN)
print(f"Loaded: {len(gdf):,} records, CRS={gdf.crs}")

gdf_utm = gdf.to_crs(DST_CRS)

print(f"Reprojected {gdf.crs} -> {gdf_utm.crs}")
assert gdf_utm.crs.to_epsg() == 32618, (
    f"CRS check FAILED: expected EPSG:32618, got {gdf_utm.crs}"
)
print(f"CRS check PASSED: resulting CRS is {gdf_utm.crs} (EPSG:{gdf_utm.crs.to_epsg()})")
print(f"Record count unchanged by reprojection: {len(gdf_utm):,}")

gdf_utm.to_file(FINAL_OUT, driver="GPKG")
print(f"\nSaved: {FINAL_OUT}")
print("\nThis file is the Phase 1 deliverable -- clean, filtered (nucleus + dry season +")
print("2020-2024 validation period), reprojected to EPSG:32618, ready to rasterize and")
print("compare against the (<=2019-trained) evaluation model's predictions in Phase 2.")
