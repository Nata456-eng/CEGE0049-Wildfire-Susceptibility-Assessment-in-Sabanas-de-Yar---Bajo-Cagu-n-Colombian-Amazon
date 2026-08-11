"""
Validation, Phase 1, Step 2 -- Clip the raw SINCHI burn-scar polygons to the study
nucleus (the dissolved 5-municipality geometry every other script in this project uses),
keeping ONLY scars that fall inside it.

Deliberately a spatial clip, not a department/municipality name filter: the nucleus is
smaller than -- and doesn't align with -- the administrative boundaries of the
departments/municipalities it overlaps, so filtering by `departamen`/`nom_munici` would
both include area outside the nucleus and potentially miss slivers of it.

Reuses the nucleus boundary already exported for the carbon-at-risk overlay
(`carbon_estimation/resampled_ArcGIS_30m/nucleus_boundary_UTM.shp`, EPSG:32618) instead
of re-querying Earth Engine, then reprojects it to the burn-scar layer's own CRS
(EPSG:4170) just for this clip -- the burn-scar layer's CRS is left untouched here;
reprojection to a final working CRS happens in a later cleaning step.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 02_clip_to_nucleus.py
"""

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
RAW_SHP = (
    HERE.parent
    / "Cicatrices_de_quema_por_regiB3n_(HistB3rico)._Escala3A100.000_SINCHI"
    / "Cicatrices_de_quema_por_región_(Histórico)._Escala%3A_1%3A100.000.shp"
)
NUCLEUS_SHP = (
    HERE.parent.parent
    / "carbon_estimation" / "resampled_ArcGIS_30m" / "nucleus_boundary_UTM.shp"
)

DATA_DIR = HERE / "data"
CLIPPED_OUT = DATA_DIR / "02_clipped_to_nucleus.gpkg"

print("=" * 70)
print("STEP 2 -- Clipping burn scars to the study nucleus")
print("=" * 70)

burn_scars = gpd.read_file(RAW_SHP)
print(f"Loaded raw burn scars: {len(burn_scars):,} records, CRS={burn_scars.crs}")

nucleus = gpd.read_file(NUCLEUS_SHP)
print(f"Loaded nucleus boundary: CRS={nucleus.crs}")

print(f"Reprojecting nucleus boundary {nucleus.crs} -> {burn_scars.crs} for the clip ...")
nucleus_reprojected = nucleus.to_crs(burn_scars.crs)

print("Clipping (spatial intersection, not an attribute filter) ...")
clipped = gpd.clip(burn_scars, nucleus_reprojected)

print()
print(f"Records before clip: {len(burn_scars):,}")
print(f"Records after clip:  {len(clipped):,}")
print(f"Dropped (outside nucleus): {len(burn_scars) - len(clipped):,}")
print(f"CRS preserved: {clipped.crs}")

DATA_DIR.mkdir(exist_ok=True)
clipped.to_file(CLIPPED_OUT, driver="GPKG")
print(f"\nSaved: {CLIPPED_OUT}")
