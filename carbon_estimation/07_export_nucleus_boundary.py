"""
Phase 5, Step 7 -- Export the study-nucleus boundary as a vector layer, reprojected to
EPSG:32618 (WGS 84 / UTM zone 18N), to sit alongside the 30 m rasters in
`resampled_ArcGIS_30m/` for ArcGIS.

Uses the same dissolved 5-municipality geometry every other script/notebook in this repo
uses to define "the nucleus" (`col_amazon_fire_utils.get_nucleus_geometry()`) -- pulled
from Earth Engine as GeoJSON (EPSG:4326), then reprojected locally with geopandas so the
boundary lines up exactly with `carbon_at_risk_UTM.tif` and the other UTM layers.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 07_export_nucleus_boundary.py
"""

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import col_amazon_fire_utils as utils

BUNDLE_DIR = HERE / "resampled_ArcGIS_30m"
OUT_SHP = BUNDLE_DIR / "nucleus_boundary_UTM.shp"

DST_CRS = "EPSG:32618"  # WGS 84 / UTM zone 18N -- matches every raster in the bundle


def main() -> None:
    BUNDLE_DIR.mkdir(exist_ok=True)

    utils.initialize_ee()
    nucleus_geom = utils.get_nucleus_geometry()
    nucleus_geojson = nucleus_geom.getInfo()
    nucleus_shape = shape(nucleus_geojson)
    print("Nucleus geometry loaded from Earth Engine, EPSG:4326 bounds:", nucleus_shape.bounds)

    gdf = gpd.GeoDataFrame(
        {"name": ["Sabanas de Yari - Bajo Caguan nucleus"]},
        geometry=[nucleus_shape],
        crs="EPSG:4326",
    )

    gdf_utm = gdf.to_crs(DST_CRS)
    print(f"Reprojected to {DST_CRS}, bounds:", tuple(gdf_utm.total_bounds))

    gdf_utm.to_file(OUT_SHP, driver="ESRI Shapefile")
    print(f"Saved: {OUT_SHP}")


if __name__ == "__main__":
    main()
