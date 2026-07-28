"""
Splits the 9-band predictor stack (`probability map/predictor_stack_v4.tif`) into 9
individual single-band GeoTIFFs, so each predictor can be opened/symbolized on its own
in ArcGIS instead of picking a band out of a multi-band raster.

Source band order (verified against col_amazon_fire_utils / predict_susceptibility_map.ipynb,
section 2c -- the notebook asserts this order before export, so it is load-bearing):
    dist_roads, dist_parks, dist_coca, dist_mosaic, temp_C, vpd_kPa, ndvi, wind_ms, oni

Nodata: the source file has no nodata value stamped in its GDAL metadata (an artifact of
the geemap export -- see 'probability map/README.md', "Technical sidenotes"), but every
band uses -9999 as its actual masked-pixel sentinel outside the study nucleus polygon
(~51.6% of the bounding box, verified per-band). This script stamps nodata=-9999 into each
output file so ArcGIS renders the mask as transparent instead of a literal value.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 01_split_predictor_bands.py
"""

from pathlib import Path

import rasterio

HERE = Path(__file__).parent
SRC_TIF = HERE.parent.parent / "probability map" / "predictor_stack_v4.tif"

PRED_COLS = [
    "dist_roads", "dist_parks", "dist_coca", "dist_mosaic",
    "temp_C", "vpd_kPa", "ndvi", "wind_ms", "oni",
]
NODATA = -9999.0


def main() -> None:
    with rasterio.open(SRC_TIF) as src:
        assert src.count == len(PRED_COLS), f"Expected {len(PRED_COLS)} bands, got {src.count}"
        profile = src.profile.copy()
        profile.update(count=1, nodata=NODATA)

        for i, name in enumerate(PRED_COLS, start=1):
            band = src.read(i)
            out_path = HERE / f"{i:02d}_{name}.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(band, 1)
                dst.set_band_description(1, name)
            print(f"Guardado: {out_path.name}")


if __name__ == "__main__":
    main()
