"""
Exports a MODIS MCD64A1-derived raster for the study nucleus, at the same 500 m /
EPSG:4326 grid as the 9 predictor layers, so it can be overlaid pixel-for-pixel with them
in ArcGIS.

What it represents -- a deliberate choice, flag if it's not what you need:
    Per-pixel COUNT of years, 2001-2024, in which MODIS/061/MCD64A1 detected a burn
    (BurnDate > 0 at any point in the calendar year). Range 0-24.

Why a frequency count and not a single year's snapshot: the 9 predictor layers already
represent a 2001-2024 climatological baseline (not any specific year -- see
'../README.md'), and 2001-2024 is also the exact year range of
`data/model_dataset/pixel_year_full.csv` (the table the model was trained on). A frequency
count is the fire-side counterpart of that same baseline: "how often did this pixel
actually burn across the years the model learned from", which is what the 9 predictors
are being compared against. A single year would need to pick one arbitrarily and
wouldn't be comparable to a multi-year climate normal.
If you instead want a specific year's burned/not-burned mask, or "years since last
fire", or a binary ever-burned mask, tell Claude and it'll regenerate this file with that
definition instead -- the GEE call is a small change.

Same burned-year definition as `col_amazon_fire_utils.get_burned_df` (annual max BurnDate
> 0), just kept per-pixel here instead of reduced to one number for the whole nucleus.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 02_export_modis_fire_frequency.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent.parent))

import geemap

import col_amazon_fire_utils as utils

START_YEAR = 2001
END_YEAR = 2024
NODATA = -9999.0
OUT_TIF = HERE / "10_modis_fire_frequency_2001_2024.tif"


def main() -> None:
    ee = utils.initialize_ee()
    nucleus_geom = utils.get_nucleus_geometry()

    modis_ba = ee.ImageCollection("MODIS/061/MCD64A1").filterBounds(nucleus_geom)

    def burned_year(year):
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        annual = modis_ba.filterDate(start, end).select("BurnDate")
        return annual.max().gt(0).unmask(0)

    years = ee.List.sequence(START_YEAR, END_YEAR)
    burned_stack = ee.ImageCollection(years.map(lambda y: burned_year(ee.Number(y))))

    fire_frequency = (
        burned_stack.sum()
        .rename("fire_frequency")
        .toFloat()
        .clip(nucleus_geom)
        .unmask(NODATA, False)  # sameFootprint=False -- same convention as predictor_stack_v4.tif
    )

    print(f"Exporting fire frequency {START_YEAR}-{END_YEAR} ...")
    geemap.ee_export_image(
        fire_frequency,
        filename=str(OUT_TIF),
        scale=500,
        crs="EPSG:4326",
        region=nucleus_geom,
        file_per_band=False,
    )
    print("Saved:", OUT_TIF)


if __name__ == "__main__":
    main()
