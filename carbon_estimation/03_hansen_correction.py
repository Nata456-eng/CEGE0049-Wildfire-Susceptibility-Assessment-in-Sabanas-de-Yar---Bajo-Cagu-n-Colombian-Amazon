"""
Phase 1, Step 3 -- Hansen correction. The AGB/carbon rasters from Steps 1-2 are a
year-2000 baseline: they say nothing about forest lost since then. This step uses Hansen
Global Forest Change (loss year 2001-2024) to zero out carbon in pixels that have since
been deforested, so the result is "carbon in forest that still exists," not "carbon that
existed in 2000."

Dataset: `UMD/hansen/global_forest_change_2024_v1_12` (confirmed to exist and to carry
the expected bands by loading it directly in Earth Engine before writing this script --
see the verification step below, not assumed).

Step by step (also documented in README.md):
1. Load the Hansen image and verify it has the bands this script depends on
   (`lossyear`, `treecover2000`) and that `lossyear` actually spans loss years
   2001-2024 (values 1-24) within the nucleus -- confirmed: min=1, max=24, all 24
   distinct values present.
2. IMPORTANT correction to the literal instruction ("pixels with lossyear==0"): the
   `lossyear` band is NOT a clean 0-24 integer band. Earth Engine MASKS (leaves as
   nodata) every pixel where no loss was ever detected -- it does not store a literal 0
   there. Checked directly: a histogram of `lossyear` inside the nucleus returns only
   keys 1-24, never "0". A naive `lossyear.eq(0)` would therefore stay masked (not
   evaluate to 1) almost everywhere loss did NOT happen -- the opposite of the intended
   mask. Fixed with `.unmask(0)` first, so masked ("no loss ever recorded") pixels become
   literal 0 before the `.eq(0)` comparison.
3. Build the mask in Earth Engine: `lossyear.unmask(0).eq(0)` -> 1 where no loss was
   recorded 2001-2024, 0 where loss was recorded in any of those years.
4. Clip to the nucleus and `.unmask(2, False)` (sameFootprint=False, same lesson learned
   in `outputs/probability_map/`'s README) so the sentinel value 2 -- deliberately
   outside the valid {0, 1} mask range -- fills the WHOLE export rectangle, not just gaps
   inside the clipped footprint, distinguishing "outside the nucleus / no Hansen data"
   from a real 0.
5. Export the mask at 30 m, EPSG:4326, directly (small file -- a byte mask, not a
   float32 continuous surface).
6. Locally: snap the downloaded mask onto the EXACT grid of `carbon_density_mg_c_ha_2000_30m.tif`
   with nearest-neighbor resampling (a categorical mask must never be interpolated), then:
   - carbon pixel is nodata -> output stays nodata (-9999)
   - carbon pixel valid, mask == 1 (no loss) -> output = carbon (unchanged)
   - carbon pixel valid, mask == 0 (loss recorded) -> output = 0.0 (real zero: that
     carbon is no longer standing forest, not "no data")
   - carbon pixel valid, mask == 2 (no Hansen data there) -> output = nodata (can't
     determine loss status, so don't guess)
7. Write the corrected raster.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 03_hansen_correction.py
"""

import sys
from pathlib import Path

import geemap
import numpy as np
import rasterio
import rasterio.warp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import col_amazon_fire_utils as utils

HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
CARBON_TIF = HERE / "carbon_density_mg_c_ha_2000_30m.tif"
MASK_TIF = HERE / "hansen_no_loss_mask_2001_2024_30m.tif"
OUT_TIF = HERE / "carbon_density_existing_forest_2024_30m.tif"

MASK_NODATA_SENTINEL = 2  # outside {0, 1} on purpose -- distinct from a real "loss" 0
NODATA = -9999.0


def build_and_export_mask(ee, nucleus_geom) -> None:
    hansen = ee.Image(HANSEN_ASSET)

    band_names = hansen.bandNames().getInfo()
    print("Hansen bands:", band_names)
    for required in ["lossyear", "treecover2000"]:
        assert required in band_names, f"Expected band '{required}' not found in {HANSEN_ASSET}"

    lossyear = hansen.select("lossyear")
    stats = lossyear.reduceRegion(
        reducer=ee.Reducer.minMax(), geometry=nucleus_geom, scale=30, maxPixels=1e10, bestEffort=True
    ).getInfo()
    ly_min, ly_max = stats["lossyear_min"], stats["lossyear_max"]
    print(f"lossyear range in nucleus: {ly_min}-{ly_max}")
    assert ly_min == 1 and ly_max == 24, (
        f"Expected lossyear to span 1-24 (loss years 2001-2024), got {ly_min}-{ly_max} -- "
        "the year coverage assumption behind this correction may no longer hold."
    )

    # unmask(0) FIRST: lossyear has no literal 0s, "no loss" pixels are masked instead
    # (see docstring point 2). Skipping this would silently invert the mask.
    no_loss_mask = lossyear.unmask(0).eq(0).rename("no_loss_2001_2024")

    mask_export = (
        no_loss_mask.clip(nucleus_geom)
        .toByte()
        .unmask(MASK_NODATA_SENTINEL, False)  # sameFootprint=False: fill the WHOLE export rectangle
    )

    # geemap.ee_export_image (a single synchronous pixel-fetch request) is capped at 48 MB
    # by Earth Engine -- this mask is ~500 MB uncompressed at 30 m over the whole nucleus,
    # so it needs geemap.download_ee_image instead, which auto-splits the region into
    # tiles under that limit, downloads them in parallel, and mosaics them back together.
    print("Exporting Hansen no-loss mask (tiled download) ...")
    geemap.download_ee_image(
        mask_export,
        filename=str(MASK_TIF),
        region=nucleus_geom,
        scale=30,
        crs="EPSG:4326",
        resampling="near",
    )
    print("Saved:", MASK_TIF)


def apply_mask_locally() -> None:
    with rasterio.open(CARBON_TIF) as csrc:
        carbon = csrc.read(1)
        carbon_transform = csrc.transform
        carbon_crs = csrc.crs
        carbon_profile = csrc.profile.copy()
        carbon_nodata = csrc.nodata
        print("Carbon grid:", csrc.width, "x", csrc.height)

    with rasterio.open(MASK_TIF) as msrc:
        mask_src_data = msrc.read(1)
        mask_transform = msrc.transform
        mask_crs = msrc.crs
        print("Mask grid (pre-snap):", msrc.width, "x", msrc.height)

    print("Snapping mask onto the carbon grid with NEAREST NEIGHBOR ...")
    mask_snapped = np.full(carbon.shape, MASK_NODATA_SENTINEL, dtype="uint8")
    rasterio.warp.reproject(
        source=mask_src_data,
        destination=mask_snapped,
        src_transform=mask_transform,
        src_crs=mask_crs,
        dst_transform=carbon_transform,
        dst_crs=carbon_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )

    corrected = np.full(carbon.shape, NODATA, dtype="float32")
    valid_carbon = carbon != carbon_nodata
    no_loss = valid_carbon & (mask_snapped == 1)
    lost = valid_carbon & (mask_snapped == 0)
    # valid_carbon & (mask_snapped == MASK_NODATA_SENTINEL) stays NODATA -- can't tell
    # loss status there, so don't guess.

    corrected[no_loss] = carbon[no_loss]
    corrected[lost] = 0.0

    carbon_profile.update(dtype="float32", nodata=NODATA)
    with rasterio.open(OUT_TIF, "w", **carbon_profile) as out:
        out.write(corrected, 1)
        out.set_band_description(1, "carbon_density_existing_forest_Mg_C_ha")

    valid = corrected[corrected != NODATA]
    n_zeroed = int(lost.sum())
    print(f"Saved: {OUT_TIF}")
    print(f"Valid pixels: {valid.size:,} | zeroed out (forest loss 2001-2024): {n_zeroed:,} "
          f"({100*n_zeroed/valid.size:.1f}% of valid pixels)")
    print(f"Corrected carbon range (Mg C/ha): [{valid.min():.1f}, {valid.max():.1f}] | mean: {valid.mean():.1f}")


def main() -> None:
    ee = utils.initialize_ee()
    nucleus_geom = utils.get_nucleus_geometry()
    build_and_export_mask(ee, nucleus_geom)
    apply_mask_locally()


if __name__ == "__main__":
    main()
