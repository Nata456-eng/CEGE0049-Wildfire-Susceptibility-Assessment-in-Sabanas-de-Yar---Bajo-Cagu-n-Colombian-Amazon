"""
Phase 2 -- Harmonization. Resamples the fire-susceptibility probability surface onto the
exact 30 m grid of `carbon_density_existing_forest_2024_30m.tif` (the final Phase 1
deliverable), and bundles both 30 m rasters together in `resampled_ArcGIS_30m/` so they
can be loaded side by side in ArcGIS.

This ONLY READS `outputs/probability_map/fire_susceptibility_probability_v4.tif` -- it
is never modified or overwritten. The resampled output is a new, separate file.

Method -- nearest neighbor, deliberately:
Nearest neighbor is the only resampling method that doesn't fabricate information. Every
30 m output pixel is assigned EXACTLY the value of the one 500 m pixel that contains it --
no interpolation, no new intermediate values invented between two neighbouring 500 m
susceptibility values.

Snap-to-grid: the destination transform, CRS, width, and height are copied directly from
`carbon_density_existing_forest_2024_30m.tif` (not independently recomputed), so origin,
cell size, and extent match that raster exactly. This is verified programmatically after
writing, not just assumed.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 04_resample_susceptibility_for_arcgis.py
"""

import shutil
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp

HERE = Path(__file__).parent

SUSCEPTIBILITY_500M = HERE.parent / "outputs" / "probability_map" / "fire_susceptibility_probability_v4.tif"
CARBON_30M = HERE / "carbon_density_existing_forest_2024_30m.tif"

OUT_DIR = HERE / "resampled_ArcGIS_30m"
OUT_SUSCEPTIBILITY = OUT_DIR / "fire_susceptibility_probability_map_v4_resampled_30m.tif"
OUT_CARBON_COPY = OUT_DIR / CARBON_30M.name

NODATA = -9999.0


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    with rasterio.open(CARBON_30M) as snap:
        dst_transform = snap.transform
        dst_crs = snap.crs
        dst_height = snap.height
        dst_width = snap.width
        print("Snap grid (30 m carbon raster):", snap.width, "x", snap.height, "| crs:", snap.crs)

    with rasterio.open(SUSCEPTIBILITY_500M) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        print("Source (500 m susceptibility):", src.width, "x", src.height, "| crs:", src.crs, "| nodata:", src_nodata)

    dst = np.full((dst_height, dst_width), NODATA, dtype="float32")

    print("Resampling 500 m -> 30 m with NEAREST NEIGHBOR, snapped to the carbon grid ...")
    rasterio.warp.reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=NODATA,
        resampling=rasterio.warp.Resampling.nearest,
    )

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(OUT_SUSCEPTIBILITY, "w", **profile) as out:
        out.write(dst, 1)
        out.set_band_description(1, "fire_susceptibility_probability_resampled_nn")

    valid = dst[dst != NODATA]
    print(f"Saved: {OUT_SUSCEPTIBILITY}")
    print(f"Shape: {dst_height} x {dst_width} | valid pixels: {valid.size:,} ({100*valid.size/dst.size:.1f}%)")
    print(f"Probability range: [{valid.min():.3f}, {valid.max():.3f}] | mean: {valid.mean():.3f}")

    # Grid-alignment verification -- required, not assumed: confirm the resampled
    # susceptibility raster and the 30 m carbon raster share the exact same origin,
    # cell size, extent, and CRS.
    with rasterio.open(OUT_SUSCEPTIBILITY) as sus, rasterio.open(CARBON_30M) as carb:
        same_transform = sus.transform == carb.transform
        same_shape = sus.shape == carb.shape
        same_crs = sus.crs == carb.crs
        same_bounds = sus.bounds == carb.bounds
        print(f"Same transform (origin + cell size): {same_transform}")
        print(f"Same shape (extent, in pixels):       {same_shape}")
        print(f"Same CRS:                              {same_crs}")
        print(f"Same bounds:                            {same_bounds}")
        assert same_transform and same_shape and same_crs and same_bounds, (
            "Grid mismatch between the resampled susceptibility raster and the carbon "
            "raster -- they must align exactly for a valid pixel-by-pixel overlay."
        )
        print("Grid alignment VERIFIED: susceptibility and carbon rasters match exactly.")

    print(f"Copying {CARBON_30M.name} into {OUT_DIR.name}/ so both final products sit together ...")
    shutil.copy2(CARBON_30M, OUT_CARBON_COPY)
    print(f"Saved: {OUT_CARBON_COPY}")


if __name__ == "__main__":
    main()
