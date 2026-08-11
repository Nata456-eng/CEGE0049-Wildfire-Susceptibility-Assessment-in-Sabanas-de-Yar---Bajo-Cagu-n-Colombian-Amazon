"""
Phase 5, Step 8 -- Smoothed fire-susceptibility surface, for stakeholder presentation only.

`fire_susceptibility_probability_v4.tif` is native 500 m (EPSG:4326). Every analysis
raster derived from it in this repo (`04_resample_susceptibility_for_arcgis.py`,
`05_reproject_to_utm_and_stock.py`) deliberately uses NEAREST NEIGHBOR resampling, so the
30 m grid used for the Jenks thresholds/carbon-at-risk math never invents values between
two real 500 m probabilities. That's correct for analysis, but it also means the 30 m
version LOOKS like a hard 500 m checkerboard when displayed -- fine for a modeller, hard
to read for a non-technical stakeholder audience.

This script produces a SEPARATE, presentation-only layer: it does not touch, and is not
used by, the carbon-at-risk analysis. It:
  1. Applies a Gaussian low-pass filter to the native 500 m grid, using NORMALIZED
     CONVOLUTION (smooth the data and the valid-pixel mask separately, then divide) so
     NoData pixels never bleed zeros into valid pixels near the nucleus boundary -- a
     naive `gaussian_filter` on the raw array would darken every edge pixel.
  2. Resamples the smoothed 500 m surface onto the SAME fixed 30 m UTM grid as the rest
     of the `resampled_ArcGIS_30m/` bundle, using BILINEAR interpolation (appropriate
     here -- this output is never thresholded/classified, only displayed).
  3. Clips the result back to [0, 1] (a probability) and writes it as a new GeoTIFF,
     clearly named so it's never mistaken for the analysis-grade layer.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 08_smooth_susceptibility_for_presentation.py
"""

from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).parent
BUNDLE_DIR = HERE / "resampled_ArcGIS_30m"

SUSCEPTIBILITY_500M = HERE.parent / "outputs" / "probability_map" / "fire_susceptibility_probability_v4.tif"
# Reuse the exact same 30 m UTM grid as the rest of the bundle (fixed reference grid).
SNAP_GRID_PATH = BUNDLE_DIR / "carbon_stock_per_pixel_2024_30m_UTM.tif"

OUT_PATH = BUNDLE_DIR / "fire_susceptibility_smoothed_presentation_30m_UTM.tif"

NODATA = -9999.0
GAUSSIAN_SIGMA = 1.2  # in units of 500 m source pixels -- ~600 m smoothing radius


def main() -> None:
    print("=" * 70)
    print("STEP 1 -- Load native 500 m susceptibility surface")
    print("=" * 70)
    with rasterio.open(SUSCEPTIBILITY_500M) as src:
        raw = src.read(1, masked=True)
        src_transform = src.transform
        src_crs = src.crs
    print(f"Source: {SUSCEPTIBILITY_500M.name} | shape={raw.shape} | "
          f"valid range=[{raw.min():.3f}, {raw.max():.3f}]")

    print()
    print("=" * 70)
    print("STEP 2 -- Gaussian smoothing with normalized convolution (NoData-safe)")
    print("=" * 70)
    valid_mask = (~np.ma.getmaskarray(raw)).astype("float32")
    data_zeroed = raw.filled(0.0).astype("float32")

    smoothed_data = gaussian_filter(data_zeroed, sigma=GAUSSIAN_SIGMA, mode="constant", cval=0.0)
    smoothed_mask = gaussian_filter(valid_mask, sigma=GAUSSIAN_SIGMA, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed_data / smoothed_mask
    # Pixels with (almost) no valid support in the smoothing window stay NoData
    smoothed = np.where(smoothed_mask > 0.1, smoothed, np.nan)
    smoothed = np.clip(smoothed, 0.0, 1.0)  # a probability can't leave [0, 1]

    valid_smoothed = ~np.isnan(smoothed)
    print(f"Smoothed (500 m grid) range: "
          f"[{np.nanmin(smoothed):.3f}, {np.nanmax(smoothed):.3f}] | "
          f"valid pixels: {int(valid_smoothed.sum()):,}")

    print()
    print("=" * 70)
    print("STEP 3 -- Resample smoothed surface onto the 30 m UTM bundle grid (bilinear)")
    print("=" * 70)
    with rasterio.open(SNAP_GRID_PATH) as snap:
        dst_transform = snap.transform
        dst_crs = snap.crs
        dst_height = snap.height
        dst_width = snap.width
        profile = snap.profile.copy()
    print(f"Snap grid: {SNAP_GRID_PATH.name} -> {dst_width} x {dst_height} | crs: {dst_crs}")

    src_for_warp = np.where(valid_smoothed, smoothed, NODATA).astype("float32")
    dst = np.full((dst_height, dst_width), NODATA, dtype="float32")

    rasterio.warp.reproject(
        source=src_for_warp,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=NODATA,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=NODATA,
        resampling=rasterio.warp.Resampling.bilinear,
    )
    dst = np.where(dst == NODATA, NODATA, np.clip(dst, 0.0, 1.0)).astype("float32")

    valid_dst = dst != NODATA
    print(f"30 m UTM presentation layer range: "
          f"[{dst[valid_dst].min():.3f}, {dst[valid_dst].max():.3f}] | "
          f"valid pixels: {int(valid_dst.sum()):,}")

    print()
    print("=" * 70)
    print("STEP 4 -- Write GeoTIFF")
    print("=" * 70)
    profile.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "nodata": NODATA,
        "compress": "deflate",
    })
    with rasterio.open(OUT_PATH, "w", **profile) as dst_ds:
        dst_ds.write(dst, 1)

    print(f"Saved: {OUT_PATH}")
    print("NOTE: presentation-only layer (Gaussian-smoothed + bilinear-resampled). "
          "Do NOT use this for Jenks thresholds/classification/carbon-at-risk math -- "
          "use fire_susceptibility_probability_map_v4_resampled_30m_UTM.tif for that.")


if __name__ == "__main__":
    main()
