"""
Smoothing the fire-susceptibility probability surface for stakeholder communication.

`fire_susceptibility_probability_v4.tif` is native to a 500 m Random Forest prediction
grid. When zoomed into for a presentation/report, it reads as a blocky, "pixelated"
mosaic that isn't easy for non-technical stakeholders to interpret -- the 30 m version
used in the carbon-at-risk analysis (`carbon_estimation/`) makes the blockiness even
more visually obvious, since nearest-neighbor resampling copies each 500 m block onto
many small 30 m cells without changing its value.

This script is a PRESENTATION product only, kept separate from the analytical layers:
- It does NOT feed back into `carbon_estimation/` or any threshold-based analysis --
  those correctly use nearest-neighbor / the unsmoothed continuous surface because
  classes and thresholds must never be blurred across pixel boundaries.
- It exists purely so stakeholders looking at a map can read the overall spatial pattern
  (where risk is concentrated) without being distracted by 500 m grid artifacts.

Method:
1. Gaussian smoothing on the NATIVE 500 m grid (EPSG:4326), using NaN-aware normalized
   convolution (`scipy.ndimage.gaussian_filter` on the data and on the valid-data mask
   separately, then dividing) so NoData pixels never bleed a fake "0" into their
   neighbors and never get blurred away from the coverage edge.
2. Reproject the smoothed 500 m surface to 30 m / EPSG:32618 with BILINEAR resampling
   (appropriate here, unlike the analytical layers, because this is a genuinely
   continuous field being prepared for visual display, not for classification).

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 01_smooth_susceptibility_for_stakeholders.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).parent
SRC_TIF = HERE.parent / "fire_susceptibility_probability_v4.tif"  # native 500 m, EPSG:4326

SMOOTHED_500M_OUT = HERE / "fire_susceptibility_smoothed_500m.tif"
SMOOTHED_30M_UTM_OUT = HERE / "fire_susceptibility_smoothed_30m_UTM.tif"
QUICKLOOK_OUT = HERE / "quicklook_smoothed_vs_original.png"

DST_CRS = CRS.from_epsg(32618)  # WGS 84 / UTM zone 18N -- matches every other UTM layer
TARGET_RES = 30.0  # metres
NODATA = -9999.0

# Gaussian sigma in NATIVE 500 m PIXELS. 1.5 px ~= 750 m smoothing radius -- enough to
# soften block edges for a presentation map without erasing the real spatial pattern
# (the coarser, low-vs-high-risk gradient across the nucleus is preserved).
SIGMA_PIXELS = 1.5

print("=" * 70)
print("STEP 1 -- Loading native 500 m susceptibility surface (EPSG:4326)")
print("=" * 70)

with rasterio.open(SRC_TIF) as src:
    prob = src.read(1, masked=True)
    src_transform = src.transform
    src_crs = src.crs
    src_profile = src.profile.copy()

valid = ~np.ma.getmaskarray(prob)
print(f"Shape: {prob.shape}, valid pixels: {int(valid.sum()):,}, nodata: {src_profile.get('nodata')}")

print()
print("=" * 70)
print(f"STEP 2 -- NaN-aware Gaussian smoothing (sigma={SIGMA_PIXELS} native pixels)")
print("=" * 70)

data_zeroed = np.where(valid, prob.filled(0.0), 0.0)
mask_float = valid.astype("float64")

smoothed_numerator = gaussian_filter(data_zeroed, sigma=SIGMA_PIXELS)
smoothed_denominator = gaussian_filter(mask_float, sigma=SIGMA_PIXELS)

with np.errstate(invalid="ignore", divide="ignore"):
    smoothed = np.where(smoothed_denominator > 1e-6, smoothed_numerator / smoothed_denominator, np.nan)

smoothed_500m = np.where(valid, smoothed, NODATA).astype("float32")
# Pixels that were NoData originally stay NoData -- smoothing never invents data outside
# the original valid-data footprint.
smoothed_500m = np.where(valid, smoothed_500m, NODATA).astype("float32")

print(f"Smoothed value range (valid pixels): "
      f"{np.nanmin(np.where(valid, smoothed_500m, np.nan)):.3f} - "
      f"{np.nanmax(np.where(valid, smoothed_500m, np.nan)):.3f}")

profile_500m = src_profile.copy()
profile_500m.update({"dtype": "float32", "nodata": NODATA, "compress": "deflate"})
with rasterio.open(SMOOTHED_500M_OUT, "w", **profile_500m) as dst:
    dst.write(smoothed_500m, 1)
print(f"Saved intermediate 500 m smoothed raster: {SMOOTHED_500M_OUT}")

print()
print("=" * 70)
print("STEP 3 -- Reprojecting smoothed surface to 30 m / EPSG:32618 (bilinear)")
print("=" * 70)

dst_transform, dst_width, dst_height = calculate_default_transform(
    src_crs, DST_CRS, src_profile["width"], src_profile["height"],
    *rasterio.transform.array_bounds(src_profile["height"], src_profile["width"], src_transform),
    resolution=(TARGET_RES, TARGET_RES),
)

smoothed_30m_utm = np.full((dst_height, dst_width), NODATA, dtype="float32")
reproject(
    source=smoothed_500m,
    destination=smoothed_30m_utm,
    src_transform=src_transform,
    src_crs=src_crs,
    src_nodata=NODATA,
    dst_transform=dst_transform,
    dst_crs=DST_CRS,
    dst_nodata=NODATA,
    resampling=Resampling.bilinear,
)

profile_30m_utm = {
    "driver": "GTiff",
    "height": dst_height,
    "width": dst_width,
    "count": 1,
    "dtype": "float32",
    "crs": DST_CRS,
    "transform": dst_transform,
    "nodata": NODATA,
    "compress": "deflate",
}
with rasterio.open(SMOOTHED_30M_UTM_OUT, "w", **profile_30m_utm) as dst:
    dst.write(smoothed_30m_utm, 1)
print(f"Saved: {SMOOTHED_30M_UTM_OUT}")
print(f"  shape={smoothed_30m_utm.shape}, CRS={DST_CRS}, resolution=30 m")

print()
print("=" * 70)
print("STEP 4 -- Quicklook: original (blocky) vs. smoothed, side by side")
print("=" * 70)

original_display = np.ma.masked_equal(prob.filled(NODATA), NODATA)
smoothed_display = np.ma.masked_equal(smoothed_500m, NODATA)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, grid, title in zip(
    axes,
    [original_display, smoothed_display],
    ["Original (500 m, unsmoothed)", f"Smoothed (Gaussian sigma={SIGMA_PIXELS} px)"],
):
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04,
             label="Fire susceptibility probability")
fig.suptitle("Fire susceptibility surface -- smoothing for stakeholder communication",
             fontsize=13, fontweight="bold")
plt.savefig(QUICKLOOK_OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {QUICKLOOK_OUT}")

print()
print("Done. Reminder: use the smoothed layer for presentation maps only -- keep using")
print("the original unsmoothed continuous layer for any threshold/classification analysis.")
