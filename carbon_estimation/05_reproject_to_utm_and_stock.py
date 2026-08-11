"""
Phase 1, Step 4 -- reproject the ArcGIS-ready layers in `resampled_ArcGIS_30m/` from
EPSG:4326 (geographic, degrees) to EPSG:32618 / WGS 84 UTM zone 18N (projected, metres),
with exact 30 m x 30 m pixels, and convert carbon from density (Mg/ha) to per-pixel
stock (Mg C per 30 m cell). Ready for the carbon-at-risk overlay.

Why the order matters (do not reorder these steps):
1. Reproject to a metric CRS with an EXPLICIT 30 m resolution FIRST. Area math (a pixel
   "= 0.09 ha") is only valid in a projected, metres-based CRS -- in EPSG:4326 a pixel is
   0.0002695 degrees on a side, which is not a fixed area on the ground (it shrinks
   toward the poles). Only after reprojecting to UTM does "30 m x 30 m = 900 m^2 = 0.09
   ha" become literally true for every pixel.
2. Convert density -> stock AFTER reprojecting, using the REPROJECTED raster's actual
   pixel size (captured from the output, not hardcoded), because resolution is forced
   explicitly but still verified rather than assumed.

Fixed destination grid (critical for a valid overlay): the susceptibility layer is
reprojected first with `resolution=(30, 30)` passed explicitly to
`calculate_default_transform` (not auto-derived from the source raster's own pixel
count, which would produce a "close to 30 m" value, not exactly 30 m). That layer's
resulting transform/width/height is then reused AS-IS for every other layer -- they are
NOT independently reprojected with their own `calculate_default_transform` call, which
would silently offset them by a fraction of a pixel and break the overlay.

Resampling method depends on what the raster represents:
- Susceptibility: `Resampling.nearest` -- a continuous P(burned) surface today, but
  nearest neighbor is kept consistent with how this same layer was already resampled
  500 m -> 30 m in Phase 2 (see `outputs/probability_map/README.md`), and is the right
  choice if/when it becomes a classified (Jenks) map, since classes must never be
  interpolated into values that don't correspond to any real class.
- Carbon density: `Resampling.bilinear` -- a genuinely continuous field, where smooth
  interpolation between source pixels is appropriate.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 05_reproject_to_utm_and_stock.py
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

HERE = Path(__file__).parent
BUNDLE_DIR = HERE / "resampled_ArcGIS_30m"
OLD_DIR = BUNDLE_DIR / "_old"

DST_CRS = CRS.from_epsg(32618)  # WGS 84 / UTM zone 18N
TARGET_RES = 30.0  # metres -- forced explicitly, never auto-calculated
NODATA = -9999.0

SUSCEPTIBILITY_SRC = BUNDLE_DIR / "fire_susceptibility_probability_map_v4_resampled_30m.tif"
CARBON_DENSITY_SRC = BUNDLE_DIR / "carbon_density_existing_forest_2024_30m.tif"

SUSCEPTIBILITY_OUT = BUNDLE_DIR / "fire_susceptibility_probability_map_v4_resampled_30m_UTM.tif"
CARBON_DENSITY_OUT = BUNDLE_DIR / "carbon_density_existing_forest_2024_30m_UTM.tif"
CARBON_STOCK_OUT = BUNDLE_DIR / "carbon_stock_per_pixel_2024_30m_UTM.tif"

# Anything matching these stems in BUNDLE_DIR (any extension/sidecar) is a pre-UTM
# artifact -- either an EPSG:4326 input to this script, or a prior ArcGIS-side overlay
# attempt built before reprojecting (confirmed EPSG:4326, nodata -3.4e38 -- an ArcGIS
# default, not this project's -9999 convention) -- all get moved to _old/, never deleted.
OLD_STEMS = [
    "fire_susceptibility_probability_map_v4_resampled_30m",
    "carbon_density_existing_forest_2024_30m",
    "carbon_at_risk_30m",
    "susceptibility_high_veryhigh_30m",
]


def reproject_reference_layer(src_path: Path, resampling: Resampling):
    """Reprojects src_path to DST_CRS, computing ITS OWN destination grid with an
    explicit 30 x 30 m resolution. Returns the data plus the grid definition
    (transform/width/height), which becomes the fixed reference grid for every
    other layer."""
    with rasterio.open(src_path) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds,
            resolution=(TARGET_RES, TARGET_RES),
        )
        dst = np.full((dst_height, dst_width), NODATA, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            dst_nodata=NODATA,
            resampling=resampling,
        )
    return dst, dst_transform, dst_width, dst_height


def reproject_onto_fixed_grid(src_path: Path, dst_transform, dst_width, dst_height, resampling: Resampling):
    """Reprojects src_path onto an ALREADY-DEFINED grid -- guarantees pixel-for-pixel
    alignment with whichever layer defined that grid, instead of computing its own
    (slightly different) destination transform."""
    with rasterio.open(src_path) as src:
        dst = np.full((dst_height, dst_width), NODATA, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            dst_nodata=NODATA,
            resampling=resampling,
        )
    return dst


def write_raster(path: Path, data: np.ndarray, transform, description: str) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": DST_CRS,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as out:
        out.write(data, 1)
        out.set_band_description(1, description)


def summarize(path: Path) -> None:
    with rasterio.open(path) as src:
        arr = src.read(1)
        valid = arr[arr != src.nodata]
        print(f"\n{path.name}")
        print(f"  CRS:        {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}] | mean: {valid.mean():.4f}")


def main() -> None:
    BUNDLE_DIR.mkdir(exist_ok=True)

    print("=== Step 1: reproject to EPSG:32618 at exactly 30 m, on one fixed grid ===")
    print(f"Reference layer (defines the grid): {SUSCEPTIBILITY_SRC.name}")
    sus_data, dst_transform, dst_width, dst_height = reproject_reference_layer(
        SUSCEPTIBILITY_SRC, Resampling.nearest
    )
    write_raster(SUSCEPTIBILITY_OUT, sus_data, dst_transform, "fire_susceptibility_probability_UTM")
    print(f"Saved: {SUSCEPTIBILITY_OUT.name}")

    print(f"\nReprojecting onto that SAME fixed grid: {CARBON_DENSITY_SRC.name}")
    density_data = reproject_onto_fixed_grid(
        CARBON_DENSITY_SRC, dst_transform, dst_width, dst_height, Resampling.bilinear
    )
    write_raster(CARBON_DENSITY_OUT, density_data, dst_transform, "carbon_density_existing_forest_Mg_C_ha_UTM")
    print(f"Saved: {CARBON_DENSITY_OUT.name}")

    print("\n=== Step 2: verify CRS and resolution (not assumed) ===")
    actual_res = None
    for f in [SUSCEPTIBILITY_OUT, CARBON_DENSITY_OUT]:
        with rasterio.open(f) as src:
            print(f"{f.name}: crs={src.crs} | res={src.res}")
            assert src.crs == DST_CRS, f"{f.name} is not in {DST_CRS}"
            assert abs(src.res[0] - TARGET_RES) < 1e-6 and abs(src.res[1] - TARGET_RES) < 1e-6, (
                f"{f.name} resolution {src.res} is not exactly {TARGET_RES} m"
            )
            actual_res = src.res  # captured from the real output, used below -- not hardcoded

    pixel_area_ha = (actual_res[0] * actual_res[1]) / 10_000
    print(f"\nActual pixel size: {actual_res[0]:.6f} x {actual_res[1]:.6f} m "
          f"-> pixel area = {pixel_area_ha:.6f} ha (used for the stock conversion below)")

    print("\n=== Step 3: convert carbon density (Mg/ha) -> stock per pixel (Mg C / 30 m cell) ===")
    with rasterio.open(CARBON_DENSITY_OUT) as src:
        density = src.read(1)
        density_nodata = src.nodata
        density_transform = src.transform

    valid_mask = density != density_nodata
    stock = np.full(density.shape, NODATA, dtype="float32")
    stock[valid_mask] = density[valid_mask] * pixel_area_ha  # only valid pixels are multiplied
    write_raster(CARBON_STOCK_OUT, stock, density_transform, "carbon_stock_Mg_C_per_pixel_UTM")
    print(f"Saved: {CARBON_STOCK_OUT.name}")

    print("\n=== Step 4: verify the stock conversion ===")
    density_valid = density[valid_mask]
    stock_valid = stock[stock != NODATA]
    print(f"Density range: [{density_valid.min():.2f}, {density_valid.max():.2f}] Mg C/ha")
    print(f"Stock range:   [{stock_valid.min():.4f}, {stock_valid.max():.4f}] Mg C/pixel")
    expected_max = density_valid.max() * pixel_area_ha
    print(f"Expected max stock = density_max x {pixel_area_ha} ha = {expected_max:.4f}")
    if abs(stock_valid.max() - expected_max) > 0.01:
        print("STOP: stock max does not match density_max x pixel_area -- something is wrong, check before proceeding.")
        raise SystemExit(1)
    print("Stock conversion verified OK.")

    print("\n=== Grid-alignment check across all 3 final UTM layers ===")
    with rasterio.open(SUSCEPTIBILITY_OUT) as a, rasterio.open(CARBON_DENSITY_OUT) as b, rasterio.open(CARBON_STOCK_OUT) as c:
        aligned = (a.transform == b.transform == c.transform) and (a.shape == b.shape == c.shape) and (a.crs == b.crs == c.crs)
        print(f"All 3 layers share the exact same grid: {aligned}")
        assert aligned, "Final layers are not grid-aligned -- overlay would be invalid."

    print("\n=== Step 5: move pre-UTM / EPSG:4326 artifacts to _old/ (nothing deleted) ===")
    OLD_DIR.mkdir(exist_ok=True)
    moved = 0
    for stem in OLD_STEMS:
        for f in BUNDLE_DIR.glob(f"{stem}.*"):
            if f.parent == BUNDLE_DIR:  # don't touch anything already inside _old/
                dest = OLD_DIR / f.name
                f.rename(dest)
                print(f"Moved {f.name} -> _old/")
                moved += 1
    print(f"{moved} file(s) moved to _old/.")

    print("\n=== Final summary: layers remaining in resampled_ArcGIS_30m/ ===")
    for f in [SUSCEPTIBILITY_OUT, CARBON_DENSITY_OUT, CARBON_STOCK_OUT]:
        summarize(f)


if __name__ == "__main__":
    main()
