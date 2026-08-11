"""
Phase 1, Step 1 -- extract Aboveground Live Woody Biomass Density (AGB) for the study
nucleus, from Global Forest Watch's 30 m dataset:
https://data.globalforestwatch.org/datasets/gfw::aboveground-live-woody-biomass-density/about

Source: WHRC/GFW pantropical biomass map, year-2000 baseline (Zarin et al. 2016),
~30 m resolution, units Megagrams of biomass per hectare (Mg/ha).

Why this is a local download + rasterio clip, not a GEE Export.image.toDrive() call:
this specific 30 m dataset is NOT a pre-ingested Earth Engine asset -- only a coarser
500 m national-level product (`WHRC/biomass/tropical`) is in the EE catalog. GFW instead
distributes it as 10x10-degree GeoTIFF tiles behind a signed-URL download API (confirmed
by hand: `https://data-api.globalforestwatch.org/dataset/whrc_aboveground_woody_biomass_stock_2000/v1.4/...`
redirects to a temporary-credentialed S3 URL, not a public gs:// URI Earth Engine could
read directly). Getting this into GEE would mean uploading to a Google Cloud Storage
bucket and ingesting it as an asset first -- extra setup for no benefit, since no
server-side EE computation is actually needed here (we already know exactly which 2
tiles cover the nucleus). So: download the 2 needed tiles directly, clip to the nucleus
with rasterio, and export locally -- same end result (a 30 m GeoTIFF clipped to the
nucleus, EPSG:4326) that ArcGIS can open directly, without the GEE detour.

Tiles needed (confirmed by requesting the download endpoint for each and checking the
redirect resolves, not by assumption): the nucleus spans roughly lon -75.63 to -70.48,
lat -0.74 to 2.95 (from predictor_stack_v4.tif's extent), which straddles the equator
and falls entirely within the 80W tile column -> tiles `10N_080W` and `00N_080W`.

Nodata caveat: the source tiles are uint16 with nodata=0 (declared in their own GDAL
metadata). This dataset cannot distinguish a genuine zero-biomass pixel (bare ground,
water, urban) from a true data gap -- both read as 0 -- that's a property of the source
product, not something introduced by this script.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 01_extract_agb_biomass.py
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.mask
import rasterio.merge
import rasterio.warp
import requests
from shapely.geometry import shape

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import col_amazon_fire_utils as utils

# Raw tiles are cached here (gitignored) so re-running doesn't re-download ~hundreds of
# MB every time. Never committed -- these are 10x10-degree tiles, far too large for the repo.
RAW_TILES_DIR = HERE / "raw_tiles_cache"
RAW_TILES_DIR.mkdir(exist_ok=True)

TILE_IDS = ["10N_080W", "00N_080W"]

# GFW data-api endpoint + the public demo API key exposed in GFW's own web portal (the
# same key their site uses for anonymous downloads). If this ever gets rate-limited or
# revoked, request a personal key at https://www.globalforestwatch.org/ and swap it in.
GFW_API_KEY = "2d60cd88-8348-4c0f-a6d5-bd9adb585a8c"
DOWNLOAD_URL_TEMPLATE = (
    "https://data-api.globalforestwatch.org/dataset/"
    "whrc_aboveground_woody_biomass_stock_2000/v1.4/download/geotiff"
    "?grid=10/40000&tile_id={tile_id}&pixel_meaning=Mg_ha-1&x-api-key=" + GFW_API_KEY
)

OUT_TIF = HERE / "agb_density_mg_ha_2000_30m.tif"
NODATA = -9999.0
TARGET_CRS = "EPSG:4326"
TARGET_SCALE_M = 30
DEG_PER_METER = 1 / 111_320  # same approximation already used elsewhere in this repo
TARGET_RES_DEG = TARGET_SCALE_M * DEG_PER_METER


def download_tile(tile_id: str) -> Path:
    out_path = RAW_TILES_DIR / f"{tile_id}.tif"
    if out_path.exists():
        print(f"Already downloaded: {out_path.name}")
        return out_path

    url = DOWNLOAD_URL_TEMPLATE.format(tile_id=tile_id)
    print(f"Downloading {tile_id} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")
        print()
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    ee = utils.initialize_ee()
    nucleus_geom = utils.get_nucleus_geometry()
    nucleus_geojson = nucleus_geom.getInfo()
    nucleus_shape = shape(nucleus_geojson)
    print("Nucleus geometry loaded, bounds:", nucleus_shape.bounds)

    tile_paths = [download_tile(t) for t in TILE_IDS]

    print("Merging tiles ...")
    mosaic, mosaic_transform = rasterio.merge.merge([str(p) for p in tile_paths])
    with rasterio.open(tile_paths[0]) as src0:
        src_crs = src0.crs
        src_nodata = src0.nodata

    mosaic_profile = {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": 1,
        "dtype": mosaic.dtype,
        "crs": src_crs,
        "transform": mosaic_transform,
        "nodata": src_nodata,
    }

    print("Clipping to nucleus polygon (exact clip, not just bounding box) ...")
    # Mask/clip in the SOURCE dtype (uint16) using the SOURCE's own declared nodata (0),
    # not our -9999 float sentinel -- -9999 doesn't fit uint16 and rasterio.mask would
    # crash trying to fill an unsigned array with a negative value. The source file's own
    # nodata=0 already means "no data" in this dataset -- note it also can't be told apart
    # from a genuine zero-biomass pixel (bare ground, water, urban), a known limitation of
    # this product, not something introduced here.
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**mosaic_profile) as mem_ds:
            mem_ds.write(mosaic)
            clipped, clipped_transform = rasterio.mask.mask(
                mem_ds, [nucleus_shape], crop=True, nodata=src_nodata, filled=True
            )

    print(f"Resampling to {TARGET_SCALE_M} m ({TARGET_RES_DEG:.7f} deg/pixel) in {TARGET_CRS} ...")
    left = clipped_transform.c
    top = clipped_transform.f
    height, width = clipped.shape[1], clipped.shape[2]
    right = left + width * clipped_transform.a
    bottom = top + height * clipped_transform.e

    dst_width = max(1, round((right - left) / TARGET_RES_DEG))
    dst_height = max(1, round((top - bottom) / TARGET_RES_DEG))
    dst_transform = rasterio.transform.from_origin(left, top, TARGET_RES_DEG, TARGET_RES_DEG)

    dst = np.full((dst_height, dst_width), NODATA, dtype="float32")
    rasterio.warp.reproject(
        source=clipped[0].astype("float32"),
        destination=dst,
        src_transform=clipped_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        dst_nodata=NODATA,
        resampling=rasterio.warp.Resampling.bilinear,
    )

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": dst_transform,
        "nodata": NODATA,
        # Compression matters here: at 30 m over the whole nucleus this is ~1 GB
        # uncompressed. DEFLATE + a floating-point predictor exploits the ~53% constant
        # nodata region plus spatial autocorrelation in real values -- brings it down to
        # a size that's actually practical to keep in the repo.
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(OUT_TIF, "w", **profile) as out:
        out.write(dst, 1)
        out.set_band_description(1, "AGB_Mg_ha")

    valid = dst[dst != NODATA]
    print(f"Saved: {OUT_TIF}")
    print(f"Shape: {dst_height} x {dst_width} | valid pixels: {valid.size:,} ({100*valid.size/dst.size:.1f}%)")
    print(f"AGB range (Mg/ha): [{valid.min():.1f}, {valid.max():.1f}] | mean: {valid.mean():.1f}")


if __name__ == "__main__":
    main()
