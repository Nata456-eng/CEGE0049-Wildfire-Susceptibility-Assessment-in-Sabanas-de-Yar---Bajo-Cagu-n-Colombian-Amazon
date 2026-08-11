"""
Validation, Phase 2, Step 2 -- Rasterize the SINCHI burn-scar polygons (Phase 1
deliverable) onto EXACTLY the same 500 m grid as the evaluation model's susceptibility
raster (`susceptibility_model2019_500m_UTM.tif`, generated in Step 1).

Instead of a binary "burned / not burned" cell, this computes the FRACTIONAL area of
each 500 m cell covered by burn-scar polygons (0-1), preserving the burn-intensity
gradient within each cell rather than collapsing it to a single yes/no bit -- this
matters because a cell that's 90% burned and a cell that's 5% burned should not compare
identically against the model's continuous probability output.

METHOD -- sub-pixel oversampling (a standard, exact way to compute area fraction with
GDAL/rasterio, which has no native "rasterize as fraction" mode):
1. Load the Step 1 susceptibility raster purely as a GRID TEMPLATE -- its transform,
   CRS, width and height are inherited exactly; no new grid is generated.
2. Build a finer sub-grid that exactly subdivides every 500 m template cell into
   OVERSAMPLE x OVERSAMPLE sub-cells (25 m each, matching the ~30 m native resolution of
   the Landsat-derived burn scars), sharing the SAME origin/orientation as the template
   (only the pixel size is divided by OVERSAMPLE).
3. Rasterize the burn-scar polygons onto that fine grid as a 0/1 mask (1 where a
   sub-cell's center falls inside a polygon).
4. Reshape and average every OVERSAMPLE x OVERSAMPLE block of the fine mask back down to
   one value per 500 m template cell -- this average IS the burned-area fraction.
5. Re-apply the template's own valid/NoData mask (nucleus boundary) so cells outside the
   nucleus are NoData, not a false "0 = no burn scar" -- and cells inside the nucleus
   with no burn-scar overlap correctly get an explicit 0.
6. Verify (assert) the output raster's shape, transform and CRS are identical to the
   template before saving.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 02_rasterize_burn_fraction.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent

TEMPLATE_TIF = HERE / "susceptibility_model2019_500m_UTM.tif"
BURN_SCARS_GPKG = REPO_ROOT / "validation" / "01_data_cleaning" / "data" / "05_burn_scars_validation_ready_UTM.gpkg"
OUT_TIF = HERE / "cicatrices_burnfraction_500m_UTM.tif"

NODATA = -9999.0
OVERSAMPLE = 20  # 500 m / 20 = 25 m sub-cells, matching the ~30 m native scale of the
                 # Landsat-derived SINCHI burn scars (see validation/README.md)

print("=" * 70)
print("STEP 1 -- Loading the susceptibility raster as the grid template (no new grid)")
print("=" * 70)

with rasterio.open(TEMPLATE_TIF) as tmpl:
    tmpl_transform = tmpl.transform
    tmpl_crs = tmpl.crs
    tmpl_height = tmpl.height
    tmpl_width = tmpl.width
    tmpl_data = tmpl.read(1)
    tmpl_nodata = tmpl.nodata

valid_mask = tmpl_data != tmpl_nodata
print(f"Template: {TEMPLATE_TIF.name}")
print(f"  shape=({tmpl_height}, {tmpl_width}), CRS={tmpl_crs}, "
      f"pixel size={tmpl_transform.a} x {abs(tmpl_transform.e)} m")
print(f"  transform={tmpl_transform}")
print(f"  Valid (in-nucleus) cells: {valid_mask.sum():,} / {valid_mask.size:,} "
      f"({100 * valid_mask.mean():.1f}%)")

print()
print("=" * 70)
print("STEP 2 -- Loading burn scars (Phase 1 deliverable)")
print("=" * 70)

burn_scars = gpd.read_file(BURN_SCARS_GPKG)
print(f"Loaded: {len(burn_scars):,} burn-scar polygons, CRS={burn_scars.crs}")
assert burn_scars.crs.to_epsg() == tmpl_crs.to_epsg(), (
    f"CRS mismatch: burn scars are {burn_scars.crs}, template is {tmpl_crs}"
)
print("CRS check PASSED: burn scars already match the template CRS.")

print()
print("=" * 70)
print(f"STEP 3 -- Rasterizing at {OVERSAMPLE}x oversampling "
      f"({tmpl_transform.a / OVERSAMPLE:.1f} m sub-cells)")
print("=" * 70)

fine_height = tmpl_height * OVERSAMPLE
fine_width = tmpl_width * OVERSAMPLE

# Same origin/orientation as the template; pixel size divided by OVERSAMPLE.
fine_transform = Affine(
    tmpl_transform.a / OVERSAMPLE, tmpl_transform.b, tmpl_transform.c,
    tmpl_transform.d, tmpl_transform.e / OVERSAMPLE, tmpl_transform.f,
)

print(f"Fine grid shape: ({fine_height:,}, {fine_width:,}) "
      f"= {fine_height * fine_width:,} sub-cells")

fine_mask = rasterize(
    [(geom, 1) for geom in burn_scars.geometry if geom is not None and not geom.is_empty],
    out_shape=(fine_height, fine_width),
    transform=fine_transform,
    fill=0,
    all_touched=False,  # center-of-subcell rule -- standard for area-fraction estimation
    dtype="uint8",
)
print(f"Fine mask built: {fine_mask.sum():,} burned sub-cells "
      f"({100 * fine_mask.mean():.2f}% of the fine grid)")

print()
print("=" * 70)
print("STEP 4 -- Aggregating sub-cells back to 500 m burn-area fraction")
print("=" * 70)

burn_fraction = (
    fine_mask.reshape(tmpl_height, OVERSAMPLE, tmpl_width, OVERSAMPLE)
    .mean(axis=(1, 3))
    .astype("float32")
)
print(f"Aggregated grid shape: {burn_fraction.shape} (matches template: "
      f"{burn_fraction.shape == (tmpl_height, tmpl_width)})")
print(f"Fraction range (all cells): [{burn_fraction.min():.4f}, {burn_fraction.max():.4f}]")

print()
print("=" * 70)
print("STEP 5 -- Applying the template's NoData mask (outside the nucleus)")
print("=" * 70)

out_data = np.where(valid_mask, burn_fraction, NODATA).astype("float32")

n_zero = int(((out_data == 0.0)).sum())
n_burned = int(((out_data > 0.0) & (out_data != NODATA)).sum())
n_nodata = int((out_data == NODATA).sum())
valid_vals = out_data[valid_mask]
print(f"Cells with NO burn scar overlap (fraction = 0, inside nucleus): {n_zero:,}")
print(f"Cells with SOME burn scar overlap (fraction > 0): {n_burned:,}")
print(f"NoData cells (outside nucleus): {n_nodata:,}")
print(f"Burn-fraction range (valid cells only): "
      f"[{valid_vals.min():.4f}, {valid_vals.max():.4f}], mean={valid_vals.mean():.4f}")

print()
print("=" * 70)
print("STEP 6 -- Verifying exact grid match with the template, then saving")
print("=" * 70)

assert out_data.shape == (tmpl_height, tmpl_width), (
    f"Shape mismatch: {out_data.shape} != {(tmpl_height, tmpl_width)}"
)
print(f"Shape check PASSED: {out_data.shape} == template {(tmpl_height, tmpl_width)}")

out_profile = {
    "driver": "GTiff",
    "height": tmpl_height,
    "width": tmpl_width,
    "count": 1,
    "dtype": "float32",
    "crs": tmpl_crs,
    "transform": tmpl_transform,
    "nodata": NODATA,
    "compress": "deflate",
}
with rasterio.open(OUT_TIF, "w", **out_profile) as dst:
    dst.write(out_data, 1)
    dst.set_band_description(1, "sinchi_burn_scar_area_fraction_2020_2024")

# Re-open and verify shape/transform/CRS are IDENTICAL to the template, not just equal-looking.
with rasterio.open(OUT_TIF) as check, rasterio.open(TEMPLATE_TIF) as tmpl2:
    assert check.shape == tmpl2.shape, f"Shape mismatch on disk: {check.shape} != {tmpl2.shape}"
    assert check.transform == tmpl2.transform, (
        f"Transform mismatch on disk: {check.transform} != {tmpl2.transform}"
    )
    assert check.crs == tmpl2.crs, f"CRS mismatch on disk: {check.crs} != {tmpl2.crs}"
    print("Post-write verification PASSED: shape, transform and CRS are IDENTICAL "
          "to the susceptibility raster template.")

print(f"\nSaved: {OUT_TIF}")
print("\nThis raster (cicatrices_burnfraction_500m_UTM.tif) is pixel-for-pixel aligned")
print("with susceptibility_model2019_500m_UTM.tif and ready for direct cell-by-cell")
print("comparison against the evaluation model's susceptibility probability.")
