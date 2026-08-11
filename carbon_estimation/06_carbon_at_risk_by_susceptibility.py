"""
Phase 5, Step 6 -- Carbon at risk by fire susceptibility.

Combines the continuous (unclassified) fire-susceptibility surface with the per-pixel
carbon stock raster to estimate how much forest carbon sits in high-susceptibility
zones. Both inputs were already verified to be perfectly aligned (EPSG:32618, 30 m,
identical transform/dimensions) by `05_reproject_to_utm_and_stock.py`, so this script
skips any alignment check and reads the two rasters directly, band-for-band.

CRITICAL -- carbon is already a per-pixel STOCK, not a density:
`carbon_stock_per_pixel_2024_30m_UTM.tif` holds Mg C already contained in each 30 m x 30 m
(900 m^2 = 0.09 ha) cell -- the density (Mg C/ha) -> stock (Mg C/pixel) conversion was
already done upstream. That means summing carbon-at-risk pixel values is a DIRECT sum in
Mg C. The 0.09 ha/pixel constant below is used ONLY to turn pixel counts into hectares
(for sanity-check densities and the class-distribution table) -- it must NEVER be
multiplied into the carbon-at-risk sum itself. Doing so would apply the density->stock
conversion a second time and divide the true total by ~11.1 (1 / 0.09).

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 06_carbon_at_risk_by_susceptibility.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ---------------------------------------------------------------------------
# Step 1 -- Analysis parameters
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
BUNDLE_DIR = HERE / "resampled_ArcGIS_30m"

SUSCEPTIBILITY_PATH = BUNDLE_DIR / "fire_susceptibility_probability_map_v4_resampled_30m_UTM.tif"
CARBON_STOCK_PATH = BUNDLE_DIR / "carbon_stock_per_pixel_2024_30m_UTM.tif"

CSV_OUT = HERE / "carbon_distribution_by_susceptibility_class.csv"
CARBON_AT_RISK_TIF_OUT = BUNDLE_DIR / "carbon_at_risk_UTM.tif"

# Jenks natural breaks obtained previously in ArcGIS (0-1 probability scale)
JENKS_BREAKS = [0.117, 0.25, 0.408, 0.612]

HIGH_RISK_THRESHOLD = 0.408   # high + very high classes
SENSITIVITY_THRESHOLD = 0.612  # very high class only

PIXEL_AREA_HA = 0.09  # 30 m x 30 m = 900 m^2 = 0.09 ha
                       # USED ONLY for area/ha and mean-density math below --
                       # NEVER to scale the carbon-at-risk sum itself.

# Plausible mean carbon-density range for Amazon forest (Mg C / ha), for the sanity check
PLAUSIBLE_DENSITY_MIN = 20.0
PLAUSIBLE_DENSITY_MAX = 250.0

print("=" * 70)
print("STEP 1 -- Loading aligned rasters as masked arrays (NoData respected)")
print("=" * 70)

with rasterio.open(SUSCEPTIBILITY_PATH) as src:
    susceptibility = src.read(1, masked=True)  # continuous, 0-1
    profile = src.profile.copy()
    susceptibility_nodata = src.nodata

with rasterio.open(CARBON_STOCK_PATH) as src:
    carbon_stock = src.read(1, masked=True)  # Mg C per 30 m pixel (already a stock)
    carbon_nodata = src.nodata

print(f"Susceptibility raster: {SUSCEPTIBILITY_PATH.name}")
print(f"  shape={susceptibility.shape}, nodata={susceptibility_nodata}, "
      f"valid pixels={susceptibility.count()}")
print(f"Carbon stock raster:   {CARBON_STOCK_PATH.name}")
print(f"  shape={carbon_stock.shape}, nodata={carbon_nodata}, "
      f"valid pixels={carbon_stock.count()}")
print(f"Jenks breaks: {JENKS_BREAKS}")
print(f"High-risk threshold (>= ): {HIGH_RISK_THRESHOLD}")
print(f"Sensitivity threshold (>=): {SENSITIVITY_THRESHOLD}")
print(f"Pixel area: {PIXEL_AREA_HA} ha (used for area/density math only, never for the carbon sum)")

# ---------------------------------------------------------------------------
# Step 2 -- Reclassify continuous susceptibility into 5 Jenks classes
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 2 -- Reclassifying continuous susceptibility into 5 Jenks classes")
print("=" * 70)

CLASS_LABELS = ["very_low", "low", "moderate", "high", "very_high"]

# Start from the raw (masked) array so the class raster inherits the same mask
susceptibility_class = np.ma.masked_array(
    np.zeros(susceptibility.shape, dtype=np.uint8),
    mask=np.ma.getmaskarray(susceptibility).copy(),
)

b0, b1, b2, b3 = JENKS_BREAKS
valid = ~np.ma.getmaskarray(susceptibility)
vals = susceptibility.data

susceptibility_class.data[valid & (vals < b0)] = 0                       # very low
susceptibility_class.data[valid & (vals >= b0) & (vals < b1)] = 1        # low
susceptibility_class.data[valid & (vals >= b1) & (vals < b2)] = 2        # moderate
susceptibility_class.data[valid & (vals >= b2) & (vals < b3)] = 3        # high
susceptibility_class.data[valid & (vals >= b3)] = 4                      # very high

print("Pixel counts per class (verification):")
for code, label in enumerate(CLASS_LABELS):
    n = int(np.sum(valid & (susceptibility_class.data == code)))
    print(f"  [{code}] {label:10s}: {n:>12,d} pixels")
print(f"  {'NoData':10s}: {int(np.sum(~valid)):>12,d} pixels")

# The continuous `susceptibility` array (not the class labels) is what all the
# risk masks below are built from -- cleaner than relying on the integer class codes.

# ---------------------------------------------------------------------------
# Step 3 -- High-risk zone extraction and cross with carbon
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3 -- Extracting high-risk zone and crossing with carbon")
print("=" * 70)

susceptibility_valid = ~np.ma.getmaskarray(susceptibility)
carbon_valid = ~np.ma.getmaskarray(carbon_stock)

high_risk_mask = susceptibility_valid & (susceptibility.data >= HIGH_RISK_THRESHOLD)

# Carbon-at-risk: keep carbon value only where BOTH high-risk and carbon-valid;
# everywhere else (low/moderate risk, or NoData in either layer) = 0.
carbon_at_risk = np.where(
    high_risk_mask & carbon_valid,
    carbon_stock.filled(0.0),
    0.0,
).astype("float32")

print(f"High-risk pixels (susceptibility >= {HIGH_RISK_THRESHOLD}): {int(high_risk_mask.sum()):,}")
print(f"High-risk pixels with valid carbon data: {int((high_risk_mask & carbon_valid).sum()):,}")

# ---------------------------------------------------------------------------
# Step 4 -- Total carbon at risk (DIRECT sum, no area multiplication)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 4 -- Total carbon at risk (direct sum -- already per-pixel stock)")
print("=" * 70)

total_carbon_at_risk_mgc = float(carbon_at_risk.sum())  # direct sum, NOT x pixel area
total_carbon_at_risk_million_mgc = total_carbon_at_risk_mgc / 1e6

print(f"Total carbon at risk (high + very high): {total_carbon_at_risk_mgc:,.1f} Mg C")
print(f"                                        = {total_carbon_at_risk_million_mgc:,.3f} million Mg C")

# ---------------------------------------------------------------------------
# Step 5 -- Sanity check: implied mean carbon density in the high-risk zone
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5 -- Sanity check")
print("=" * 70)

high_risk_pixel_count = int(high_risk_mask.sum())
high_risk_area_ha = high_risk_pixel_count * PIXEL_AREA_HA
implied_mean_density_mgc_ha = (
    total_carbon_at_risk_mgc / high_risk_area_ha if high_risk_area_ha > 0 else float("nan")
)

print(f"High-risk pixel count:            {high_risk_pixel_count:,}")
print(f"High-risk area:                   {high_risk_area_ha:,.1f} ha")
print(f"Implied mean carbon density:       {implied_mean_density_mgc_ha:,.2f} Mg C / ha")

if not (PLAUSIBLE_DENSITY_MIN <= implied_mean_density_mgc_ha <= PLAUSIBLE_DENSITY_MAX):
    print(
        f"WARNING: implied mean density ({implied_mean_density_mgc_ha:,.2f} Mg C/ha) falls "
        f"OUTSIDE the plausible range for Amazon forest "
        f"({PLAUSIBLE_DENSITY_MIN}-{PLAUSIBLE_DENSITY_MAX} Mg C/ha). "
        f"Something may be mis-scaled (e.g. an accidental extra area multiplication/division) "
        f"-- review before reporting this number."
    )
else:
    print(
        f"OK: implied mean density is within the plausible range "
        f"({PLAUSIBLE_DENSITY_MIN}-{PLAUSIBLE_DENSITY_MAX} Mg C/ha) for Amazon forest."
    )

# ---------------------------------------------------------------------------
# Step 6 -- Threshold sensitivity analysis (high+very high vs. very high only)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 6 -- Threshold sensitivity analysis")
print("=" * 70)

sensitivity_mask = susceptibility_valid & (susceptibility.data >= SENSITIVITY_THRESHOLD)
carbon_at_risk_sensitivity = np.where(
    sensitivity_mask & carbon_valid,
    carbon_stock.filled(0.0),
    0.0,
).astype("float32")
total_carbon_sensitivity_mgc = float(carbon_at_risk_sensitivity.sum())

pct_captured_by_very_high = (
    100.0 * total_carbon_sensitivity_mgc / total_carbon_at_risk_mgc
    if total_carbon_at_risk_mgc > 0 else float("nan")
)

print(f"Carbon at risk, threshold >= {HIGH_RISK_THRESHOLD} (high + very high): "
      f"{total_carbon_at_risk_mgc:,.1f} Mg C ({total_carbon_at_risk_million_mgc:,.3f} million Mg C)")
print(f"Carbon at risk, threshold >= {SENSITIVITY_THRESHOLD} (very high only):  "
      f"{total_carbon_sensitivity_mgc:,.1f} Mg C ({total_carbon_sensitivity_mgc / 1e6:,.3f} million Mg C)")
print(f"Percent of high-risk carbon captured by the very-high-only threshold: "
      f"{pct_captured_by_very_high:,.1f}%")
print(
    "This comparison shows how sensitive the carbon-at-risk estimate is to exactly "
    "where the high-risk boundary is drawn (0.408 vs. 0.612)."
)

# ---------------------------------------------------------------------------
# Step 7 -- Carbon distribution table across the 5 Jenks classes
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 7 -- Carbon distribution by susceptibility class")
print("=" * 70)

carbon_filled = carbon_stock.filled(0.0)
carbon_valid_data = np.where(carbon_valid, carbon_filled, 0.0)

total_carbon_all_mgc = float(carbon_valid_data.sum())  # total valid carbon, any class

rows = []
for code, label in enumerate(CLASS_LABELS):
    class_mask = valid & (susceptibility_class.data == code)
    n_pixels = int(class_mask.sum())
    area_ha = n_pixels * PIXEL_AREA_HA

    class_carbon_mgc = float(carbon_valid_data[class_mask & carbon_valid].sum())
    class_carbon_million_mgc = class_carbon_mgc / 1e6

    mean_density_mgc_ha = class_carbon_mgc / area_ha if area_ha > 0 else np.nan
    pct_of_total_carbon = (
        100.0 * class_carbon_mgc / total_carbon_all_mgc if total_carbon_all_mgc > 0 else np.nan
    )

    rows.append({
        "class": label,
        "n_pixels": n_pixels,
        "area_ha": area_ha,
        "carbon_mgc": class_carbon_mgc,
        "carbon_million_mgc": class_carbon_million_mgc,
        "mean_density_mgc_ha": mean_density_mgc_ha,
        "pct_of_total_carbon": pct_of_total_carbon,
    })

distribution_df = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print(distribution_df.to_string(index=False))

distribution_df.to_csv(CSV_OUT, index=False)
print(f"\nSaved: {CSV_OUT}")

# ---------------------------------------------------------------------------
# Step 8 -- Export carbon-at-risk raster (high + very high threshold)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 8 -- Exporting carbon-at-risk GeoTIFF")
print("=" * 70)

out_profile = profile.copy()
out_profile.update({
    "dtype": "float32",
    "count": 1,
    "nodata": 0.0,
    "compress": "deflate",
})

with rasterio.open(CARBON_AT_RISK_TIF_OUT, "w", **out_profile) as dst:
    dst.write(carbon_at_risk, 1)

print(f"Saved: {CARBON_AT_RISK_TIF_OUT}")
print(f"  CRS: {out_profile['crs']}, resolution: 30 m, NoData: 0.0")
print("Done.")
