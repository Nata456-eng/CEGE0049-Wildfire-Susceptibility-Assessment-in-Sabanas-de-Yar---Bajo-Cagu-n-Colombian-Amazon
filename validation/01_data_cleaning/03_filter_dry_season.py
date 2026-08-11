"""
Validation, Phase 1, Step 3 -- Filter the clipped burn scars down to the dry-season
months this project's model was actually trained on.

Dry-season definition (must match training exactly): Dec(Y-1) + Jan(Y) + Feb(Y) -- the
same definition used everywhere else in this project (see
`col_amazon_fire_utils.get_climate_df`'s `dry_season_predictors`, and
`model/logistic_regression/README.md`: "24 dry seasons, Dec[Y-1]-Feb[Y]"). Burn scars
recorded in any other month are dropped -- comparing the susceptibility map (fit on
Dec-Jan-Feb fire behavior) against burn scars from, say, August would not be a fair
out-of-sample test.

Builds a numeric month column by mapping the Spanish month names in `mes` to 1-12,
normalizing case/whitespace first (`.str.strip().str.lower()`) so variants like
' Enero', 'ENERO', or 'enero ' all map correctly instead of silently becoming NaN.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 03_filter_dry_season.py
"""

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CLIPPED_IN = DATA_DIR / "02_clipped_to_nucleus.gpkg"
DRY_SEASON_OUT = DATA_DIR / "03_dry_season_filtered.gpkg"

# Spanish month name -> number. Keys are normalized (stripped, lowercased) before lookup.
MONTH_MAP = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9,  # both spellings seen in Spanish-language datasets
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

# Dry season used throughout this project's training: Dec(Y-1) + Jan(Y) + Feb(Y)
DRY_SEASON_MONTHS = [12, 1, 2]

print("=" * 70)
print("STEP 3 -- Filtering to dry-season months (Dec-Jan-Feb, matches model training)")
print("=" * 70)

gdf = gpd.read_file(CLIPPED_IN)
print(f"Loaded clipped burn scars: {len(gdf):,} records, CRS={gdf.crs}")

# Normalize before mapping: strip whitespace, lowercase -- guards against ' Enero',
# 'ENERO', 'enero ', etc. all being treated as distinct/unmapped values.
mes_normalized = gdf["mes"].astype(str).str.strip().str.lower()
gdf["mes_num"] = mes_normalized.map(MONTH_MAP)

unmapped = gdf[gdf["mes_num"].isna()]
if len(unmapped) > 0:
    print(f"WARNING: {len(unmapped):,} records had an unrecognized 'mes' value and "
          f"could not be mapped: {sorted(unmapped['mes'].unique().tolist())}")
else:
    print("All 'mes' values mapped successfully to a month number (0 unmapped).")

print()
print("Record count by month number (1-12), before filtering:")
print(gdf["mes_num"].value_counts().sort_index())

dry_season = gdf[gdf["mes_num"].isin(DRY_SEASON_MONTHS)].copy()

print()
print(f"Records before dry-season filter: {len(gdf):,}")
print(f"Records after dry-season filter (Dec, Jan, Feb only): {len(dry_season):,}")
print(f"Dropped (outside Dec-Jan-Feb): {len(gdf) - len(dry_season):,}")
print()
print("Breakdown of retained records by month:")
print(dry_season["mes_num"].value_counts().sort_index())

dry_season.to_file(DRY_SEASON_OUT, driver="GPKG")
print(f"\nSaved: {DRY_SEASON_OUT}")
