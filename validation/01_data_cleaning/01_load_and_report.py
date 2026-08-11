"""
Validation, Phase 1, Step 1 -- Load the raw SINCHI burn-scar shapefile and report its
baseline shape (record count, CRS, temporal range) BEFORE any cleaning.

This is the verifiable starting point for the whole Phase 1 cleaning process -- every
later filtering step in this phase reports how many records it drops, always relative
to this initial count.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 01_load_and_report.py
"""

from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
RAW_SHP = (
    HERE.parent
    / "Cicatrices_de_quema_por_regiB3n_(HistB3rico)._Escala3A100.000_SINCHI"
    / "Cicatrices_de_quema_por_región_(Histórico)._Escala%3A_1%3A100.000.shp"
)

print("=" * 70)
print("STEP 1 -- Loading raw SINCHI burn-scar shapefile")
print("=" * 70)

gdf = gpd.read_file(RAW_SHP)

print(f"Source file: {RAW_SHP.name}")
print(f"Initial record count: {len(gdf):,}")
print(f"CRS: {gdf.crs}")
print(f"Columns: {list(gdf.columns)}")

print()
print("Temporal range ('periodo' unique values):")
periodos = sorted(gdf["periodo"].dropna().unique(), key=str)
print(periodos)
print(f"  -> {len(periodos)} distinct periods, "
      f"min={min(periodos)}, max={max(periodos)}")
