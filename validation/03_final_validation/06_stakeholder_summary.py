"""
Validation, Phase 3, Step 6 -- Plain-language stakeholder summary table (IDEAM, UNGRD,
local communities), built from the same Step 3-5 rasters, but expressed WITHOUT
statistical jargon ("capture fraction", "enrichment ratio", "out-of-sample", etc. only
appear in these code comments, never in the printed/saved table).

Column meanings (stakeholder-facing names -> technical equivalent, for this script's
internal bookkeeping only):
  - "Risk level"              -> Jenks susceptibility class
  - "% of territory"          -> class area / total common-valid area
  - "% of total fire"         -> the capture fraction from Step 5, as a percent
  - "Burn intensity (%)"      -> burn rate = burned area in class / class area
                                ("out of every 100 ha in this zone, how many burned")

Ordered from Very high -> Very low (most decision-relevant risk class first, which is
the natural reading order for a decision-maker scanning top-to-bottom for "where should
we focus"), rather than the low-to-high gradient order used in the technical Step 5 table.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 06_stakeholder_summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

CLASS_TIF = DATA_DIR / "susceptibility_class_common.tif"
BURNED_AREA_TIF = DATA_DIR / "burned_area_observed_ha.tif"
COMMON_MASK_TIF = DATA_DIR / "common_valid_mask.tif"
RESULT_CSV = HERE / "validation_stakeholder_summary.csv"

CELL_AREA_HA = 25.0

# code -> (technical label, plain English label), in Very high -> Very low display order
CLASS_ORDER = [
    (4, "very_high", "Very high"),
    (3, "high", "High"),
    (2, "moderate", "Moderate"),
    (1, "low", "Low"),
    (0, "very_low", "Very low"),
]

FOOTNOTE = (
    "This validation compares the map against real fires observed independently "
    "(Landsat satellite, SINCHI historical data) that occurred between 2020 and 2024 "
    "-- years the model NEVER saw during training. It's a fair test: the model had "
    "to anticipate fire that hadn't happened yet when it was built."
)

print("=" * 70)
print("STEP 1 -- Loading the classified raster, burned-area raster, and common mask")
print("=" * 70)

with rasterio.open(CLASS_TIF) as src:
    class_codes = src.read(1)

with rasterio.open(BURNED_AREA_TIF) as src:
    burned_area_ha = src.read(1)

with rasterio.open(COMMON_MASK_TIF) as src:
    common_valid = src.read(1).astype(bool)

n_common = int(common_valid.sum())
total_area_ha = n_common * CELL_AREA_HA
total_burned_ha = float(burned_area_ha[common_valid].sum())

print(f"Total territory considered: {total_area_ha:,.2f} ha")
print(f"Total fire observed (2020-2024, independent SINCHI/Landsat source): "
      f"{total_burned_ha:,.2f} ha")

print()
print("=" * 70)
print("STEP 2 -- Building the plain-language table (full precision, rounded at the end)")
print("=" * 70)

rows = []
for code, tech_label, es_label in CLASS_ORDER:
    class_mask = (class_codes == code) & common_valid
    class_area_ha = float(class_mask.sum()) * CELL_AREA_HA
    burned_in_class_ha = float(burned_area_ha[class_mask].sum())

    pct_territorio = 100 * class_area_ha / total_area_ha
    pct_fuego = 100 * burned_in_class_ha / total_burned_ha
    intensidad = 100 * burned_in_class_ha / class_area_ha  # burn rate within the class

    rows.append({
        "Risk level": es_label,
        "% of territory": round(pct_territorio, 1),
        "% of total fire": round(pct_fuego, 1),
        "Burn intensity (%)": round(intensidad, 1),
        "_tech_label": tech_label,  # kept only for the ratio calc below, not saved to CSV
        "_intensidad_raw": intensidad,
    })

table = pd.DataFrame(rows)

print()
print(table[["Risk level", "% of territory", "% of total fire",
             "Burn intensity (%)"]].to_string(index=False))

print()
print("=" * 70)
print("STEP 3 -- Verifying the fire percentages sum to 100%")
print("=" * 70)

sum_fuego = table["% of total fire"].sum()
sum_territorio = table["% of territory"].sum()
print(f"Sum of '% of total fire' (all classes): {sum_fuego:.1f}% (expected ~100.0%)")
print(f"Sum of '% of territory' (all classes): {sum_territorio:.1f}% (expected ~100.0%)")

print()
print("=" * 70)
print("STEP 4 -- Headline figures for stakeholders")
print("=" * 70)

alto_muyalto = table[table["_tech_label"].isin(["high", "very_high"])]
pct_fuego_alto_muyalto = round(alto_muyalto["% of total fire"].sum(), 1)
pct_territorio_alto_muyalto = round(alto_muyalto["% of territory"].sum(), 1)

frase_titular = (
    f"Out of every 100 hectares burned, {pct_fuego_alto_muyalto} were in areas marked "
    f"as high or very high risk, which occupy only {pct_territorio_alto_muyalto}% of "
    f"the territory."
)
print(frase_titular)

intensidad_muy_alto = table.loc[table["_tech_label"] == "very_high", "_intensidad_raw"].iloc[0]
intensidad_muy_bajo = table.loc[table["_tech_label"] == "very_low", "_intensidad_raw"].iloc[0]
ratio_intensidad = intensidad_muy_alto / intensidad_muy_bajo

frase_contraste = (
    f"Fire was {ratio_intensidad:.0f} times more intense in very-high-risk areas than "
    f"in very-low-risk areas."
)
print(frase_contraste)

print()
print("Footnote:")
print(FOOTNOTE)

print()
print("=" * 70)
print("STEP 5 -- Saving the stakeholder table (plain-language columns only)")
print("=" * 70)

table_out = table[["Risk level", "% of territory", "% of total fire",
                    "Burn intensity (%)"]].copy()
table_out.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

with open(RESULT_CSV, "a", encoding="utf-8-sig") as f:
    f.write(f"\n# Total % of total fire (all classes): {sum_fuego:.1f}%\n")
    f.write(f"# {frase_titular}\n")
    f.write(f"# {frase_contraste}\n")
    f.write(f"# {FOOTNOTE}\n")

print(f"Saved: {RESULT_CSV}")
