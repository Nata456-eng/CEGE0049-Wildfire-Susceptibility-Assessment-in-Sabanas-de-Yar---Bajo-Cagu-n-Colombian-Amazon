"""
Phase 1, Step 2 -- convert Aboveground Live Woody Biomass Density (AGB, Mg/ha) into
Aboveground Carbon Density (Mg C/ha), by multiplying every valid pixel by 0.5.

Why 0.5: dry woody biomass is conventionally assumed to be ~50% carbon by mass (IPCC
2006 Guidelines for National Greenhouse Gas Inventories, Vol. 4, default carbon fraction
CF = 0.47 (a bit lower than 0.5 is also commonly cited, tree-species-dependent); 0.5 is
the standard simplified default used across most tropical forest carbon studies when a
biome-/species-specific carbon fraction isn't available -- which is the case here, since
the source AGB map doesn't carry species information). This is a fixed multiplicative
factor, not a re-derived model -- every valid pixel is scaled by exactly the same 0.5.

Scope: this is ABOVEGROUND carbon only -- belowground (root) biomass, dead wood, litter,
and soil organic carbon are not included and would need separate data/factors to add.

Re-run: "C:\\Users\\Natal\\.conda\\envs\\fire_thesis\\python.exe" 02_convert_biomass_to_carbon.py
"""

from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).parent

AGB_TIF = HERE / "agb_density_mg_ha_2000_30m.tif"
OUT_TIF = HERE / "carbon_density_mg_c_ha_2000_30m.tif"

CARBON_FRACTION = 0.5
NODATA = -9999.0


def main() -> None:
    with rasterio.open(AGB_TIF) as src:
        agb = src.read(1)
        profile = src.profile.copy()
        src_nodata = src.nodata
        print("AGB source:", src.width, "x", src.height, "| nodata:", src_nodata)

    valid_mask = agb != src_nodata
    carbon = np.full(agb.shape, NODATA, dtype="float32")
    carbon[valid_mask] = agb[valid_mask] * CARBON_FRACTION

    profile.update(dtype="float32", nodata=NODATA)
    with rasterio.open(OUT_TIF, "w", **profile) as out:
        out.write(carbon, 1)
        out.set_band_description(1, "carbon_density_Mg_C_ha")

    valid = carbon[carbon != NODATA]
    print(f"Saved: {OUT_TIF}")
    print(f"Valid pixels: {valid.size:,} ({100*valid.size/carbon.size:.1f}%)")
    print(f"Carbon density range (Mg C/ha): [{valid.min():.1f}, {valid.max():.1f}] | mean: {valid.mean():.1f}")


if __name__ == "__main__":
    main()
