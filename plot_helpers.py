import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


sns.set_style("whitegrid") 
plt.rcParams['figure.dpi'] = 120


# Charting the burned area and climate predictors
def plot_fire_and_climate(burned_df, climate_df, output_path='fire_and_drivers_panel.png'):
    """Create the fire and dry-season drivers panel chart."""
    df = burned_df.merge(climate_df, on='year')

# ENSO phase by Dec–Feb season (NOAA CPC ONI; year = year of Jan/Feb)
    el_nino = [2003, 2005, 2007, 2010, 2015, 2016, 2019, 2020, 2024]
    la_nina = [2001, 2006, 2008, 2009, 2011, 2012, 2018, 2021, 2022, 2023, 2025]

    def add_enso(ax):
            """Shade El Niño (red) and La Niña (blue) years, mark the 2016 Peace Agreement."""
        for y in el_nino:
            ax.axvspan(y - 0.4, y + 0.4, alpha=0.10, color='red', zorder=0)
        for y in la_nina:
            ax.axvspan(y - 0.4, y + 0.4, alpha=0.10, color='blue', zorder=0)
        ax.axvline(x=2016, color='black', linestyle=':', linewidth=1.2, alpha=0.6)

    predictors = [
        ('temp_C',      'Temperature (°C)',                 '#D97706', 'o'),
        ('precip_mm',   'Precipitation (mm)',               '#1E40AF', 's'),
        ('wind_ms',     'Wind speed (m/s)',                 '#6B21A8', 'v'),
        ('rh_pct',      'Relative humidity (%)',            '#0E7C7B', 'o'),
        ('vpd_kPa',     'Vapour pressure deficit (kPa)',    '#B91C1C', '^'),
        ('solar_MJ_m2', 'Solar radiation (MJ/m², monthly)', '#CA8A04', 's'),
        ('ndvi',        'NDVI (greenness)',                 '#2E7D32', 'D'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    axes = axes.flatten()

    # Panel 0: the target (burned area) as bars
    ax = axes[0]
    ax.bar(df['year'], df['burned_area_ha'] / 1000, color='#C84B31',
           edgecolor='black', linewidth=0.4, alpha=0.85)
    add_enso(ax)
    ax.set_title('Annual burned area (TARGET)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Burned area (thousand ha)', fontsize=9)

    # Panels 1–7: the predictors as lines
    for ax, (col, label, color, marker) in zip(axes[1:], predictors):
        ax.plot(df['year'], df[col], color=color, linewidth=2.2, marker=marker, markersize=5)
        valid = df[['year', col]].dropna()
        if len(valid) > 2:
            z = np.polyfit(valid['year'], valid[col], 1)
            ax.plot(valid['year'], np.poly1d(z)(valid['year']),
                    color=color, linewidth=1.1, linestyle='--', alpha=0.55)
        add_enso(ax)
        r = df['burned_area_ha'].corr(df[col])
        ax.set_title(f'{label}\n(r with burned area = {r:+.2f})', fontsize=10, fontweight='bold')
        ax.set_ylabel(label, fontsize=9)

    for ax in axes[4:]:
        ax.set_xticks(df['year'])
        ax.tick_params(labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('Year', fontsize=9, fontweight='bold')

    fig.legend(
        handles=[
            mpatches.Patch(color='red', alpha=0.25, label='El Niño (Dec–Feb)'),
            mpatches.Patch(color='blue', alpha=0.25, label='La Niña (Dec–Feb)'),
            Line2D([0], [0], color='black', linestyle=':', label='Peace Agreement (2016)')
        ],
        loc='lower center', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        'Fire and dry-season (Dec–Feb) drivers — Sabanas del Yarí–Bajo Caguán núcleo, 2001–2025\n'
        'MODIS MCD64A1 burned area · ERA5-Land climate · MODIS NDVI',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    return fig
