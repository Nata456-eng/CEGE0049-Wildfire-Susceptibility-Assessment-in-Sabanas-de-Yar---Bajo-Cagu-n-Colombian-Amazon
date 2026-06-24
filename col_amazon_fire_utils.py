import pandas as pd


def _ee():
    import ee
    return ee


NUCLEUS_MUNICIPALITIES = [
    'Cartagena Del Chaira',
    'San Vicente Del Caguan',
    'Solano',
    'La Macarena',
    'San Jose Del Guaviare'
]


def initialize_ee(project='col-amazon-fire-susceptibility'):
    """Initialize Google Earth Engine with the project context."""
    ee = _ee()
    ee.Initialize(project=project)
    return ee


def get_nucleus_geometry():
    """Build and return the dissolved geometry of the 5 target municipalities."""
    ee = _ee()
    admin = ee.FeatureCollection('FAO/GAUL/2015/level2')
    colombia = admin.filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))
    nucleus_fc = colombia.filter(ee.Filter.inList('ADM2_NAME', NUCLEUS_MUNICIPALITIES))
    nucleus_geom = nucleus_fc.geometry().dissolve(maxError=100)
    return nucleus_geom


def get_burned_df(nucleus_geom, start_year=2001, end_year=2025):
    """Compute annual burned area (hectares) inside the nucleus using MODIS MCD64A1."""
    ee = _ee()
    modis_ba = ee.ImageCollection('MODIS/061/MCD64A1').filterBounds(nucleus_geom)

    def burned_area_one_year(year):
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)
        annual = modis_ba.filterDate(start, end).select('BurnDate')
        burned_mask = annual.max().gt(0)
        result = burned_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=nucleus_geom,
            scale=500,
            maxPixels=1e13
        ).get('BurnDate').getInfo()
        return result / 10000 if result is not None else None

    records = []
    for year in range(start_year, end_year + 1):
        print(f'Computing burned area for {year}...')
        records.append({'year': year, 'burned_area_ha': burned_area_one_year(year)})

    return pd.DataFrame(records)


def get_climate_df(nucleus_geom, start_year=2001, end_year=2025):
    """Compute dry-season climate predictors for the nucleus geometry."""
    ee = _ee()

# Builds every dry-season predictor for a single year.
    def dry_season_predictors(year):
        start = ee.Date.fromYMD(year - 1, 12, 1) # Dry season of year Y = December(Y-1) + January(Y) + February(Y)
        end = ee.Date.fromYMD(year, 3, 1)

    # ERA5-Land monthly data, kept to only those three months
        era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(start, end)
    
        #For each monthly image, build the variables we cannot read directly from the dataset
        def add_derived(img):
            u = img.select('u_component_of_wind_10m')
            v = img.select('v_component_of_wind_10m')
            wind = u.hypot(v).rename('wind_speed')  # sqrt(u^2 + v^2)

            t = img.select('temperature_2m').subtract(273.15)   # Kelvin -> Celsius
            td = img.select('dewpoint_temperature_2m').subtract(273.15)

            es = t.expression(
                '0.6108 * exp(17.27 * T / (T + 237.3))',
                {'T': t}
            )
            e = td.expression(
                '0.6108 * exp(17.27 * Td / (Td + 237.3))',
                {'Td': td}
            )

            rh = e.divide(es).multiply(100).rename('rel_humidity')
            vpd = es.subtract(e).rename('vpd')
            return img.addBands([wind, rh, vpd])

        era5 = era5.map(add_derived)

        #Collapse the three monthly values into ONE number per variable
        era5_img = (
            era5.select('temperature_2m').mean()
            .addBands(era5.select('total_precipitation_sum').sum())
            .addBands(era5.select('wind_speed').mean())
            .addBands(era5.select('rel_humidity').mean())
            .addBands(era5.select('vpd').mean())
            .addBands(era5.select('surface_solar_radiation_downwards_sum').mean())
        )
    # Collapse all pixels inside the nucleus into one average value

        era5_stats = era5_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=nucleus_geom,
            scale=10000, # ERA5-Land pixel ~ 9-11 km
            maxPixels=1e13 
        ).getInfo()

        # NDVI comes from a different satellite (MODIS) -> handled separately
        ndvi = (
            ee.ImageCollection('MODIS/061/MOD13A1')
            .filterDate(start, end)
            .select('NDVI')
            .mean()
            .multiply(0.0001) #MOD13A1 stores NDVI x 10000
        )

        ndvi_stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=nucleus_geom,
            scale=500, # MODIS pixel ~ 500 m
            maxPixels=1e13
        ).getInfo()

        # Return one tidy row for this year
        return {
            'year': year,
            'temp_K': era5_stats.get('temperature_2m'),
            'precip_m': era5_stats.get('total_precipitation_sum'),
            'wind_ms': era5_stats.get('wind_speed'),
            'rh_pct': era5_stats.get('rel_humidity'),
            'vpd_kPa': era5_stats.get('vpd'),
            'solar_Jm2': era5_stats.get('surface_solar_radiation_downwards_sum'),
            'ndvi': ndvi_stats.get('NDVI'),
        }

# Run it for every year and collect the rows into one table
    records = []
    for year in range(start_year, end_year + 1):
        print(f'Processing dry season {year}...')
        records.append(dry_season_predictors(year))

# Convert raw units to readable units — temperature and precipitation
    climate_df = pd.DataFrame(records)
    climate_df['temp_C'] = climate_df['temp_K'] - 273.15 # K -> C
    climate_df['precip_mm'] = climate_df['precip_m'] * 1000 # m -> mm
    climate_df['solar_MJ_m2'] = climate_df['solar_Jm2'] / 1e6 # J -> MJ

    return climate_df[['year', 'temp_C', 'precip_mm', 'wind_ms',
                       'rh_pct', 'vpd_kPa', 'solar_MJ_m2', 'ndvi']]
