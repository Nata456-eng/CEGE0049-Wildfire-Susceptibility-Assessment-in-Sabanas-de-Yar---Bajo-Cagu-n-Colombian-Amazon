# Logistic Regression — Baseline Model

**Notebook:** [`baseline_comparison.ipynb`](baseline_comparison.ipynb)
**Component:** Transparent baseline model for the baseline-vs-ML comparison
**Project:** Wildfire susceptibility and socio-ecological exposure — Sabanas del Yarí–Bajo Caguán deforestation nucleus, Colombian Amazon

---

## 1. Purpose

This baseline exists to answer a methodological question posed by the supervisor: *why use machine learning instead of a simpler, transparent method?* The logistic regression is the simple, interpretable reference. Random Forest and XGBoost must beat it under identical data and validation to justify their added complexity. All three models read the **same frozen dataset** ([`data/model_dataset/model_dataset.csv`](../../data/model_dataset/model_dataset.csv)), so any performance difference comes from the model, not the data.

This notebook does **two jobs in sequence** (read the cells top to bottom):
1. **Builds** the frozen dataset from Google Earth Engine (only runs once — if the CSV already exists it is loaded directly, skipping the slow GEE calls).
2. **Trains and evaluates** the logistic regression baseline on that dataset.

---

## 2. Frozen dataset

| Property | Value |
|---|---|
| Design | Pixel-year (one row per site × year) |
| Sites sampled | 8,000 requested → survived NDVI masking |
| Years | 2001–2024 (24 dry seasons, Dec[Y-1]–Feb[Y]) |
| Full pixel-year table | 83,184 rows, 2,077 burned (2.5%) |
| **Final stratified dataset** | **6,231 rows** |
| Presences (burned pixel-years) | 2,077 (100% kept) |
| Pseudo-absences | 4,154 |
| Presence : absence ratio | 1 : 2 |
| Exclusion buffer | 3 km (same-year matching) |
| Events per predictor | 231 (rule: ≥10) |

**Target:** `burned` = 1 if the pixel burned in that dry season (MODIS MCD64A1 BurnDate > 0), else 0.

---

## 3. Predictors (9)

**Anthropogenic — distance rasters (time-invariant, present conditions):**

| Predictor | Source | Search radius | Notes |
|---|---|---|---|
| `dist_roads` | OpenStreetMap (asset `osm_roads_v2`) | 10 km | Replaced an empty official layer (6 → 8,731 segments) |
| `dist_parks` | National Natural Parks | 100 km | Larger radius: avoidance operates broad-scale |
| `dist_coca` | UNODC-SIMCI grid (historical presence, any year 2001–2023) | 10 km | Distance to *ever*-coca cells |
| `dist_mosaic` | MapBiomas Col. 3, group 5, `fastDistanceTransform` | 10 km | Real agropecuario frontier proxy (no pure "pasture" class in nucleus) |

**Climatic — dry-season averages (pixel-varying):**

| Predictor | Source |
|---|---|
| `temp_C` | ERA5-Land 2 m temperature |
| `vpd_kPa` | Derived (Tetens) from T + dewpoint |
| `ndvi` | MODIS MOD13A1 |
| `wind_ms` | ERA5-Land hypot(u, v) |

**Temporal:**

| Predictor | Source |
|---|---|
| `oni` | ENSO Oceanic Niño Index, DJF, NOAA CPC (spatially constant per year) |

**Metadata (NOT predictors):** `lon`, `lat` (spatial blocks), `year` (temporal split).

---

## 4. Diagnostics performed and decisions

**Spearman (predictor redundancy).** Strong pairs (|ρ| > 0.7): `rh_pct ↔ vpd_kPa` = −0.94, `precip_mm ↔ rh_pct` = +0.77, `precip_mm ↔ vpd_kPa` = −0.75. → The humidity block (RH, precip, VPD) measures one dimension.

**VIF (multicollinearity).** Initial 10-predictor set: severe (temp_C 929, rh_pct 612, ndvi 131, vpd_kPa 112, precip 42, wind 27). → **Dropped `rh_pct` and `precip_mm`**, kept `vpd_kPa` as the single "dryness" representative (most fire-relevant). Reduced 8-predictor VIF: all 1.2–2.0 (healthy). Anthropogenic predictors always healthy (2.5–4.1).

**Moran's I (target autocorrelation).** Global I = 0.329, p = 0.001 → fire is significantly spatially clustered → justifies spatial-block validation. Correlogram: I = 0.46 (1 km) → 0.41 (3 km) → 0.35 (7.5 km); elbow ≈ 3 km → **exclusion buffer set to 3 km**.

---

## 5. Sampling strategy

- **Stratified with pseudo-absences:** all presences retained; absences drawn at 2× presences.
- **Rationale for 1:2:** natural rate (~2.5%) makes the model ignore the minority class; 1:1 over-predicts susceptibility; 1:2 is the middle ground, consistent with Barbet-Massin et al. (2012) and recent susceptibility studies (1:1–1:2).
- **Exclusion buffer (3 km):** absences within 3 km of a same-year presence are removed to avoid ambiguous negatives (possible unrecorded fire). Removed 2,229 of 81,107 candidates.
- **Sample-size justification:** not a fraction of the population (449,160 pixels), but statistical sufficiency — 231 events per predictor (Peduzzi et al., 1996: ≥10) and spatial redundancy proven by Moran's I.

---

## 6. Validation design

**Spatial block CV** — `GroupKFold` (5 folds) on 0.25° (~27 km) blocks (~110 blocks). Whole blocks stay in one fold → tests generalisation to *new places*, removes the inflation of a random split.

**Temporal split** — train ≤ 2019 (1,828 events, ~5,484 rows), test ≥ 2020 (249 events, ~747 rows). Tests generalisation to *new years*. Note: 2023 and 2024 have low fire activity (10 events each) — the temporal metric is dominated by 2020–2022; declared as a limitation.

---

## 7. Metrics

| Metric | Role |
|---|---|
| **AUC-ROC** | Primary — ranking ability, robust to imbalance |
| **PR-AUC** (average precision) | Primary — appropriate for rare events |
| **F1** | Contextual only — expected low under imbalance; reported for transparency, not as the decision metric |

Threshold-dependent metrics (F1) are secondary because the product is a **susceptibility ranking**, not a hard classification. Outputs are relative (pseudo-)probabilities, to be presented in susceptibility classes (quintiles), not as calibrated probabilities.

---

## 8. Model configuration

- `LogisticRegression(max_iter=1000, class_weight='balanced')`
- `StandardScaler` fit **on the training fold only** (no leakage), applied to test.
- `lon`, `lat`, `year` excluded from predictors.

---

## 9. Log — chronological record of key decisions

1. **Target = MODIS MCD64A1** burned area, dry-season Dec–Feb window.
2. **Real predictors are distances, not raw layers** — RF/XGBoost are not spatially aware; distance encodes proximity/gradient.
3. **Roads fixed via OpenStreetMap** — official layer had 6 segments in the nucleus; OSM gave 8,731. Median burned-pixel distance dropped from 67 km (broken) to 1.1 km (valid).
4. **Coca = historical presence** (any year 2001–2023), not a single-year snapshot, because the coca footprint persists.
5. **Search radius by feature scale** — 10 km for local effects (roads, coca, mosaic; 95th pct of burned distances ≤5.6 km), 100 km for parks (broad-scale avoidance).
6. **Logistic over linear regression** — binary target; the sigmoid keeps output in [0,1] as a probability.
7. **Climate reduced to 4 predictors** — VIF removed rh_pct and precip_mm (redundant with VPD).
8. **ONI added as the temporal predictor** — continuous index (intensity), not a categorical ENSO label; `year` itself is NOT a predictor (would not generalise to the 2026 snapshot).
9. **Pixel-year spatio-temporal design** — same table serves both spatial-block and temporal validation; averaged climate reserved only for painting the final 2026 snapshot.
10. **Stratified 1:2 with 3 km exclusion buffer** — buffer size from the Moran's I correlogram.
11. **Resolution harmonised to 500 m** (nearest-neighbour resampling in GEE); ERA5-Land retains coarse effective resolution (~9 km). Training at 500 m (matches MODIS target); final map may be painted at 250 m for legibility, with the caveat that fine detail comes from anthropogenic/vegetation predictors.

---

## 10. Declared limitations

- Averaged dry-season climate loses intra-seasonal extremes and interannual timing (the "when" between years); mitigable with extreme aggregates.
- OSM roads are a 2023 snapshot, not a temporal series (declared as accumulated state).
- ONI is spatially constant → aids temporal, not spatial, discrimination.
- Stratification changes the base rate → outputs are relative susceptibility, presented in quintiles.
- 2023–2024 have few fire events → temporal metric is dominated by 2020–2022.

---

## Results (baseline, before tuning)

| Validation | AUC-ROC | PR-AUC | F1 |
|---|---|---|---|
| Spatial block CV | 0.836 ± 0.029 | 0.692 ± 0.101 | 0.666 ± 0.059 |
| Temporal hold-out (≥2020) | 0.808 | 0.620 | 0.521 |

The **tuned** version of this model (hyperparameters chosen via GridSearchCV) is evaluated
in [`tuning/v1/tune_rf.ipynb`](../../tuning/v1/tune_rf.ipynb), together with the tuned
Random Forest, in the same notebook so both use identical code paths.

> **Note:** this baseline uses the frozen 2:1 dataset above. A later sensitivity
> analysis ([`tuning/v3/`](../../tuning/v3/README.md)) found the 1:1 ratio performs
> substantially better, and the final deployed model
> ([`tuning/v4/`](../../tuning/v4/README.md),
> [`outputs/probability_map/`](../../outputs/probability_map/README.md)) uses a
> different, 1:1 ring-sampled (7–15 km) dataset. This README documents the original
> baseline exactly as it was built and evaluated — don't mistake its 2:1 dataset for the
> one behind the final map.
