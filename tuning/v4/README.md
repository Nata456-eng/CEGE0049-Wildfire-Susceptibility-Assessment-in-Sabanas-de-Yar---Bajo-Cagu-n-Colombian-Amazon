# Tuning — v4 (7–15 km annulus, ratio 1:1, three-model comparison)

**Notebook:** [`v4_comparison_buffer_7_15km.ipynb`](v4_comparison_buffer_7_15km.ipynb)
**Outputs:** [`v4_buffer_7_15km/`](v4_buffer_7_15km/) (metrics, hyperparameters, run record) and [`graphs/`](graphs/) (comparison charts)

## Why v4 exists
`v1`–`v3` all drew pseudo-absences from a simple **3 km exclusion buffer** ("anywhere
beyond 3 km from a same-year fire pixel"). Two diagnostics from that design motivated a
re-sampling:

1. **The Moran's I correlogram never flattened inside 3 km** — global Moran's I on the
   target was +0.329 (p=0.001), and by distance band:

   | Band | 1 km | 2 km | 3 km | 5 km | 7.5 km | 10 km | 15 km |
   |---|---|---|---|---|---|---|---|
   | Moran's I | +0.460 | +0.459 | +0.407 | +0.419 | +0.354 | +0.332 | +0.324 |

   At the 3 km buffer used through v3, residual spatial dependence was still strong
   (I ≈ 0.41) — pseudo-absences drawn just outside that ring were still inside the
   presences' zone of autocorrelation, inflating spatial-CV performance.
2. **Static human-pressure predictors under-performed in SHAP** — `dist_roads`,
   `dist_mosaic`, and `dist_coca` ranked below fuel/fire-weather variables, contrary to
   the literature. A 3 km buffer draws non-fire points from the same near-field envelope
   as fire points, so both classes end up with near-identical distances to roads/mosaic,
   leaving the model nothing to separate on.

## The v4 sampling design
Three deliberate changes from v1–v3, everything else (base table `pixel_year_full.csv`,
seed `42`, per-year exclusion criterion, unstratified global absence draw among eligible
rows, `0.25°` spatial blocks) held identical so the buffer geometry is the only real
variable:

- **7–15 km annulus** instead of a 3 km buffer — pseudo-absences are drawn from a ring at
  least 7 km and at most 15 km from any same-year fire point. The 7 km inner radius
  clears the residual autocorrelation zone found above; the 15 km outer radius keeps
  absences in the same broad environmental/socio-political setting as the presences
  (otherwise the model starts separating *regions* rather than *fire conditions*).
- **Ratio kept at 1:1** — per Barbet-Massin et al. (2012), classification/ML methods
  (Random Forest, boosted trees) perform best with balanced presence:pseudo-absence
  samples (regression-based methods are the ones that benefit from a larger absence
  pool). 1:1 also keeps PR-AUC's no-skill baseline fixed at 0.50, making it directly
  comparable across the three models.
- **Proportional year-stratification** — absences are sampled proportionally to the
  annual distribution of presences (peak-fire years like 2004/2007/2018 get
  proportionally more absences, quiet years like 2023/2024 get fewer). Several
  predictors (`oni`, `temp_C`, `vpd_kPa`) are constant within a year — without
  stratification the model could separate classes by *year* (and hence by that year's
  climate) instead of by genuine fire conditions.

Dataset: **4,154 rows, 2,077 presences / 2,077 absences**, drawn from 14,928 ring-eligible
candidates, across 84 spatial blocks. Year-balance check confirms presence and absence
shares match exactly (`abs_diff = 0.0`) in every year. SHAP and other interpretability
analysis were deliberately left out of this notebook — it establishes *which model on
which dataset*; interpretation happens afterward in
[`outputs/shap/`](../../outputs/shap/README.md).

## Method
Best hyperparameters for all three models were reused as-is from their respective
earlier tuning rounds (v1/v2 for LR/RF, the XGBoost notebooks for XGBoost) rather than
re-searched — this notebook's purpose is a clean three-way comparison on the improved
sampling geometry, not a new hyperparameter search. Evaluation: 5-fold spatial-block CV
(`0.25°` blocks) + temporal hold-out (train ≤2019, test ≥2020), same protocol as every
other version.

| Model | Best hyperparameters |
|---|---|
| Logistic Regression | `C=0.01`, `penalty='l2'` |
| Random Forest | `n_estimators=500`, `max_features=0.5`, `min_samples_leaf=5`, `max_depth=None` |
| XGBoost | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=1.0` |

## Results

| Model | AUC spatial | PR-AUC spatial | F1 spatial | AUC temporal | PR-AUC temporal | F1 temporal |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.751 ± 0.031 | 0.737 ± 0.034 | 0.669 ± 0.068 | 0.662 | 0.709 | 0.628 |
| **Random Forest** | **0.783 ± 0.032** | **0.774 ± 0.036** | **0.706 ± 0.058** | 0.675 | 0.692 | 0.636 |
| XGBoost | 0.773 ± 0.043 | 0.763 ± 0.051 | 0.688 ± 0.055 | 0.663 | 0.659 | 0.647 |

(full table in [`v4_buffer_7_15km/v4_results_table.md`](v4_buffer_7_15km/v4_results_table.md) and
[`v4_buffer_7_15km/v4_model_comparison_metrics.csv`](v4_buffer_7_15km/v4_model_comparison_metrics.csv);
charts in [`graphs/`](graphs/))

As expected, every spatial score dropped relative to v3's 3 km-buffer numbers — that's
the point of v4, not a regression: the v3 numbers were partly inflated by pseudo-absences
drawn from inside the presences' autocorrelation zone. Random Forest wins on both spatial
AUC and spatial PR-AUC, and is competitive on every temporal metric too (XGBoost edges it
narrowly on temporal PR-AUC and F1, by 0.003–0.011 — within fold-to-fold noise).

## Random Forest selected as the final model
Random Forest is the model deployed in [`outputs/probability_map/`](../../outputs/probability_map/README.md)
and interpreted in [`outputs/shap/`](../../outputs/shap/README.md): it's the strongest or
joint-strongest model on every metric reported here, and the most established choice in
the fire-susceptibility literature for this kind of tabular, mixed-predictor problem.

## A discrepancy worth flagging
[`tuning/tuning_comparison.ipynb`](../tuning_comparison.ipynb) — a later notebook that
consolidates every version's metrics into one table — has a **different** hardcoded set
of numbers for v4 (e.g. RF spatial PR-AUC 0.800 and temporal PR-AUC 0.710, vs. 0.774 and
0.692 here). That notebook's own header states its numbers are manually transcribed from
each version's executed output, "not a re-run" — which means it can silently drift out of
sync if a version gets re-run later with a fix. This README uses the numbers in
`v4_buffer_7_15km/v4_results_table.md` and `v4_model_comparison_metrics.csv` directly (the
actual current output files sitting next to this v4 notebook), since those are the most
primary, current source. Reconciling `tuning_comparison.ipynb`'s hardcoded table — and its
downstream narrative, which was written around the older numbers — is a substantive
analysis decision, not a documentation cleanup, so it wasn't done here; flagging it for a
deliberate look rather than silently rewriting your analysis notebook.
