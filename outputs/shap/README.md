# `outputs/shap/` — Model interpretability (SHAP)

SHAP (SHapley Additive exPlanations) and permutation-importance results, computed on the
tuned Random Forest.

## What is SHAP, simply
Impurity-based importance (the kind already computed in
[`model/random_forest/`](../../model/random_forest/README.md)) has a bias: it favours
continuous variables (like `ndvi` or `vpd_kPa`) over variables that behave more like
distance thresholds (like `dist_mosaic`). SHAP measures, for each individual prediction,
how much each variable "pushed" the fire probability up or down — it's fairer, and is
the current standard in interpretable Machine Learning literature.

## Two SHAP runs — read the right one for the right model
- **`shap_v1.ipynb`** + the top-level `shap_bar.png`, `shap_beeswarm.png`,
  `shap_dependence_dist_mosaic.png`, `shap_dependence_dist_roads.png`,
  `shap_importance_comparison.csv` — SHAP on the **v1-tuned** Random Forest (2:1 ratio,
  3 km exclusion buffer).
- **`shap_v2_rf_v4.ipynb`** + the **`SHAP_V2/`** subfolder — SHAP on the **final v4**
  Random Forest (1:1 ratio, 7–15 km annulus — see
  [`tuning/v4/README.md`](../../tuning/v4/README.md)), rebuilt with the exact same
  recipe (seed 42, hyperparameters from `tuning/v4`) used in
  [`outputs/probability_map/`](../../outputs/probability_map/README.md). **This is the
  current, authoritative run** — it exists specifically to check whether the anthropogenic
  predictors (`dist_roads`, `dist_mosaic`, `dist_coca`), which looked artificially
  suppressed at the old 3 km buffer, come back once that sampling bias is fixed.

## Results (v4 model, `SHAP_V2/`)

| Predictor | SHAP rank | Impurity rank | Permutation rank | SHAP \|mean\| |
|---|---|---|---|---|
| `ndvi` | 1 | 1 | 1 | 0.1431 |
| `wind_ms` | 2 | 2 | 2 | 0.0664 |
| `dist_parks` | 3 | 3 | 3 | 0.0308 |
| `vpd_kPa` | 4 | 4 | 4 | 0.0284 |
| `dist_roads` | 5 | 5 | 5 | 0.0206 |
| `oni` | 6 | 8 | 6 | 0.0177 |
| `dist_coca` | 7 | 7 | 7 | 0.0167 |
| `dist_mosaic` | 8 | 9 | 9 | 0.0162 |
| `temp_C` | 9 | 6 | 8 | 0.0161 |

(full numbers in [`SHAP_V2/shap_importance_comparison.csv`](SHAP_V2/shap_importance_comparison.csv);
plots: [`SHAP_V2/shap_beeswarm.png`](SHAP_V2/shap_beeswarm.png),
[`SHAP_V2/shap_bar.png`](SHAP_V2/shap_bar.png),
[`SHAP_V2/shap_dependence_dist_mosaic.png`](SHAP_V2/shap_dependence_dist_mosaic.png),
[`SHAP_V2/shap_dependence_dist_roads.png`](SHAP_V2/shap_dependence_dist_roads.png))

### What this confirms/refutes from the two hypotheses in `tuning/v1/README.md`

**`oni` (ENSO index):** confirmed as a genuinely mid-table predictor, not a noise
variable — SHAP rank 6 of 9, permutation rank 6 of 9, consistent with the earlier
hypothesis that the linear model's L1 regularisation zeroed its coefficient because the
relationship is real but **non-linear**, not because it's unimportant. Its impurity rank
(8th, near-bottom) is the one outlier metric here — a further sign that impurity
specifically undersells `oni`, exactly the bias SHAP is meant to correct for.

**Anthropogenic predictors resurface once the sampling bias is fixed.** This was the
second, explicit motivation for building v4 (see
[`tuning/v4/README.md`](../../tuning/v4/README.md)): at the old 3 km buffer,
`dist_roads`/`dist_mosaic`/`dist_coca` looked artificially unimportant because
pseudo-absences shared almost the same road/mosaic/coca distances as the fire points
they were paired against. In the v4 model, `dist_roads` and `dist_coca` now rank
**identically** across all three importance methods — `dist_roads` ranks 5th under SHAP,
impurity, and permutation alike; `dist_coca` ranks 7th under all three — a stable,
method-independent signal that wasn't there before. `dist_mosaic` remains the weakest anthropogenic predictor (SHAP #8,
impurity #9, permutation #9), but note it ranks **above** `temp_C` under SHAP while
ranking **below** it under impurity/permutation — mildly consistent with the original
multicollinearity hypothesis (its signal overlaps with `dist_roads`/`dist_coca`, and
SHAP handles correlated features more gracefully than split-based impurity).

## Files
- `SHAP_V2/shap_importance_comparison.csv` — the table above, full precision, plus
  `shap_mean_abs`, `impurity`, `permutation`, and `permutation_std` columns.
- `SHAP_V2/shap_beeswarm.png` — per-prediction SHAP values for every predictor (global
  importance + direction of effect).
- `SHAP_V2/shap_bar.png` — mean |SHAP value| per predictor, the ranking in the table
  above as a bar chart.
- `SHAP_V2/shap_dependence_dist_roads.png`, `shap_dependence_dist_mosaic.png` — how each
  predictor's SHAP contribution changes with its own value (checks for non-linearity/
  thresholds).
- The top-level `shap_*.png`/`.csv` and `shap_v1.ipynb` are the earlier v1-model run,
  kept for reference — don't mix its numbers with the v4 table above, they're different
  models on different datasets.
