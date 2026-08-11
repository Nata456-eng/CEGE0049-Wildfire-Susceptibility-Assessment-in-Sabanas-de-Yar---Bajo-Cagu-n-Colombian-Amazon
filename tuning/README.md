# `tuning/` — Hyperparameter search

Each subfolder (`v1/`, `v2/`, ...) is one round of hyperparameter tuning with
`GridSearchCV`/`RandomizedSearchCV`. Rounds are numbered (rather than overwritten) to
keep a history of what was tried and what it produced — useful for the thesis and for
the defence.

| Version | What was tuned | Result |
|---|---|---|
| [`v1/`](v1/) | Logistic Regression + Random Forest | See [`v1/README.md`](v1/README.md) — final comparison table for both tuned models. |
| [`v2/`](v2/) | Logistic Regression + Random Forest, with the temporal-leakage fix (`GridSearchCV` only sees `year<=2019`) | See [`v2/README.md`](v2/README.md) — same grid as v1, but with tuning correctly isolated from the `>=2020` hold-out. |
| [`v3/`](v3/) | Pseudo-absence ratio sensitivity analysis (1:1, 2:1, 3:1), same pipeline as v2 | See [`v3/README.md`](v3/README.md) — ratio 1:1 clearly beats 2:1 (used through v2) and 3:1. |
| [`v4/`](v4/) | Logistic Regression + Random Forest + XGBoost, rebuilt with a 7–15 km annulus pseudo-absence sampling (a stronger fix for spatial autocorrelation than the 3 km buffer used in v1–v3), ratio 1:1 | See [`v4/README.md`](v4/README.md) — **Random Forest selected as the final model**, later interpreted with SHAP and deployed as the study-nucleus susceptibility map. |

Now that XGBoost is tuned (in `v4/`), any future round should follow the same pattern:
notebook + `README.md` documenting the grid used, the best parameters found, and the
interpretation of the results.
