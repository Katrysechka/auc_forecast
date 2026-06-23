# catboost-optuna

- **Split protocol:** time-based 80/20 (sort by hour_start with row_index tiebreaker) + 5-fold campaign CV on train (seed=42)
- **Leak-safe:** True
- **Features:** leak-safe 39-feat aggregates
- **Source:** notebook 03_evaluation_and_results
- **Date:** 2026-06-18

## Cross-validation (5 folds)

| Metric | Value |
|---|---|
| SMLAR (flat, mean) | 41.24% |
| SMLAR (flat, std) | 4.95 |
| Monotone violation | 6.95% |

## Holdout (20% time-based)

| Metric | Value |
|---|---|
| SMLAR (flat) | 33.66% |
| Monotone violation | 4.95% |

## Confidence intervals (split_cp, α = 0.1)

| Target | Coverage | Mean width |
|---|---|---|
| at_least_one | 0.882 | 0.1209 |
| at_least_two | 0.893 | 0.0880 |
| at_least_three | 0.896 | 0.0712 |
