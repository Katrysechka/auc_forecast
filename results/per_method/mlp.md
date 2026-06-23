# mlp-smlar_smooth

- **Split protocol:** time-based 80/20 (sort by hour_start with row_index tiebreaker) + 5-fold campaign CV on train (seed=42)
- **Leak-safe:** True
- **Features:** leak-safe 39-feat aggregates
- **Source:** notebook 03_evaluation_and_cp
- **Date:** 2026-06-18

## Cross-validation (5 folds)

| Metric | Value |
|---|---|
| SMLAR (flat, mean) | 29.17% |
| SMLAR (flat, std) | 1.99 |
| Monotone violation | 0.00% |

### SMLAR per target

| Target | SMLAR |
|---|---|
| at_least_one | 34.08% |
| at_least_two | 27.44% |
| at_least_three | 26.16% |

## Holdout (20% time-based)

| Metric | Value |
|---|---|
| SMLAR (flat) | 22.63% |
| Monotone violation | 0.00% |

## Confidence intervals (split_cp, α = 0.1)

| Target | Coverage | Mean width |
|---|---|---|
| at_least_one | 0.919 | 0.1428 |
| at_least_two | 0.906 | 0.0765 |
| at_least_three | 0.900 | 0.0627 |
