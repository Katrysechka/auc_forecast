# monte_carlo+beta

- **Split protocol:** time-based 80/20 (sort by hour_start with row_index tiebreaker) + 5-fold campaign CV on train (seed=42)
- **Leak-safe:** True
- **Features:** leak-safe 39-feat aggregates + per-user history rates
- **Source:** notebook 03_holdout_cp_credible, S=500, alpha_disp=2.21
- **Date:** 2026-06-18

## Cross-validation (5 folds)

| Metric | Value |
|---|---|
| SMLAR (flat, mean) | 28.24% |
| SMLAR (flat, std) | 1.56 |
| Monotone violation | 0.00% |

### SMLAR per target

| Target | SMLAR |
|---|---|
| at_least_one | 29.67% |
| at_least_two | 28.86% |
| at_least_three | 26.26% |

## Holdout (20% time-based)

| Metric | Value |
|---|---|
| SMLAR (flat) | 23.85% |
| Monotone violation | 0.00% |

## Confidence intervals (split_cp, α = 0.1)

| Target | Coverage | Mean width |
|---|---|---|
| at_least_one | 0.904 | 0.1085 |
| at_least_two | 0.913 | 0.0983 |
| at_least_three | 0.901 | 0.0704 |
