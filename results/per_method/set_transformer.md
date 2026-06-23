# set_transformer-A_full

- **Split protocol:** time-based 80/20 (sort by hour_start) + 5-fold campaign CV (seed=42) — same protocol as core/splits.py
- **Leak-safe:** True
- **Features:** leak-safe set of users + per-user history features
- **Source:** local-cpu eval via src/eval.py
- **Date:** 2026-06-18

## Cross-validation (5 folds)

| Metric | Value |
|---|---|
| SMLAR (flat, mean) | 31.66% |
| SMLAR (flat, std) | 2.40 |
| Monotone violation | 0.00% |

### SMLAR per target

| Target | SMLAR |
|---|---|
| at_least_one | 36.62% |
| at_least_two | 34.68% |
| at_least_three | 24.07% |

## Holdout (20% time-based)

| Metric | Value |
|---|---|
| SMLAR (flat) | 21.21% |
| Monotone violation | 0.00% |

## Confidence intervals (split_cp, α = 0.1)

| Target | Coverage | Mean width |
|---|---|---|
| at_least_one | 0.905 | 0.1189 |
| at_least_two | 0.911 | 0.0796 |
| at_least_three | 0.903 | 0.0589 |
