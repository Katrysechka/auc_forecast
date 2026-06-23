# Final comparison — all methods on unified leak-safe protocol

All methods evaluated on the SAME 806 train / 202 holdout campaign split (time-based)
with 5-fold campaign CV on train (seed=42) and Split Conformal Prediction at α=0.10.

| Method | Leak-safe | CV SMLAR % | Holdout SMLAR % | Mono violation % | CP coverage | CP width |
|---|---|---|---|---|---|---|
| catboost-default | ✓ | 47.59 ± 5.49 | 40.88 | 4.72 | 0.896 | 0.1058 |
| catboost-optuna | ✓ | 43.55 ± 5.88 | 34.49 | 6.45 | 0.895 | 0.0952 |
| mlp-mse | ✓ | 67.86 ± 4.00 | 54.14 | 10.43 | 0.916 | 0.1172 |
| mlp-mae | ✓ | 37.80 ± 1.67 | 26.78 | 2.48 | 0.922 | 0.1073 |
| mlp-huber | ✓ | 50.55 ± 2.51 | 39.25 | 6.08 | 0.920 | 0.1059 |
| mlp-msle | ✓ | 30.78 ± 1.44 | 24.27 | 1.49 | 0.902 | 0.0924 |
| mlp-rmsle | ✓ | 42.91 ± 2.69 | 32.54 | 2.23 | 0.926 | 0.1054 |
| mlp-smlar_smooth | ✓ | 29.17 ± 1.99 | 22.63 | 0.00 | 0.908 | 0.0940 |
| monte_carlo+beta | ✓ | 28.24 ± 1.56 | 23.85 | 0.00 | 0.906 | 0.0924 |
| set_transformer-A_full | ✓ | 27.53 ± 2.12 | — | 0.00 | — | — |
| set_transformer-B_no_attn | ✓ | 28.94 ± 2.30 | — | 0.00 | — | — |
| set_transformer-C_no_distr | ✓ | 34.28 ± 10.88 | — | 0.00 | — | — |
| set_transformer-E_none | ✓ | 61.26 ± 3.07 | — | 0.37 | — | — |
| set_transformer-D_no_smlar | ✓ | 81.86 ± 13.88 | — | 0.00 | — | — |
