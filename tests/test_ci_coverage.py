import numpy as np

from core.conformal import SplitCP


def test_split_cp_synthetic_coverage_near_target():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 5))
    theta = np.array([0.2, -0.1, 0.5, 0.3, 0.0])
    y = x @ theta + rng.normal(size=n)

    n_train = 1500
    n_cal = 300

    Xtr, Xcal, Xte = x[:n_train], x[n_train:n_train + n_cal], x[n_train + n_cal:]
    ytr, ycal, yte = y[:n_train], y[n_train:n_train + n_cal], y[n_train + n_cal:]

    beta_hat = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    pred_cal = Xcal @ beta_hat
    pred_te = Xte @ beta_hat

    cp = SplitCP(alpha=0.10).calibrate(ycal, pred_cal)
    lo, hi = cp.predict(pred_te)
    cov = ((yte >= lo) & (yte <= hi)).mean()
    assert 0.86 <= cov <= 0.94, f"Empirical coverage {cov:.3f} outside [0.86, 0.94] band"
