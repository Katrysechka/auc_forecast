import numpy as np

from core.conformal import MultiTargetSplitCP, SplitCP


def test_split_cp_coverage_on_iid_synthetic():
    rng = np.random.default_rng(0)
    n = 5000
    y_true = rng.normal(0.5, 0.1, size=n)
    y_pred = y_true + rng.normal(0, 0.05, size=n)
    n_cal = 2000
    cp = SplitCP(alpha=0.10).calibrate(y_true[:n_cal], y_pred[:n_cal])
    lo, hi = cp.predict(y_pred[n_cal:])
    coverage = ((y_true[n_cal:] >= lo) & (y_true[n_cal:] <= hi)).mean()
    assert 0.86 <= coverage <= 0.94, f"coverage={coverage:.3f}"


def test_multi_target_split_cp_shape():
    rng = np.random.default_rng(1)
    y_true = rng.uniform(0, 1, size=(200, 3))
    y_pred = y_true + rng.normal(0, 0.02, size=(200, 3))
    cp = MultiTargetSplitCP(alpha=0.10).calibrate(y_true[:100], y_pred[:100])
    lo, hi = cp.predict(y_pred[100:])
    assert lo.shape == (100, 3) and hi.shape == (100, 3)
    assert np.all(lo <= hi + 1e-12)
    assert np.all(lo >= 0) and np.all(hi <= 1)
