import numpy as np
import pytest

from core.config import EPS
from core.metrics import (
    coverage_at_alpha,
    monotonicity_violation,
    smlar,
    smlar_per_target,
)


def test_smlar_perfect_prediction_is_zero():
    y = np.array([0.1, 0.5, 0.9])
    assert smlar(y, y) == pytest.approx(0.0, abs=1e-10)


def test_smlar_eps_matches_spec():
    assert EPS == 0.005


def test_smlar_symmetric_in_log_space():
    y_true = np.array([0.2, 0.3])
    y_high = np.array([0.4, 0.6]) 
    y_low = np.array([0.1, 0.15]) 
    s_high = smlar(y_true, y_high)
    s_low = smlar(y_true, y_low)
    assert s_high == pytest.approx(s_low, rel=0.05)


def test_smlar_known_value():
    y_true = np.array([0.4])
    y_pred = np.array([0.6])
    expected_log_ratio = float(np.log((0.6 + EPS) / (0.4 + EPS)))
    expected = 100.0 * (np.exp(abs(expected_log_ratio)) - 1.0)
    assert smlar(y_true, y_pred) == pytest.approx(expected)


def test_smlar_handles_zeros():
    y_true = np.array([0.0, 0.5])
    y_pred = np.array([0.0, 0.5])
    assert smlar(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_smlar_per_target_keys():
    y_true = np.array([[0.5, 0.3, 0.1], [0.6, 0.4, 0.2]])
    y_pred = np.array([[0.5, 0.3, 0.1], [0.7, 0.5, 0.3]])
    res = smlar_per_target(y_true, y_pred, ["a", "b", "c"])
    assert set(res.keys()) == {"a", "b", "c"}
    assert res["a"] > 0  # first target has error
    assert all(v >= 0 for v in res.values())


def test_monotonicity_violation():
    # 1st row violates y1 >= y2; 2nd row OK; 3rd row violates y2 >= y3.
    y = np.array([[0.3, 0.5, 0.1], [0.9, 0.5, 0.2], [0.9, 0.3, 0.5]])
    assert monotonicity_violation(y) == pytest.approx(2 / 3)


def test_monotonicity_clean_data():
    y = np.array([[0.9, 0.5, 0.2], [0.8, 0.7, 0.6]])
    assert monotonicity_violation(y) == 0.0


def test_coverage_at_alpha():
    y_true = np.array([0.5, 0.5, 0.5, 0.5])
    lo = np.array([0.4, 0.4, 0.6, 0.4])  # 3rd is above the truth
    hi = np.array([0.6, 0.6, 0.7, 0.45])  # 4th is below the truth
    assert coverage_at_alpha(y_true, lo, hi) == pytest.approx(0.5)
