from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core.config import ALPHA


@dataclass
class SplitCP:
    alpha: float = ALPHA
    quantile_: float | None = None

    def calibrate(self, y_true, y_pred) -> "SplitCP":
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        residuals = np.abs(y_true - y_pred)
        n = residuals.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        self.quantile_ = float(np.quantile(residuals, q_level))
        return self

    def predict(self, y_pred):
        assert self.quantile_ is not None, "Call .calibrate() first"
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return y_pred - self.quantile_, y_pred + self.quantile_


@dataclass
class MultiTargetSplitCP:
    alpha: float = ALPHA
    per_target_: list[SplitCP] | None = None

    def calibrate(self, y_true_2d, y_pred_2d) -> "MultiTargetSplitCP":
        y_true = np.asarray(y_true_2d, dtype=np.float64)
        y_pred = np.asarray(y_pred_2d, dtype=np.float64)
        assert y_true.shape == y_pred.shape
        self.per_target_ = [
            SplitCP(alpha=self.alpha).calibrate(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
        return self

    def predict(self, y_pred_2d):
        assert self.per_target_ is not None
        y_pred = np.asarray(y_pred_2d, dtype=np.float64)
        lo = np.empty_like(y_pred)
        hi = np.empty_like(y_pred)
        for i, cp in enumerate(self.per_target_):
            lo[:, i], hi[:, i] = cp.predict(y_pred[:, i])
        return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
