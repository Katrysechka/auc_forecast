from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
import numpy as np
from core.conformal import MultiTargetSplitCP
from core.config import ALPHA, INNER_CALIB_FRAC, SEED


class FitPredict(Protocol):
    def fit(self, X, y) -> Any: ...
    def predict(self, X) -> Any: ...


def seeded_train_calib_split(
    n: int, calib_frac: float = INNER_CALIB_FRAC, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_calib = int(round(n * calib_frac))
    calib = np.sort(idx[:n_calib])
    train_sub = np.sort(idx[n_calib:])
    return train_sub, calib


@dataclass
class ConformalEvaluation:
    y_true: np.ndarray
    y_pred: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantiles_per_target: list[float]

    @property
    def coverage_per_target(self) -> dict[str, float]:
        from core.config import TARGETS

        out: dict[str, float] = {}
        for i, t in enumerate(TARGETS):
            hit = (self.y_true[:, i] >= self.lower[:, i]) & (self.y_true[:, i] <= self.upper[:, i])
            out[t] = float(hit.mean())
        return out

    @property
    def width_per_target(self) -> dict[str, float]:
        from core.config import TARGETS

        return {t: float((self.upper[:, i] - self.lower[:, i]).mean()) for i, t in enumerate(TARGETS)}


@dataclass
class ConformalRunner:
    factory: Callable[[], FitPredict]
    alpha: float = ALPHA
    inner_calib_frac: float = INNER_CALIB_FRAC
    inner_seed: int = SEED
    predict_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None
    evaluations: list[ConformalEvaluation] = field(default_factory=list)

    def _predict(self, model, X):
        if self.predict_fn is not None:
            return self.predict_fn(model, X)
        return np.asarray(model.predict(X), dtype=np.float64)

    def run_fold(self, X_train, y_train, X_val, y_val) -> ConformalEvaluation:
        n = X_train.shape[0]
        sub_idx, cal_idx = seeded_train_calib_split(n, self.inner_calib_frac, self.inner_seed)
        model = self.factory()
        if hasattr(X_train, "iloc"):
            model.fit(X_train.iloc[sub_idx], y_train.iloc[sub_idx] if hasattr(y_train, "iloc") else y_train[sub_idx])
            y_pred_cal = self._predict(model, X_train.iloc[cal_idx])
            y_pred_val = self._predict(model, X_val)
        else:
            model.fit(X_train[sub_idx], y_train[sub_idx])
            y_pred_cal = self._predict(model, X_train[cal_idx])
            y_pred_val = self._predict(model, X_val)

        y_cal_true = y_train.iloc[cal_idx].to_numpy() if hasattr(y_train, "iloc") else y_train[cal_idx]
        y_val_true = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)

        cp = MultiTargetSplitCP(alpha=self.alpha).calibrate(y_cal_true, y_pred_cal)
        lo, hi = cp.predict(y_pred_val)
        quantiles = [float(c.quantile_) for c in cp.per_target_]  # type: ignore[union-attr]

        ev = ConformalEvaluation(
            y_true=np.asarray(y_val_true, dtype=np.float64),
            y_pred=np.asarray(y_pred_val, dtype=np.float64),
            lower=lo, upper=hi,
            quantiles_per_target=quantiles,
        )
        self.evaluations.append(ev)
        return ev


def aggregate_cp_coverage(evaluations: list[ConformalEvaluation]) -> dict:
    from core.config import TARGETS
    cov_per_t = {t: [] for t in TARGETS}
    wid_per_t = {t: [] for t in TARGETS}
    for ev in evaluations:
        for t, v in ev.coverage_per_target.items():
            cov_per_t[t].append(v)
        for t, v in ev.width_per_target.items():
            wid_per_t[t].append(v)
    return {
        "coverage_per_target": {t: float(np.mean(v)) for t, v in cov_per_t.items()},
        "width_per_target": {t: float(np.mean(v)) for t, v in wid_per_t.items()},
        "coverage_flat": float(np.mean([c for v in cov_per_t.values() for c in v])),
        "width_flat": float(np.mean([c for v in wid_per_t.values() for c in v])),
    }
