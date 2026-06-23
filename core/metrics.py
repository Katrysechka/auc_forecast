from __future__ import annotations
from typing import Sequence
import numpy as np
from core.config import EPS, TARGETS


def smlar(y_true, y_pred, epsilon: float = EPS) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    ratio = (y_pred + epsilon) / (y_true + epsilon)
    ratio = np.clip(ratio, 1e-12, None)
    smlr = float(np.mean(np.abs(np.log(ratio))))
    return 100.0 * (np.exp(smlr) - 1.0)


def smlar_per_target(y_true, y_pred, target_names: Sequence[str] = TARGETS) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape and y_true.shape[1] == len(target_names)
    return {name: smlar(y_true[:, i], y_pred[:, i]) for i, name in enumerate(target_names)}


def monotonicity_violation(y_pred) -> float:
    y = np.asarray(y_pred, dtype=np.float64)
    assert y.shape[1] == 3, f"expected 3 targets, got shape {y.shape}"
    bad = (y[:, 0] < y[:, 1]) | (y[:, 1] < y[:, 2])
    return float(bad.mean())


def monotonicity_violation_per_pair(y_pred) -> dict[str, float]:
    y = np.asarray(y_pred, dtype=np.float64)
    assert y.shape[1] == 3, f"expected 3 targets, got shape {y.shape}"
    return {
        "y1_vs_y2": float((y[:, 0] < y[:, 1]).mean()),
        "y2_vs_y3": float((y[:, 1] < y[:, 2]).mean()),
    }


def coverage_at_alpha(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    return float(((y_true >= lo) & (y_true <= hi)).mean())

def _torch():
    import torch

    return torch


def mse_loss():
    return _torch().nn.MSELoss()


def mae_loss():
    return _torch().nn.L1Loss()


def huber_loss(delta: float = 0.1):
    return _torch().nn.HuberLoss(delta=delta)


def msle_loss(eps: float = EPS):
    torch = _torch()

    class MSLE(torch.nn.Module):
        def forward(self, pred, target):
            ratio = (pred + eps) / (target + eps)
            ratio = torch.clamp(ratio, min=1e-8)
            return torch.mean(torch.log(ratio) ** 2)

    return MSLE()


def smlar_smooth_loss(eps: float = EPS, alpha: float = 0.01):
    torch = _torch()

    class SMLARSmooth(torch.nn.Module):
        def forward(self, pred, target):
            ratio = (pred + eps) / (target + eps)
            ratio = torch.clamp(ratio, min=1e-8)
            log_ratio = torch.log(ratio)
            return torch.mean(torch.sqrt(log_ratio ** 2 + alpha ** 2) - alpha)

    return SMLARSmooth()


def rmsle_loss():
    torch = _torch()

    class RMSLE(torch.nn.Module):
        def forward(self, pred, target):
            pred_clipped = torch.clamp(pred, min=0)
            return torch.sqrt(torch.mean((torch.log1p(pred_clipped) - torch.log1p(target)) ** 2))

    return RMSLE()


LOSS_REGISTRY = {
    "mse": mse_loss,
    "mae": mae_loss,
    "huber": huber_loss,
    "msle": msle_loss,
    "rmsle": rmsle_loss,
    "smlar_smooth": smlar_smooth_loss,
}
