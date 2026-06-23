"""Anti-leakage smoke tests for the Set Transformer pipeline.

Run these before trusting any reported number — they verify that:
  1. Per-user features for campaign j do NOT see history with hour >= hour_start_j.
  2. The time-based holdout is strictly later than the train portion.
  3. Normalizers fit on the train fold are unaffected by perturbing the holdout.

All tests use the project-wide `core.*` utilities so any drift between this method
and the unified split protocol would surface immediately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import HOLDOUT_FRAC, SEED
from core.data_io import load_raw
from core.leak_safe_features import UserHistoryIndex
from core.splits import time_based_split

from .data import _as_user_id_array, fit_normalizers, precompute_campaign_features


def _load_campaigns() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users, history, val, _ = load_raw()
    campaigns = precompute_campaign_features(val)
    return users, history, campaigns


def test_no_future_history_in_user_features(n_check_campaigns: int = 20, seed: int = 1) -> dict:
    """For random campaigns, perturb history AT/AFTER the cutoff and check features are unchanged."""
    users, history, campaigns = _load_campaigns()
    hist_base = UserHistoryIndex.build(history)

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(campaigns), n_check_campaigns, replace=False)
    max_abs_diff = 0.0
    for r in rows:
        row = campaigns.iloc[r]
        cutoff = int(row["hour_start"])
        h2 = history.copy()
        mask = h2["hour"] >= cutoff
        h2.loc[mask, "cpm"] = rng.uniform(1, 999, size=int(mask.sum()))
        h2.loc[mask, "publisher"] = rng.integers(1, 21, size=int(mask.sum()))
        hist_pert = UserHistoryIndex.build(h2)

        uids = _as_user_id_array(row["user_ids"])
        for uid in uids[:30]:
            f_base = hist_base.user_features_before(int(uid), cutoff)
            f_pert = hist_pert.user_features_before(int(uid), cutoff)
            max_abs_diff = max(max_abs_diff, float(np.abs(f_base - f_pert).max()))
    return {
        "max_abs_diff": max_abs_diff,
        "passed": max_abs_diff < 1e-6,
        "checked_campaigns": int(len(rows)),
    }


def test_holdout_split_temporal() -> dict:
    """Holdout campaigns must have hour_start >= max hour_start of train."""
    _, _, campaigns = _load_campaigns()
    train_idx, holdout_idx = time_based_split(campaigns, holdout_frac=HOLDOUT_FRAC)
    max_train_hs = int(campaigns.iloc[train_idx]["hour_start"].max())
    min_hold_hs = int(campaigns.iloc[holdout_idx]["hour_start"].min())
    return {
        "max_train_hour_start": max_train_hs,
        "min_holdout_hour_start": min_hold_hs,
        "passed": max_train_hs <= min_hold_hs,
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
    }


def test_normalizers_train_only() -> dict:
    """Normalizers fit on train must be identical after perturbing holdout campaigns."""
    _, history, campaigns = _load_campaigns()
    hist_index = UserHistoryIndex.build(history)
    train_idx, holdout_idx = time_based_split(campaigns, holdout_frac=HOLDOUT_FRAC)

    norm1 = fit_normalizers(campaigns, train_idx, hist_index, sample_seed=SEED)
    campaigns2 = campaigns.copy()
    campaigns2.loc[holdout_idx, "cpm"] = -999.0
    norm2 = fit_normalizers(campaigns2, train_idx, hist_index, sample_seed=SEED)
    same = all(
        np.allclose(norm1[k], norm2[k]) for k in ("camp_mean", "camp_std", "user_mean", "user_std")
    )
    return {"passed": same}


def run_all(n_check_campaigns: int = 20) -> dict:
    results = {
        "no_future_history": test_no_future_history_in_user_features(n_check_campaigns),
        "temporal_holdout": test_holdout_split_temporal(),
        "normalizers_train_only": test_normalizers_train_only(),
    }
    results["all_passed"] = all(r["passed"] for r in results.values() if isinstance(r, dict))
    return results
