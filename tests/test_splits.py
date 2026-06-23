import numpy as np
import pandas as pd

from core.config import N_FOLDS
from core.splits import build_unified_split, kfold_by_campaign, time_based_split


def _make_val(hour_starts):
    return pd.DataFrame({"hour_start": hour_starts})


def test_train_holdout_total_is_input_length():
    val = _make_val(list(range(1008)))
    tr, ho = time_based_split(val)
    assert len(tr) + len(ho) == 1008
    assert len(set(tr) & set(ho)) == 0


def test_default_split_is_806_202():
    val = _make_val(list(range(1008)))
    tr, ho = time_based_split(val)
    assert len(tr) == 806
    assert len(ho) == 202


def test_holdout_is_latest_hours():
    val = _make_val([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])
    tr, ho = time_based_split(val, holdout_frac=0.2)
    assert sorted(ho.tolist()) == [8, 9]


def test_tiebreaker_on_equal_hour_start_is_row_index():
    """When many campaigns share the same hour_start, partition must depend on the row index
    (the secondary key), so the result is independent of input row order *as long as the
    underlying campaigns are the same*."""
    val = _make_val([100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    tr, ho = time_based_split(val, holdout_frac=0.2)
    # last 2 of 10 = last 2 row indices
    assert sorted(ho.tolist()) == [8, 9]


def test_folds_are_5_disjoint_partitions():
    train = np.arange(806)
    folds = list(kfold_by_campaign(train, n_splits=N_FOLDS, seed=42))
    assert len(folds) == 5
    all_val_indices = np.concatenate([va for _, va in folds])
    assert len(all_val_indices) == 806
    assert sorted(all_val_indices.tolist()) == sorted(train.tolist())
    # each (train, val) is a partition of `train`
    for tr, va in folds:
        assert len(set(tr) & set(va)) == 0
        assert sorted(np.concatenate([tr, va]).tolist()) == sorted(train.tolist())


def test_folds_are_deterministic_under_same_seed():
    train = np.arange(806)
    a = [(tr.copy(), va.copy()) for tr, va in kfold_by_campaign(train, seed=42)]
    b = [(tr.copy(), va.copy()) for tr, va in kfold_by_campaign(train, seed=42)]
    for (atr, ava), (btr, bva) in zip(a, b):
        assert (atr == btr).all() and (ava == bva).all()


def test_build_unified_split_protocol_string_includes_seed():
    val = _make_val(list(range(1008)))
    s = build_unified_split(val)
    assert "seed=42" in s["protocol"]
    assert len(s["folds"]) == N_FOLDS
