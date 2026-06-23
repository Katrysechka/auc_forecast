import numpy as np
import pandas as pd

from core.splits import build_unified_split


def test_two_callers_get_identical_folds():
    val = pd.DataFrame({"hour_start": list(range(1008))})
    s1 = build_unified_split(val)
    s2 = build_unified_split(val)
    assert (s1["train_idx"] == s2["train_idx"]).all()
    assert (s1["holdout_idx"] == s2["holdout_idx"]).all()
    for (a_tr, a_va), (b_tr, b_va) in zip(s1["folds"], s2["folds"]):
        assert (a_tr == b_tr).all() and (a_va == b_va).all()


def test_train_and_holdout_are_disjoint():
    val = pd.DataFrame({"hour_start": list(range(1008))})
    s = build_unified_split(val)
    overlap = np.intersect1d(s["train_idx"], s["holdout_idx"])
    assert overlap.size == 0


def test_fold_indices_are_subset_of_train():
    val = pd.DataFrame({"hour_start": list(range(1008))})
    s = build_unified_split(val)
    train_set = set(s["train_idx"].tolist())
    for tr, va in s["folds"]:
        assert set(tr.tolist()).issubset(train_set)
        assert set(va.tolist()).issubset(train_set)
