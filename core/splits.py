from __future__ import annotations
from typing import Iterator
import numpy as np
import pandas as pd

from core.config import HOLDOUT_FRAC, N_FOLDS, SEED


def time_based_split(
    val_df: pd.DataFrame,
    holdout_frac: float = HOLDOUT_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    hour_start = val_df["hour_start"].to_numpy()
    row_index = np.arange(len(val_df), dtype=np.int64)
    order = np.lexsort((row_index, hour_start)) 
    n = len(val_df)
    n_hold = int(round(n * holdout_frac))
    holdout = order[-n_hold:] if n_hold > 0 else np.array([], dtype=np.int64)
    train = order[:-n_hold] if n_hold > 0 else order
    return np.sort(train), np.sort(holdout)


def kfold_by_campaign(
    train_idx: np.ndarray,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(train_idx, copy=True)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_splits)
    for k in range(n_splits):
        val_f = np.sort(folds[k])
        train_f = np.sort(np.concatenate([folds[i] for i in range(n_splits) if i != k]))
        yield train_f, val_f


def build_unified_split(
    val_df: pd.DataFrame,
    holdout_frac: float = HOLDOUT_FRAC,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
) -> dict:
    train_idx, holdout_idx = time_based_split(val_df, holdout_frac=holdout_frac)
    folds = list(kfold_by_campaign(train_idx, n_splits=n_splits, seed=seed))
    return {
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
        "folds": folds,
        "protocol": (
            f"time-based 80/20 (sort by hour_start with row_index tiebreaker) "
            f"+ {n_splits}-fold campaign CV on train (seed={seed})"
        ),
    }
