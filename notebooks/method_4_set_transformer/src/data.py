"""Leak-safe set-based dataset for the Set Transformer.

Reuses the project-wide `core.leak_safe_features.UserHistoryIndex` (with prefix-sum
acceleration) and the unified split from `core.splits` so that the 806/202 partition
and the 5 CV folds are bit-identical to every other method in the repo.

Anti-leakage invariants:
  1. Time-based holdout via `core.splits.time_based_split` (with row-index tiebreaker).
  2. Per-campaign history cutoff at hour_start (handled by `UserHistoryIndex`).
  3. Normalizers fit on the TRAIN fold only (validated by `leak_tests.py`).
  4. Cold users (no history before cutoff) get an `is_cold` flag instead of NaN/zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.config import N_PUBLISHERS, TARGETS
from core.leak_safe_features import UserHistoryIndex


CAMPAIGN_CONT_COLS = ["cpm", "hour_start", "duration", "audience_size", "n_publishers"]
N_USER_FEATURES = 9  # 4 history-std + 1 cold flag + 4 demographic (sex/age/city/unknown)


def precompute_campaign_features(val_df: pd.DataFrame) -> pd.DataFrame:
    """Add `duration` and `n_publishers` columns without leaking future info.

    Accepts either the parsed form (lists in `publishers`/`user_ids`, what
    `core.runner.load_unified_data` produces) or the raw string form.
    """
    df = val_df.copy()
    df["duration"] = df["hour_end"] - df["hour_start"]
    pubs = df["publishers"]
    if len(pubs) > 0 and isinstance(pubs.iloc[0], (list, tuple)):
        df["n_publishers"] = pubs.apply(len).astype(np.int32)
    else:
        df["n_publishers"] = pubs.astype(str).str.split(",").str.len().astype(np.int32)
    return df


def _as_user_id_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.int32)
    return np.array([int(x) for x in str(value).split(",") if x], dtype=np.int32)


def _publisher_multihot(value, n_pubs: int = N_PUBLISHERS) -> np.ndarray:
    out = np.zeros(n_pubs, dtype=np.float32)
    iterable = value if isinstance(value, (list, tuple, np.ndarray)) else str(value).split(",")
    for p in iterable:
        idx = int(p) - 1  # publishers are 1-indexed in raw data
        if 0 <= idx < n_pubs:
            out[idx] = 1.0
    return out


@dataclass
class _DemoLookup:
    sex: np.ndarray
    age: np.ndarray
    city: np.ndarray
    known: np.ndarray
    max_uid: int

    @classmethod
    def build(cls, users_df: pd.DataFrame, age_max: int = 120, city_max: int = 3000) -> "_DemoLookup":
        u = users_df.set_index("user_id")
        max_uid = int(u.index.max())
        sex = np.zeros(max_uid + 2, dtype=np.float32)
        age = np.zeros(max_uid + 2, dtype=np.float32)
        city = np.zeros(max_uid + 2, dtype=np.float32)
        known = np.zeros(max_uid + 2, dtype=np.float32)
        for uid, r in u.iterrows():
            sex[uid] = float(r["sex"]) / 2.0
            age[uid] = float(r["age"]) / age_max
            city[uid] = float(r["city_id"]) / city_max
            known[uid] = 1.0
        return cls(sex=sex, age=age, city=city, known=known, max_uid=max_uid)

    def demo_block(self, user_ids: np.ndarray) -> np.ndarray:
        """[n_users, 4] -> (sex, age, city, is_unknown_user)."""
        demo = np.zeros((len(user_ids), 4), dtype=np.float32)
        for j, uid in enumerate(user_ids):
            if 0 <= uid <= self.max_uid and self.known[uid] > 0:
                demo[j, 0] = self.sex[uid]
                demo[j, 1] = self.age[uid]
                demo[j, 2] = self.city[uid]
            else:
                demo[j, 3] = 1.0
        return demo


class CampaignSetDataset(Dataset):
    """Each sample = one campaign represented as a SET of (variable-size) users.

    Per-campaign tensors are pre-computed once at construction (the per-user history
    cutoff is determined by `hour_start`, which is fixed across epochs).
    """

    def __init__(
        self,
        campaign_rows: pd.DataFrame,
        users_df: pd.DataFrame,
        hist_index: UserHistoryIndex,
        targets_df: pd.DataFrame | None,
        camp_feat_mean: np.ndarray,
        camp_feat_std: np.ndarray,
        user_feat_mean: np.ndarray,
        user_feat_std: np.ndarray,
        n_publishers: int = N_PUBLISHERS,
    ):
        self.campaigns = campaign_rows.reset_index(drop=True)
        self.targets_df = targets_df.reset_index(drop=True) if targets_df is not None else None
        self.n_publishers = n_publishers
        demo = _DemoLookup.build(users_df)

        self._user_feat_cache: list[torch.Tensor] = []
        self._camp_feat_cache: list[torch.Tensor] = []
        self._pub_cache: list[torch.Tensor] = []
        self._target_cache: list[torch.Tensor] = []
        self._n_users_cache: list[int] = []

        for idx in range(len(self.campaigns)):
            row = self.campaigns.iloc[idx]
            cutoff = int(row["hour_start"])
            user_ids = _as_user_id_array(row["user_ids"])

            hist_feats = np.empty((len(user_ids), 5), dtype=np.float32)
            for j, uid in enumerate(user_ids):
                hist_feats[j] = hist_index.user_features_before(int(uid), cutoff)
            hist_std = (hist_feats[:, :4] - user_feat_mean[:4]) / (user_feat_std[:4] + 1e-6)

            demo_block = demo.demo_block(user_ids)
            user_feats = np.concatenate(
                [hist_std, hist_feats[:, 4:5], demo_block], axis=1
            ).astype(np.float32)
            assert user_feats.shape[1] == N_USER_FEATURES

            camp_raw = np.array([row[c] for c in CAMPAIGN_CONT_COLS], dtype=np.float32)
            camp_feats = ((camp_raw - camp_feat_mean) / (camp_feat_std + 1e-6)).astype(np.float32)
            pub_mh = _publisher_multihot(row["publishers"], n_publishers)

            self._user_feat_cache.append(torch.from_numpy(user_feats))
            self._camp_feat_cache.append(torch.from_numpy(camp_feats))
            self._pub_cache.append(torch.from_numpy(pub_mh))
            self._n_users_cache.append(int(len(user_ids)))
            if self.targets_df is not None:
                self._target_cache.append(torch.tensor(
                    [self.targets_df.loc[idx, t] for t in TARGETS], dtype=torch.float32
                ))

    def __len__(self) -> int:
        return len(self.campaigns)

    def __getitem__(self, idx: int) -> dict:
        out = {
            "user_feats": self._user_feat_cache[idx],
            "camp_feats": self._camp_feat_cache[idx],
            "pub_mh": self._pub_cache[idx],
            "n_users": self._n_users_cache[idx],
        }
        if self.targets_df is not None:
            out["target"] = self._target_cache[idx]
        return out


def collate_pad(batch: list[dict]) -> dict:
    """Pad variable-size user sets to the max-in-batch size with a boolean mask."""
    max_n = max(b["n_users"] for b in batch)
    B = len(batch)
    D_user = batch[0]["user_feats"].shape[1]
    user_feats = torch.zeros(B, max_n, D_user)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    camp_feats = torch.stack([b["camp_feats"] for b in batch])
    pub_mh = torch.stack([b["pub_mh"] for b in batch])
    has_target = "target" in batch[0]
    targets = torch.stack([b["target"] for b in batch]) if has_target else None
    for i, b in enumerate(batch):
        n = b["n_users"]
        user_feats[i, :n] = b["user_feats"]
        mask[i, :n] = True
    return {
        "user_feats": user_feats,
        "mask": mask,
        "camp_feats": camp_feats,
        "pub_mh": pub_mh,
        "target": targets,
    }


def fit_normalizers(
    campaigns: pd.DataFrame,
    train_idx: np.ndarray,
    hist_index: UserHistoryIndex,
    sample_seed: int = 42,
    user_sample_cap: int = 200,
) -> dict:
    """Compute mean/std on TRAIN fold campaigns only. Deterministic w.r.t. `sample_seed`."""
    train_camps = campaigns.iloc[train_idx]
    camp_X = train_camps[CAMPAIGN_CONT_COLS].to_numpy(dtype=np.float32)
    camp_mean = camp_X.mean(axis=0)
    camp_std = camp_X.std(axis=0) + 1e-6

    rng = np.random.default_rng(sample_seed)
    user_feat_samples = []
    for _, row in train_camps.iterrows():
        cutoff = int(row["hour_start"])
        uids = _as_user_id_array(row["user_ids"])
        if len(uids) > user_sample_cap:
            uids = rng.choice(uids, user_sample_cap, replace=False)
        for uid in uids:
            user_feat_samples.append(hist_index.user_features_before(int(uid), cutoff))
    U = np.stack(user_feat_samples, axis=0)
    user_mean = U[:, :4].mean(axis=0)
    user_std = U[:, :4].std(axis=0) + 1e-6
    return {
        "camp_mean": camp_mean,
        "camp_std": camp_std,
        "user_mean": user_mean,
        "user_std": user_std,
    }
