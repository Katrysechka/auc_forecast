from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

AUD_COLUMNS = (
    "aud_n_known",
    "aud_mean_imp",
    "aud_median_imp",
    "aud_std_imp",
    "aud_mean_cpm",
    "aud_median_cpm",
    "aud_std_cpm",
    "aud_mean_pub_diversity",
    "aud_mean_hour",
    "aud_morning_ratio",
    "aud_afternoon_ratio",
    "aud_evening_ratio",
    "aud_night_ratio",
)
PUB_COLUMNS = (
    "pub_n_known",
    "pub_mean_imp",
    "pub_mean_users",
    "pub_mean_cpm",
    "pub_median_cpm",
    "pub_std_cpm",
    "pub_mean_age",
    "pub_male_ratio",
    "pub_imp_per_user",
    "pub_premium_ratio",
    "pub_high_volume_ratio",
)
TIME_COLS = ("coverage_morning", "coverage_afternoon", "coverage_evening", "coverage_night")
CAMPAIGN_COLS = (
    "cpm",
    "audience_size",
    "hour_start",
    "hour_end",
    "campaign_duration",
    "num_publishers",
    "hour_start_of_day",
    "hour_end_of_day",
)
CROSS_COLS = ("cpm_x_duration", "cpm_x_audience", "audience_known_ratio")

ALL_FEATURE_COLUMNS = (
    CAMPAIGN_COLS + TIME_COLS + AUD_COLUMNS + PUB_COLUMNS + CROSS_COLS
)  

@dataclass
class UserHistoryIndex:
    hours: dict[int, np.ndarray]
    pubs: dict[int, np.ndarray]
    cpms: dict[int, np.ndarray]
    cpm_prefix: dict[int, np.ndarray]
    hod_prefix: dict[int, np.ndarray]
    morn_prefix: dict[int, np.ndarray]
    aft_prefix: dict[int, np.ndarray]
    eve_prefix: dict[int, np.ndarray]
    night_prefix: dict[int, np.ndarray]

    @classmethod
    def build(cls, history: pd.DataFrame) -> "UserHistoryIndex":
        h = history.sort_values(["user_id", "hour"], kind="mergesort")
        hours, pubs, cpms = {}, {}, {}
        cpm_prefix, hod_prefix = {}, {}
        morn_prefix, aft_prefix, eve_prefix, night_prefix = {}, {}, {}, {}
        for uid, grp in h.groupby("user_id", sort=False):
            hh = grp["hour"].to_numpy(dtype=np.int32)
            pp = grp["publisher"].to_numpy(dtype=np.int16)
            cc = grp["cpm"].to_numpy(dtype=np.float32)
            hours[uid] = hh
            pubs[uid] = pp
            cpms[uid] = cc
            cpm_prefix[uid] = np.concatenate([[0.0], np.cumsum(cc, dtype=np.float64)])
            hod = (hh % 24).astype(np.int32)
            hod_prefix[uid] = np.concatenate([[0], np.cumsum(hod, dtype=np.int64)])
            morn = ((hod >= 6) & (hod < 12)).astype(np.int32)
            aft = ((hod >= 12) & (hod < 18)).astype(np.int32)
            eve = ((hod >= 18) & (hod < 24)).astype(np.int32)
            night = ((hod >= 0) & (hod < 6)).astype(np.int32)
            morn_prefix[uid] = np.concatenate([[0], np.cumsum(morn, dtype=np.int64)])
            aft_prefix[uid] = np.concatenate([[0], np.cumsum(aft, dtype=np.int64)])
            eve_prefix[uid] = np.concatenate([[0], np.cumsum(eve, dtype=np.int64)])
            night_prefix[uid] = np.concatenate([[0], np.cumsum(night, dtype=np.int64)])
        return cls(
            hours, pubs, cpms,
            cpm_prefix, hod_prefix,
            morn_prefix, aft_prefix, eve_prefix, night_prefix,
        )

    def user_features_before(self, user_id: int, cutoff_hour: int) -> np.ndarray:
        if user_id not in self.hours:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        hh = self.hours[user_id]
        idx = int(np.searchsorted(hh, cutoff_hour, side="left"))
        if idx == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        n_imps = float(idx)
        mean_cpm = float(self.cpm_prefix[user_id][idx] / idx)
        n_unique_pubs = float(np.unique(self.pubs[user_id][:idx]).size)
        recency = float(cutoff_hour - hh[idx - 1])
        return np.array([n_imps, mean_cpm, n_unique_pubs, recency, 0.0], dtype=np.float32)

    def user_aggregates_for_audience(
        self, user_ids: np.ndarray, cutoff_hour: int
    ) -> dict:
        n_imps, mean_cpms, n_unique_pubs = [], [], []
        mean_hods = []
        morn, aft, eve, night = [], [], [], []
        n_known = 0

        for uid in user_ids:
            uid = int(uid)
            if uid not in self.hours:
                continue
            hh = self.hours[uid]
            idx = int(np.searchsorted(hh, cutoff_hour, side="left"))
            if idx == 0:
                continue
            n_known += 1
            n_imps.append(float(idx))
            mean_cpms.append(float(self.cpm_prefix[uid][idx] / idx))
            n_unique_pubs.append(float(np.unique(self.pubs[uid][:idx]).size))
            mean_hods.append(float(self.hod_prefix[uid][idx] / idx))
            morn.append(float(self.morn_prefix[uid][idx] / idx))
            aft.append(float(self.aft_prefix[uid][idx] / idx))
            eve.append(float(self.eve_prefix[uid][idx] / idx))
            night.append(float(self.night_prefix[uid][idx] / idx))

        if n_known == 0:
            return {
                "aud_n_known": 0,
                "aud_mean_imp": 0.0, "aud_median_imp": 0.0, "aud_std_imp": 0.0,
                "aud_mean_cpm": 0.0, "aud_median_cpm": 0.0, "aud_std_cpm": 0.0,
                "aud_mean_pub_diversity": 0.0, "aud_mean_hour": 12.0,
                "aud_morning_ratio": 0.0, "aud_afternoon_ratio": 0.0,
                "aud_evening_ratio": 0.0, "aud_night_ratio": 0.0,
            }
        n_imps_a = np.array(n_imps)
        mean_cpms_a = np.array(mean_cpms)
        return {
            "aud_n_known": n_known,
            "aud_mean_imp": float(n_imps_a.mean()),
            "aud_median_imp": float(np.median(n_imps_a)),
            "aud_std_imp": float(n_imps_a.std(ddof=0)),
            "aud_mean_cpm": float(mean_cpms_a.mean()),
            "aud_median_cpm": float(np.median(mean_cpms_a)),
            "aud_std_cpm": float(mean_cpms_a.std(ddof=0)),
            "aud_mean_pub_diversity": float(np.mean(n_unique_pubs)),
            "aud_mean_hour": float(np.mean(mean_hods)),
            "aud_morning_ratio": float(np.mean(morn)),
            "aud_afternoon_ratio": float(np.mean(aft)),
            "aud_evening_ratio": float(np.mean(eve)),
            "aud_night_ratio": float(np.mean(night)),
        }

@dataclass
class PublisherHistoryIndex:
    hours: dict[int, np.ndarray]
    user_ids: dict[int, np.ndarray]
    cpms: dict[int, np.ndarray]
    ages: dict[int, np.ndarray]
    sexes: dict[int, np.ndarray]
    cpm_prefix: dict[int, np.ndarray]
    age_prefix: dict[int, np.ndarray]
    sex_prefix: dict[int, np.ndarray]

    @classmethod
    def build(cls, history: pd.DataFrame, users_df: pd.DataFrame) -> "PublisherHistoryIndex":
        u = users_df.set_index("user_id")
        max_uid = int(u.index.max())
        age_arr = np.zeros(max_uid + 2, dtype=np.float32)
        sex_arr = np.zeros(max_uid + 2, dtype=np.float32)
        for uid, r in u.iterrows():
            age_arr[uid] = float(r["age"])
            sex_arr[uid] = 1.0 if float(r["sex"]) == 1 else 0.0  # male=1, female=0

        h = history.copy()
        h["age"] = age_arr[h["user_id"].clip(0, max_uid + 1)]
        h["male"] = sex_arr[h["user_id"].clip(0, max_uid + 1)]
        h = h.sort_values(["publisher", "hour"], kind="mergesort")

        hours, uids, cpms, ages, sexes = {}, {}, {}, {}, {}
        cpm_prefix, age_prefix, sex_prefix = {}, {}, {}
        for pid, grp in h.groupby("publisher", sort=False):
            hh = grp["hour"].to_numpy(dtype=np.int32)
            uu = grp["user_id"].to_numpy(dtype=np.int32)
            cc = grp["cpm"].to_numpy(dtype=np.float32)
            aa = grp["age"].to_numpy(dtype=np.float32)
            ss = grp["male"].to_numpy(dtype=np.float32)
            hours[pid] = hh
            user_ids = uu
            uids[pid] = user_ids
            cpms[pid] = cc
            ages[pid] = aa
            sexes[pid] = ss
            cpm_prefix[pid] = np.concatenate([[0.0], np.cumsum(cc, dtype=np.float64)])
            age_prefix[pid] = np.concatenate([[0.0], np.cumsum(aa, dtype=np.float64)])
            sex_prefix[pid] = np.concatenate([[0.0], np.cumsum(ss, dtype=np.float64)])
        return cls(hours, uids, cpms, ages, sexes, cpm_prefix, age_prefix, sex_prefix)

    def publisher_aggregates(
        self, publisher_ids: list[int], cutoff_hour: int
    ) -> dict:
        n_imps, mean_users, mean_cpms = [], [], []
        median_cpms, std_cpms = [], []
        mean_ages, male_ratios = [], []
        imp_per_user, premium_flags, high_vol_flags = [], [], []
        n_known = 0

        for pid in publisher_ids:
            pid = int(pid)
            if pid not in self.hours:
                continue
            hh = self.hours[pid]
            idx = int(np.searchsorted(hh, cutoff_hour, side="left"))
            if idx == 0:
                continue
            n_known += 1
            n_imps.append(float(idx))
            uniq_users = float(np.unique(self.user_ids[pid][:idx]).size)
            mean_users.append(uniq_users)
            cpm_slice = self.cpms[pid][:idx]
            mean_cpms.append(float(self.cpm_prefix[pid][idx] / idx))
            median_cpms.append(float(np.median(cpm_slice)))
            std_cpms.append(float(cpm_slice.std(ddof=0)))
            mean_ages.append(float(self.age_prefix[pid][idx] / idx))
            male_ratios.append(float(self.sex_prefix[pid][idx] / idx))
            imp_per_user.append(float(idx / max(uniq_users, 1.0)))
            premium_flags.append(1.0 if float(self.cpm_prefix[pid][idx] / idx) > 50 else 0.0)
            high_vol_flags.append(1.0 if idx > 10000 else 0.0)

        if n_known == 0:
            return {
                "pub_n_known": 0,
                "pub_mean_imp": 0.0, "pub_mean_users": 0.0,
                "pub_mean_cpm": 0.0, "pub_median_cpm": 0.0, "pub_std_cpm": 0.0,
                "pub_mean_age": 0.0, "pub_male_ratio": 0.5,
                "pub_imp_per_user": 0.0, "pub_premium_ratio": 0.0, "pub_high_volume_ratio": 0.0,
            }
        return {
            "pub_n_known": n_known,
            "pub_mean_imp": float(np.mean(n_imps)),
            "pub_mean_users": float(np.mean(mean_users)),
            "pub_mean_cpm": float(np.mean(mean_cpms)),
            "pub_median_cpm": float(np.median(median_cpms)),
            "pub_std_cpm": float(np.mean(std_cpms)),
            "pub_mean_age": float(np.mean(mean_ages)),
            "pub_male_ratio": float(np.mean(male_ratios)),
            "pub_imp_per_user": float(np.mean(imp_per_user)),
            "pub_premium_ratio": float(np.mean(premium_flags)),
            "pub_high_volume_ratio": float(np.mean(high_vol_flags)),
        }

def _time_coverage(start_hour: int, end_hour: int) -> dict:
    if end_hour <= start_hour:
        return {c: 0.0 for c in TIME_COLS}
    hh = np.arange(start_hour, end_hour) % 24
    n = float(len(hh))
    return {
        "coverage_morning": float(((hh >= 6) & (hh < 12)).sum() / n),
        "coverage_afternoon": float(((hh >= 12) & (hh < 18)).sum() / n),
        "coverage_evening": float(((hh >= 18) & (hh < 24)).sum() / n),
        "coverage_night": float(((hh >= 0) & (hh < 6)).sum() / n),
    }


def build_leak_safe_aggregates(
    validate: pd.DataFrame,
    user_index: UserHistoryIndex,
    pub_index: PublisherHistoryIndex,
) -> pd.DataFrame:
    rows = []
    for _, row in validate.iterrows():
        cutoff = int(row["hour_start"])
        d = {
            "cpm": float(row["cpm"]),
            "audience_size": int(row["audience_size"]),
            "hour_start": int(row["hour_start"]),
            "hour_end": int(row["hour_end"]),
            "campaign_duration": int(row["hour_end"] - row["hour_start"]),
            "num_publishers": int(len(row["publishers"])),
            "hour_start_of_day": int(row["hour_start"]) % 24,
            "hour_end_of_day": int(row["hour_end"]) % 24,
        }
        d.update(_time_coverage(int(row["hour_start"]), int(row["hour_end"])))
        d.update(user_index.user_aggregates_for_audience(np.array(row["user_ids"]), cutoff))
        d.update(pub_index.publisher_aggregates(list(row["publishers"]), cutoff))
        d["cpm_x_duration"] = d["cpm"] * d["campaign_duration"]
        d["cpm_x_audience"] = d["cpm"] * d["audience_size"]
        d["audience_known_ratio"] = d["aud_n_known"] / max(d["audience_size"], 1)
        rows.append(d)
    X = pd.DataFrame(rows)
    X = X[list(ALL_FEATURE_COLUMNS)]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X
