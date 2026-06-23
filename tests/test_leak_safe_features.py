import numpy as np
import pandas as pd

from core.leak_safe_features import (
    ALL_FEATURE_COLUMNS,
    PublisherHistoryIndex,
    UserHistoryIndex,
    build_leak_safe_aggregates,
)


def _toy_history():
    return pd.DataFrame({
        "user_id": [10, 10, 10, 10, 20, 20, 30, 30, 30],
        "hour": [10, 50, 200, 800, 30, 400, 5, 100, 700],
        "publisher": [1, 2, 1, 3, 2, 2, 1, 1, 2],
        "cpm": [10.0, 20.0, 15.0, 25.0, 50.0, 60.0, 5.0, 10.0, 15.0],
    })


def _toy_users():
    return pd.DataFrame({"user_id": [10, 20, 30], "age": [25, 35, 45], "sex": [1, 2, 1], "city_id": [1, 2, 3]})


def test_user_features_change_with_cutoff():
    idx = UserHistoryIndex.build(_toy_history())
    f_early = idx.user_features_before(10, cutoff_hour=100)  # sees first 2 impressions
    f_late = idx.user_features_before(10, cutoff_hour=900)   # sees all 4
    assert f_early[0] == 2  # n_imps
    assert f_late[0] == 4
    assert not np.allclose(f_early, f_late), "Same user at different cutoffs must produce different features"


def test_cold_start_before_first_impression():
    idx = UserHistoryIndex.build(_toy_history())
    f = idx.user_features_before(30, cutoff_hour=3)
    assert f[4] == 1.0, "is_cold flag should be 1"
    assert f[0] == 0.0


def test_unknown_user_is_cold():
    idx = UserHistoryIndex.build(_toy_history())
    f = idx.user_features_before(999, cutoff_hour=500)
    assert f[4] == 1.0


def test_build_leak_safe_aggregates_returns_39_columns():
    hist = _toy_history()
    users = _toy_users()
    uidx = UserHistoryIndex.build(hist)
    pidx = PublisherHistoryIndex.build(hist, users)
    val = pd.DataFrame({
        "cpm": [10.0, 20.0],
        "hour_start": [100, 500],
        "hour_end": [120, 520],
        "audience_size": [3, 3],
        "user_ids": [[10, 20, 30], [10, 20, 30]],
        "publishers": [[1, 2], [1, 2]],
    })
    X = build_leak_safe_aggregates(val, uidx, pidx)
    assert X.shape == (2, len(ALL_FEATURE_COLUMNS))
    assert not X.isna().any().any()
    # Audience-mean-imp must differ between an early and a late campaign on the same audience
    assert X.iloc[0]["aud_mean_imp"] != X.iloc[1]["aud_mean_imp"]
