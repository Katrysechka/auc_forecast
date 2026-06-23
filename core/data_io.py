from __future__ import annotations
from ast import literal_eval
from pathlib import Path

import pandas as pd

from core.config import (
    DATA_DIR,
    HISTORY_TSV,
    USERS_TSV,
    VALIDATE_ANSWERS_TSV,
    VALIDATE_TSV,
)


def _parse_list(x):
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("["):
        return literal_eval(s)
    return [int(p) for p in s.split(",") if p]


def load_users(data_dir: Path | str | None = None) -> pd.DataFrame:
    path = Path(data_dir) / "users.tsv" if data_dir else USERS_TSV
    return pd.read_csv(path, sep="\t")


def load_history(data_dir: Path | str | None = None) -> pd.DataFrame:
    path = Path(data_dir) / "history.tsv" if data_dir else HISTORY_TSV
    return pd.read_csv(path, sep="\t")


def load_validate(data_dir: Path | str | None = None, parse_lists: bool = True) -> pd.DataFrame:
    path = Path(data_dir) / "validate.tsv" if data_dir else VALIDATE_TSV
    df = pd.read_csv(path, sep="\t")
    if parse_lists:
        df["publishers"] = df["publishers"].apply(_parse_list)
        df["user_ids"] = df["user_ids"].apply(_parse_list)
    return df


def load_validate_answers(data_dir: Path | str | None = None) -> pd.DataFrame:
    path = Path(data_dir) / "validate_answers.tsv" if data_dir else VALIDATE_ANSWERS_TSV
    return pd.read_csv(path, sep="\t")


def load_validate_full(data_dir: Path | str | None = None) -> pd.DataFrame:
    v = load_validate(data_dir, parse_lists=True)
    a = load_validate_answers(data_dir)
    assert len(v) == len(a), f"validate {len(v)} vs answers {len(a)}"
    return pd.concat([v.reset_index(drop=True), a.reset_index(drop=True)], axis=1)


def load_raw(data_dir: Path | str | None = None):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    users = pd.read_csv(data_dir / "users.tsv", sep="\t")
    history = pd.read_csv(data_dir / "history.tsv", sep="\t")
    val = pd.read_csv(data_dir / "validate.tsv", sep="\t")
    ans = pd.read_csv(data_dir / "validate_answers.tsv", sep="\t")
    return users, history, val, ans
