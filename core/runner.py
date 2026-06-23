from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Protocol
import pandas as pd

from core.config import DATA_DIR
from core.data_io import load_raw
from core.leak_safe_features import PublisherHistoryIndex, UserHistoryIndex
from core.reporting import save_method_results, write_final_comparison
from core.splits import build_unified_split


class Method(Protocol):
    name: str

    def run(self, data: dict, split: dict) -> dict: ...


def load_unified_data(data_dir: Path | str = DATA_DIR) -> dict:
    users, history, val, ans = load_raw(data_dir)
    val_parsed = val.copy()
    val_parsed["publishers"] = val_parsed["publishers"].apply(
        lambda s: [int(x) for x in str(s).split(",") if x]
    )
    val_parsed["user_ids"] = val_parsed["user_ids"].apply(
        lambda s: [int(x) for x in str(s).split(",") if x]
    )
    user_index = UserHistoryIndex.build(history)
    pub_index = PublisherHistoryIndex.build(history, users)
    split = build_unified_split(val_parsed)
    return {
        "users": users,
        "history": history,
        "val": val_parsed,
        "ans": ans,
        "user_index": user_index,
        "pub_index": pub_index,
        "split": split,
    }


def run_all_methods(
    methods: list[Method],
    data: dict | None = None,
    write_comparison: bool = True,
) -> dict[str, dict]:
    if data is None:
        data = load_unified_data()
    split = data["split"]
    out: dict[str, dict] = {}
    for m in methods:
        result = m.run(data, split)
        save_method_results(m.name, result)
        out[m.name] = result
    if write_comparison:
        write_final_comparison()
    return out


if __name__ == "__main__":
    write_final_comparison()
