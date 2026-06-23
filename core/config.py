from __future__ import annotations
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PARQUET_DIR = ROOT / "parquet_files"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
RESULTS_FIGS_DIR = RESULTS_DIR / "figures"
RESULTS_PER_METHOD_DIR = RESULTS_DIR / "per_method"
NOTEBOOKS_DIR = ROOT / "notebooks"
OUTPUTS_DIR = ROOT / "outputs"

USERS_TSV = DATA_DIR / "users.tsv"
HISTORY_TSV = DATA_DIR / "history.tsv"
VALIDATE_TSV = DATA_DIR / "validate.tsv"
VALIDATE_ANSWERS_TSV = DATA_DIR / "validate_answers.tsv"

EPS = 0.005
SEED = 42
TARGETS = ("at_least_one", "at_least_two", "at_least_three")
N_FOLDS = 5
HOLDOUT_FRAC = 0.2
N_PUBLISHERS = 21

ALPHA = 0.10
INNER_CALIB_FRAC = 0.25
MC_N_SIMS = 500
MC_N_SIMS_HIGH = 1000
K_MAX = 6


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
