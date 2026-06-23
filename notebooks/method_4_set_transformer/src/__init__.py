"""Set Transformer local implementation (CPU/MPS/CUDA — runnable in Jupyter).

Public surface:
  data:        CampaignSetDataset, collate_pad, fit_normalizers, build_user_index
  model:       SetTransformerReach
  train:       train_one_fold, run_cv, run_holdout, get_device
  leak_tests:  run_all (smoke tests for leakage)
"""
from .data import (
    CampaignSetDataset,
    collate_pad,
    fit_normalizers,
    CAMPAIGN_CONT_COLS,
    N_USER_FEATURES,
)
from .model import SetTransformerReach
from .train import (
    train_one_fold,
    run_cv,
    run_holdout,
    get_device,
    BASE_CONFIG,
    VARIANTS,
)

__all__ = [
    "CampaignSetDataset",
    "collate_pad",
    "fit_normalizers",
    "CAMPAIGN_CONT_COLS",
    "N_USER_FEATURES",
    "SetTransformerReach",
    "train_one_fold",
    "run_cv",
    "run_holdout",
    "get_device",
    "BASE_CONFIG",
    "VARIANTS",
]
