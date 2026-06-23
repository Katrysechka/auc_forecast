"""Training loop with SMLAR-smooth loss + 5-fold CV (campaign-level) + holdout.

Anti-leak invariants enforced in code:
  - normalizers fit on TRAIN fold only (`fit_normalizers`)
  - per-campaign history cutoff at hour_start (handled by `UserHistoryIndex`)
  - calibration/HPO never sees holdout

Device selection respects an explicit env override:
  AUC_TORCH_DEVICE=cuda|mps|cpu  -> force that device
otherwise: CUDA if available, else CPU. MPS is intentionally NOT picked by default
because masked attention over variable-length sets is unstable on Apple's stack
(see PyTorch issue tracker). Set the env var to opt in.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.config import N_PUBLISHERS, SEED, TARGETS, set_seed
from core.leak_safe_features import UserHistoryIndex
from core.metrics import (
    monotonicity_violation,
    smlar,
    smlar_per_target,
    smlar_smooth_loss,
    mse_loss,
    mae_loss,
)
from core.splits import build_unified_split

from .data import (
    CAMPAIGN_CONT_COLS,
    CampaignSetDataset,
    N_USER_FEATURES,
    collate_pad,
    fit_normalizers,
    precompute_campaign_features,
)
from .model import SetTransformerReach


# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------

_LOSSES = {
    "smlar_smooth": smlar_smooth_loss,
    "mse": mse_loss,
    "mae": mae_loss,
}


def _get_loss(name: str) -> torch.nn.Module:
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss {name!r}. Available: {sorted(_LOSSES)}")
    return _LOSSES[name]()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the torch device respecting AUC_TORCH_DEVICE override."""
    override = os.environ.get("AUC_TORCH_DEVICE", "").lower()
    if override in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("AUC_TORCH_DEVICE=cuda but CUDA is not available.")
        return torch.device("cuda")
    if override == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("AUC_TORCH_DEVICE=mps but MPS is not available.")
        return torch.device("mps")
    if override == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

BASE_CONFIG: dict = {
    "batch_size": 16,
    "lr": 1e-3,
    "wd": 1e-4,
    "epochs": 50,
    "d_hidden": 96,
    "n_isab": 2,
    "m_ind": 16,
    "n_heads": 4,
    "k_max": 5,
    "dropout": 0.1,
}

# 5 variants from the ablation table — same as the original Colab run.
VARIANTS: list[tuple[str, dict]] = [
    ("A_full",      {"use_attention": True,  "use_distribution": True,  "loss": "smlar_smooth"}),
    ("B_no_attn",   {"use_attention": False, "use_distribution": True,  "loss": "smlar_smooth"}),
    ("C_no_distr",  {"use_attention": True,  "use_distribution": False, "loss": "smlar_smooth"}),
    ("D_no_smlar",  {"use_attention": True,  "use_distribution": True,  "loss": "mse"}),
    ("E_none",      {"use_attention": False, "use_distribution": False, "loss": "mse"}),
]


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def _build_model(config: dict) -> SetTransformerReach:
    return SetTransformerReach(
        d_user=N_USER_FEATURES,
        d_camp=len(CAMPAIGN_CONT_COLS),
        n_publishers=N_PUBLISHERS,
        d_hidden=config["d_hidden"],
        n_isab=config["n_isab"],
        m_ind=config["m_ind"],
        n_heads=config["n_heads"],
        k_max=config["k_max"],
        dropout=config["dropout"],
        use_attention=config["use_attention"],
        use_distribution=config["use_distribution"],
    )


def train_one_fold(
    campaigns: pd.DataFrame,
    targets: pd.DataFrame,
    users_df: pd.DataFrame,
    hist_index: UserHistoryIndex,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    config: dict,
    seed: int = SEED,
    verbose: bool = False,
    save_checkpoint: Path | None = None,
) -> tuple[dict, SetTransformerReach, dict]:
    """Train one fold; return (best metrics + arrays, trained model, normalizers)."""
    set_seed(seed)
    device = get_device()

    norm = fit_normalizers(campaigns, train_idx, hist_index, sample_seed=seed)

    train_ds = CampaignSetDataset(
        campaigns.iloc[train_idx], users_df, hist_index, targets.iloc[train_idx],
        norm["camp_mean"], norm["camp_std"], norm["user_mean"], norm["user_std"],
    )
    val_ds = CampaignSetDataset(
        campaigns.iloc[val_idx], users_df, hist_index, targets.iloc[val_idx],
        norm["camp_mean"], norm["camp_std"], norm["user_mean"], norm["user_std"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_pad, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_pad, num_workers=0,
    )

    model = _build_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"])
    loss_fn = _get_loss(config["loss"]).to(device)

    best: dict = {"smlar": float("inf"), "epoch": -1, "pred": None, "true": None, "state_dict": None}
    for ep in range(config["epochs"]):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out["y_hat"], batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                out = model(batch)
                all_pred.append(out["y_hat"].detach().cpu().numpy())
                all_true.append(batch["target"].detach().cpu().numpy())
        y_pred = np.clip(np.concatenate(all_pred, axis=0), 0.0, 1.0)
        y_true = np.concatenate(all_true, axis=0)
        sm = smlar(y_true, y_pred)
        if sm < best["smlar"]:
            best = {
                "smlar": sm,
                "epoch": ep,
                "pred": y_pred,
                "true": y_true,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if verbose and (ep % 5 == 0 or ep == config["epochs"] - 1):
            print(
                f"    ep {ep:3d}  trainL={np.mean(train_losses):.4f}  "
                f"valSMLAR={sm:.2f}%  best={best['smlar']:.2f}%"
            )

    if save_checkpoint is not None and best["state_dict"] is not None:
        save_checkpoint = Path(save_checkpoint)
        save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best["state_dict"], "config": config, "norm": norm}, save_checkpoint)

    return best, model, norm


# ---------------------------------------------------------------------------
# 5-fold CV (campaign-level)
# ---------------------------------------------------------------------------

def run_cv(
    config: dict,
    *,
    data: dict | None = None,
    n_folds: int = 5,
    seed: int = SEED,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """Run 5-fold CV on the unified split.

    `data` should be the dict returned by `core.runner.load_unified_data`. If None,
    it is loaded here (slower if called multiple times — pre-load once and pass in).
    """
    if data is None:
        from core.runner import load_unified_data
        data = load_unified_data()
    campaigns = precompute_campaign_features(data["val"])
    hist_index = data["user_index"]
    users_df = data["users"]
    ans = data["ans"]
    split = data["split"] if "split" in data else build_unified_split(data["val"])

    fold_results = []
    for fold_i, (tr, va) in enumerate(split["folds"]):
        t0 = time.time()
        best, _, _ = train_one_fold(
            campaigns, ans, users_df, hist_index, tr, va,
            config=config, seed=seed + fold_i, verbose=verbose,
        )
        elapsed = time.time() - t0
        per_t = smlar_per_target(best["true"], best["pred"])
        mono = monotonicity_violation(best["pred"])
        fold_results.append({
            "fold": fold_i,
            "smlar": best["smlar"],
            "best_epoch": best["epoch"],
            "smlar_y1": per_t["at_least_one"],
            "smlar_y2": per_t["at_least_two"],
            "smlar_y3": per_t["at_least_three"],
            "monotone_violation": mono,
            "time_s": elapsed,
        })
        if verbose:
            print(
                f"  fold {fold_i} best epoch {best['epoch']}  "
                f"SMLAR={best['smlar']:.2f}%  ({elapsed:.1f}s)"
            )

    fold_df = pd.DataFrame(fold_results)
    summary = {
        "mean_smlar": float(fold_df["smlar"].mean()),
        "std_smlar": float(fold_df["smlar"].std()),
        "mean_mono_violation": float(fold_df["monotone_violation"].mean()),
        "config": config,
    }
    return fold_df, summary, split["train_idx"], split["holdout_idx"]


# ---------------------------------------------------------------------------
# Holdout (train on full 806, evaluate on 202 — call ONCE per variant)
# ---------------------------------------------------------------------------

def run_holdout(
    config: dict,
    *,
    data: dict | None = None,
    seed: int = SEED,
    verbose: bool = False,
    save_checkpoint: Path | None = None,
) -> dict:
    if data is None:
        from core.runner import load_unified_data
        data = load_unified_data()
    campaigns = precompute_campaign_features(data["val"])
    hist_index = data["user_index"]
    users_df = data["users"]
    ans = data["ans"]
    split = data["split"] if "split" in data else build_unified_split(data["val"])

    best, _, _ = train_one_fold(
        campaigns, ans, users_df, hist_index,
        split["train_idx"], split["holdout_idx"],
        config=config, seed=seed, verbose=verbose,
        save_checkpoint=save_checkpoint,
    )
    per_t = smlar_per_target(best["true"], best["pred"])
    return {
        "holdout_smlar": best["smlar"],
        "best_epoch": best["epoch"],
        "smlar_y1": per_t["at_least_one"],
        "smlar_y2": per_t["at_least_two"],
        "smlar_y3": per_t["at_least_three"],
        "monotone_violation": monotonicity_violation(best["pred"]),
        "pred": best["pred"],
        "true": best["true"],
    }
