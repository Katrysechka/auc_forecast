"""Full-metric evaluation for Set Transformer.

Computes the SAME metric set as MLP/CatBoost/MC notebooks, in one run:

  CV (5 folds, sub-trained per fold):
    - smlar_flat (mean ± std)
    - smlar_per_target (mean / std per fold)
    - monotone_viol (mean) and monotone_viol_per_pair (mean)
    - Split CP coverage_per_target + width_per_target (averaged over folds)

  Holdout (full-train):
    - smlar_flat
    - smlar_per_target
    - monotone_viol + monotone_viol_per_pair

Protocol matches notebooks/method_2_mlp/03_evaluation_and_cp.ipynb:
  - 5 outer folds (campaign-CV from core.splits).
  - Per fold: inner seeded split of fold's train into (sub, calib).
  - Train on sub, predict on val + calib. SMLAR from val. CP calibrated on calib,
    evaluated on val. Coverage aggregated over folds.
  - HO: train on full 806 train campaigns, predict on 202 holdout.
"""
from __future__ import annotations

import time
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    # fallback: tqdm-заглушка, если не установлен
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            print(f"  [{desc}] step {i+1}/{total or '?'}", flush=True)
            yield item

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.config import ALPHA, INNER_CALIB_FRAC, SEED, TARGETS
from core.ci import ConformalEvaluation, aggregate_cp_coverage, seeded_train_calib_split
from core.conformal import MultiTargetSplitCP
from core.metrics import (
    monotonicity_violation,
    monotonicity_violation_per_pair,
    smlar,
    smlar_per_target,
)

from .data import (
    CampaignSetDataset,
    collate_pad,
    precompute_campaign_features,
)
from .train import BASE_CONFIG, _build_model, get_device, train_one_fold


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_indices(
    state_dict: dict,
    norm: dict,
    config: dict,
    campaigns: pd.DataFrame,
    targets: pd.DataFrame,
    users_df: pd.DataFrame,
    hist_index,
    indices: np.ndarray,
) -> np.ndarray:
    """Run inference on an arbitrary index set using the best-epoch weights.

    `targets` is only needed to satisfy CampaignSetDataset's constructor; values are
    ignored downstream (model output is the only thing we use).
    """
    device = get_device()
    model = _build_model(config).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    model.eval()
    ds = CampaignSetDataset(
        campaigns.iloc[indices], users_df, hist_index, targets.iloc[indices],
        norm["camp_mean"], norm["camp_std"], norm["user_mean"], norm["user_std"],
    )
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_pad)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch)
            preds.append(out["y_hat"].detach().cpu().numpy())
    return np.clip(np.concatenate(preds, axis=0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(
    variant_name: str,
    variant_overrides: dict,
    *,
    data: dict,
    alpha: float = ALPHA,
    verbose: bool = False,
    do_cv: bool = True,
    do_holdout: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Full-metric evaluation of one ablation variant.

    Returns a dict with `cv`, `holdout`, `ci` payloads matching the keys consumed by
    core.reporting.make_result.
    """
    config = dict(BASE_CONFIG, **variant_overrides)
    campaigns = precompute_campaign_features(data["val"])
    hist_index = data["user_index"]
    users_df = data["users"]
    ans = data["ans"]
    split = data["split"]

    train_idx = split["train_idx"]
    holdout_idx = split["holdout_idx"]
    folds = split["folds"]

    # ---------------- CV ----------------
    cv_payload, ci_payload, cv_fold_records = None, None, None
    if do_cv:
        fold_smlar, fold_pt, fold_mono, fold_mp = [], [], [], []
        fold_records, cp_evals = [], []
        fold_iter = tqdm(
            enumerate(folds),
            total=len(folds),
            desc=f"{variant_name} CV",
            leave=True,
        )
        for fi, (tr, va) in fold_iter:
            print(f"\n  [{variant_name}] fold {fi+1}/{len(folds)} — обучение...", flush=True)
            t0 = time.time()
            sub_loc, cal_loc = seeded_train_calib_split(len(tr), INNER_CALIB_FRAC, SEED + fi)
            sub_idx = tr[sub_loc]
            cal_idx = tr[cal_loc]
            best, _, norm = train_one_fold(
                campaigns, ans, users_df, hist_index, sub_idx, va,
                config=config, seed=SEED + fi, verbose=verbose,
            )
            p_va = best["pred"]
            y_va = best["true"]
            p_cal = _predict_indices(
                best["state_dict"], norm, config,
                campaigns, ans, users_df, hist_index, cal_idx,
            )
            y_cal = ans.iloc[cal_idx][list(TARGETS)].values.astype(np.float64)

            sm_flat = float(smlar(y_va, p_va))
            pt = smlar_per_target(y_va, p_va)
            mono = float(monotonicity_violation(p_va))
            mp = monotonicity_violation_per_pair(p_va)

            cp = MultiTargetSplitCP(alpha=alpha).calibrate(y_cal, p_cal)
            lo, hi = cp.predict(p_va)
            cp_evals.append(ConformalEvaluation(
                y_true=y_va.astype(np.float64),
                y_pred=p_va.astype(np.float64),
                lower=lo, upper=hi,
                quantiles_per_target=[float(c.quantile_) for c in cp.per_target_],
            ))

            fold_smlar.append(sm_flat)
            fold_pt.append(pt)
            fold_mono.append(mono)
            fold_mp.append(mp)
            elapsed = time.time() - t0
            fold_records.append({
                "fold": fi, "smlar": sm_flat, "best_epoch": best["epoch"],
                "smlar_y1": pt["at_least_one"], "smlar_y2": pt["at_least_two"],
                "smlar_y3": pt["at_least_three"],
                "monotone_violation": mono,
                "mono_y1_vs_y2": mp["y1_vs_y2"], "mono_y2_vs_y3": mp["y2_vs_y3"],
                "time_s": elapsed,
            })
            if verbose:
                fold_iter.set_postfix({
                    "ep": best["epoch"],
                    "SMLAR": f"{sm_flat:.2f}%",
                    "t": f"{elapsed:.0f}s",
                })
            print(
                f"  [{variant_name}] fold {fi+1}/{len(folds)} ✓  "
                f"ep={best['epoch']}  SMLAR={sm_flat:.2f}%  "
                f"y1={pt['at_least_one']:.2f}  y2={pt['at_least_two']:.2f}  "
                f"y3={pt['at_least_three']:.2f}  mono={mono*100:.2f}%  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

        cp_agg = aggregate_cp_coverage(cp_evals)
        cv_payload = {
            "smlar_flat_mean": float(np.mean(fold_smlar)),
            "smlar_flat_std": float(np.std(fold_smlar, ddof=0)),
            "smlar_per_target": {
                t: float(np.mean([d[t] for d in fold_pt])) for t in TARGETS
            },
            "smlar_per_target_std": {
                t: float(np.std([d[t] for d in fold_pt], ddof=0)) for t in TARGETS
            },
            "monotone_viol_mean": float(np.mean(fold_mono)),
            "monotone_viol_per_pair": {
                k: float(np.mean([d[k] for d in fold_mp]))
                for k in ("y1_vs_y2", "y2_vs_y3")
            },
        }
        ci_payload = {
            "type": "split_cp",
            "alpha": alpha,
            "coverage_per_target": cp_agg["coverage_per_target"],
            "width_per_target": cp_agg["width_per_target"],
        }
        cv_fold_records = fold_records

    # ---------------- Holdout ----------------
    ho_payload = None
    ckpt_path = None
    if do_holdout:
        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir) / f"holdout_{variant_name}.pt"
        print(f"\n  [{variant_name}] holdout — обучение на всём train ({len(train_idx)} кампаний)...", flush=True)
        t0 = time.time()
        best_ho, _, _ = train_one_fold(
            campaigns, ans, users_df, hist_index, train_idx, holdout_idx,
            config=config, seed=SEED, verbose=verbose,
            save_checkpoint=ckpt_path,
        )
        elapsed = time.time() - t0
        p_ho = best_ho["pred"]
        y_ho = best_ho["true"]
        ho_payload = {
            "smlar_flat": float(best_ho["smlar"]),
            "smlar_per_target": smlar_per_target(y_ho, p_ho),
            "monotone_viol": float(monotonicity_violation(p_ho)),
            "monotone_viol_per_pair": monotonicity_violation_per_pair(p_ho),
            "best_epoch": int(best_ho["epoch"]),
            "elapsed_s": float(elapsed),
        }
        if verbose:
            print(
                f"  [{variant_name}] holdout ✓  "
                f"ep={best_ho['epoch']}  SMLAR={best_ho['smlar']:.2f}%  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    return {
        "variant": variant_name,
        "config": config,
        "cv": cv_payload,
        "holdout": ho_payload,
        "ci": ci_payload,
        "cv_fold_records": cv_fold_records,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def fold_records_to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
