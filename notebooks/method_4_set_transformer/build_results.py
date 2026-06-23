"""Build unified results.json for Set Transformer from full-metric evaluation.

Two code paths:

  1. `build_results_from_eval(eval_results)` — preferred. Takes a list of dicts
     produced by `src.eval.evaluate_variant` (one per variant) and produces a
     unified result with A_full as primary + all variants in `extra.variants`.
     This path gives full per-target / per-pair / CP coverage / HO numbers.

  2. `build_results_from_frozen(source_sha)` — fallback. Reads the frozen Colab
     summary (results_frozen/ablation_summary_frozen.csv) and produces a result
     with flat CV numbers only (per-target / per-pair / CP all `null`). Kept for
     reference / regression testing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import ALPHA, RESULTS_PER_METHOD_DIR as METHODS_DIR
from core.reporting import make_result, save_method_results


FROZEN_DIR = Path(__file__).parent / "results_frozen"
PRIMARY_VARIANT = "A_full"
PROTOCOL = (
    "time-based 80/20 (sort by hour_start) + 5-fold campaign CV "
    "(seed=42) — same protocol as core/splits.py"
)
FEATURES = "leak-safe set of users + per-user history features"


# ---------------------------------------------------------------------------
# Preferred path — full local evaluation
# ---------------------------------------------------------------------------

def _variant_block(ev: dict) -> dict:
    """Render one eval result into the `extra.variants[]` schema."""
    cfg = ev["config"]
    cv = ev.get("cv") or {}
    ho = ev.get("holdout") or {}
    ci = ev.get("ci") or {}
    return {
        "name": f"set_transformer-{ev['variant']}",
        "cv": {
            "smlar_flat_mean": cv.get("smlar_flat_mean"),
            "smlar_flat_std": cv.get("smlar_flat_std"),
            "smlar_per_target": cv.get("smlar_per_target"),
            "smlar_per_target_std": cv.get("smlar_per_target_std"),
            "monotone_viol_mean": cv.get("monotone_viol_mean"),
            "monotone_viol_per_pair": cv.get("monotone_viol_per_pair"),
        },
        "holdout": {
            "smlar_flat": ho.get("smlar_flat"),
            "smlar_per_target": ho.get("smlar_per_target"),
            "monotone_viol": ho.get("monotone_viol"),
            "monotone_viol_per_pair": ho.get("monotone_viol_per_pair"),
            "best_epoch": ho.get("best_epoch"),
        },
        "ci": {
            "type": ci.get("type", "not_computed"),
            "alpha": ci.get("alpha"),
            "coverage_per_target": ci.get("coverage_per_target"),
            "width_per_target": ci.get("width_per_target"),
        },
        "use_attention": bool(cfg["use_attention"]),
        "use_distribution": bool(cfg["use_distribution"]),
        "loss": cfg["loss"],
    }


def build_results_from_eval(
    eval_results: list[dict],
    *,
    primary_variant: str = PRIMARY_VARIANT,
    source: str = "local-cpu eval via src/eval.py",
    note: str | None = None,
) -> dict:
    """Assemble the unified result.json from per-variant eval payloads.

    `eval_results` is a list of dicts returned by `evaluate_variant`. Order is
    preserved in `extra.variants`. The variant whose name matches
    `primary_variant` is also used for the top-level cv/holdout/ci numbers.
    """
    by_name = {ev["variant"]: ev for ev in eval_results}
    if primary_variant not in by_name:
        raise ValueError(
            f"Primary variant {primary_variant!r} not found in eval_results; "
            f"have {sorted(by_name)}"
        )
    primary = by_name[primary_variant]
    cv = primary.get("cv") or {}
    ho = primary.get("holdout") or {}
    ci = primary.get("ci") or {}

    variants = [_variant_block(ev) for ev in eval_results]

    extra = {
        "variants": variants,
        "note": note or (
            "Full-metric local CPU evaluation via src/eval.py. "
            "CV: 5-fold campaign CV with per-fold inner sub/calib split for Split CP. "
            "Holdout: trained on full 806 train campaigns. "
            "Monotonicity is 0 by construction for variants with use_distribution=True "
            "(tail-cumsum head guarantees y1 >= y2 >= y3)."
        ),
        "holdout_smlar_per_target": ho.get("smlar_per_target"),
        "holdout_monotone_viol_per_pair": ho.get("monotone_viol_per_pair"),
        "cv_smlar_per_target_std": cv.get("smlar_per_target_std"),
        "cv_monotone_viol_per_pair": cv.get("monotone_viol_per_pair"),
    }

    result = make_result(
        method=f"set_transformer-{primary_variant}",
        split_protocol=PROTOCOL,
        leak_safe=True,
        features=FEATURES,
        cv_smlar_flat_mean=cv.get("smlar_flat_mean"),
        cv_smlar_flat_std=cv.get("smlar_flat_std"),
        cv_smlar_per_target=cv.get("smlar_per_target"),
        cv_monotone_viol_mean=cv.get("monotone_viol_mean"),
        holdout_smlar=ho.get("smlar_flat"),
        holdout_monotone_viol=ho.get("monotone_viol"),
        ci_type=ci.get("type", "split_cp"),
        ci_alpha=ci.get("alpha", ALPHA),
        ci_coverage_per_target=ci.get("coverage_per_target"),
        ci_width_per_target=ci.get("width_per_target"),
        source=source,
        extra=extra,
    )
    return result


# ---------------------------------------------------------------------------
# Fallback path — frozen Colab CSV
# ---------------------------------------------------------------------------

def build_results_from_frozen(source_sha: str = "4376bab") -> dict:
    """Legacy: build result.json from the frozen ablation summary (CV-only, flat)."""
    summary = pd.read_csv(FROZEN_DIR / "ablation_summary_frozen.csv")
    summary = summary.sort_values("cv_smlar_mean").reset_index(drop=True)

    a_full = summary[summary["variant"] == PRIMARY_VARIANT].iloc[0]
    variants = []
    for _, row in summary.iterrows():
        variants.append({
            "name": f"set_transformer-{row['variant']}",
            "cv": {
                "smlar_flat_mean": float(row["cv_smlar_mean"]),
                "smlar_flat_std": float(row["cv_smlar_std"]),
                "smlar_per_target": None,
                "monotone_viol_mean": float(row["monotone_violation"]),
            },
            "holdout": {"smlar_flat": None, "monotone_viol": None},
            "ci": {
                "type": "not_computed", "alpha": None,
                "coverage_per_target": None, "width_per_target": None,
            },
            "use_attention": bool(row["use_attention"]),
            "use_distribution": bool(row["use_distribution"]),
            "loss": str(row["loss"]),
        })

    result = make_result(
        method=f"set_transformer-{PRIMARY_VARIANT}",
        split_protocol=PROTOCOL,
        leak_safe=True,
        features=FEATURES,
        cv_smlar_flat_mean=float(a_full["cv_smlar_mean"]),
        cv_smlar_flat_std=float(a_full["cv_smlar_std"]),
        cv_smlar_per_target=None,
        cv_monotone_viol_mean=float(a_full["monotone_violation"]),
        holdout_smlar=None,
        holdout_monotone_viol=None,
        ci_type="not_computed",
        ci_alpha=ALPHA,
        ci_coverage_per_target=None,
        ci_width_per_target=None,
        source=f"results_frozen/ablation_summary_frozen.csv (logs/ablation.log @ {source_sha})",
        extra={
            "variants": variants,
            "note": (
                "CV numbers frozen from logs/ablation.log (original 5-fold Colab run). "
                "Holdout / CP / per-target NOT available from frozen log — re-run "
                "via src/eval.py:evaluate_variant for the full metric set."
            ),
        },
    )
    return result


# Kept for backward compat with the existing notebook cell.
build_results_json = build_results_from_frozen


def main():
    result = build_results_from_frozen()
    save_method_results("set_transformer", result)
    print("Wrote methods/set_transformer/results.{json,md} from frozen CSV.")
    print(f"  A_full CV SMLAR: {result['cv']['smlar_flat_mean']:.2f}% ± {result['cv']['smlar_flat_std']:.2f}")
    print(f"  Monotone violations: {result['cv']['monotone_viol_mean'] * 100:.2f}% (by construction)")
    print(f"  {len(result['extra']['variants'])} ablation variants stored in extra.")


if __name__ == "__main__":
    main()
