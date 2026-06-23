from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from core.config import RESULTS_DIR, RESULTS_PER_METHOD_DIR


SCHEMA_VERSION = "1.0"


def make_result(
    method: str,
    split_protocol: str,
    leak_safe: bool,
    features: str,
    cv_smlar_flat_mean: float,
    cv_smlar_flat_std: float,
    cv_smlar_per_target: dict[str, float] | None,
    cv_monotone_viol_mean: float,
    holdout_smlar: float | None,
    holdout_monotone_viol: float | None,
    ci_type: str,
    ci_alpha: float,
    ci_coverage_per_target: dict[str, float] | None,
    ci_width_per_target: dict[str, float] | None,
    source: str,
    extra: dict | None = None,
) -> dict:
    out: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "method": method,
        "split_protocol": split_protocol,
        "leak_safe": bool(leak_safe),
        "features": features,
        "cv": {
            "smlar_flat_mean": _to_py(cv_smlar_flat_mean),
            "smlar_flat_std": _to_py(cv_smlar_flat_std),
            "smlar_per_target": _to_py(cv_smlar_per_target) if cv_smlar_per_target else None,
            "monotone_viol_mean": _to_py(cv_monotone_viol_mean),
        },
        "holdout": {
            "smlar_flat": _to_py(holdout_smlar),
            "monotone_viol": _to_py(holdout_monotone_viol),
        },
        "ci": {
            "type": ci_type,
            "alpha": ci_alpha,
            "coverage_per_target": _to_py(ci_coverage_per_target) if ci_coverage_per_target else None,
            "width_per_target": _to_py(ci_width_per_target) if ci_width_per_target else None,
        },
        "source": source,
        "date": str(date.today()),
    }
    if extra:
        out["extra"] = _to_py(extra)
    return out


def _to_py(v):
    if isinstance(v, dict):
        return {k: _to_py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def save_method_results(method_name: str, result: dict) -> Path:
    RESULTS_PER_METHOD_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_PER_METHOD_DIR / f"{method_name}.json"
    md_path = RESULTS_PER_METHOD_DIR / f"{method_name}.md"
    json_path.write_text(json.dumps(result, indent=2))
    md_path.write_text(result_to_markdown(result))
    return json_path


def result_to_markdown(r: dict) -> str:
    lines = [
        f"# {r['method']}",
        "",
        f"- **Split protocol:** {r['split_protocol']}",
        f"- **Leak-safe:** {r['leak_safe']}",
        f"- **Features:** {r['features']}",
        f"- **Source:** {r['source']}",
        f"- **Date:** {r['date']}",
        "",
        "## Cross-validation (5 folds)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| SMLAR (flat, mean) | {r['cv']['smlar_flat_mean']:.2f}% |",
        f"| SMLAR (flat, std) | {r['cv']['smlar_flat_std']:.2f} |",
        f"| Monotone violation | {r['cv']['monotone_viol_mean'] * 100:.2f}% |",
    ]
    if r["cv"].get("smlar_per_target"):
        lines += ["", "### SMLAR per target", "", "| Target | SMLAR |", "|---|---|"]
        for t, v in r["cv"]["smlar_per_target"].items():
            lines.append(f"| {t} | {v:.2f}% |")
    lines += ["", "## Holdout (20% time-based)", ""]
    h = r.get("holdout") or {}
    if h.get("smlar_flat") is not None:
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| SMLAR (flat) | {h['smlar_flat']:.2f}% |",
            f"| Monotone violation | {(h.get('monotone_viol') or 0.0) * 100:.2f}% |",
        ]
    else:
        lines.append("_Not run for this method._")
    lines += [
        "",
        f"## Confidence intervals ({r['ci']['type']}, α = {r['ci']['alpha']})",
        "",
    ]
    if r["ci"].get("coverage_per_target"):
        lines += [
            "| Target | Coverage | Mean width |",
            "|---|---|---|",
        ]
        for t in r["ci"]["coverage_per_target"]:
            cov = r["ci"]["coverage_per_target"][t]
            wid = (r["ci"].get("width_per_target") or {}).get(t, float("nan"))
            lines.append(f"| {t} | {cov:.3f} | {wid:.4f} |")
    else:
        lines.append("_No CI for this method._")
    return "\n".join(lines) + "\n"


def collect_method_results() -> list[dict]:
    out = []
    if RESULTS_PER_METHOD_DIR.exists():
        for f in sorted(RESULTS_PER_METHOD_DIR.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                continue
    expanded: list[dict] = []
    for r in out:
        variants = (r.get("extra") or {}).get("variants")
        if variants:
            for v in variants:
                row = {
                    **r,
                    "method": v.get("name", r["method"]),
                    "cv": v.get("cv", r["cv"]),
                    "holdout": v.get("holdout", r.get("holdout", {})),
                    "ci": v.get("ci", r.get("ci", {})),
                }
                expanded.append(row)
        else:
            expanded.append(r)
    return expanded


def write_final_comparison() -> tuple[Path, Path]:
    rows = collect_method_results()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_rows = []
    for r in rows:
        cv = r.get("cv", {})
        ho = r.get("holdout", {}) or {}
        ci = r.get("ci", {}) or {}
        cov = ci.get("coverage_per_target") or {}
        wid = ci.get("width_per_target") or {}
        cov_mean = float(np.mean(list(cov.values()))) if cov else float("nan")
        wid_mean = float(np.mean(list(wid.values()))) if wid else float("nan")
        df_rows.append({
            "method": r["method"],
            "leak_safe": r.get("leak_safe"),
            "cv_smlar_mean": cv.get("smlar_flat_mean"),
            "cv_smlar_std": cv.get("smlar_flat_std"),
            "cv_monotone_viol": cv.get("monotone_viol_mean"),
            "holdout_smlar": ho.get("smlar_flat"),
            "holdout_monotone_viol": ho.get("monotone_viol"),
            "ci_type": ci.get("type"),
            "ci_alpha": ci.get("alpha"),
            "ci_coverage_mean": cov_mean,
            "ci_width_mean": wid_mean,
        })
    df = pd.DataFrame(df_rows)
    csv_path = RESULTS_DIR / "final_comparison.csv"
    md_path = RESULTS_DIR / "final_comparison.md"
    df.to_csv(csv_path, index=False)

    md = ["# Final comparison — all methods on unified leak-safe protocol", "",
          "All methods evaluated on the SAME 806 train / 202 holdout campaign split (time-based)",
          "with 5-fold campaign CV on train (seed=42) and Split Conformal Prediction at α=0.10.",
          "",
          "| Method | Leak-safe | CV SMLAR % | Holdout SMLAR % | Mono violation % | CP coverage | CP width |",
          "|---|---|---|---|---|---|---|"]
    for r in df_rows:
        md.append(
            f"| {r['method']} | "
            f"{'' if r['leak_safe'] else ''} | "
            f"{_fmt(r['cv_smlar_mean'])} ± {_fmt(r['cv_smlar_std'])} | "
            f"{_fmt(r['holdout_smlar'])} | "
            f"{_fmt_pct(r['cv_monotone_viol'])} | "
            f"{_fmt(r['ci_coverage_mean'], digits=3)} | "
            f"{_fmt(r['ci_width_mean'], digits=4)} |"
        )
    md_path.write_text("\n".join(md) + "\n")
    return md_path, csv_path


def _fmt(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x * 100:.2f}"
