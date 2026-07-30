from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


ENDPOINT_FIELDS = (
    "avg_at_n",
    "pass_at_n",
    "all_fail_rate",
    "all_correct_rate",
    "nondegenerate_group_rate",
    "format_rate",
    "mean_tokens",
    "cap_hit_rate",
    "generation_seconds",
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _flatten_scalars(value: Mapping[str, Any], prefix: str = "signal") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}"
        if isinstance(item, Mapping):
            output.update(_flatten_scalars(item, name))
        elif isinstance(item, bool):
            output[name] = float(item)
        elif isinstance(item, (int, float)) and math.isfinite(float(item)):
            output[name] = float(item)
    return output


def _evaluation_index(plan: Mapping[str, Any], panel: str) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for spec in plan.get("evaluation_specs", []):
        if spec.get("panel") != panel:
            continue
        model_id = str(spec["model_id"])
        if model_id in index:
            raise ValueError(f"duplicate evaluation for model={model_id}, panel={panel}")
        index[model_id] = spec
    return index


def _load_summary(spec: Mapping[str, Any], root: Path) -> dict:
    output_dir = Path(str(spec["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    path = output_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_json(path)


def _validate_pair(left: Mapping[str, Any], right: Mapping[str, Any], panel: str) -> None:
    for name, summary in (("opd", left), ("grpo", right)):
        if summary.get("panel") != panel:
            raise ValueError(f"{name} summary panel does not match {panel}")
    for key in ("dataset_sha256", "rows", "n", "max_tokens", "seed"):
        if left.get(key) != right.get(key):
            raise ValueError(f"paired endpoint mismatch for {key}: {left.get(key)!r} != {right.get(key)!r}")


def build_surface(
    plan: Mapping[str, Any],
    *,
    root: Path,
    panel: str | None = None,
    metric: str = "avg_at_n",
    signals: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    panel = panel or str(plan.get("primary_panel", ""))
    if not panel:
        raise ValueError("panel is required when the evaluation plan has no primary_panel")
    evaluations = _evaluation_index(plan, panel)
    comparisons = plan.get("primary_comparisons", [])
    if not comparisons:
        raise ValueError("evaluation plan contains no primary comparisons")

    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        checkpoint = int(comparison["branch_point"])
        opd_id = str(comparison["opd_model_id"])
        grpo_id = str(comparison["rl_model_id"])
        if opd_id not in evaluations or grpo_id not in evaluations:
            raise ValueError(f"missing paired evaluation specs at branch point {checkpoint}")
        opd = _load_summary(evaluations[opd_id], root)
        grpo = _load_summary(evaluations[grpo_id], root)
        _validate_pair(opd, grpo, panel)
        if metric not in opd or metric not in grpo:
            raise KeyError(f"endpoint metric {metric!r} is absent at branch point {checkpoint}")

        row: dict[str, Any] = {
            "branch_point": checkpoint,
            "horizon": int(comparison.get("horizon", plan.get("branch_horizon", 0))),
            "target_step": int(comparison.get("target_step", checkpoint)),
            "panel": panel,
            "metric": metric,
            "opd_model_id": opd_id,
            "grpo_model_id": grpo_id,
            "opd_endpoint": float(opd[metric]),
            "grpo_endpoint": float(grpo[metric]),
            "grpo_minus_opd": float(grpo[metric]) - float(opd[metric]),
        }
        for field in ENDPOINT_FIELDS:
            if field in opd and field in grpo:
                row[f"opd_{field}"] = float(opd[field])
                row[f"grpo_{field}"] = float(grpo[field])
        if signals is not None:
            signal_value = signals.get(str(checkpoint), signals.get(checkpoint))
            if signal_value is not None:
                if not isinstance(signal_value, Mapping):
                    raise TypeError(f"signals[{checkpoint}] must be an object")
                row.update(_flatten_scalars(signal_value))
        rows.append(row)
    return sorted(rows, key=lambda item: item["branch_point"])


def write_surface(rows: list[dict[str, Any]], output_dir: Path, panel: str, metric: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "handoff_surface.json"
    csv_path = output_dir / "handoff_surface.csv"
    payload = {"schema_version": 1, "panel": panel, "metric": metric, "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OPD-to-GRPO handoff surface from paired endpoints.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--panel")
    parser.add_argument("--metric", default="avg_at_n")
    parser.add_argument("--signals", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = _load_json(args.plan)
    signals = _load_json(args.signals) if args.signals else None
    panel = args.panel or str(plan.get("primary_panel", ""))
    rows = build_surface(plan, root=args.root, panel=panel, metric=args.metric, signals=signals)
    paths = write_surface(rows, args.output_dir, panel, args.metric)
    print(json.dumps({"comparisons": len(rows), **paths}, indent=2))


if __name__ == "__main__":
    main()
