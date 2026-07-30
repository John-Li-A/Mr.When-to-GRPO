from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_KEYS = (
    "schema_version",
    "project.protocol_id",
    "project.output_dir",
    "paths.student_model",
    "paths.teacher_model",
    "paths.train_data",
    "paths.source_root",
    "paths.verl_root",
    "paths.external_eval",
    "data.prompt_batch_size",
    "rollout.n",
    "rollout.max_prompt_length",
    "rollout.max_response_length",
    "training.opd_estimator",
    "training.rl_estimator",
    "training.trunk_updates",
    "training.branch_points",
    "training.branch_horizon",
    "evaluation.primary_panel",
    "evaluation.external_panels",
    "evaluation.primary_n",
    "evaluation.max_response_length",
    "runtime.n_gpus_per_node",
    "runtime.model_dtype",
    "runtime.rollout_gpu_memory_utilization",
    "runtime.actor_micro_batch_size_per_gpu",
    "runtime.teacher_micro_batch_size_per_gpu",
)


def _get(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def inspect_config(config: Mapping[str, Any], *, check_paths: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED_KEYS:
        try:
            _get(config, key)
        except KeyError:
            errors.append(f"missing required key: {key}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    if int(config["schema_version"]) != 1:
        errors.append("schema_version must be 1")
    n = int(_get(config, "rollout.n"))
    if n < 2:
        errors.append("rollout.n must be at least 2 for group-relative RL")
    prompt_batch = int(_get(config, "data.prompt_batch_size"))
    if prompt_batch <= 0:
        errors.append("data.prompt_batch_size must be positive")
    response_cap = int(_get(config, "rollout.max_response_length"))
    if response_cap <= 0:
        errors.append("rollout.max_response_length must be positive")
    if int(_get(config, "evaluation.max_response_length")) != response_cap:
        errors.append("training and evaluation response caps must match")

    points = [int(value) for value in _get(config, "training.branch_points")]
    if not points:
        errors.append("training.branch_points must contain at least one candidate checkpoint")
    elif points != sorted(set(points)) or any(value < 0 for value in points):
        errors.append("training.branch_points must be sorted, unique, and non-negative")
    horizon = int(_get(config, "training.branch_horizon"))
    trunk = int(_get(config, "training.trunk_updates"))
    if horizon <= 0:
        errors.append("training.branch_horizon must be positive")
    if points and max(points) + horizon > trunk:
        errors.append("the last branch endpoint exceeds training.trunk_updates")
    saved = {int(value) for value in config.get("training", {}).get("checkpoint_updates", [])}
    missing_checkpoints = sorted(value for value in points if value > 0 and value not in saved)
    if missing_checkpoints:
        errors.append(f"branch points are not present in training.checkpoint_updates: {missing_checkpoints}")

    forbidden = set(config.get("training", {}).get("forbidden_estimators", []))
    for key in ("training.opd_estimator", "training.rl_estimator"):
        if str(_get(config, key)) in forbidden:
            errors.append(f"{key} is listed as forbidden")
    if str(_get(config, "training.opd_estimator")) == str(_get(config, "training.rl_estimator")):
        errors.append("OPD and RL estimators must differ")

    if int(_get(config, "runtime.n_gpus_per_node")) < 1:
        errors.append("runtime.n_gpus_per_node must be positive")
    utilization = float(_get(config, "runtime.rollout_gpu_memory_utilization"))
    if not 0 < utilization <= 1:
        errors.append("runtime.rollout_gpu_memory_utilization must be in (0, 1]")
    for key in ("runtime.actor_micro_batch_size_per_gpu", "runtime.teacher_micro_batch_size_per_gpu"):
        if int(_get(config, key)) < 1:
            errors.append(f"{key} must be positive")

    path_keys = ("student_model", "teacher_model", "train_data", "source_root", "verl_root")
    for key in path_keys:
        raw = config.get("paths", {}).get(key)
        if raw is None:
            continue
        if "CHANGE_ME" in str(raw):
            message = f"unresolved path placeholder: paths.{key}"
            (errors if check_paths else warnings).append(message)
        elif check_paths and not Path(str(raw)).exists():
            errors.append(f"path does not exist: paths.{key}={raw}")

    eval_paths = list(config.get("paths", {}).get("external_eval", []))
    panels = list(config.get("evaluation", {}).get("external_panels", []))
    if len(eval_paths) != len(panels):
        errors.append("paths.external_eval and evaluation.external_panels must have equal length")
    if len(set(map(str, panels))) != len(panels):
        errors.append("evaluation.external_panels must be unique")
    if int(config.get("evaluation", {}).get("primary_n", 0)) < 1:
        errors.append("evaluation.primary_n must be positive")
    if float(config.get("evaluation", {}).get("equivalence_band", 0.0)) < 0:
        errors.append("evaluation.equivalence_band must be non-negative")
    primary = str(_get(config, "evaluation.primary_panel"))
    if panels and primary not in panels:
        errors.append("evaluation.primary_panel must appear in evaluation.external_panels")
    for index, raw in enumerate(eval_paths):
        if "CHANGE_ME" in str(raw):
            message = f"unresolved path placeholder: paths.external_eval[{index}]"
            (errors if check_paths else warnings).append(message)
        elif check_paths and not Path(str(raw)).exists():
            errors.append(f"evaluation path does not exist: {raw}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "protocol": {
            "branch_points": points,
            "branch_horizon": horizon,
            "trunk_updates": trunk,
            "prompt_batch_size": prompt_batch,
            "n": n,
            "response_cap": response_cap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a composed When-to-GRPO configuration.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = inspect_config(config, check_paths=args.check_paths)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
