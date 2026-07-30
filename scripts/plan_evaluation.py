from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_plan(config: dict, config_path: str = "configs/experiment.yaml") -> dict:
    output = Path(config["project"]["output_dir"])
    root = output.parent
    runs = root / "runs"
    merged = root / "merged"
    protocol_id = str(config["project"]["protocol_id"])

    def run_id(name: str) -> str:
        return f"{protocol_id}_{name}"

    def model_id(name: str) -> str:
        return f"{protocol_id}_{name}"

    horizon = int(config["training"]["branch_horizon"])
    branch_points = [int(value) for value in config["training"]["branch_points"]]
    endpoints = []
    for checkpoint in branch_points:
        target = checkpoint + horizon
        opd_run = run_id("discovery_trunk_opd") if checkpoint == 0 else run_id(f"branch_t{checkpoint}_opd")
        opd_actor = runs / opd_run / "checkpoint" / f"global_step_{target}" / "actor"
        rl_actor = runs / run_id(f"branch_t{checkpoint}_rl") / "checkpoint" / f"global_step_{target}" / "actor"
        opd_model_id = (
            model_id(f"trunk_opd_t{target}")
            if checkpoint == 0
            else model_id(f"branch_t{checkpoint}_opd_t{target}")
        )
        endpoints.extend(
            [
                {
                    "model_id": opd_model_id,
                    "arm": "opd",
                    "branch_point": checkpoint,
                    "target_step": target,
                    "actor_dir": str(opd_actor),
                    "merged_model": str(merged / opd_model_id),
                },
                {
                    "model_id": model_id(f"branch_t{checkpoint}_rl_t{target}"),
                    "arm": "grpo",
                    "branch_point": checkpoint,
                    "target_step": target,
                    "actor_dir": str(rl_actor),
                    "merged_model": str(merged / model_id(f"branch_t{checkpoint}_rl_t{target}")),
                },
            ]
        )

    models = [
        {
            "model_id": "student_base",
            "arm": "baseline",
            "merged_model": config["paths"]["student_model"],
            "merge_required": False,
        },
        {
            "model_id": "teacher_reference",
            "arm": "teacher_reference",
            "merged_model": config["paths"]["teacher_model"],
            "merge_required": False,
        },
        *[{**item, "merge_required": True} for item in endpoints],
    ]

    panels = list(config["evaluation"]["external_panels"])
    datasets = list(config["paths"]["external_eval"])
    if len(panels) != len(datasets):
        raise ValueError("evaluation panels and dataset paths must have equal length")
    panel_paths = dict(zip(panels, datasets, strict=True))
    primary_panel = str(config["evaluation"]["primary_panel"])
    if primary_panel not in panel_paths:
        raise ValueError("primary panel has no matching dataset path")

    eval_specs = []

    def add_eval(model: dict, panel: str, n: int) -> None:
        eval_id = f"{model['model_id']}__{panel.lower()}"
        output_dir = root / "evaluations" / eval_id
        argv = [
            "python",
            "scripts/evaluate.py",
            "--config",
            config_path,
            "--model-path",
            model["merged_model"],
            "--dataset",
            panel_paths[panel],
            "--panel",
            panel,
            "--n",
            str(n),
            "--output-dir",
            str(output_dir),
        ]
        eval_specs.append(
            {
                "eval_id": eval_id,
                "model_id": model["model_id"],
                "model_path": model["merged_model"],
                "panel": panel,
                "dataset": panel_paths[panel],
                "n": n,
                "output_dir": str(output_dir),
                "argv": argv,
            }
        )

    for model in models:
        add_eval(model, primary_panel, int(config["evaluation"]["primary_n"]))

    if branch_points:
        terminal_checkpoint = max(branch_points)
        terminal_target = terminal_checkpoint + horizon
        terminal_ids = {
            (
                model_id(f"trunk_opd_t{terminal_target}")
                if terminal_checkpoint == 0
                else model_id(f"branch_t{terminal_checkpoint}_opd_t{terminal_target}")
            ),
            model_id(f"branch_t{terminal_checkpoint}_rl_t{terminal_target}"),
        }
        for model in models:
            if model["model_id"] not in terminal_ids:
                continue
            for panel in panels:
                if panel != primary_panel:
                    add_eval(model, panel, int(config["evaluation"]["external_n"]))

    merge_specs = [
        {
            "model_id": model["model_id"],
            "actor_dir": model["actor_dir"],
            "target_dir": model["merged_model"],
            "argv": [
                "python",
                "scripts/merge_checkpoint.py",
                "--config",
                config_path,
                "--actor-dir",
                model["actor_dir"],
                "--target-dir",
                model["merged_model"],
            ],
        }
        for model in models
        if model["merge_required"]
    ]
    comparisons = []
    for checkpoint in branch_points:
        target = checkpoint + horizon
        comparisons.append(
            {
                "branch_point": checkpoint,
                "horizon": horizon,
                "target_step": target,
                "opd_model_id": (
                    model_id(f"trunk_opd_t{target}")
                    if checkpoint == 0
                    else model_id(f"branch_t{checkpoint}_opd_t{target}")
                ),
                "rl_model_id": model_id(f"branch_t{checkpoint}_rl_t{target}"),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "primary_panel": primary_panel,
        "primary_metric": "avg_at_n",
        "branch_horizon": horizon,
        "models": models,
        "merge_specs": merge_specs,
        "evaluation_specs": eval_specs,
        "primary_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = build_plan(config, str(args.config))
    output = Path(config["project"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(plan["models"]), "evaluations": len(plan["evaluation_specs"]), "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
