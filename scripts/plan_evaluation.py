from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_plan(config: dict) -> dict:
    output = Path(config["project"]["output_dir"])
    root = output.parent
    runs = root / "runs"
    merged = root / "merged_v5_math"
    protocol_id = str(config["project"]["protocol_id"])

    def run_id(name: str) -> str:
        return f"{protocol_id}_{name}"

    def model_id(name: str) -> str:
        return f"{protocol_id}_{name}"

    horizon = int(config["training"]["branch_horizon"])
    endpoints = []
    for checkpoint in config["training"]["branch_points"]:
        target = checkpoint + horizon
        opd_run = run_id("discovery_trunk_opd") if checkpoint == 0 else run_id(f"branch_t{checkpoint}_opd")
        opd_actor = runs / opd_run / "checkpoint" / f"global_step_{target}" / "actor"
        branch_actor = runs / run_id(f"branch_t{checkpoint}_rl") / "checkpoint" / f"global_step_{target}" / "actor"
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
                    "arm": "rl",
                    "branch_point": checkpoint,
                    "target_step": target,
                    "actor_dir": str(branch_actor),
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
    primary_path = config["paths"]["external_eval"][0]
    eval_specs = []

    def add_eval(model: dict, panel: str, dataset: str, n: int) -> None:
        eval_id = f"{model['model_id']}__{panel.lower()}"
        output_dir = root / "evaluations_v5" / eval_id
        argv = [
            "python",
            "scripts/evaluate_math.py",
            "--config",
            "configs/experiment.yaml",
            "--model-path",
            model["merged_model"],
            "--dataset",
            dataset,
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
                "dataset": dataset,
                "n": n,
                "output_dir": str(output_dir),
                "argv": argv,
            }
        )

    for model in models:
        add_eval(model, "MATH-500", primary_path, config["evaluation"]["primary_n"])
    terminal_ids = {
        model_id("branch_t40_opd_t50"),
        model_id("branch_t40_rl_t50"),
    }
    external_paths = config["paths"]["external_eval"][1:]
    for model in models:
        if model["model_id"] not in terminal_ids:
            continue
        for dataset in external_paths:
            panel = Path(dataset).parent.name
            add_eval(model, panel, dataset, config["evaluation"]["external_n"])
    merge_specs = [
        {
            "model_id": model["model_id"],
            "actor_dir": model["actor_dir"],
            "target_dir": model["merged_model"],
            "argv": [
                "python",
                "scripts/merge_checkpoint.py",
                "--config",
                "configs/experiment.yaml",
                "--actor-dir",
                model["actor_dir"],
                "--target-dir",
                model["merged_model"],
            ],
        }
        for model in models
        if model["merge_required"]
    ]
    return {
        "models": models,
        "merge_specs": merge_specs,
        "evaluation_specs": eval_specs,
        "primary_comparisons": [
            {
                "branch_point": checkpoint,
                "opd_model_id": (
                    model_id(f"trunk_opd_t{checkpoint + horizon}")
                    if checkpoint == 0
                    else model_id(f"branch_t{checkpoint}_opd_t{checkpoint + horizon}")
                ),
                "rl_model_id": model_id(f"branch_t{checkpoint}_rl_t{checkpoint + horizon}"),
            }
            for checkpoint in config["training"]["branch_points"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = build_plan(config)
    output = Path(config["project"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation_plan_v5.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(plan["models"]), "evaluations": len(plan["evaluation_specs"]), "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
