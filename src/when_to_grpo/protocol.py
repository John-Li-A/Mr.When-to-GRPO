from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .core import branch_prompt_window, canonical_json, guard_estimator, sha256_bytes, sha256_file


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_specs(config: dict) -> list[dict]:
    output = Path(config["project"]["output_dir"])
    runs_root = output.parent / "runs"
    protocol_id = str(config["project"].get("protocol_id", ""))
    if not protocol_id or not protocol_id.replace("_", "").isalnum():
        raise ValueError("project.protocol_id must be a non-empty filesystem-safe identifier")

    def run_id(name: str) -> str:
        return f"{protocol_id}_{name}"

    trunk_run_id = run_id("discovery_trunk_opd")
    manifest = load_json(output / "canonical_manifest.json")
    discovery_queue = load_json(output / "discovery_prompt_queue.json")
    training = config["training"]
    forbidden = training.get("forbidden_estimators", [])
    guard_estimator(training["opd_estimator"], forbidden)
    guard_estimator(training["rl_estimator"], forbidden)
    if config["rollout"]["max_response_length"] <= 0:
        raise ValueError("training response cap must be positive")
    if config["rollout"]["n"] < 2:
        raise ValueError("group-relative RL requires n >= 2")
    trajectories_per_update = config["data"]["prompt_batch_size"] * config["rollout"]["n"]
    mini_batch_size = int(training.get("ppo_mini_batch_size", config["data"]["prompt_batch_size"]))
    if trajectories_per_update % mini_batch_size:
        raise ValueError("trajectory batch must be divisible by ppo_mini_batch_size")
    optimizer_steps_per_update = (
        trajectories_per_update // mini_batch_size * int(training.get("ppo_epochs", 1))
    )

    def dose(updates: int, prompt_batches: list[list[str]]) -> dict:
        return {
            "trainer_updates": updates,
            "optimizer_steps": updates * optimizer_steps_per_update,
            "prompts": updates * config["data"]["prompt_batch_size"],
            "trajectories": updates * trajectories_per_update,
            "max_generated_tokens_upper_bound": (
                updates * trajectories_per_update * config["rollout"]["max_response_length"]
            ),
            "prompt_batches_sha256": sha256_bytes(canonical_json(prompt_batches).encode("utf-8")),
        }

    common = {
        "manifest_sha256": sha256_file(output / "canonical_manifest.json"),
        "n": config["rollout"]["n"],
        "max_response_length": config["rollout"]["max_response_length"],
        "prompt_batch_size": config["data"]["prompt_batch_size"],
        "dataloader_num_workers": config["data"]["dataloader_num_workers"],
        "rollout_seed": config["rollout"]["seeds"]["discovery"],
        "recipe_version": protocol_id,
    }
    trunk_batches = discovery_queue[: training["trunk_updates"]]
    specs = [{
        **common,
        "run_id": trunk_run_id,
        "phase": "discovery_trunk",
        "estimator": training["opd_estimator"],
        "objective": "opd",
        "start_step": 0,
        "updates": training["trunk_updates"],
        "target_global_step": training["trunk_updates"],
        "prompt_batches": trunk_batches,
        "planned_dose": dose(training["trunk_updates"], trunk_batches),
        "save_steps": training["checkpoint_updates"],
        "train_file": str(output / "discovery_schedule.parquet"),
        "resume_mode": "disable",
        "default_local_dir": str(runs_root / trunk_run_id / "checkpoint"),
        "final_checkpoint_actor_dir": str(
            runs_root / trunk_run_id / "checkpoint"
            / f"global_step_{training['trunk_updates']}" / "actor"
        ),
    }]
    counterfactual_mode = training.get("counterfactual_mode")
    if counterfactual_mode != "fork_resumed_opd":
        raise ValueError(
            "the paired protocol requires fork_resumed_opd so resumed OPD/GRPO arms "
            "share process-reset semantics"
        )
    reuse_fresh_t0 = bool(training.get("reuse_fresh_trunk_t0", False))
    for checkpoint in training["branch_points"]:
        future = branch_prompt_window(discovery_queue, checkpoint, training["branch_horizon"])
        arms = [("rl", training["rl_estimator"])]
        if checkpoint > 0 or not reuse_fresh_t0:
            arms.insert(0, ("opd", training["opd_estimator"]))
        for arm, estimator in arms:
            resume_path = runs_root / trunk_run_id / "checkpoint" / f"global_step_{checkpoint}"
            paired_opd_run_id = (
                trunk_run_id if checkpoint == 0 else run_id(f"branch_t{checkpoint}_opd")
            )
            branch_run_id = run_id(f"branch_t{checkpoint}_{arm}")
            specs.append({
                **common,
                "run_id": branch_run_id,
                "phase": "discovery_branch",
                "checkpoint": checkpoint,
                "estimator": estimator,
                "objective": training.get("rl_objective", "grpo") if arm == "rl" else "opd",
                "opd_counterfactual": (
                    {
                        "source_run_id": paired_opd_run_id,
                        "start_step": checkpoint,
                        "target_global_step": checkpoint + training["branch_horizon"],
                    }
                    if arm == "rl"
                    else None
                ),
                "paired_arm_run_id": (
                    paired_opd_run_id if arm == "rl" else run_id(f"branch_t{checkpoint}_rl")
                ),
                "resume_mode": "disable" if checkpoint == 0 else "resume_path",
                "resume_from_path": None if checkpoint == 0 else str(resume_path),
                "updates": training["branch_horizon"],
                "target_global_step": checkpoint + training["branch_horizon"],
                "prompt_batches": future,
                "planned_dose": dose(training["branch_horizon"], future),
                "train_file": str(output / "discovery_schedule.parquet"),
                "default_local_dir": str(runs_root / branch_run_id / "checkpoint"),
                "final_checkpoint_actor_dir": str(
                    runs_root / branch_run_id / "checkpoint"
                    / f"global_step_{checkpoint + training['branch_horizon']}" / "actor"
                ),
            })
    validation_updates = training.get("validation_updates")
    if validation_updates is not None:
        validation_queue = load_json(output / "validation_prompt_queue.json")
        for arm, estimator in (("always_opd", training["opd_estimator"]), ("always_rl", training["rl_estimator"])):
            specs.append({
                **common,
                "run_id": run_id(f"validation_{arm}"),
                "phase": "validation",
                "estimator": estimator,
                "updates": validation_updates,
                "target_global_step": validation_updates,
                "prompt_batches": validation_queue,
                "train_file": str(output / "validation_schedule.parquet"),
                "rollout_seed": config["rollout"]["seeds"]["validation"],
                "resume_mode": "disable",
                "default_local_dir": str(runs_root / run_id(f"validation_{arm}") / "checkpoint"),
            })
        specs.extend([
            {**common, "run_id": run_id("validation_fixed_handoff"), "phase": "validation", "estimator": "scheduled_native_opd_to_grpo", "updates": validation_updates, "target_global_step": validation_updates, "prompt_batches": validation_queue, "train_file": str(output / "validation_schedule.parquet"), "rollout_seed": config["rollout"]["seeds"]["validation"], "resume_mode": "disable", "default_local_dir": str(runs_root / run_id("validation_fixed_handoff") / "checkpoint"), "controller_pending": True},
            {**common, "run_id": run_id("validation_monitor_handoff"), "phase": "validation", "estimator": "monitor_controlled_native_opd_to_grpo", "updates": validation_updates, "target_global_step": validation_updates, "prompt_batches": validation_queue, "train_file": str(output / "validation_schedule.parquet"), "rollout_seed": config["rollout"]["seeds"]["validation"], "resume_mode": "disable", "default_local_dir": str(runs_root / run_id("validation_monitor_handoff") / "checkpoint"), "controller_pending": True},
        ])
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "discovery", "validation"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("GPU execution is intentionally disabled until the native verl adapter and RNG resume gates pass")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    specs = run_specs(config)
    if args.phase == "discovery":
        specs = [item for item in specs if item["phase"].startswith("discovery")]
    elif args.phase == "validation":
        specs = [item for item in specs if item["phase"] == "validation"]
    output = Path(config["project"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_specs.json").write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(specs), "run_ids": [item["run_id"] for item in specs]}, indent=2))


if __name__ == "__main__":
    main()
