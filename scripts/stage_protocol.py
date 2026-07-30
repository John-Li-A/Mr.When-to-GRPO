from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from when_to_grpo.core import canonical_json, sha256_bytes


def _dose(config: dict, updates: int, prompt_batches: list[list[str]]) -> dict:
    prompt_batch_size = int(config["data"]["prompt_batch_size"])
    n = int(config["rollout"]["n"])
    trajectories_per_update = prompt_batch_size * n
    mini_batch_size = int(config["training"].get("ppo_mini_batch_size", prompt_batch_size))
    optimizer_steps_per_update = (
        trajectories_per_update
        // mini_batch_size
        * int(config["training"].get("ppo_epochs", 1))
    )
    return {
        "trainer_updates": updates,
        "optimizer_steps": updates * optimizer_steps_per_update,
        "prompts": updates * prompt_batch_size,
        "trajectories": updates * trajectories_per_update,
        "max_generated_tokens_upper_bound": (
            updates * trajectories_per_update * int(config["rollout"]["max_response_length"])
        ),
        "prompt_batches_sha256": sha256_bytes(
            canonical_json(prompt_batches).encode("utf-8")
        ),
    }


def build_execution_specs(config: dict, formal_specs: list[dict]) -> list[dict]:
    protocol_id = str(config["project"]["protocol_id"])
    trunk_id = f"{protocol_id}_discovery_trunk_opd"
    by_id = {item["run_id"]: item for item in formal_specs}
    if trunk_id not in by_id:
        raise ValueError(f"formal trunk spec is missing: {trunk_id}")

    trunk = by_id[trunk_id]
    initial_stop = int(config["training"]["staged_trunk_initial_stop"])
    final_stop = int(trunk["target_global_step"])
    checkpoint_frequency = int(config["training"]["checkpoint_frequency"])
    horizon = int(config["training"]["branch_horizon"])
    if initial_stop != horizon:
        raise ValueError("initial trunk stop must equal the t0 counterfactual horizon")
    if not 0 < initial_stop < final_stop:
        raise ValueError("initial trunk stop must lie strictly inside the formal trunk")
    if initial_stop % checkpoint_frequency:
        raise ValueError("initial trunk stop must be a complete checkpoint boundary")

    first_batches = trunk["prompt_batches"][:initial_stop]
    remaining_batches = trunk["prompt_batches"][initial_stop:final_stop]
    if len(first_batches) != initial_stop or len(remaining_batches) != final_stop - initial_stop:
        raise ValueError("formal trunk prompt queue does not cover both execution stages")

    first = copy.deepcopy(trunk)
    first.update(
        {
            "execution_id": f"{trunk_id}_t0_t{initial_stop}",
            "scientific_run_id": trunk_id,
            "execution_stage": "initial_t0_endpoint",
            "start_step": 0,
            "updates": initial_stop,
            "target_global_step": initial_stop,
            "prompt_batches": first_batches,
            "planned_dose": _dose(config, initial_stop, first_batches),
            "save_steps": [initial_stop],
            "identity_group": f"{protocol_id}_branch_t0",
            "pre_intervention_step": 0,
            "final_checkpoint_actor_dir": str(
                Path(trunk["default_local_dir"])
                / f"global_step_{initial_stop}"
                / "actor"
            ),
        }
    )

    continuation = copy.deepcopy(trunk)
    continuation.update(
        {
            "execution_id": f"{trunk_id}_t{initial_stop}_t{final_stop}",
            "scientific_run_id": trunk_id,
            "execution_stage": "post_t0_continuation",
            "start_step": initial_stop,
            "pre_intervention_step": initial_stop,
            "updates": final_stop - initial_stop,
            "target_global_step": final_stop,
            "prompt_batches": remaining_batches,
            "planned_dose": _dose(config, final_stop - initial_stop, remaining_batches),
            "resume_mode": "resume_path",
            "resume_from_path": str(
                Path(trunk["default_local_dir"]) / f"global_step_{initial_stop}"
            ),
            "process_boundary": {
                "planned": True,
                "at_global_step": initial_stop,
                "known_effect": "live vLLM generation stream is reinitialized",
                "scientific_disposition": (
                    "documented trunk-only boundary; paired resumed t20/t40 arms still share "
                    "identical process-reset semantics"
                ),
            },
        }
    )

    t0_rl_id = f"{protocol_id}_branch_t0_rl"
    ordered: list[dict] = [first]
    if t0_rl_id in by_id:
        t0_rl = copy.deepcopy(by_id[t0_rl_id])
        t0_rl["execution_id"] = t0_rl["run_id"]
        t0_rl["scientific_run_id"] = t0_rl["run_id"]
        ordered.append(t0_rl)
    ordered.append(continuation)
    for spec in formal_specs:
        if spec["run_id"] in {trunk_id, t0_rl_id}:
            continue
        item = copy.deepcopy(spec)
        item["execution_id"] = item["run_id"]
        item["scientific_run_id"] = item["run_id"]
        ordered.append(item)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["project"]["output_dir"])
    formal_specs = json.loads((output / "run_specs.json").read_text(encoding="utf-8"))
    specs = build_execution_specs(config, formal_specs)
    path = output / "execution_specs.json"
    path.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"executions": [item["execution_id"] for item in specs], "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
