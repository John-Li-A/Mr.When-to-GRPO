from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def merge_argv(config: dict, actor_dir: Path, target_dir: Path) -> list[str]:
    return [
        "python",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
        "--use_cpu_initialization",
    ]


def validate_actor_checkpoint(actor_dir: Path) -> dict:
    required = [
        actor_dir / "model_world_size_1_rank_0.pt",
        actor_dir / "extra_state_world_size_1_rank_0.pt",
        actor_dir / "fsdp_config.json",
        actor_dir / "huggingface/config.json",
        actor_dir / "huggingface/tokenizer.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete actor checkpoint: {missing}")
    return {"actor_dir": str(actor_dir), "required_files": [str(path) for path in required]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actor-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    argv = merge_argv(config, args.actor_dir, args.target_dir)
    payload = {"actor_dir": str(args.actor_dir), "target_dir": str(args.target_dir), "argv": argv}
    command_dir = Path(config["project"]["output_dir"]) / "merge_commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    command_id = args.target_dir.name
    (command_dir / f"{command_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.execute:
        validate_actor_checkpoint(args.actor_dir)
        os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
