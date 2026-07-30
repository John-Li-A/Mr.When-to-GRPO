from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def merge(left: dict, right: dict) -> dict:
    """Recursively merge configuration fragments and reject type drift."""
    output = dict(left)
    for key, value in right.items():
        if key in output and isinstance(output[key], dict) and isinstance(value, dict):
            output[key] = merge(output[key], value)
        elif key in output and isinstance(output[key], dict) != isinstance(value, dict):
            raise TypeError(f"configuration type drift at {key!r}")
        else:
            output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fragments", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config: dict = {}
    for path in args.fragments:
        fragment = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(fragment, dict):
            raise TypeError(f"top-level YAML value must be a mapping: {path}")
        config = merge(config, fragment)
    if config.get("schema_version") != 1:
        raise ValueError("composed config must declare schema_version: 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
