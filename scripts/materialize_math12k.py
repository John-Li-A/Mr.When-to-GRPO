from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from when_to_grpo.core import sha256_file


OPD_COMMIT = "4532fd35ccfdde82adc918b265e4c964534e83d1"
SOURCE_JSON_RELATIVE = Path("datasets/test_data/MATH/train.json")
SOURCE_PARQUET_RELATIVE = Path("datasets/test_data/MATH/train.parquet")
SOURCE_JSON_SHA256 = "ad41fe4ffc830efcdac9fc58b477d9d91a74d5c4c687e275a800af2fa58ae5b3"
SOURCE_PARQUET_SHA256 = "53e815b8781e3b4513c7cf9eb4a003b4f4af27198f1d06458336785626b0229d"
INSTRUCTION = " Please reason step by step, and put your final answer within \\boxed{}."


def expected_row(source: Mapping[str, Any], index: int) -> dict[str, Any]:
    required = {"prompt", "answer", "level", "id"}
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"source row {index} is missing fields: {missing}")
    return {
        "prompt": [{"role": "user", "content": str(source["prompt"]) + INSTRUCTION}],
        "level": int(source["level"]),
        "id": str(source["id"]),
        "data_source": "MATH",
        "ability": "math",
        "reward_model": {"ground_truth": str(source["answer"]), "style": "rule"},
        "extra_info": {"index": f"MATH-{index}", "split": "train"},
    }


def normalized_native_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prompt = row["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError("native prompt must contain exactly one message")
    return {
        "prompt": [
            {
                "role": str(prompt[0]["role"]),
                "content": str(prompt[0]["content"]),
            }
        ],
        "level": int(row["level"]),
        "id": str(row["id"]),
        "data_source": str(row["data_source"]),
        "ability": str(row["ability"]),
        "reward_model": {
            "ground_truth": str(row["reward_model"]["ground_truth"]),
            "style": str(row["reward_model"]["style"]),
        },
        "extra_info": {
            "index": str(row["extra_info"]["index"]),
            "split": str(row["extra_info"]["split"]),
        },
    }


def validate_mapping(source_rows: Sequence[Mapping[str, Any]], native_frame: pd.DataFrame) -> None:
    expected_columns = [
        "prompt",
        "level",
        "id",
        "data_source",
        "ability",
        "reward_model",
        "extra_info",
    ]
    if list(native_frame.columns) != expected_columns:
        raise ValueError(
            f"native parquet column drift: {list(native_frame.columns)} != {expected_columns}"
        )
    if len(source_rows) != len(native_frame):
        raise ValueError(f"row-count drift: {len(source_rows)} != {len(native_frame)}")
    for index, (source, native) in enumerate(
        zip(source_rows, native_frame.to_dict(orient="records"), strict=True)
    ):
        expected = expected_row(source, index)
        actual = normalized_native_row(native)
        if actual != expected:
            raise ValueError(f"native mapping drift at row {index}: {actual!r} != {expected!r}")


def assert_pinned_checkout(opd_root: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(opd_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"OPD root is not a readable Git checkout: {opd_root}") from error
    actual = result.stdout.strip()
    if actual != OPD_COMMIT:
        raise ValueError(f"OPD commit drift: {actual} != {OPD_COMMIT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and copy the exact MATH-12K native-verl artifact from pinned OPD."
    )
    parser.add_argument("--opd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert_pinned_checkout(args.opd_root)
    source_json = args.opd_root / SOURCE_JSON_RELATIVE
    source_parquet = args.opd_root / SOURCE_PARQUET_RELATIVE
    for path, expected_hash in (
        (source_json, SOURCE_JSON_SHA256),
        (source_parquet, SOURCE_PARQUET_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source hash drift for {path}: {actual_hash} != {expected_hash}")

    source_rows = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise TypeError("MATH-12K source JSON must contain a list")
    native_frame = pd.read_parquet(source_parquet)
    validate_mapping(source_rows, native_frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if source_parquet.resolve() != args.output.resolve():
        shutil.copyfile(source_parquet, args.output)
    output_hash = sha256_file(args.output)
    if output_hash != SOURCE_PARQUET_SHA256:
        raise ValueError(f"materialized artifact hash drift: {output_hash}")
    print(
        json.dumps(
            {
                "opd_commit": OPD_COMMIT,
                "rows": len(native_frame),
                "source_json_sha256": SOURCE_JSON_SHA256,
                "output": str(args.output),
                "output_sha256": output_hash,
                "mapping_audit": "all_rows_exact",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
