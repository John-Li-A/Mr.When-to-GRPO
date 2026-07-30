from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from transformers import AutoTokenizer

from when_to_grpo.core import (
    audit_and_deduplicate_records,
    assert_no_split_leakage,
    build_prompt_queue,
    canonical_json,
    deterministic_split,
    normalize_prompt,
    problem_id,
    sha256_bytes,
    sha256_file,
)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    return config


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_training_data(frame: pd.DataFrame, train_path: Path, config: dict) -> dict:
    data = config["data"]
    expected_rows = int(data["expected_train_rows"])
    if len(frame) != expected_rows:
        raise ValueError(f"training row count drift: {len(frame)} != {expected_rows}")
    actual_hash = sha256_file(train_path)
    expected_hash = str(data["expected_train_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"training data hash drift: {actual_hash} != {expected_hash}")
    required_columns = set(data["required_columns"])
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"training data is missing required columns: {missing}")

    provenance = config["provenance"]["training_data"]
    expected_unique = int(provenance["unique_prompts"])
    identity_counts = Counter(problem_id(record) for record in frame.to_dict(orient="records"))
    actual_unique = len(identity_counts)
    if actual_unique != expected_unique:
        raise ValueError(f"unique prompt count drift: {actual_unique} != {expected_unique}")
    expected_duplicate_groups = int(provenance["duplicate_prompt_groups"])
    actual_duplicate_groups = sum(count > 1 for count in identity_counts.values())
    if actual_duplicate_groups != expected_duplicate_groups:
        raise ValueError(
            f"duplicate prompt group drift: {actual_duplicate_groups} != {expected_duplicate_groups}"
        )
    actual_duplicate_rows_removed = len(frame) - actual_unique
    expected_duplicate_rows_removed = int(provenance["exact_duplicate_rows_removed"])
    if actual_duplicate_rows_removed != expected_duplicate_rows_removed:
        raise ValueError(
            "duplicate row count drift: "
            f"{actual_duplicate_rows_removed} != {expected_duplicate_rows_removed}"
        )
    expected_levels = {
        int(level): int(count)
        for level, count in config["provenance"]["training_data"]["level_counts"].items()
    }
    actual_levels = {int(level): int(count) for level, count in frame["level"].value_counts().to_dict().items()}
    if actual_levels != expected_levels:
        raise ValueError(f"MATH level distribution drift: {actual_levels} != {expected_levels}")

    required = tuple(data.get("required_prompt_substrings", ()))
    forbidden = tuple(data.get("forbidden_prompt_substrings", ()))
    violations: list[str] = []
    for row_index, prompt in enumerate(frame["prompt"]):
        normalized = normalize_prompt(prompt)
        if len(normalized) != 1 or normalized[0]["role"] != "user":
            violations.append(f"row {row_index}: prompt must contain exactly one user message")
            continue
        content = normalized[0]["content"]
        for token in required:
            if content.count(token) != 1:
                violations.append(f"row {row_index}: required substring must occur once: {token!r}")
        for token in forbidden:
            if token in content:
                violations.append(f"row {row_index}: forbidden pre-rendered token: {token!r}")
        if len(violations) >= 20:
            break
    if violations:
        raise ValueError("prompt contract violations: " + "; ".join(violations))
    return {
        "rows": len(frame),
        "sha256": actual_hash,
        "required_columns": sorted(required_columns),
        "unique_prompts": actual_unique,
        "duplicate_prompt_groups": actual_duplicate_groups,
        "exact_duplicate_rows_removed": actual_duplicate_rows_removed,
        "level_counts": actual_levels,
        "prompt_contract": "one-raw-user-message-with-single-reasoning-and-boxed-instructions",
    }


def prompt_ids(frame: pd.DataFrame) -> set[str]:
    return {problem_id(record) for record in frame.to_dict(orient="records")}


def audit_external_eval_overlap(
    train_frame: pd.DataFrame, eval_paths: Iterable[str], *, fail_on_overlap: bool
) -> list[dict]:
    train_ids = prompt_ids(train_frame)
    audits = []
    for raw_path in eval_paths:
        path = Path(raw_path)
        eval_frame = pd.read_parquet(path)
        if "prompt" not in eval_frame.columns:
            raise ValueError(f"external eval is missing prompt column: {path}")
        overlap = sorted(train_ids & prompt_ids(eval_frame))
        audit = {
            "path": str(path),
            "rows": len(eval_frame),
            "sha256": sha256_file(path),
            "exact_prompt_overlap_count": len(overlap),
            "overlap_prompt_ids": overlap[:20],
        }
        audits.append(audit)
        if overlap and fail_on_overlap:
            raise ValueError(f"train/eval exact prompt overlap in {path}: {len(overlap)} rows")
    return audits


def materialize_schedule(frame: pd.DataFrame, id_to_row: dict[str, int], queue: list[list[str]], path: Path) -> None:
    rows = []
    for update, batch in enumerate(queue):
        for slot, item in enumerate(batch):
            row = frame.iloc[id_to_row[item]].to_dict()
            row["problem_id"] = item
            row["schedule_update"] = update
            row["schedule_slot"] = slot
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def filter_overlong_prompts(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    tokenizer = AutoTokenizer.from_pretrained(config["paths"]["student_model"], local_files_only=True)
    rendered = [
        tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=config["rollout"]["enable_thinking"],
        )
        for prompt in frame["prompt"]
    ]
    lengths = []
    batch_size = 512
    for start in range(0, len(rendered), batch_size):
        encoded = tokenizer(
            rendered[start : start + batch_size],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths.extend(int(value) for value in encoded["length"])
    cap = int(config["rollout"]["max_prompt_length"])
    keep = [length <= cap for length in lengths]
    filtered = frame.loc[keep].reset_index(drop=True)
    retained_lengths = [length for length, include in zip(lengths, keep) if include]
    audit = {
        "prompt_cap": cap,
        "overlong_prompt_rows_removed": int(len(frame) - len(filtered)),
        "max_prompt_tokens_before_filter": int(max(lengths)),
        "max_prompt_tokens_after_filter": int(max(retained_lengths)),
    }
    return filtered, audit


def model_manifest(model_dir: Path) -> dict:
    required = ["config.json", "tokenizer.json"]
    for name in required:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    files = {}
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and (path.suffix in {".json", ".jinja", ".txt"} or path.name.endswith(".safetensors")):
            if ".partial" in path.name:
                continue
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"path": str(model_dir), "files": files}


def tokenizer_compatibility(student_dir: Path, teacher_dir: Path) -> dict:
    student_path = student_dir / "tokenizer.json"
    teacher_path = teacher_dir / "tokenizer.json"
    student_hash = sha256_file(student_path)
    teacher_hash = sha256_file(teacher_path)
    student = AutoTokenizer.from_pretrained(student_dir, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True)
    student_vocab = student.get_vocab()
    teacher_vocab = teacher.get_vocab()
    if student_vocab != teacher_vocab:
        mismatches = sorted(
            token
            for token in set(student_vocab) | set(teacher_vocab)
            if student_vocab.get(token) != teacher_vocab.get(token)
        )
        raise ValueError(f"student/teacher token-to-id mappings differ: {mismatches[:20]}")
    if student.special_tokens_map != teacher.special_tokens_map:
        raise ValueError("student and teacher special token maps differ")
    if student.added_tokens_decoder != teacher.added_tokens_decoder:
        raise ValueError("student and teacher added-token decoders differ")
    if student.chat_template != teacher.chat_template:
        raise ValueError("student and teacher chat templates differ")
    probe = "Token compatibility probe: 1 + 1 = 2; \\boxed{2}."
    student_ids = student.encode(probe, add_special_tokens=False)
    teacher_ids = teacher.encode(probe, add_special_tokens=False)
    if student_ids != teacher_ids:
        raise ValueError("student and teacher tokenize the compatibility probe differently")
    special_ids = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        left = getattr(student, name)
        right = getattr(teacher, name)
        if left != right:
            raise ValueError(f"student/teacher {name} mismatch: {left} != {right}")
        special_ids[name] = left
    return {
        "student_tokenizer_json_sha256": student_hash,
        "teacher_tokenizer_json_sha256": teacher_hash,
        "tokenizer_json_files_identical": student_hash == teacher_hash,
        "token_to_id_mapping_sha256": sha256_bytes(canonical_json(student_vocab).encode("utf-8")),
        "token_to_id_mapping_identical": True,
        "chat_template_identical": True,
        "added_tokens_identical": True,
        "probe_token_ids_sha256": sha256_bytes(canonical_json(student_ids).encode("utf-8")),
        "vocab_size": len(student),
        "special_token_ids": special_ids,
    }


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(text.encode("utf-8"))


def source_manifest(source_root: Path, provenance: dict) -> dict:
    selected = [
        source_root / "on_policy_distillation.sh",
        source_root / "grpo.sh",
        source_root / "verl/verl/trainer/ppo/core_algos.py",
        source_root / "verl/verl/trainer/ppo/ray_trainer.py",
        source_root / "verl/verl/workers/config/rollout.py",
        source_root / "verl/verl/workers/actor/dp_actor.py",
        source_root / "verl/verl/utils/torch_functional.py",
        source_root / "verl/verl/workers/fsdp_workers.py",
        source_root / "verl/verl/utils/reward_score/ttrl_math/__init__.py",
    ]
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical source files: {missing}")
    upstream = provenance["upstream_normalized_sha256"]
    patches = provenance.get("permitted_local_patches", {})
    files = {}
    for path in selected:
        relative = path.relative_to(source_root).as_posix()
        normalized_hash = normalized_text_sha256(path)
        if relative in patches:
            expected = patches[relative]["normalized_sha256"]
            status = "permitted-local-patch"
        else:
            expected = upstream[relative]
            status = "exact-upstream-text"
        if normalized_hash != expected:
            raise ValueError(f"unapproved source drift in {relative}: {normalized_hash} != {expected}")
        files[relative] = {
            "sha256": sha256_file(path),
            "normalized_text_sha256": normalized_hash,
            "status": status,
            "upstream_normalized_text_sha256": upstream[relative],
            "patch_disposition": patches.get(relative, {}).get("disposition"),
        }
    return {
        "provenance_status": f"audited-upstream-commit-with-{len(patches)}-locked-local-patches",
        "repo": provenance["repo"],
        "commit": provenance["commit"],
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(config["paths"]["train_data"])
    frame = pd.read_parquet(train_path)
    raw_data_audit = validate_training_data(frame, train_path, config)
    external_eval_audit = audit_external_eval_overlap(
        frame,
        config["paths"]["external_eval"],
        fail_on_overlap=bool(config["data"].get("fail_on_external_eval_overlap", True)),
    )
    raw_records = frame.to_dict(orient="records")
    retained_indices, dataset_audit = audit_and_deduplicate_records(raw_records)
    frame = frame.iloc[retained_indices].reset_index(names="source_row")
    frame, length_audit = filter_overlong_prompts(frame, config)
    dataset_audit.update(length_audit)
    dataset_audit["retained_rows_after_prompt_filter"] = len(frame)
    records = frame.to_dict(orient="records")
    splits, ordered_ids = deterministic_split(records, config["data"]["splits"], config["project"]["seed"])
    assert_no_split_leakage(splits)

    id_to_row = {problem_id(record): index for index, record in enumerate(records)}
    if len(id_to_row) != len(records):
        raise ValueError("problem IDs are not unique")
    for split_name, ids in splits.items():
        split_frame = frame.iloc[[id_to_row[item] for item in ids]].copy()
        split_frame.insert(0, "problem_id", ids)
        split_frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)

    training = config["training"]
    data = config["data"]
    branch_points = training.get("branch_points", [])
    last_branch_update = (
        max(branch_points) + training["branch_horizon"] if branch_points else 0
    )
    discovery_updates = max(training["trunk_updates"], last_branch_update)
    discovery_queue = build_prompt_queue(
        splits["train_trunk"], seed=data["queue_seeds"]["discovery"],
        updates=discovery_updates, prompt_batch_size=data["prompt_batch_size"]
    )
    validation_updates = training.get("validation_updates")
    validation_queue = None
    if validation_updates is not None:
        validation_queue = build_prompt_queue(
            splits["train_trunk"], seed=data["queue_seeds"]["validation"],
            updates=validation_updates, prompt_batch_size=data["prompt_batch_size"]
        )
    write_json(output_dir / "discovery_prompt_queue.json", discovery_queue)
    if validation_queue is not None:
        write_json(output_dir / "validation_prompt_queue.json", validation_queue)
    materialize_schedule(frame, id_to_row, discovery_queue, output_dir / "discovery_schedule.parquet")
    if validation_queue is not None:
        materialize_schedule(frame, id_to_row, validation_queue, output_dir / "validation_schedule.parquet")

    manifest = {
        "schema_version": 1,
        "config_sha256": sha256_bytes(canonical_json(config).encode("utf-8")),
        "train_data": {"path": str(train_path), "bytes": train_path.stat().st_size, "sha256": sha256_file(train_path)},
        "raw_data_audit": raw_data_audit,
        "dataset_audit": dataset_audit,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_hashes": {name: sha256_bytes(canonical_json(ids).encode("utf-8")) for name, ids in splits.items()},
        "discovery_queue_sha256": sha256_file(output_dir / "discovery_prompt_queue.json"),
        "validation_queue_sha256": (
            sha256_file(output_dir / "validation_prompt_queue.json") if validation_queue is not None else None
        ),
        "discovery_schedule_sha256": sha256_file(output_dir / "discovery_schedule.parquet"),
        "validation_schedule_sha256": (
            sha256_file(output_dir / "validation_schedule.parquet") if validation_queue is not None else None
        ),
        "student": {**model_manifest(Path(config["paths"]["student_model"])), **config["provenance"]["student"]},
        "teacher": {**model_manifest(Path(config["paths"]["teacher_model"])), **config["provenance"]["teacher"]},
        "tokenizer_compatibility": tokenizer_compatibility(
            Path(config["paths"]["student_model"]), Path(config["paths"]["teacher_model"])
        ),
        "source": source_manifest(Path(config["paths"]["source_root"]), config["provenance"]["source"]),
        "external_eval": external_eval_audit,
        "all_problem_ids_sha256": sha256_bytes(canonical_json(ordered_ids).encode("utf-8")),
    }
    write_json(output_dir / "canonical_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "split_counts": manifest["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
