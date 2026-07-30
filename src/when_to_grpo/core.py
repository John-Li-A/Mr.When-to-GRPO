from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


FORBIDDEN_ESTIMATORS = {"token_reward_direct_plus_grpo"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


EMPTY_SHA256 = sha256_bytes(b"")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_evaluation_protocol(config: Mapping[str, Any], n: int) -> dict[str, Any]:
    """Return the evaluation fields that must be fixed before endpoint runs."""
    evaluation = config["evaluation"]
    rollout = config["rollout"]
    max_tokens = int(evaluation["max_response_length"])
    if max_tokens != int(rollout["max_response_length"]):
        raise ValueError("evaluation and training response caps must match")
    if int(n) < 1:
        raise ValueError("evaluation sample count must be positive")
    verifier = config.get("verifier", {})
    verifier_path = Path(config["paths"]["verl_root"]) / str(
        verifier.get("train_relative_path", "verl/utils/reward_score/ttrl_math/__init__.py")
    )
    if not verifier_path.is_file():
        raise FileNotFoundError(verifier_path)
    return {
        "n": int(n),
        "temperature": float(evaluation["temperature"]),
        "top_p": float(evaluation["top_p"]),
        "top_k": int(evaluation["top_k"]),
        "repetition_penalty": float(evaluation["repetition_penalty"]),
        "max_tokens": max_tokens,
        "max_prompt_length": int(rollout["max_prompt_length"]),
        "max_model_len": int(rollout["max_prompt_length"]) + max_tokens,
        "seed": int(evaluation["seed"]),
        "enable_thinking": bool(rollout["enable_thinking"]),
        "verifier": {
            "label": str(verifier.get("label", "configured-verifier")),
            "module": str(verifier.get("eval_module", "verl.utils.reward_score.ttrl_math")),
            "function": str(verifier.get("eval_function", "compute_score")),
            "score_field": str(verifier.get("score_field", "score")),
            "format_field": str(verifier.get("format_field", "format_score")),
            "source_sha256": sha256_file(verifier_path),
        },
    }


def file_set_identity(root: str | Path, paths: Sequence[str | Path]) -> dict[str, Any]:
    root = Path(root)
    files: dict[str, dict[str, Any]] = {}
    for raw_path in sorted((Path(path) for path in paths), key=lambda path: path.as_posix()):
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(root).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not files:
        raise ValueError(f"identity file set is empty under {root}")
    return {
        "root": str(root),
        "files": files,
        "sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }


def tree_identity(
    root: str | Path,
    *,
    include: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"when2grpo_identity.json", ".when2grpo_identity.json"}
        and (include is None or include(path.relative_to(root)))
    ]
    return file_set_identity(root, paths)


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except json.JSONDecodeError:
            import ast

            prompt = ast.literal_eval(prompt)
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty message list")
    normalized = []
    for message in prompt:
        if not isinstance(message, Mapping):
            raise ValueError("each prompt message must be a mapping")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            raise ValueError("prompt messages require non-empty role and content")
        normalized.append({"role": role, "content": " ".join(content.split())})
    return normalized


def problem_id(record: Mapping[str, Any]) -> str:
    # Identity is intentionally prompt-only. If the same prompt has conflicting
    # labels, treating prompt+answer as separate identities would permit leakage.
    identity = {"prompt": normalize_prompt(record["prompt"])}
    return sha256_bytes(canonical_json(identity).encode("utf-8"))


def record_answer(record: Mapping[str, Any]) -> str:
    reward_model = record.get("reward_model") or {}
    if not isinstance(reward_model, Mapping):
        raise ValueError("reward_model must be a mapping")
    return str(reward_model.get("ground_truth", "")).strip()


def audit_and_deduplicate_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[int], dict[str, int]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for index, record in enumerate(records):
        groups.setdefault(problem_id(record), []).append((index, record_answer(record)))
    retained: list[int] = []
    conflicting_groups = 0
    conflicting_rows = 0
    exact_duplicate_rows_removed = 0
    for entries in groups.values():
        answers = {answer for _, answer in entries}
        if len(answers) != 1:
            conflicting_groups += 1
            conflicting_rows += len(entries)
            continue
        retained.append(min(index for index, _ in entries))
        exact_duplicate_rows_removed += len(entries) - 1
    retained.sort()
    audit = {
        "input_rows": len(records),
        "unique_prompt_groups": len(groups),
        "retained_rows": len(retained),
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "conflicting_prompt_groups_removed": conflicting_groups,
        "conflicting_rows_removed": conflicting_rows,
    }
    if len(retained) + exact_duplicate_rows_removed + conflicting_rows != len(records):
        raise AssertionError("deduplication accounting mismatch")
    return retained, audit


def deterministic_split(
    records: Sequence[Mapping[str, Any]], split_sizes: Mapping[str, int], seed: int
) -> tuple[dict[str, list[str]], list[str]]:
    ids = [problem_id(record) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt identity detected before splitting; audit and deduplicate first")
    ranked = sorted(ids, key=lambda item: sha256_bytes(f"{seed}:{item}".encode("ascii")))
    requested = sum(int(size) for size in split_sizes.values())
    if requested >= len(ranked):
        raise ValueError("held-out splits must leave at least one train_trunk example")
    output: dict[str, list[str]] = {}
    cursor = 0
    for name, size in split_sizes.items():
        size = int(size)
        if size <= 0:
            raise ValueError(f"split {name} must be positive")
        output[name] = ranked[cursor : cursor + size]
        cursor += size
    output["train_trunk"] = ranked[cursor:]
    assert_no_split_leakage(output)
    return output, ids


def assert_no_split_leakage(splits: Mapping[str, Sequence[str]]) -> None:
    owner: dict[str, str] = {}
    for split_name, ids in splits.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate identity inside split {split_name}")
        for item in ids:
            if item in owner:
                raise ValueError(f"identity appears in both {owner[item]} and {split_name}: {item}")
            owner[item] = split_name


def build_prompt_queue(
    train_ids: Sequence[str], *, seed: int, updates: int, prompt_batch_size: int
) -> list[list[str]]:
    if not train_ids:
        raise ValueError("train_ids cannot be empty")
    if updates <= 0 or prompt_batch_size <= 0:
        raise ValueError("updates and prompt_batch_size must be positive")
    rng = np.random.default_rng(seed)
    queue: list[list[str]] = []
    permutation: list[str] = []
    while len(queue) < updates:
        if len(permutation) < prompt_batch_size:
            permutation.extend(np.asarray(train_ids)[rng.permutation(len(train_ids))].tolist())
        queue.append(permutation[:prompt_batch_size])
        permutation = permutation[prompt_batch_size:]
    return queue


def branch_prompt_window(queue: Sequence[Sequence[str]], checkpoint: int, horizon: int) -> list[list[str]]:
    if checkpoint < 0 or horizon <= 0 or checkpoint + horizon > len(queue):
        raise ValueError("branch window falls outside the canonical prompt queue")
    return [list(batch) for batch in queue[checkpoint : checkpoint + horizon]]


def guard_estimator(name: str, additionally_forbidden: Iterable[str] = ()) -> None:
    forbidden = FORBIDDEN_ESTIMATORS | set(additionally_forbidden)
    if name in forbidden:
        raise ValueError(f"forbidden estimator: {name}")


def group_reward_signals(rewards: np.ndarray) -> dict[str, float]:
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("GRPO rewards must have shape [prompt_batch, n] with n >= 2")
    successes = rewards > 0.5
    count = successes.sum(axis=1)
    n = rewards.shape[1]
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    return {
        "pass_rate": float(successes.mean()),
        "all_fail_rate": float(np.mean(count == 0)),
        "nondegenerate_group_rate": float(np.mean((count > 0) & (count < n))),
        "all_correct_rate": float(np.mean(count == n)),
        "mean_abs_grpo_advantage": float(np.mean(np.abs(centered))),
        "reward_variance": float(np.mean(np.var(rewards, axis=1, ddof=1))),
    }


def masked_scalar_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim == mask.ndim + 1:
        mask = np.broadcast_to(mask[..., None], values.shape)
    if values.shape != mask.shape:
        raise ValueError("values and mask are not broadcast-compatible")
    selected = values[mask]
    if selected.size == 0:
        raise ValueError("mask selects no values")
    return {
        "mean_abs": float(np.mean(np.abs(selected))),
        "p50_abs": float(np.quantile(np.abs(selected), 0.50)),
        "p95_abs": float(np.quantile(np.abs(selected), 0.95)),
        "p99_abs": float(np.quantile(np.abs(selected), 0.99)),
        "positive_rate": float(np.mean(selected > 0)),
        "negative_rate": float(np.mean(selected < 0)),
        "zero_rate": float(np.mean(selected == 0)),
    }


@dataclass(frozen=True)
class CheckpointIdentity:
    global_step: int
    actor_hash: str
    optimizer_hash: str
    trainer_state_hash: str
    dataloader_hash: str
    rollout_seed: int
    config_hash: str
    prompt_window_hash: str
    source_manifest_hash: str

    def validate(self) -> None:
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")
        for field, value in asdict(self).items():
            if field in {"global_step", "rollout_seed"}:
                continue
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{field} must be a SHA256 hex digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{field} must be a SHA256 hex digest") from error

    @property
    def sha256(self) -> str:
        self.validate()
        return sha256_bytes(canonical_json(asdict(self)).encode("utf-8"))


def checkpoint_identity(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
    rollout_seed: int,
    config_hash: str,
    prompt_window_hash: str,
    source_manifest_hash: str,
) -> tuple[CheckpointIdentity, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    actor_dir = checkpoint_dir / "actor"
    if not actor_dir.is_dir():
        raise FileNotFoundError(actor_dir)
    actor_paths = [
        path
        for path in actor_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith("optim_")
        and not path.name.startswith("extra_state_")
    ]
    optimizer_paths = list(actor_dir.glob("optim_*.pt"))
    trainer_state_paths = list(actor_dir.glob("extra_state_*.pt"))
    dataloader_path = checkpoint_dir / "data.pt"
    actor = file_set_identity(checkpoint_dir, actor_paths)
    optimizer = file_set_identity(checkpoint_dir, optimizer_paths)
    trainer_state = file_set_identity(checkpoint_dir, trainer_state_paths)
    dataloader = file_set_identity(checkpoint_dir, [dataloader_path])
    identity = CheckpointIdentity(
        global_step=global_step,
        actor_hash=actor["sha256"],
        optimizer_hash=optimizer["sha256"],
        trainer_state_hash=trainer_state["sha256"],
        dataloader_hash=dataloader["sha256"],
        rollout_seed=rollout_seed,
        config_hash=config_hash,
        prompt_window_hash=prompt_window_hash,
        source_manifest_hash=source_manifest_hash,
    )
    identity.validate()
    return identity, {
        "checkpoint_dir": str(checkpoint_dir),
        "actor": actor,
        "optimizer": optimizer,
        "trainer_state": trainer_state,
        "dataloader": dataloader,
    }


def compare_branch_identities(left: CheckpointIdentity, right: CheckpointIdentity) -> None:
    left.validate()
    right.validate()
    if left != right:
        mismatches = [key for key in asdict(left) if getattr(left, key) != getattr(right, key)]
        raise ValueError(f"branch identities differ before intervention: {mismatches}")


def branch_label(score_rl: float, score_opd: float, equivalence_band: float) -> str:
    delta = score_rl - score_opd
    if abs(delta) <= equivalence_band:
        return "indistinguishable"
    return "rl" if delta > 0 else "opd"


@dataclass(frozen=True)
class BranchResult:
    checkpoint: int
    horizon: int
    checkpoint_identity_hash: str
    prompt_queue_hash: str
    score_start: float
    score_opd: float
    score_rl: float
    generated_tokens_opd: int
    generated_tokens_rl: int
    wall_seconds_opd: float
    wall_seconds_rl: float

    def validate(self) -> None:
        if self.checkpoint < 0 or self.horizon <= 0:
            raise ValueError("checkpoint and horizon are invalid")
        for name in ("checkpoint_identity_hash", "prompt_queue_hash"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a SHA256 digest")
        if min(self.generated_tokens_opd, self.generated_tokens_rl) < 0:
            raise ValueError("generated token counts cannot be negative")
        if min(self.wall_seconds_opd, self.wall_seconds_rl) < 0:
            raise ValueError("wall time cannot be negative")

    @property
    def delta_future(self) -> float:
        return (self.score_rl - self.score_start) - (self.score_opd - self.score_start)


def budget_projection(
    *, trunk_updates: int, checkpoints: int, horizon: int, validation_arms: int,
    validation_updates: int, benchmark_seconds_per_update: float, overhead_fraction: float = 0.25
) -> dict[str, float]:
    discovery_updates = trunk_updates + checkpoints * 2 * horizon
    validation_total = validation_arms * validation_updates
    total_updates = discovery_updates + validation_total
    core_hours = total_updates * benchmark_seconds_per_update / 3600.0
    return {
        "discovery_updates": float(discovery_updates),
        "validation_updates": float(validation_total),
        "total_updates": float(total_updates),
        "projected_core_gpu_hours": core_hours,
        "projected_gpu_hours_with_overhead": core_hours * (1.0 + overhead_fraction),
    }
