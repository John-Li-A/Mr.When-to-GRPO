from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORBIDDEN_ESTIMATORS = {"token_reward_direct_plus_grpo"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def topk_support_signals(
    student_ids: np.ndarray,
    student_logp: np.ndarray,
    teacher_ids: np.ndarray,
    teacher_logp: np.ndarray,
    token_mask: np.ndarray,
) -> dict[str, float]:
    student_ids = np.asarray(student_ids)
    teacher_ids = np.asarray(teacher_ids)
    student_logp = np.asarray(student_logp, dtype=float)
    teacher_logp = np.asarray(teacher_logp, dtype=float)
    token_mask = np.asarray(token_mask, dtype=bool)
    if student_ids.shape != student_logp.shape or teacher_ids.shape != teacher_logp.shape:
        raise ValueError("each top-k id tensor must match its log-prob tensor")
    if student_ids.shape[:-1] != teacher_ids.shape[:-1] or token_mask.shape != student_ids.shape[:-1]:
        raise ValueError("student, teacher, and token mask prefixes must agree")
    overlaps = []
    teacher_mass_covered = []
    teacher_supported_student_low_mass = []
    for index in np.argwhere(token_mask):
        prefix = tuple(index)
        s_ids = student_ids[prefix]
        t_ids = teacher_ids[prefix]
        s_lp = student_logp[prefix]
        t_lp = teacher_logp[prefix]
        common = np.isin(t_ids, s_ids)
        overlaps.append(float(common.mean()))
        teacher_mass_covered.append(float(np.exp(t_lp[common]).sum()))
        s_by_id = {int(token): float(lp) for token, lp in zip(s_ids, s_lp)}
        low_mass = [math.exp(s_by_id.get(int(token), -math.inf)) < 1e-3 for token in t_ids]
        teacher_supported_student_low_mass.append(float(np.exp(t_lp[np.asarray(low_mass)]).sum()))
    if not overlaps:
        raise ValueError("token_mask selects no tokens")
    return {
        "topk_token_overlap": float(np.mean(overlaps)),
        "teacher_topk_mass_covered_by_student_topk": float(np.mean(teacher_mass_covered)),
        "teacher_supported_student_low_mass": float(np.mean(teacher_supported_student_low_mass)),
    }


def gradient_relation(opd_grads: Sequence[np.ndarray], rl_grads: Sequence[np.ndarray]) -> dict[str, float]:
    if len(opd_grads) != len(rl_grads) or not opd_grads:
        raise ValueError("OPD and RL gradient lists must be non-empty and aligned")
    opd = np.concatenate([np.asarray(item, dtype=float).ravel() for item in opd_grads])
    rl = np.concatenate([np.asarray(item, dtype=float).ravel() for item in rl_grads])
    if opd.shape != rl.shape:
        raise ValueError("OPD and RL gradient vectors must have identical shapes")
    opd_norm = float(np.linalg.norm(opd))
    rl_norm = float(np.linalg.norm(rl))
    cosine = float(np.dot(opd, rl) / (opd_norm * rl_norm)) if opd_norm and rl_norm else float("nan")
    return {"opd_grad_norm": opd_norm, "rl_grad_norm": rl_norm, "gradient_cosine": cosine}


@dataclass(frozen=True)
class CheckpointIdentity:
    global_step: int
    actor_hash: str
    optimizer_hash: str
    scheduler_hash: str
    dataloader_hash: str
    driver_rng_hash: str
    rollout_seed: int
    config_hash: str
    prompt_queue_hash: str
    source_manifest_hash: str

    def validate(self) -> None:
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")
        for field, value in asdict(self).items():
            if field in {"global_step", "rollout_seed"}:
                continue
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{field} must be a SHA256 hex digest")


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
