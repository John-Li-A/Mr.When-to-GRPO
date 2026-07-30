from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


BOXED_PATTERN = re.compile(r"\\boxed\s*\{")


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _json_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def trajectory_outcome(token_rewards: object) -> float:
    if not isinstance(token_rewards, list) or not token_rewards:
        raise ValueError("true_reward_score must be a non-empty per-token list")
    values = np.asarray(token_rewards, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("true_reward_score contains a non-finite value")
    return float(np.max(values))


def _row_outcome(row: dict) -> float:
    """Recover the verifier outcome from either OPD or native-RL dumps.

    OPD dumps contain the padded per-token ``true_reward_score`` tensor, while
    native GRPO dumps only retain the scalar ``verifier_reward``.  When both
    are available, require them to agree instead of silently choosing one.
    """
    token_outcome = None
    if "true_reward_score" in row:
        token_outcome = trajectory_outcome(row["true_reward_score"])

    scalar_outcome = None
    if "verifier_reward" in row:
        scalar_outcome = float(row["verifier_reward"])
        if not np.isfinite(scalar_outcome):
            raise ValueError("verifier_reward contains a non-finite value")

    if token_outcome is None and scalar_outcome is None:
        raise ValueError(
            "row must contain true_reward_score or verifier_reward"
        )
    if token_outcome is not None and scalar_outcome is not None:
        if not math.isclose(token_outcome, scalar_outcome, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "true_reward_score and verifier_reward disagree: "
                f"{token_outcome} != {scalar_outcome}"
            )
    return token_outcome if token_outcome is not None else scalar_outcome


def _group_rows(rows: Sequence[dict], n: int) -> list[list[dict]]:
    if n < 2:
        raise ValueError("n must be at least 2")
    groups: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        step = int(row["step"])
        key = (step, _json_key(row["input"]), _json_key(row["gts"]))
        groups[key].append(row)
    bad = {key: len(group) for key, group in groups.items() if len(group) != n}
    if bad:
        preview = list(bad.items())[:3]
        raise ValueError(f"rollout groups must contain exactly n={n} rows: {preview}")
    return list(groups.values())


def _stored_or_supplied_lengths(
    rows: Sequence[dict], response_lengths: Sequence[int] | None
) -> np.ndarray:
    if response_lengths is None:
        if all("response_length" in row for row in rows):
            response_lengths = [int(row["response_length"]) for row in rows]
        elif all("response_token_count" in row for row in rows):
            response_lengths = [int(row["response_token_count"]) for row in rows]
        else:
            raise ValueError(
                "exact response lengths are not stored in verl rollout JSON; "
                "supply tokenizer-reencoded response_lengths"
            )
    if len(response_lengths) != len(rows):
        raise ValueError("response_lengths must align one-to-one with rows")
    lengths = np.asarray(response_lengths, dtype=float)
    if np.any(lengths <= 0):
        raise ValueError("response token lengths must be positive")
    return lengths


def analyze_rows(
    rows: Sequence[dict],
    n: int,
    response_cap: int,
    response_lengths: Sequence[int] | None = None,
    response_lengths_exact: bool = True,
) -> dict[str, object]:
    if not rows:
        raise ValueError("rollout file contains no rows")
    if response_cap <= 0:
        raise ValueError("response_cap must be positive")

    required = {"input", "gts", "output", "score", "step"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing fields: {sorted(missing)}")

    token_reward_presence = ["true_reward_score" in row for row in rows]
    if any(token_reward_presence) and not all(token_reward_presence):
        raise ValueError(
            "rollout mixes OPD rows with and without true_reward_score"
        )
    has_token_rewards = all(token_reward_presence)

    groups = _group_rows(rows, n)
    lengths = _stored_or_supplied_lengths(rows, response_lengths)
    outcome_values = np.asarray([_row_outcome(row) for row in rows], dtype=float)
    outcomes = outcome_values > 0.5
    raw_scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("score contains a non-finite value")
    per_token_scores = raw_scores / lengths

    group_success_counts = np.asarray(
        [sum(_row_outcome(row) > 0.5 for row in group) for group in groups],
        dtype=int,
    )
    distribution = Counter(int(value) for value in group_success_counts)
    mixed = (group_success_counts > 0) & (group_success_counts < n)
    all_fail = group_success_counts == 0
    all_correct = group_success_counts == n

    length_summary = {
        "mean": float(lengths.mean()),
        "p50": _quantile(lengths, 0.50),
        "p95": _quantile(lengths, 0.95),
        "p99": _quantile(lengths, 0.99),
        "max": int(lengths.max()),
        (
            "cap_hit_rate" if response_lengths_exact else "at_or_above_cap_rate"
        ): float(np.mean(lengths >= response_cap)),
    }
    length_key = (
        "response_length" if response_lengths_exact else "response_length_reencoded_proxy"
    )

    result = {
        "steps": sorted({int(row["step"]) for row in rows}),
        "trajectory_count": len(rows),
        "group_count": len(groups),
        "unique_input_count": len({_json_key(row["input"]) for row in rows}),
        "outcome_source": (
            "true_reward_score+verifier_reward_validated"
            if has_token_rewards and all("verifier_reward" in row for row in rows)
            else "true_reward_score"
            if has_token_rewards
            else "verifier_reward"
        ),
        "verified_correct_trajectories": int(outcomes.sum()),
        "trajectory_pass_rate": float(outcomes.mean()),
        "group_success_count_distribution": {
            str(count): int(distribution.get(count, 0)) for count in range(n + 1)
        },
        "all_fail_group_rate": float(all_fail.mean()),
        "mixed_group_rate": float(mixed.mean()),
        "all_correct_group_rate": float(all_correct.mean()),
        "zero_native_grpo_task_gradient_group_rate": float((all_fail | all_correct).mean()),
        length_key: length_summary,
        "boxed_marker_rate": float(
            np.mean([bool(BOXED_PATTERN.search(str(row["output"]))) for row in rows])
        ),
    }
    if has_token_rewards:
        result.update({
        "padded_reward_widths": sorted(
            {len(row["true_reward_score"]) for row in rows}
        ),
        "opd_trajectory_score": {
            "mean": float(raw_scores.mean()),
            "mean_abs": float(np.abs(raw_scores).mean()),
            "p50": _quantile(raw_scores, 0.50),
            "p05": _quantile(raw_scores, 0.05),
            "p95": _quantile(raw_scores, 0.95),
        },
        "opd_score_per_response_token_proxy": {
            "mean": float(per_token_scores.mean()),
            "mean_abs": float(np.abs(per_token_scores).mean()),
            "p50": _quantile(per_token_scores, 0.50),
            "p05": _quantile(per_token_scores, 0.05),
            "p95": _quantile(per_token_scores, 0.95),
        },
        "raw_score_length_pearson": _pearson(raw_scores, lengths),
        "per_token_score_length_pearson": _pearson(per_token_scores, lengths),
        })
    return result


def load_jsonl(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from error
    return rows


def _numeric_path_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return math.inf, path.name


def _reencoded_lengths(rows: Sequence[dict], tokenizer_path: Path) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)
    lengths: list[int] = []
    texts = [str(row["output"]) for row in rows]
    for start in range(0, len(texts), 64):
        encoded = tokenizer(
            texts[start : start + 64],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        lengths.extend(len(item) for item in encoded)
    return lengths


def audit_directory(
    rollout_dir: Path,
    n: int,
    response_cap: int,
    tokenizer_path: Path | None = None,
) -> dict[str, object]:
    paths = sorted(rollout_dir.glob("*.jsonl"), key=_numeric_path_key)
    if not paths:
        raise ValueError(f"no JSONL rollout files found in {rollout_dir}")
    per_file = {}
    all_rows: list[dict] = []
    all_lengths: list[int] = []
    for path in paths:
        rows = load_jsonl([path])
        lengths = (
            _reencoded_lengths(rows, tokenizer_path)
            if tokenizer_path is not None
            else None
        )
        per_file[path.name] = analyze_rows(
            rows,
            n=n,
            response_cap=response_cap,
            response_lengths=lengths,
            response_lengths_exact=tokenizer_path is None,
        )
        all_rows.extend(rows)
        if lengths is not None:
            all_lengths.extend(lengths)
    aggregate = analyze_rows(
        all_rows,
        n=n,
        response_cap=response_cap,
        response_lengths=all_lengths if tokenizer_path is not None else None,
        response_lengths_exact=tokenizer_path is None,
    )
    return {
        "rollout_dir": str(rollout_dir),
        "n": n,
        "response_cap": response_cap,
        "response_length_source": (
            f"reencoded_output:{tokenizer_path}"
            if tokenizer_path is not None
            else "row.response_length_or_response_token_count"
        ),
        "files": per_file,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--response-cap", type=int, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_directory(
        args.rollout_dir, args.n, args.response_cap, tokenizer_path=args.tokenizer
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
