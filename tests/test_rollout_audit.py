import json

import pytest

from when_to_grpo.rollout_audit import analyze_rows, audit_directory, trajectory_outcome


def row(prompt, sample, outcome, length, score, step=1):
    rewards = [0.0] * length
    if outcome:
        rewards[-1] = 1.0
    return {
        "input": prompt,
        "gts": "42",
        "output": f"sample {sample} " + (r"\boxed{42}" if outcome else "wrong"),
        "score": score,
        "step": step,
        "true_reward_score": rewards,
        "response_token_count": length,
    }


def test_group_and_length_signals_are_recovered():
    rows = [
        row("a", 0, False, 2, -4.0),
        row("a", 1, True, 4, -4.0),
        row("a", 2, False, 4, -8.0),
        row("a", 3, False, 6, -6.0),
        row("b", 0, False, 2, -2.0),
        row("b", 1, False, 4, -8.0),
        row("b", 2, False, 6, -12.0),
        row("b", 3, False, 8, -8.0),
    ]
    result = analyze_rows(rows, n=4, response_cap=8)

    assert result["trajectory_count"] == 8
    assert result["group_count"] == 2
    assert result["verified_correct_trajectories"] == 1
    assert result["trajectory_pass_rate"] == pytest.approx(1 / 8)
    assert result["group_success_count_distribution"] == {
        "0": 1,
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    assert result["all_fail_group_rate"] == pytest.approx(0.5)
    assert result["mixed_group_rate"] == pytest.approx(0.5)
    assert result["zero_native_grpo_task_gradient_group_rate"] == pytest.approx(0.5)
    assert result["response_length"]["mean"] == pytest.approx(4.5)
    assert result["response_length"]["cap_hit_rate"] == pytest.approx(1 / 8)
    assert result["boxed_marker_rate"] == pytest.approx(1 / 8)


def test_groups_are_keyed_by_step_input_and_ground_truth():
    rows = []
    for step in (1, 2):
        rows.extend(row("same", sample, False, 2, -2.0, step=step) for sample in range(4))
    result = analyze_rows(rows, n=4, response_cap=8)
    assert result["group_count"] == 2
    assert result["steps"] == [1, 2]


def test_incomplete_group_is_rejected():
    rows = [row("a", sample, False, 2, -2.0) for sample in range(3)]
    with pytest.raises(ValueError, match="exactly n=4"):
        analyze_rows(rows, n=4, response_cap=8)


def test_empty_or_nonfinite_reward_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        trajectory_outcome([])
    with pytest.raises(ValueError, match="non-finite"):
        trajectory_outcome([0.0, float("nan")])


def test_padded_reward_width_is_not_used_as_response_length():
    rows = [row("a", sample, False, 2, -2.0) for sample in range(4)]
    for item in rows:
        item.pop("response_token_count")
        item["true_reward_score"] = [0.0] * 8
    with pytest.raises(ValueError, match="tokenizer-reencoded"):
        analyze_rows(rows, n=4, response_cap=8)

    result = analyze_rows(rows, n=4, response_cap=8, response_lengths=[2] * 4)
    assert result["response_length"]["mean"] == 2
    assert result["padded_reward_widths"] == [8]


def test_exact_dumped_response_length_takes_precedence():
    rows = [row("a", sample, False, 2, -2.0) for sample in range(4)]
    for item in rows:
        item["response_length"] = 3
        item["response_token_count"] = 7
    result = analyze_rows(rows, n=4, response_cap=8)
    assert result["response_length"]["mean"] == 3


def test_reencoded_lengths_are_explicitly_labeled_as_proxies():
    rows = [row("a", sample, False, 2, -2.0) for sample in range(4)]
    result = analyze_rows(
        rows,
        n=4,
        response_cap=8,
        response_lengths=[9, 2, 2, 2],
        response_lengths_exact=False,
    )
    assert "response_length" not in result
    assert result["response_length_reencoded_proxy"]["max"] == 9
    assert result["response_length_reencoded_proxy"]["at_or_above_cap_rate"] == 0.25


def test_directory_report_contains_per_file_and_aggregate(tmp_path):
    for step, prompt in ((1, "a"), (2, "b")):
        path = tmp_path / f"{step}.jsonl"
        rows = [row(prompt, sample, sample == 0, 3, -3.0, step=step) for sample in range(4)]
        path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    report = audit_directory(tmp_path, n=4, response_cap=8)
    assert list(report["files"]) == ["1.jsonl", "2.jsonl"]
    assert report["aggregate"]["trajectory_count"] == 8
    assert report["aggregate"]["group_count"] == 2
    assert report["aggregate"]["raw_score_length_pearson"] is None
    json.dumps(report, allow_nan=False)


def test_directory_ignores_blank_jsonl_lines(tmp_path):
    path = tmp_path / "1.jsonl"
    rows = [row("a", sample, sample == 0, 3, -3.0) for sample in range(4)]
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n\n", encoding="utf-8")
    assert audit_directory(tmp_path, n=4, response_cap=8)["aggregate"]["group_count"] == 1


def test_native_rl_dump_uses_scalar_verifier_reward_without_opd_fields():
    rows = [row("a", sample, sample == 0, 3, float(sample == 0)) for sample in range(4)]
    for item in rows:
        outcome = trajectory_outcome(item.pop("true_reward_score"))
        item["verifier_reward"] = outcome
        item["response_length"] = item.pop("response_token_count")

    result = analyze_rows(rows, n=4, response_cap=8)

    assert result["outcome_source"] == "verifier_reward"
    assert result["trajectory_pass_rate"] == pytest.approx(0.25)
    assert result["mixed_group_rate"] == pytest.approx(1.0)
    assert "padded_reward_widths" not in result
    assert "opd_trajectory_score" not in result


def test_scalar_and_token_outcomes_must_agree_when_both_are_dumped():
    rows = [row("a", sample, False, 3, -3.0) for sample in range(4)]
    for item in rows:
        item["verifier_reward"] = 0.0
    rows[0]["verifier_reward"] = 1.0

    with pytest.raises(ValueError, match="disagree"):
        analyze_rows(rows, n=4, response_cap=8)


def test_mixed_opd_and_rl_rows_are_rejected():
    rows = [row("a", sample, False, 3, -3.0) for sample in range(4)]
    rows[0]["verifier_reward"] = trajectory_outcome(rows[0].pop("true_reward_score"))

    with pytest.raises(ValueError, match="mixes OPD rows"):
        analyze_rows(rows, n=4, response_cap=8)
