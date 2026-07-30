import json

import pytest

from when_to_grpo.surface import build_surface, write_surface


def _summary(panel: str, score: float, model_id: str = "model") -> dict:
    return {
        "panel": panel,
        "model_id": model_id,
        "model_identity_sha256": ("b" if model_id.startswith("opd") else "c") * 64,
        "evaluation_protocol_sha256": "d" * 64,
        "dataset_sha256": "a" * 64,
        "rows": 4,
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": 512,
        "max_prompt_length": 128,
        "max_model_len": 640,
        "seed": 7,
        "enable_thinking": False,
        "verifier": {"label": "demo", "source_sha256": "e" * 64},
        "avg_at_n": score,
        "pass_at_n": score + 0.1,
        "cap_hit_rate": 0.0,
    }


def test_surface_joins_paired_endpoints_and_signals(tmp_path) -> None:
    for model, score in (("opd_t10", 0.25), ("rl_t10", 0.35)):
        directory = tmp_path / model
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps(_summary("demo", score, model)), encoding="utf-8"
        )
    plan = {
        "primary_panel": "demo",
        "branch_horizon": 10,
        "evaluation_specs": [
            {"model_id": "opd_t10", "panel": "demo", "output_dir": "opd_t10"},
            {"model_id": "rl_t10", "panel": "demo", "output_dir": "rl_t10"},
        ],
        "primary_comparisons": [
            {
                "branch_point": 0,
                "horizon": 10,
                "target_step": 10,
                "opd_model_id": "opd_t10",
                "rl_model_id": "rl_t10",
            }
        ],
    }
    rows = build_surface(
        plan,
        root=tmp_path,
        signals={"0": {"mixed_group_rate": 0.5, "length": {"cap_hit_rate": 0.1}}},
    )
    assert rows[0]["grpo_minus_opd"] == pytest.approx(0.10)
    assert rows[0]["preferred_arm"] == "rl"
    assert rows[0]["signal_mixed_group_rate"] == 0.5
    assert rows[0]["signal_length_cap_hit_rate"] == 0.1
    paths = write_surface(rows, tmp_path / "surface", "demo", "avg_at_n")
    assert json.loads(open(paths["json"], encoding="utf-8").read())["rows"][0]["branch_point"] == 0


def test_surface_rejects_unmatched_evaluation_identity(tmp_path) -> None:
    for model, seed in (("opd", 1), ("rl", 2)):
        directory = tmp_path / model
        directory.mkdir()
        summary = _summary("demo", 0.2, model)
        summary["seed"] = seed
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    plan = {
        "primary_panel": "demo",
        "evaluation_specs": [
            {"model_id": "opd", "panel": "demo", "output_dir": "opd"},
            {"model_id": "rl", "panel": "demo", "output_dir": "rl"},
        ],
        "primary_comparisons": [{"branch_point": 0, "opd_model_id": "opd", "rl_model_id": "rl"}],
    }
    with pytest.raises(ValueError, match="seed"):
        build_surface(plan, root=tmp_path)


def test_surface_rejects_swapped_model_summary(tmp_path) -> None:
    for directory_name, summary_model in (("opd", "rl"), ("rl", "opd")):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps(_summary("demo", 0.2, summary_model)), encoding="utf-8"
        )
    plan = {
        "primary_panel": "demo",
        "evaluation_specs": [
            {"model_id": "opd", "panel": "demo", "output_dir": "opd"},
            {"model_id": "rl", "panel": "demo", "output_dir": "rl"},
        ],
        "primary_comparisons": [
            {"branch_point": 0, "opd_model_id": "opd", "rl_model_id": "rl"}
        ],
    }
    with pytest.raises(ValueError, match="model_id"):
        build_surface(plan, root=tmp_path)


def test_surface_rejects_sampling_protocol_drift(tmp_path) -> None:
    for model in ("opd", "rl"):
        directory = tmp_path / model
        directory.mkdir()
        summary = _summary("demo", 0.2, model)
        if model == "rl":
            summary["temperature"] = 1.0
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    plan = {
        "primary_panel": "demo",
        "evaluation_specs": [
            {"model_id": "opd", "panel": "demo", "output_dir": "opd"},
            {"model_id": "rl", "panel": "demo", "output_dir": "rl"},
        ],
        "primary_comparisons": [
            {"branch_point": 0, "opd_model_id": "opd", "rl_model_id": "rl"}
        ],
    }
    with pytest.raises(ValueError, match="temperature"):
        build_surface(plan, root=tmp_path)


def test_surface_rejects_drift_shared_by_both_arms(tmp_path) -> None:
    protocol_fields = {
        "n",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_tokens",
        "max_prompt_length",
        "max_model_len",
        "seed",
        "enable_thinking",
        "verifier",
    }
    protocol = {
        key: value
        for key, value in _summary("demo", 0.2, "opd").items()
        if key in protocol_fields
    }
    for model in ("opd", "rl"):
        directory = tmp_path / model
        directory.mkdir()
        summary = _summary("demo", 0.2, model)
        summary["temperature"] = 1.0
        summary["dataset"] = "panel.parquet"
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    plan = {
        "primary_panel": "demo",
        "evaluation_specs": [
            {
                "model_id": model,
                "panel": "demo",
                "output_dir": model,
                "dataset": "panel.parquet",
                "dataset_sha256": "a" * 64,
                "evaluation_protocol": protocol,
                "evaluation_protocol_sha256": "d" * 64,
            }
            for model in ("opd", "rl")
        ],
        "primary_comparisons": [
            {"branch_point": 0, "opd_model_id": "opd", "rl_model_id": "rl"}
        ],
    }
    with pytest.raises(ValueError, match="temperature"):
        build_surface(plan, root=tmp_path)
