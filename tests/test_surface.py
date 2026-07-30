import json

import pytest

from when_to_grpo.surface import build_surface, write_surface


def _summary(panel: str, score: float) -> dict:
    return {
        "panel": panel,
        "dataset_sha256": "a" * 64,
        "rows": 4,
        "n": 4,
        "max_tokens": 512,
        "seed": 7,
        "avg_at_n": score,
        "pass_at_n": score + 0.1,
        "cap_hit_rate": 0.0,
    }


def test_surface_joins_paired_endpoints_and_signals(tmp_path) -> None:
    for model, score in (("opd_t10", 0.25), ("rl_t10", 0.35)):
        directory = tmp_path / model
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps(_summary("demo", score)), encoding="utf-8")
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
    assert rows[0]["signal_mixed_group_rate"] == 0.5
    assert rows[0]["signal_length_cap_hit_rate"] == 0.1
    paths = write_surface(rows, tmp_path / "surface", "demo", "avg_at_n")
    assert json.loads(open(paths["json"], encoding="utf-8").read())["rows"][0]["branch_point"] == 0


def test_surface_rejects_unmatched_evaluation_identity(tmp_path) -> None:
    for model, seed in (("opd", 1), ("rl", 2)):
        directory = tmp_path / model
        directory.mkdir()
        summary = _summary("demo", 0.2)
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
