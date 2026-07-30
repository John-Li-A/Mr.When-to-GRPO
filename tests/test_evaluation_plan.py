import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plan_evaluation.py"
SPEC = importlib.util.spec_from_file_location("plan_evaluation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_plan_uses_configured_panels_and_dynamic_terminal_branch(tmp_path) -> None:
    verifier = tmp_path / "verl" / "utils" / "reward_score" / "ttrl_math" / "__init__.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("def compute_score(*args): return 0\n", encoding="utf-8")
    primary = tmp_path / "primary.parquet"
    external_panel = tmp_path / "external.parquet"
    primary.write_bytes(b"primary")
    external_panel.write_bytes(b"external")
    config = {
        "project": {"output_dir": "artifacts/demo", "protocol_id": "demo"},
        "paths": {
            "student_model": "student",
            "teacher_model": "teacher",
            "verl_root": str(tmp_path),
            "external_eval": [str(primary), str(external_panel)],
        },
        "rollout": {
            "max_prompt_length": 128,
            "max_response_length": 512,
            "enable_thinking": False,
        },
        "training": {"branch_horizon": 5, "branch_points": [0, 7]},
        "evaluation": {
            "primary_panel": "Primary",
            "external_panels": ["Primary", "External"],
            "primary_n": 4,
            "external_n": 8,
            "equivalence_band": 0.01,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_response_length": 512,
            "seed": 7,
        },
    }
    plan = MODULE.build_plan(config, "my-config.yaml")
    assert plan["primary_panel"] == "Primary"
    assert plan["primary_comparisons"][-1]["target_step"] == 12
    external = [item for item in plan["evaluation_specs"] if item["panel"] == "External"]
    assert {item["model_id"] for item in external} == {
        "demo_branch_t7_opd_t12",
        "demo_branch_t7_rl_t12",
    }
    assert all("scripts/evaluate.py" in item["argv"] for item in plan["evaluation_specs"])
    assert all("--execute" in item["argv"] for item in plan["evaluation_specs"])
    assert all("--model-id" in item["argv"] for item in plan["evaluation_specs"])
    assert all(len(item["dataset_sha256"]) == 64 for item in plan["evaluation_specs"])
    assert all(len(item["evaluation_protocol_sha256"]) == 64 for item in plan["evaluation_specs"])
    assert plan["equivalence_band"] == 0.01
