import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plan_evaluation.py"
SPEC = importlib.util.spec_from_file_location("plan_evaluation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_plan_uses_configured_panels_and_dynamic_terminal_branch() -> None:
    config = {
        "project": {"output_dir": "artifacts/demo", "protocol_id": "demo"},
        "paths": {
            "student_model": "student",
            "teacher_model": "teacher",
            "external_eval": ["primary.parquet", "external.parquet"],
        },
        "training": {"branch_horizon": 5, "branch_points": [0, 7]},
        "evaluation": {
            "primary_panel": "Primary",
            "external_panels": ["Primary", "External"],
            "primary_n": 4,
            "external_n": 8,
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
