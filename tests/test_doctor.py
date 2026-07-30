from when_to_grpo.doctor import inspect_config


def config() -> dict:
    return {
        "schema_version": 1,
        "project": {"protocol_id": "demo", "output_dir": "artifacts/demo"},
        "paths": {
            "student_model": "student",
            "teacher_model": "teacher",
            "train_data": "train.parquet",
            "source_root": "source",
            "verl_root": "source/verl",
            "external_eval": ["eval.parquet"],
        },
        "data": {"prompt_batch_size": 8},
        "rollout": {"n": 4, "max_prompt_length": 128, "max_response_length": 512},
        "training": {
            "opd_estimator": "token_reward_direct",
            "rl_estimator": "grpo",
            "forbidden_estimators": ["token_reward_direct_plus_grpo"],
            "trunk_updates": 30,
            "checkpoint_updates": [0, 10, 20, 30],
            "branch_points": [0, 10, 20],
            "branch_horizon": 10,
        },
        "evaluation": {
            "primary_panel": "demo",
            "external_panels": ["demo"],
            "primary_n": 4,
            "max_response_length": 512,
        },
        "runtime": {
            "n_gpus_per_node": 1,
            "model_dtype": "bfloat16",
            "rollout_gpu_memory_utilization": 0.2,
            "actor_micro_batch_size_per_gpu": 1,
            "teacher_micro_batch_size_per_gpu": 1,
        },
    }


def test_valid_config_reports_protocol_shape() -> None:
    report = inspect_config(config())
    assert report["ok"]
    assert report["protocol"]["branch_points"] == [0, 10, 20]


def test_invalid_handoff_geometry_is_rejected() -> None:
    value = config()
    value["training"]["branch_points"] = [0, 25]
    report = inspect_config(value)
    assert not report["ok"]
    assert any("exceeds" in error for error in report["errors"])


def test_placeholders_are_warnings_without_path_checks() -> None:
    value = config()
    value["paths"]["student_model"] = "CHANGE_ME/student"
    report = inspect_config(value)
    assert report["ok"]
    assert any("student_model" in warning for warning in report["warnings"])


def test_placeholders_are_errors_with_path_checks() -> None:
    value = config()
    value["paths"]["student_model"] = "CHANGE_ME/student"
    report = inspect_config(value, check_paths=True)
    assert not report["ok"]
    assert any("student_model" in error for error in report["errors"])


def test_unsaved_branch_point_is_rejected() -> None:
    value = config()
    value["training"]["branch_points"] = [0, 15]
    report = inspect_config(value)
    assert not report["ok"]
    assert any("checkpoint_updates" in error for error in report["errors"])
