import json

import pytest

from when_to_grpo.core import canonical_json, sha256_bytes
from when_to_grpo.verl import record_pre_intervention_identity


def test_launcher_locks_paired_pre_intervention_identity(tmp_path) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    model = tmp_path / "student"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    source = {"commit": "demo"}
    config_hash = "a" * 64
    (output / "canonical_manifest.json").write_text(
        json.dumps({"config_sha256": config_hash, "source": source}), encoding="utf-8"
    )
    config = {
        "project": {"output_dir": str(output)},
        "paths": {"student_model": str(model)},
    }
    source_hash = sha256_bytes(canonical_json(source).encode("utf-8"))
    base_spec = {
        "run_id": "opd",
        "identity_group": "branch_t0",
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "rollout_seed": 7,
        "planned_dose": {"prompt_batches_sha256": "b" * 64},
    }
    left = record_pre_intervention_identity(config, base_spec)
    right = record_pre_intervention_identity(config, {**base_spec, "run_id": "rl"})
    assert left["sha256"] == right["sha256"]

    (model / "model.safetensors").write_bytes(b"drifted")
    with pytest.raises(ValueError, match="actor_hash"):
        record_pre_intervention_identity(config, {**base_spec, "run_id": "drift"})
