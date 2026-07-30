import json
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parents[1] / "results" / "reference_evidence.json"


def test_reference_evidence_arithmetic_and_hash_shape() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evaluation = evidence["math500_evaluation"]
    for model in ("student", "teacher"):
        assert evaluation[model]["trajectory_count"] == evaluation["rows"] * evaluation["n"]
    pilot = evidence["ten_update_opd_pilot"]
    assert pilot["prompts"] == pilot["updates"] * evidence["protocol"]["prompt_batch_size"]
    assert pilot["trajectories"] == pilot["prompts"] * evidence["protocol"]["rollouts_per_prompt"]
    assert pilot["mixed_groups"] + pilot["all_fail_groups"] == pilot["prompts"]

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("sha256"):
                    assert isinstance(item, str) and len(item) == 64
                    int(item, 16)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(evidence)
