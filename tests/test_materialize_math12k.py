import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "materialize_math12k.py"
SPEC = importlib.util.spec_from_file_location("materialize_math12k", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_exact_native_mapping_contract() -> None:
    source = [{"prompt": "What is 1+1?", "answer": "2", "level": 1, "id": "demo/1"}]
    native = pd.DataFrame([MODULE.expected_row(source[0], 0)])
    MODULE.validate_mapping(source, native)


def test_mapping_rejects_prompt_drift() -> None:
    source = [{"prompt": "What is 1+1?", "answer": "2", "level": 1, "id": "demo/1"}]
    row = MODULE.expected_row(source[0], 0)
    row["prompt"][0]["content"] += " changed"
    with pytest.raises(ValueError, match="row 0"):
        MODULE.validate_mapping(source, pd.DataFrame([row]))
