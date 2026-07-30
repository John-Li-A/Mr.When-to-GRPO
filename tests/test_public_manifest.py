import json
from pathlib import Path

from when_to_grpo.core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_public_manifest_patch_hashes_match_repository() -> None:
    manifest = json.loads(
        (ROOT / "data" / "manifests" / "canonical_manifest.public.json").read_text(
            encoding="utf-8"
        )
    )
    for name, expected in manifest["source"]["patch_sha256"].items():
        assert sha256_file(ROOT / "patches" / name) == expected
