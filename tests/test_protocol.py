import json
import tempfile
import unittest
from pathlib import Path

from when_to_grpo.core import guard_estimator
from when_to_grpo.protocol import run_specs


class LauncherGuardTests(unittest.TestCase):
    def test_only_native_estimators_pass_guard(self):
        guard_estimator("token_reward_direct")
        guard_estimator("grpo")
        with self.assertRaises(ValueError):
            guard_estimator("token_reward_direct_plus_grpo")

    def test_uncommitted_validation_is_omitted_and_only_branch_points_fork(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "artifacts"
            output.mkdir()
            (output / "canonical_manifest.json").write_text(
                json.dumps({"config_sha256": "a" * 64, "source": {"commit": "demo"}}) + "\n",
                encoding="utf-8",
            )
            queue = [[f"p{step}"] for step in range(12)]
            (output / "discovery_prompt_queue.json").write_text(
                json.dumps(queue) + "\n", encoding="utf-8"
            )
            config = {
                "project": {"output_dir": str(output), "protocol_id": "testp"},
                "rollout": {
                    "n": 4,
                    "max_response_length": 7168,
                    "eval_max_response_length": 31744,
                    "seeds": {"discovery": 1, "validation": 2},
                },
                "data": {"prompt_batch_size": 64, "dataloader_num_workers": 0},
                "training": {
                    "opd_estimator": "token_reward_direct",
                    "rl_estimator": "grpo",
                    "forbidden_estimators": ["token_reward_direct_plus_grpo"],
                    "trunk_updates": 10,
                    "checkpoint_updates": [0, 5, 10],
                    "branch_points": [5],
                    "branch_horizon": 2,
                    "counterfactual_mode": "fork_resumed_opd",
                    "reuse_fresh_trunk_t0": True,
                    "validation_updates": None,
                },
            }
            specs = run_specs(config)
            self.assertEqual(
                [item["run_id"] for item in specs],
                ["testp_discovery_trunk_opd", "testp_branch_t5_opd", "testp_branch_t5_rl"],
            )
            self.assertFalse(any(item["phase"] == "validation" for item in specs))


if __name__ == "__main__":
    unittest.main()
