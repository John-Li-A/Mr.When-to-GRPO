import tempfile
import unittest
from pathlib import Path

import numpy as np

from when_to_grpo.core import (
    BranchResult,
    CheckpointIdentity,
    audit_and_deduplicate_records,
    assert_no_split_leakage,
    branch_label,
    branch_prompt_window,
    budget_projection,
    build_prompt_queue,
    compare_branch_identities,
    deterministic_split,
    gradient_relation,
    group_reward_signals,
    guard_estimator,
    masked_scalar_stats,
    sha256_file,
    topk_support_signals,
)


def record(index: int):
    return {
        "prompt": [{"role": "user", "content": f"Problem {index}"}],
        "reward_model": {"ground_truth": str(index)},
    }


class HandoffCoreTests(unittest.TestCase):
    def test_split_is_deterministic_and_leak_free(self):
        records = [record(i) for i in range(40)]
        left, _ = deterministic_split(records, {"signal": 5, "eval": 7}, 11)
        right, _ = deterministic_split(records, {"signal": 5, "eval": 7}, 11)
        self.assertEqual(left, right)
        assert_no_split_leakage(left)
        self.assertEqual(len(left["train_trunk"]), 28)

    def test_duplicate_problem_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate prompt"):
            deterministic_split([record(1), record(1), record(2)], {"signal": 1}, 3)

    def test_dedup_drops_conflicting_labels(self):
        duplicate = record(1)
        conflict = record(2)
        conflict_other = record(2)
        conflict_other["reward_model"] = {"ground_truth": "different"}
        retained, audit = audit_and_deduplicate_records([duplicate, duplicate, conflict, conflict_other, record(3)])
        self.assertEqual(retained, [0, 4])
        self.assertEqual(audit["exact_duplicate_rows_removed"], 1)
        self.assertEqual(audit["conflicting_prompt_groups_removed"], 1)
        self.assertEqual(audit["conflicting_rows_removed"], 2)

    def test_prompt_queue_and_counterfactual_window(self):
        queue = build_prompt_queue([str(i) for i in range(20)], seed=2, updates=10, prompt_batch_size=3)
        self.assertEqual(queue, build_prompt_queue([str(i) for i in range(20)], seed=2, updates=10, prompt_batch_size=3))
        self.assertEqual(branch_prompt_window(queue, 4, 3), queue[4:7])

    def test_group_signals_cover_dead_and_live_zones(self):
        rewards = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
        result = group_reward_signals(rewards)
        self.assertAlmostEqual(result["all_fail_rate"], 1 / 3)
        self.assertAlmostEqual(result["nondegenerate_group_rate"], 1 / 3)
        self.assertAlmostEqual(result["all_correct_rate"], 1 / 3)

    def test_masked_stats_support_topk_axis(self):
        values = np.array([[[1.0, -2.0], [99.0, 99.0]]])
        result = masked_scalar_stats(values, np.array([[1, 0]]))
        self.assertEqual(result["mean_abs"], 1.5)
        self.assertEqual(result["positive_rate"], 0.5)

    def test_topk_support(self):
        result = topk_support_signals(
            np.array([[[1, 2]]]), np.log(np.array([[[0.8, 0.2]]])),
            np.array([[[2, 3]]]), np.log(np.array([[[0.6, 0.3]]])),
            np.array([[True]]),
        )
        self.assertAlmostEqual(result["topk_token_overlap"], 0.5)
        self.assertAlmostEqual(result["teacher_topk_mass_covered_by_student_topk"], 0.6)
        self.assertAlmostEqual(result["teacher_supported_student_low_mass"], 0.3)

    def test_gradient_relation(self):
        result = gradient_relation([np.array([1.0, 0.0])], [np.array([1.0, 1.0])])
        self.assertAlmostEqual(result["opd_grad_norm"], 1.0)
        self.assertAlmostEqual(result["gradient_cosine"], 2 ** -0.5)

    def test_checkpoint_identity_mismatch_is_rejected(self):
        digest = "a" * 64
        left = CheckpointIdentity(10, digest, digest, digest, digest, digest, 1, digest, digest, digest)
        right = CheckpointIdentity(10, digest, digest, digest, digest, digest, 2, digest, digest, digest)
        with self.assertRaisesRegex(ValueError, "rollout_seed"):
            compare_branch_identities(left, right)

    def test_branch_result_delta(self):
        digest = "b" * 64
        result = BranchResult(10, 8, digest, digest, 0.2, 0.3, 0.35, 10, 12, 1.0, 1.2)
        result.validate()
        self.assertAlmostEqual(result.delta_future, 0.05)

    def test_forbidden_plus_estimator(self):
        with self.assertRaisesRegex(ValueError, "forbidden estimator"):
            guard_estimator("token_reward_direct_plus_grpo")
        guard_estimator("token_reward_direct")
        guard_estimator("grpo")

    def test_branch_label_and_budget(self):
        self.assertEqual(branch_label(0.51, 0.50, 0.02), "indistinguishable")
        self.assertEqual(branch_label(0.6, 0.5, 0.02), "rl")
        budget = budget_projection(
            trunk_updates=50, checkpoints=6, horizon=8, validation_arms=4,
            validation_updates=50, benchmark_seconds_per_update=60,
        )
        self.assertEqual(budget["total_updates"], 346)

    def test_file_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
