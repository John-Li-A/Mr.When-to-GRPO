# verl patches

These patches apply to `thunlp/OPD` commit
`4532fd35ccfdde82adc918b265e4c964534e83d1` in filename order.

1. `01_rollout_observability.patch` makes rollout JSON serialization robust and
   records exact response length and verifier outcome.
2. `02_rollout_seed.patch` exposes the rollout seed already consumed by the
   vLLM path.
3. `03_actor_selected_token_memory.patch` avoids the full-vocabulary
   differentiable allocation on the selected-token path and supports an
   optional chunked top-k diagnostic.
4. `04_chunked_logprob_helper.patch` implements selected-token normalized
   log-probabilities through chunked `logsumexp`.
5. `05_teacher_fused_alignment.patch` aligns fused teacher response scores,
   makes entropy opt-in, and fixes non-contiguous reshaping.

The original experimental tree contained one additional dormant edit in the
unvalidated `token_reward_direct_plus_grpo` estimator. It was never used by the
native OPD or Dr.GRPO arms and is intentionally excluded.
