# Recipe audit

The experiment uses published components where their semantics are clear and
keeps the handoff question in the experimental design, not in a new loss.

## Adopted components

- **OPD implementation:** THUNLP OPD at commit `4532fd35...`.
- **Training estimator:** native sampled-token reverse-KL OPD
  (`token_reward_direct`, `log_prob_top_k=0`). This follows the low-memory
  estimator analyzed in *Rethinking On-Policy Distillation*; top-k statistics
  remain diagnostics rather than the training loss.
- **RL objective:** native Dr.GRPO settings, including no per-group standard
  deviation normalization and sequence-level loss normalization.
- **Evaluation:** the JustRL-style vLLM generation recipe with the same response
  cap as training and the locked `ttrl_math` rule verifier.
- **Data discipline:** deterministic prompt-only identities, duplicate/conflict
  audit, fixed held-out splits, and a materialized prompt queue.
- **Counterfactual:** both resumed arms load the same complete checkpoint and
  consume the same future queue window.

## Deliberately rejected

`token_reward_direct_plus_grpo` is forbidden. In the audited fork it was a
direct sum of two objectives rather than a validated handoff mechanism, and an
old top-k path broadcast the GRPO term across the `K` axis, changing its dose.
The public patch set removes that dormant local edit entirely.

We also do not treat a manually chosen decay schedule as evidence of an
adaptive boundary. A schedule may be a useful baseline, but it cannot answer
which observable signal predicts future relative gain.

## Single-card memory recipe

The failed long-context runs materialized differentiable full-vocabulary logits
and FP32 normalization during actor update. The dominant allocation was not the
model parameters or vLLM KV cache. The accepted recipe therefore uses the
sampled-token estimator, fused selected-token log-probability paths, teacher
parameter offload, activation offload, SDPA, and optional teacher entropy.

On one 96GB RTX PRO 6000, `B=64`, `n=4`, and response cap `7168` completed three
consecutive OPD updates with a one-second NVML peak of 47,138 MiB. This is a
resource gate, not a quality result.

The exact local diffs are in `patches/`, and the upstream file hashes are in the
training configuration and public manifest.

