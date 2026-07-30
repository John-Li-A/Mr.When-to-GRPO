# Related work

Mr. When-to-GRPO does not introduce another OPD/RL objective. It supplies the
paired experiment needed to test claims about *when* one objective should give
way to the other.

## Foundations used by the reference path

- [Generalized Knowledge Distillation for Auto-regressive Sequence
  Models](https://arxiv.org/abs/2306.13649) established student-on-policy
  distillation and discussed combining it with reinforcement learning.
- [MiniLLM](https://arxiv.org/abs/2306.08543) is an early reverse-KL,
  on-policy distillation method for generative language models.
- [THUNLP OPD](https://github.com/thunlp/OPD) provides the sampled-token OPD
  implementation pinned by this repository.
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) and
  its [reference code](https://github.com/sail-sg/understand-r1-zero) motivate
  the Dr.GRPO arm and the use of verifier-group diagnostics.

## Why handoff measurement matters

Several recent proposals use different evidence for reducing or gating teacher
influence:

- [G-OPD / ExOPD](https://arxiv.org/abs/2602.12125) connects OPD to dense,
  KL-constrained RL and separates reference choice from reward scale.
- [ATOD](https://arxiv.org/abs/2606.27814) anneals OPD toward RL and weights
  supervision by turn-level disagreement and uncertainty.
- [Reward-Gated On-Policy Distillation](https://arxiv.org/abs/2607.04037) uses
  verifier outcomes to decide when teacher logits are trustworthy.
- [Direct-OPD](https://arxiv.org/abs/2607.05394) distills a teacher's policy
  shift rather than its absolute distribution.
- [Distilled Reinforcement Learning for LLM
  Post-training](https://arxiv.org/abs/2607.17247) combines teacher and reward
  information inside one post-training objective.
- [Demystifying On-Policy
  Distillation](https://arxiv.org/abs/2607.13399) studies teacher mismatch,
  length effects, and the distinction between better exploration and a higher
  capability ceiling.

These works do not imply one universal switch rule. A verifier gate, annealing
schedule, policy-gap statistic, or entropy heuristic remains a hypothesis
until it predicts relative future gain under matched data, dose, and endpoint
evaluation. That paired counterfactual is the narrow role of this toolkit.

All 2026 items above are arXiv preprints as of July 2026; reported results
should be treated as author claims until independently replicated.
