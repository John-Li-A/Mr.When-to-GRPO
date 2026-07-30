# Scope and non-goals

## Scope

Mr. When-to-GRPO is a paired-intervention test bench for a specific question:
from a shared checkpoint and future prompt window, does switching to GRPO or
continuing OPD produce the better held-out endpoint after the same update
horizon?

The toolkit provides:

- deterministic prompt queues and branch specifications;
- matched OPD/GRPO launch plans on a pinned OPD/verl stack;
- configuration and path checks before execution;
- checkpoint, tokenizer, dataset, verifier, and evaluation identity controls;
- rollout diagnostics that can serve as candidate handoff signals;
- assembly of paired endpoint results into a handoff surface.

The included Qwen3 Math setup is a reference recipe. Other models and datasets
can be used when they satisfy the same configuration, verifier, and endpoint
contracts, but they should be validated with their own resource and signal
gates.

## Non-goals

The project does not:

- prescribe a universal OPD-to-GRPO handoff step or signal threshold;
- claim that OPD is always preferable early or GRPO always preferable late;
- introduce a new distillation loss, RL estimator, or adaptive controller;
- hide differences in prompt order, rollout dose, response cap, or evaluation
  behind a nominally paired comparison;
- support every post-training framework through a generic plugin layer;
- redistribute model weights, datasets, raw trajectories, or checkpoints.

## Interpretation boundaries

A handoff surface is local to its student, teacher, task distribution,
verifier, prompt template, rollout recipe, branch horizon, and endpoint panel.
Equal update count also need not mean equal token or wall-clock cost, so those
quantities should be reported alongside endpoint gain.

Rule verifiers can fail around extraction and formatting. Format rate,
truncation, and cap-hit rate therefore remain separate diagnostics rather than
being silently folded into task reward. A post-RL teacher may also confound
teacher quality with capacity or objective history. These limitations do not
invalidate the paired intervention, but they constrain what can be inferred
from it.

The tool makes a handoff hypothesis testable. Generality must come from
replication across protocols, not from the name of a signal.
