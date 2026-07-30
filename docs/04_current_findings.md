# Current findings

## Capability and signal checks

With four samples per MATH-500 problem, the Qwen3-1.7B-Base student reached
25.50% avg@4 and the Qwen3-4B-GRPO teacher reached 84.30%. The teacher improved
the four-sample mean on 442 of 500 problems. Of 213 student all-fail problems,
the teacher solved at least one sample on 178.

On the fixed 64-problem matched signal panel, the student produced 79 correct
trajectories out of 256 (30.86%). Forty-seven of 64 groups were mixed (73.44%).
This panel can measure both improvement and degradation over a short horizon.

The contrast with early DAPO rollouts was large: 185 of 192 groups were
all-fail, six had one correct sample, and one had two. Only 3.65% of groups
supplied native GRPO task contrast. Dataset choice is therefore part of the
causal design, not a cosmetic benchmark decision.

## Systems checks

- `B=64`, `n=4`, response cap 7168 sampled-token OPD passed a three-update
  resource gate on one 96GB GPU.
- One-second NVML peak was 47,138 MiB during that gate.
- A complete OPD checkpoint was resumed under native Dr.GRPO while preserving
  model, optimizer, scheduler, RNG, and dataloader continuation semantics.
- A formal ten-update OPD pilot processed 640 prompts and 2,560 trajectories in
  2:44:35; all 640 scheduled prompts matched the canonical queue exactly.
- The corresponding fresh ten-update RL run completed, but paired held-out
  endpoint evaluation was not completed before the experiment was stopped.

## What these results do not show

They do not identify a handoff step, validate a monitor, or show that OPD is
better early and RL better late. They establish a viable model/task pair and a
counterfactual protocol capable of testing those claims.

