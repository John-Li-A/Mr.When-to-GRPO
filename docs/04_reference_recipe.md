# Qwen3 Math reference recipe

The repository includes one concrete recipe for exercising the switch-point
harness. It is a reproducible starting point, not a recommended universal
handoff schedule.

## Locked setup

- student: Qwen3-1.7B-Base;
- teacher: Qwen3-4B-GRPO;
- training prompts: MATH-12K, with a deterministic queue and held-out split;
- objectives: sampled-token OPD and native Dr.GRPO from the pinned THUNLP OPD
  source;
- rollout shape: `B=64`, `n=4`, response cap 7168;
- reference hardware: one 96GB RTX PRO 6000.
- reference runtime: Python 3.12, PyTorch 2.8.0, vLLM 0.11.0, Ray 2.56.1,
  Transformers 4.55.4, and SDPA.

Model, tokenizer, dataset, source, verifier, prompt template, and evaluation
identities are recorded in the generated protocol manifest. A change to any of
them defines a new protocol rather than a continuation of this recipe.

## Why this model/task pair is usable

Under the locked four-sample MATH-500 evaluation, the student reached 25.50%
avg@4 and the teacher reached 84.30%. On the fixed 64-problem signal panel, 47
of 64 student rollout groups were mixed. The pair therefore provides both a
teacher gap and enough verifier contrast for OPD and GRPO to be meaningfully
compared over a short branch horizon.

This choice followed a failed distribution check on early DAPO prompts: 185 of
192 groups were all-fail and only 3.65% supplied native GRPO task contrast. The
harness exposes these rates because a switch-point study is not informative
when one objective receives almost no usable signal.

## Exercised systems path

The following parts of the reference recipe have been exercised:

- three exact-shape OPD updates at `B=64`, `n=4`, cap 7168 on one 96GB GPU;
- a checkpoint-resume gate from OPD into native Dr.GRPO, covering model,
  optimizer, scheduler, RNG, and dataloader state;
- a ten-update OPD pilot covering 640 prompts and 2,560 trajectories;
- exact prompt-queue and rollout-to-schedule audits;
- matched student/teacher evaluation and rollout signal extraction.

The one-second NVML peak during the three-update resource gate was 47,138 MiB.
That number is evidence for the pinned stack and exact batch shape only; users
should rerun a resource gate after changing hardware, framework, model, context
length, or rollout count.

## Using it as a reference

Compose the four fragments in `configs/`, replace only their `CHANGE_ME` paths,
and run `when2grpo-doctor` before materializing the protocol. The generated
branch plan defines matched OPD and GRPO futures at every candidate checkpoint.
After endpoint evaluation, `when2grpo-surface` joins those futures into the
handoff surface.

The checked observations above validate the execution path. Their compact
machine-readable record is `results/reference_evidence.json`. They do not
identify a handoff point or validate any monitoring signal.
