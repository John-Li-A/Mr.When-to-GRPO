# Mr. When-to-GRPO

**A paired-branch recipe and training launcher for testing when OPD should
hand off to GRPO.**

On-policy distillation (OPD) gives a student a dense teacher signal. Verifier
RL gives it a sparse external outcome signal. This project asks a narrower
question than “which objective is better?”:

> At a given post-training checkpoint, which observable signals predict that
> another fixed budget of verifier RL will improve held-out performance more
> than the same budget of continued OPD?

The unit of evidence is a paired intervention. From the same checkpoint, OPD
and native Dr.GRPO consume the same future prompt window for the same number of
updates. Their held-out gain is then compared. Training statistics are treated
as candidate predictors, not as proof of a handoff rule.

## Status

This is a work in progress. The data audit, deterministic prompt queue,
checkpoint identity checks, native OPD-to-RL resume path, long-context memory
recipe, and rollout signal audit have been exercised. Fresh 10-update OPD and
RL pilot arms also completed. Paired endpoint evaluation and the later
checkpoint interventions are not complete, so this repository does **not**
claim a universal handoff point.

Verified observations so far:

| Check | Observation |
|---|---|
| MATH-500 starting point | Qwen3-1.7B-Base avg@4 25.50%; Qwen3-4B-GRPO teacher avg@4 84.30% |
| Matched 64-problem signal panel | 79/256 correct trajectories; 73.44% mixed rollout groups |
| Early DAPO batches | 185/192 groups all-fail; only 3.65% supplied native GRPO task contrast |
| Single-card OPD resource gate | B=64, n=4, cap=7168 completed for three updates; 47,138 MiB one-second NVML peak |
| Formal OPD pilot | 10 updates, 640 prompts, 2,560 trajectories; exact queue audit passed |

These numbers establish that task distribution changes the availability of an
RL signal, and that the intended training shape is feasible. They do not yet
establish when switching objectives is optimal.

## Protocol in one diagram

```text
student checkpoint S_t
        |
        +-- continue native sampled-token OPD for H updates -- evaluate E_OPD
        |
        +-- switch to native Dr.GRPO for H updates ---------- evaluate E_RL

local future gain label: E_RL - E_OPD
candidate signals at S_t: verifier contrast, OPD score, response length and
truncation, teacher/student support, and OPD/RL gradient relation
```

The primary comparison uses checkpoint state, prompt order, rollout settings,
response cap, update count, and evaluation panel as controlled variables. See
[the research question](docs/01_research_question.md) and
[the protocol notes](docs/02_recipe_audit.md).

## CPU quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest

when2grpo-audit \
  --rollout-dir data/examples \
  --n 4 \
  --response-cap 32
```

To prepare the full experiment, install `.[data]`, edit the path placeholders
in the four configuration fragments, and compose them:

```bash
python scripts/compose_config.py \
  configs/models/qwen3_math.yaml \
  configs/train/paired_handoff.yaml \
  configs/gates/rtx_pro_6000_96gb.yaml \
  configs/eval/math.yaml \
  --output configs/experiment.yaml

python scripts/prepare_protocol.py --config configs/experiment.yaml
when2grpo-plan --config configs/experiment.yaml --dry-run
python scripts/stage_protocol.py --config configs/experiment.yaml
```

GPU execution is deliberately guarded by `HANDOFF_GPU_AUTHORIZED=1`. The
repository pins the upstream OPD commit and keeps local verl changes as small
patches rather than vendoring the training framework. Detailed setup is in
[reproduction](docs/appendix/reproduction.md).

## Repository map

- `src/when_to_grpo/`: split, queue, branch, signal, and command invariants
- `scripts/`: data preparation, staged execution, checkpoint merge, evaluation
- `configs/`: model, training, resource, and evaluation fragments
- `patches/`: minimal diffs against the pinned upstream OPD repository
- `data/`: acquisition instructions, public manifest, and a synthetic schema example
- `results/`: checked observations only; no raw trajectories or checkpoints
- `docs/`: motivation, protocol evolution, current findings, and limitations

The full experimental working directory is intentionally not published. It
contains checkpoints, raw generations, machine paths, and retired launchers
that would obscure rather than improve reproducibility.
