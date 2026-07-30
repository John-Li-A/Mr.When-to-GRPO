# Mr. When-to-GRPO

**A paired-intervention toolkit for OPD→GRPO switch-point discovery.**

Recent OPD and hybrid post-training work increasingly relies on signals such as
verifier contrast, teacher–student support gap, entropy, or fixed decay
schedules to decide how long a student should keep following a teacher. These
signals are difficult to compare when the dataset, prompt stream, response cap,
training dose, or endpoint evaluation changes at the same time.

Mr. When-to-GRPO provides the missing test bench. Given a set of candidate
checkpoints, it forks each checkpoint into two matched futures—continued OPD and
switched GRPO—and measures which one yields greater held-out gain.

It does not prescribe a universal handoff rule. It makes handoff hypotheses
testable.

## What it produces

```text
candidate checkpoint S_t
        |
        +-- continue sampled-token OPD for H updates -- evaluate E_OPD
        |
        +-- switch to Dr.GRPO for H updates ----------- evaluate E_GRPO

paired label at t:  Delta(t) = E_GRPO - E_OPD
```

Sweeping several checkpoints produces a handoff surface:

| switch step | mixed-group rate | OPD score | cap-hit rate | OPD endpoint | GRPO endpoint | GRPO − OPD |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | … | … | … | … | … | … |
| 20 | … | … | … | … | … | … |
| 40 | … | … | … | … | … | … |

Candidate signals are measured before the branch. Endpoint gain is measured on
the same held-out panel after an equal update horizon. The resulting table can
be used to inspect a fixed schedule, calibrate a proposed signal, or compare
several handoff heuristics.

## Included tools

- `when2grpo-doctor`: validate branch geometry, response caps, panels, and paths
- `when2grpo-plan`: materialize matched OPD/GRPO branches over candidate points
- `when2grpo-launch`: emit or execute the pinned verl command for one branch
- `when2grpo-audit`: recover verifier contrast, truncation, format, and OPD signals from rollouts
- `when2grpo-surface`: join paired endpoint summaries and optional signals into CSV/JSON

The harness also locks prompt order, rollout seeds, checkpoint identity,
dataloader continuation, model/tokenizer provenance, and train/eval leakage
checks. Those controls are the difference between a switch-point plot and a
paired intervention.

## Quick start: inspect the CPU-only surface demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest

when2grpo-audit \
  --rollout-dir data/examples \
  --n 4 \
  --response-cap 32

when2grpo-surface \
  --plan data/examples/surface_plan.json \
  --signals data/examples/signals.json \
  --root . \
  --output-dir /tmp/when2grpo-surface
```

No GPU, model weights, or external datasets are needed for this demo.

## Run a handoff sweep

The repository ships a Qwen3 Math reference recipe as four composable fragments.
Edit the `CHANGE_ME` paths, then compose and inspect the experiment:

```bash
pip install -e ".[data,test]"

python scripts/compose_config.py \
  configs/models/qwen3_math.yaml \
  configs/train/paired_handoff.yaml \
  configs/gates/rtx_pro_6000_96gb.yaml \
  configs/eval/math.yaml \
  --output configs/experiment.yaml

when2grpo-doctor --config configs/experiment.yaml --check-paths
python scripts/prepare_protocol.py --config configs/experiment.yaml
when2grpo-plan --config configs/experiment.yaml --dry-run
python scripts/stage_protocol.py --config configs/experiment.yaml
python scripts/plan_evaluation.py --config configs/experiment.yaml
```

Inspect a generated command before authorizing GPU execution:

```bash
when2grpo-launch \
  --config configs/experiment.yaml \
  --run-id paired_handoff_v1_discovery_trunk_opd_t0_t10 \
  --dry-run

HANDOFF_GPU_AUTHORIZED=1 when2grpo-launch \
  --config configs/experiment.yaml \
  --run-id paired_handoff_v1_discovery_trunk_opd_t0_t10
```

After the planned endpoint evaluations write their `summary.json` files:

```bash
when2grpo-surface \
  --plan artifacts/handoff_math/evaluation_plan.json \
  --signals artifacts/handoff_math/checkpoint_signals.json \
  --root . \
  --output-dir results/my-sweep
```

`--signals` is optional. Its JSON object maps each branch step to any nested
numeric signal dictionary; scalar fields are flattened into the surface table.

## Built-in signal families

- **Verifier availability:** all-fail, mixed, all-correct, and zero-GRPO-contrast group rates
- **Teacher pressure:** sampled-token OPD score and length-normalized score
- **Generation health:** response length, cap-hit rate, and output-format rate
- **Support diagnostics:** student/teacher top-k overlap and teacher mass outside student support
- **Objective relation:** OPD/GRPO gradient norms and cosine on frozen trajectories

Users can add candidate signals without changing the paired branch protocol.
The tool treats them as predictors to be calibrated against future relative
gain, not as handoff rules by definition.

## Reference recipe

The included recipe uses Qwen3-1.7B-Base as student, a Qwen3-4B-GRPO teacher,
MATH-12K training prompts, and a pinned THUNLP OPD/verl commit. It has exercised:

- exact `B=64`, `n=4`, response-cap-7168 sampled-token OPD on one 96GB GPU;
- complete checkpoint continuation from OPD into native Dr.GRPO;
- deterministic prompt queues and exact rollout-to-schedule audits;
- matched student/teacher evaluation and rollout signal recovery.

These artifacts validate the harness and provide a reproducible starting point.
They are not presented as a universal OPD→GRPO boundary. See
[the reference recipe](docs/04_reference_recipe.md) and
[scope and non-goals](docs/05_scope_and_non_goals.md).

## Repository map

- `src/when_to_grpo/`: branch planning, config checks, rollout audits, and surface assembly
- `scripts/`: data preparation, staged execution, checkpoint merge, and evaluation
- `configs/`: model, training, hardware, and evaluation fragments
- `patches/`: minimal diffs against the pinned OPD/verl source
- `data/examples/`: synthetic CPU-only examples of the rollout and surface schemas
- `results/`: checked reference observations; no raw trajectories or checkpoints
- `docs/`: causal question, recipe audit, design lessons, reference recipe, and scope

The full experiment workspace is intentionally excluded. Model weights,
checkpoints, raw generations, local machine paths, and retired launchers do not
belong in a reusable switch-point tool.
