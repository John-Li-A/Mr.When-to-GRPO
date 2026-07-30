# Mr. When-to-GRPO

**A paired-intervention toolkit for OPD→GRPO switch-point discovery.**

Many post-training recipes use verifier contrast, teacher–student disagreement,
entropy, or a fixed schedule to decide when a student should stop following a
teacher and rely on reinforcement learning. Those signals are hard to compare
when the prompt stream, rollout dose, response cap, or endpoint evaluation also
changes.

Mr. When-to-GRPO turns the question into a matched experiment. At each candidate
checkpoint, it forks the same student state into two futures:

```text
checkpoint S_t
    ├── continue sampled-token OPD for H updates ── evaluate E_OPD
    └── switch to Dr.GRPO for H updates ─────────── evaluate E_GRPO

paired outcome: Δ(t) = E_GRPO - E_OPD
```

It does not claim a universal handoff rule. It provides the launcher, identity
checks, rollout diagnostics, and surface builder needed to test one.

## Outputs

A sweep over configurable checkpoints such as `[0, 20, 40]` produces one row
per matched fork:

| branch point | candidate signal(s) | OPD endpoint | GRPO endpoint | GRPO − OPD | label |
|---:|---:|---:|---:|---:|---|
| 0 | … | … | … | … | … |
| 20 | … | … | … | … | … |
| 40 | … | … | … | … | … |

The label uses a configurable equivalence band. It is a display convention,
not a statistical significance claim.

The paired comparison locks:

- source checkpoint, actor, optimizer, scheduler/RNG, and dataloader state;
- future prompt batches, rollout seed, sample count, and response cap;
- dataset, tokenizer, verifier implementation, and sampling parameters;
- endpoint model identity and held-out evaluation panel.

## Included commands

- `when2grpo-doctor` — validate branch geometry, runtime fields, panels, and paths
- `when2grpo-plan` — materialize matched OPD/GRPO run specifications
- `when2grpo-launch` — hash the pre-intervention state, then emit or execute one pinned verl run
- `when2grpo-audit` — recover verifier contrast, OPD score, length, truncation, and format signals
- `when2grpo-surface` — join fully matched endpoint summaries into CSV and JSON

## CPU-only demo

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

The demo uses synthetic records and needs no model, GPU, or external dataset.

## Reference recipe

The included recipe is one tested starting point:

- student: `Qwen/Qwen3-1.7B-Base`;
- teacher: `lllyx/Qwen3-4B-Base-GRPO` at revision
  `1f3b2966edfb75f2f98a00617588c1f748088422`;
- training data: the MATH-12K native-verl artifact in pinned THUNLP OPD;
- objectives: sampled-token OPD and native Dr.GRPO;
- rollout shape: `B=64`, `n=4`, response cap 7168;
- exercised hardware: one 96GB RTX PRO 6000.

The checked evidence covers exact-shape OPD updates, an OPD→Dr.GRPO
checkpoint-resume gate, prompt-queue audits, and matched student/teacher
evaluation. It validates the execution path, not a preferred switch point. See
[reference evidence](results/reference_evidence.json) and the
[scope statement](docs/05_scope_and_non_goals.md).

## Full setup

Clone the pinned training source and install the tested runtime:

```bash
git clone https://github.com/thunlp/OPD.git external/OPD
git -C external/OPD checkout 4532fd35ccfdde82adc918b265e4c964534e83d1

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/reference-runtime.txt
pip install -e external/OPD/verl --no-deps
pip install -e ".[data,test]"

for patch in patches/*.patch; do
  git -C external/OPD apply "$(realpath "$patch")"
done
```

The reference path uses SDPA and does not require FlashAttention. CUDA driver
compatibility still depends on the host; the recorded stack uses PyTorch 2.8.0
and CUDA 12.8-compatible wheels.

Materialize the exact training artifact from the pinned checkout. The script
checks the Git commit, both upstream file hashes, and all 12,000 JSON→parquet
row mappings before copying the byte-identical parquet:

```bash
python scripts/materialize_math12k.py \
  --opd-root external/OPD \
  --output data/MATH/train.parquet
```

Download the two models and evaluation panels listed in
[`data/README.md`](data/README.md), then replace the `CHANGE_ME` paths and
compose the machine-local config:

```bash
python scripts/compose_config.py \
  configs/models/qwen3_math.yaml \
  configs/train/paired_handoff.yaml \
  configs/gates/rtx_pro_6000_96gb.yaml \
  configs/eval/math.yaml \
  --output configs/experiment.yaml

when2grpo-doctor --config configs/experiment.yaml --check-paths
python scripts/prepare_protocol.py --config configs/experiment.yaml
when2grpo-plan --config configs/experiment.yaml
python scripts/stage_protocol.py --config configs/experiment.yaml
python scripts/plan_evaluation.py --config configs/experiment.yaml
```

Inspect one command and its recorded pre-intervention identity before running:

```bash
when2grpo-launch \
  --config configs/experiment.yaml \
  --run-id paired_handoff_v1_discovery_trunk_opd_t0_t10 \
  --dry-run

HANDOFF_GPU_AUTHORIZED=1 when2grpo-launch \
  --config configs/experiment.yaml \
  --run-id paired_handoff_v1_discovery_trunk_opd_t0_t10
```

Each evaluation command generated by `plan_evaluation.py` is executable and
writes a model identity manifest plus a summary. After every paired endpoint is
present:

```bash
when2grpo-surface \
  --plan artifacts/handoff_math/evaluation_plan.json \
  --signals artifacts/handoff_math/checkpoint_signals.json \
  --root . \
  --output-dir results/my-sweep
```

`--signals` is optional. Its JSON object maps branch points to nested numeric
diagnostics; scalar values are flattened into the output table.

## What the rollout audit currently measures

- all-fail, mixed, all-correct, and zero-GRPO-contrast group rates;
- sampled-token OPD trajectory score and length-normalized proxy;
- response length, cap-hit rate, verifier outcome, and output-format rate.

Additional candidate signals can be joined as external JSON. The repository
does not advertise a signal family until its extraction path is implemented.

## Repository map

- `src/when_to_grpo/` — protocol checks, identity locking, rollout audit, and surface assembly
- `scripts/` — data materialization, protocol staging, checkpoint merge, and evaluation
- `configs/` — model, training, hardware, and evaluation fragments
- `patches/` — five audited diffs against the pinned OPD/verl source
- `data/examples/` — synthetic CPU-only schema examples
- `results/` — compact reference-path evidence; no claimed handoff result
- `docs/` — research question, recipe audit, scope, provenance, and [related work](docs/06_related_work.md)

Model weights, checkpoints, raw trajectories, local paths, and generated
experiment workspaces are intentionally excluded.
