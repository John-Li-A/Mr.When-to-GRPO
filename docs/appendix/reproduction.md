# Reproduction

## 1. Install the public utilities

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,test]"
pytest
```

That environment is sufficient for the CPU demo. For the GPU reference path,
use Python 3.12 and install the exercised stack:

```bash
pip install -r requirements/reference-runtime.txt
```

## 2. Acquire external assets

Clone the training source at the pinned commit, install its verl package, and
apply the five public patches:

```bash
git clone https://github.com/thunlp/OPD.git external/OPD
git -C external/OPD checkout 4532fd35ccfdde82adc918b265e4c964534e83d1
pip install -e external/OPD/verl --no-deps

for patch in patches/*.patch; do
  git -C external/OPD apply "$(realpath "$patch")"
done
```

Materialize MATH-12K directly from that checkout:

```bash
python scripts/materialize_math12k.py \
  --opd-root external/OPD \
  --output data/MATH/train.parquet
```

The command checks the pinned Git commit, raw JSON and parquet hashes, and all
12,000 JSON→native-verl row mappings. Download the student, teacher, and
evaluation panels listed in `data/README.md`. `prepare_protocol.py` separately
checks row count, prompt identity, columns, level distribution, and SHA-256.

The intentionally excluded historical edit to
`compute_token_reward_direct_plus_grpo_advantage` is not part of the protocol.

## 3. Compose machine-local configuration

Edit only the `CHANGE_ME` paths in the model, training, and evaluation
fragments. Then run:

```bash
python scripts/compose_config.py \
  configs/models/qwen3_math.yaml \
  configs/train/paired_handoff.yaml \
  configs/gates/rtx_pro_6000_96gb.yaml \
  configs/eval/math.yaml \
  --output configs/experiment.yaml

when2grpo-doctor --config configs/experiment.yaml --check-paths
```

`configs/experiment.yaml` is ignored because it may contain local paths.

## 4. Materialize and audit the protocol

```bash
python scripts/prepare_protocol.py --config configs/experiment.yaml
when2grpo-plan --config configs/experiment.yaml
python scripts/stage_protocol.py --config configs/experiment.yaml
```

This produces held-out splits, a canonical manifest, a deterministic discovery
queue, materialized schedules, run specifications, and dose accounting. The
preparation step fails on dataset drift, prompt leakage, tokenizer mismatch, or
unapproved source changes.

## 5. Inspect commands before GPU execution

```bash
when2grpo-launch \
  --config configs/experiment.yaml \
  --run-id paired_handoff_v1_discovery_trunk_opd_t0_t10 \
  --dry-run
```

Before emitting the command, the launcher hashes the complete pre-intervention
state. Paired arms at the same branch point must match on actor, optimizer,
scheduler/RNG, dataloader, future prompt window, source, config, and rollout
seed. The command and identity are recorded as structured JSON. Actual
execution requires explicit authorization:

```bash
HANDOFF_GPU_AUTHORIZED=1 when2grpo-launch \
  --config configs/experiment.yaml \
  --run-id paired_handoff_v1_discovery_trunk_opd_t0_t10
```

Run resource gates before a long trunk when the GPU model, framework, context
length, or batch shape changes. The provided runtime fragment is evidence for a
single 96GB RTX PRO 6000 only.

## 6. Audit rollouts and evaluate endpoints

```bash
when2grpo-audit \
  --rollout-dir rollouts/RUN_ID \
  --n 4 \
  --response-cap 7168 \
  --output artifacts/RUN_ID.signal.json

python scripts/evaluate.py \
  --config configs/experiment.yaml \
  --model-path merged/MODEL_ID \
  --model-id MODEL_ID \
  --dataset CHANGE_ME/data/MATH-500/test.parquet \
  --panel MATH-500 \
  --n 4 \
  --output-dir evaluations/MODEL_ID__math500 \
  --execute
```

Endpoint claims require the paired OPD and RL summaries, not training-batch
accuracy. Preserve the generated manifests and hashes with every result.

## 7. Assemble the handoff surface

Plan endpoint evaluations and run the generated `argv` for every registered
pair. Each generated evaluation command already includes `--execute`:

```bash
python scripts/plan_evaluation.py --config configs/experiment.yaml
```

After each evaluation directory contains `summary.json`, combine the matched
endpoints and optional pre-branch signals:

```bash
when2grpo-surface \
  --plan artifacts/handoff_math/evaluation_plan.json \
  --signals artifacts/handoff_math/checkpoint_signals.json \
  --root . \
  --output-dir results/my-sweep
```

Omit `--signals` when only the paired endpoint surface is required. The command
checks the planned model ID and model-directory hash for each endpoint, then
refuses to join summaries whose dataset, row count, sampling recipe, response
cap, thinking mode, verifier source, or evaluation protocol differ.
