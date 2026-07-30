# Data

No training or evaluation dataset is redistributed in this repository.

The expected training artifact is published in `thunlp/OPD` at commit
`4532fd35ccfdde82adc918b265e4c964534e83d1`. Run
`scripts/materialize_math12k.py` against that checkout. It verifies the raw
JSON, the native parquet, and every row mapping before copying the artifact.
The locked identities are:

- source JSON SHA-256: `ad41fe4ffc830efcdac9fc58b477d9d91a74d5c4c687e275a800af2fa58ae5b3`
- rows: 12,000
- unique prompt identities: 11,998
- native parquet SHA-256: `53e815b8781e3b4513c7cf9eb4a003b4f4af27198f1d06458336785626b0229d`
- required columns: `prompt`, `level`, `id`, `data_source`, `ability`,
  `reward_model`, `extra_info`

Evaluation panels are MATH-500, AMC23, AIME24, and AIME25 in the native OPD
parquet schema. `prepare_protocol.py` checks exact prompt overlap between train
and evaluation data and stops on leakage.

Users are responsible for obtaining each dataset from its original source and
following its license. Everything in `examples/` is synthetic. The JSONL file
documents the rollout-audit schema; the plan, signals, and endpoint summaries
exercise `when2grpo-surface` without a GPU.
