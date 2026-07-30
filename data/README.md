# Data

No training or evaluation dataset is redistributed in this repository.

The expected training artifact is the MATH-12K native-verl parquet produced by
the conversion recipe in `sail-sg/understand-r1-zero` at commit
`dfca49dd460ee7cc8e4a5a162c876a7fd6993b87`. Its expected identity is:

- rows: 12,000
- unique prompt identities: 11,998
- SHA-256: `53e815b8781e3b4513c7cf9eb4a003b4f4af27198f1d06458336785626b0229d`
- required columns: `prompt`, `level`, `id`, `data_source`, `ability`,
  `reward_model`, `extra_info`

Evaluation panels are MATH-500, AMC23, AIME24, and AIME25 in the native OPD
parquet schema. `prepare_protocol.py` checks exact prompt overlap between train
and evaluation data and stops on leakage.

Users are responsible for obtaining each dataset from its original source and
following its license. Everything in `examples/` is synthetic. The JSONL file
documents the rollout-audit schema; the plan, signals, and endpoint summaries
exercise `when2grpo-surface` without a GPU.
