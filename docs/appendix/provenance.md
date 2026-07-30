# Provenance

## Code

- THUNLP OPD: <https://github.com/thunlp/OPD>, commit
  `4532fd35ccfdde82adc918b265e4c964534e83d1`.
- MATH-12K source JSON and native-verl parquet: the same pinned THUNLP OPD
  commit. `scripts/materialize_math12k.py` checks both file hashes and audits
  all 12,000 mappings. The underlying MATH-12K collection is also distributed
  by `sail-sg/understand-r1-zero`.
- Evaluation structure follows the JustRL vLLM evaluation scripts; reward
  computation uses the pinned OPD `ttrl_math` verifier.

The training config records normalized upstream and permitted target hashes;
the public manifest records the five patch-file hashes. The original experiment
also contained a dormant
four-line edit to the unvalidated plus estimator. It was never used by the
formal native OPD or native Dr.GRPO arms and is deliberately absent here.

## Models

- Student: `Qwen/Qwen3-1.7B-Base`.
- Teacher: `lllyx/Qwen3-4B-Base-GRPO`, revision
  `1f3b2966edfb75f2f98a00617588c1f748088422`; base model
  `Qwen/Qwen3-4B-Base`.

Local model files are identified by SHA-256 during preparation. Model weights
are not redistributed.

## Data

The protocol uses MATH-12K for training and MATH-500, AMC23, AIME24, and AIME25
for evaluation. The conversion used in the experiment contained 12,000 rows,
11,998 unique prompt identities, and two exact duplicate rows. After prompt
length filtering, 11,989 rows remained before held-out splitting.

Dataset files are not included. See `data/README.md` for identity checks and
license responsibility.

## Reference runtime

The exercised stack used Python 3.12, PyTorch 2.8.0 with CUDA 12.8 support,
vLLM 0.11.0, Ray 2.56.1, Transformers 4.55.4, and the pinned editable OPD/verl
source. Exact Python package pins are in `requirements/reference-runtime.txt`.
The reference path uses SDPA; FlashAttention is not required.
