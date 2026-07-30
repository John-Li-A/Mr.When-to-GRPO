# Metric dictionary

| Metric | Definition | Role |
|---|---|---|
| trajectory pass rate | correct sampled completions / all completions | capability and verifier health |
| mixed-group rate | prompt groups with at least one correct and one incorrect sample | availability of native group-relative RL contrast |
| all-fail rate | prompt groups with zero correct samples | RL dead-zone indicator |
| zero native-GRPO task-gradient rate | all-fail or all-correct groups / all groups | direct measure of absent task contrast before other terms |
| OPD score | teacher log-probability minus student log-probability on sampled tokens, aggregated by the pinned estimator | dense teacher signal magnitude |
| cap-hit rate | completions whose exact token length reaches the response cap | truncation diagnostic |
| avg@n | mean verifier correctness over `n` samples per problem | primary endpoint metric |
| pass@n | problems with at least one correct sample / all problems | secondary endpoint metric |
| local future gain | paired RL endpoint score minus paired OPD endpoint score | handoff label |

Training-batch pass rate is not compared across steps as a learning curve unless
the prompt set is fixed.

