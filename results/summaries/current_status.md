# Current status

Completed:

- student and teacher MATH-500 baselines under the locked generation recipe;
- dataset, tokenizer, source, and train/eval leakage audits;
- exact `B=64`, `n=4`, cap-7168 OPD resource gate on one 96GB GPU;
- complete-checkpoint OPD-to-RL resume gate;
- ten-update fresh OPD and fresh RL pilot executions from the base student;
- exact prompt-queue audit for the ten-update OPD pilot;
- CPU tests for split, queue, branch, estimator, and rollout-audit semantics.

Not completed:

- paired held-out endpoint evaluation for the two ten-update arms;
- branch interventions at later checkpoints;
- calibration of any monitoring signal against local future gain;
- a fixed or adaptive handoff controller;
- multi-seed or cross-model validation.

The strongest defensible claim is that the intended causal experiment is now
well specified and technically runnable. A handoff law has not been found.

