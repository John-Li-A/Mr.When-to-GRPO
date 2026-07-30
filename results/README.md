# Results

This directory contains compact reference observations and the output schema
for handoff surfaces. Checkpoints, full trajectories, machine logs, and
generated datasets are intentionally excluded.

- `tables/validated_checks.csv` records the empirical gates exercised by the
  Qwen3 Math reference recipe.
- `summaries/reference_recipe.md` explains what those gates do and do not
  validate.
- `figures/README.md` defines which comparisons are suitable for plotting.

Users should write each sweep to a separate output directory with
`when2grpo-surface`; generated experiment results are not bundled as a claimed
universal boundary.
