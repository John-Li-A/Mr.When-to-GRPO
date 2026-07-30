# Figures

The primary visualization for this toolkit is a paired handoff surface:

- x-axis: registered branch checkpoint;
- y-axis: held-out `GRPO endpoint - OPD endpoint` after the same horizon;
- optional overlays: candidate signals measured before each branch.

Raw per-step training-batch correctness should not be plotted as a comparable
learning curve because successive batches contain different prompts. Figures
should be derived from paired held-out summaries produced under the same panel,
sampling count, response cap, and seed.
