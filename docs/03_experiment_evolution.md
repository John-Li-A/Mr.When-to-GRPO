# Experiment evolution

The project changed substantially as assumptions were tested.

1. **A nominal recipe was not a measured recipe.** Early smoke tests used fewer
   prompts, fewer samples, and a shorter response cap than the proposed formal
   run. Resource gates were rebuilt around the true `B x n x cap` shape.
2. **Task difficulty controlled RL signal availability.** On early DAPO
   batches, 185 of 192 groups were all-fail. That made native group-relative RL
   nearly silent and could not support a short paired study. MATH-12K became the
   training distribution only after a matched MATH-500 panel showed substantial
   nondegenerate student behavior and a large teacher gap.
3. **Short training caps and long evaluation caps were invalid.** A 512-token
   training cap did not match 7k-token evaluation and risked measuring
   truncation rather than learning. The protocol now locks both to 7168.
4. **The long-context OOM was localized to actor update.** Full-vocabulary
   top-k normalization, not KV cache, caused the peak. Sampled-token OPD removed
   this allocation without changing the task or response length.
5. **A switch must be a paired causal comparison.** Looking at OPD and RL curves
   from different states or prompt sequences does not identify a handoff. The
   current protocol hashes checkpoint identity and materializes future prompt
   windows before either arm runs.
6. **Training-batch accuracy is not an endpoint metric.** Different batches have
   different problems. Step-to-step training accuracy is kept as health context;
   the handoff label comes from a fixed held-out panel.

The result is intentionally less ambitious than an immediate learned
controller, but more defensible: first produce trustworthy local future-gain
labels, then ask whether a monitor generalizes.

