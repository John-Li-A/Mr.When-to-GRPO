# Research question

Let `S_t` denote the complete training state at update `t`: actor, optimizer,
scheduler, dataloader cursor, random states, rollout seed, configuration, and
prompt queue identity. From `S_t`, run two interventions over the same next `H`
prompt batches:

- `I_OPD`: continue native sampled-token OPD;
- `I_RL`: switch to native Dr.GRPO.

Evaluate both endpoints on the same held-out panel. The local label is

```text
Delta_future(t, H) = score(I_RL(S_t), t+H) - score(I_OPD(S_t), t+H)
```

The starting score cancels algebraically but is retained in reports to show
absolute capability. A positive value favors RL for that checkpoint and
horizon; a negative value favors continued OPD. Values inside a preregistered
equivalence band are reported as indistinguishable.

## Candidate signals

Signals are measured at or before the branch point and must not use endpoint
labels. The current registry includes:

- fraction of all-fail, mixed, and all-correct rollout groups;
- mean absolute group-relative verifier advantage;
- sampled-token OPD score and its length-normalized proxy;
- response length, cap-hit rate, and boxed-format rate;
- student/teacher top-k overlap and teacher mass outside student support on a
  frozen diagnostic probe;
- OPD and RL gradient norms and cosine on the same frozen trajectories.

The first goal is calibration: determine whether any signal is monotonically
related to `Delta_future` across paired checkpoints. Only then is it sensible to
fit or hand-design a switching policy.

## Controls

The comparison fixes prompt order, update count, rollout count, response cap,
sampling settings, verifier, tokenizer, initial state, and endpoint evaluation.
OPD and RL keep their native loss aggregation and advantage semantics; making
their formulas identical would erase the intervention being studied. Compute
and generated-token counts are reported as secondary dose measures.

