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
- sampled-token OPD score and its length-normalized proxy;
- response length, cap-hit rate, verifier outcome, and boxed-marker rate.

Other signals may be supplied to the surface builder as external JSON. They are
not described as built in until the repository contains their extraction path.

The first goal is calibration: determine whether any signal is monotonically
related to `Delta_future` across paired checkpoints. Only then is it sensible to
fit or hand-design a switching policy.

## Controls

The comparison fixes prompt order, update count, rollout count, response cap,
sampling settings, verifier, tokenizer, initial state, and endpoint evaluation.
OPD and RL keep their native loss aggregation and advantage semantics; making
their formulas identical would erase the intervention being studied. Compute
and generated-token counts are reported as secondary dose measures.
