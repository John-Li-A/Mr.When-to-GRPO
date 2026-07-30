# Motivation

OPD and verifier RL solve different failures in post-training.

OPD can provide a dense token-level direction before a small student has enough
successful rollouts for outcome RL. It is useful for cold start and credit
assignment, but its target remains bounded by a particular teacher and may keep
reinforcing teacher-specific errors or style after the student has acquired the
relevant skill.

Verifier RL supplies an external task oracle. It can reject teacher errors and,
in principle, let the student exceed the teacher. Its signal is nevertheless
sparse: when every sampled answer to a prompt is wrong, group-relative RL has no
within-group task contrast.

This suggests a temporal question. Early training may benefit from a dense
teacher prior; later training may benefit more from the verifier. A fixed switch
step is only a schedule. The scientific object here is whether signals visible
during training can predict the local future value of switching.

The project therefore separates three things that are often conflated:

1. **signal availability** — does the current batch contain useful teacher or
   verifier contrast?
2. **local causal value** — from the same checkpoint, which objective produces
   more held-out gain over the next fixed horizon?
3. **a deployable controller** — can a rule learned on discovery branches choose
   the objective out of sample?

Only the first two are in the current completed scope. A controller is not
claimed before enough paired branch labels exist.

