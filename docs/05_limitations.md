# Limitations

- The study currently covers one student/teacher pair and mathematical
  reasoning. A boundary found here need not transfer to another model scale,
  teacher quality, verifier, or domain.
- Only early pilot arms have run. There are not enough paired branch labels to
  estimate a stable relation between monitoring signals and future gain.
- The student baseline has a substantial 7168-token cap-hit rate on full
  MATH-500. Length and truncation must be reported with every endpoint.
- The teacher is a post-RL 4B model. Conclusions may conflate objective choice
  with teacher quality and capacity; a weaker or differently trained teacher
  could change the result.
- Rule verifiers are fallible around answer extraction and formatting. The
  current work audits verifier inputs and keeps format rate separate, but does
  not eliminate verifier error.
- Equal update count is not equal wall-clock cost. Generated tokens and wall
  time are recorded so conclusions can be restated under a compute budget.
- The low-memory verl patches are tested against a single pinned upstream
  commit. They should not be assumed to apply cleanly to later versions.
- Raw trajectories and checkpoints are not redistributed. Reproduction requires
  acquiring the models and datasets under their own licenses.

