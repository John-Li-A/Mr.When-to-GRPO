# Reference recipe checks

These observations document the Qwen3 Math execution path shipped with the
repository. They validate the harness; they are not a handoff result.

| Check | Observation |
|---|---|
| Student baseline | Qwen3-1.7B-Base: 25.50% avg@4 on MATH-500 |
| Teacher baseline | Qwen3-4B-GRPO: 84.30% avg@4 on MATH-500 |
| Fixed signal panel | 79/256 correct trajectories; 47/64 mixed groups |
| Early DAPO contrast check | 185/192 all-fail groups; 3.65% mixed groups |
| Exact-shape OPD gate | Three updates at B=64, n=4, cap 7168 on one 96GB GPU |
| Observed memory peak | 47,138 MiB at one-second NVML sampling |
| Checkpoint handoff gate | Complete OPD checkpoint resumed under native Dr.GRPO |
| OPD pilot | Ten updates; 640 prompts; 2,560 trajectories; exact queue match |

The repository does not publish a switch-point claim from these checks. A
handoff claim requires paired held-out OPD and GRPO endpoint summaries at
registered branch points, assembled by `when2grpo-surface`.
