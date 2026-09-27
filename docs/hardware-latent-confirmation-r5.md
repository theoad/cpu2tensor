# Hardware latent confirmation R5

Preregistered 2026-09-27. R5 prospectively confirms the benign-only R4 geometry
on one new physical-host session. It does not retrain models and receives no
effect or vulnerability data.

## Frozen capture

- Exact retained boot `31179aea-9a43-4c5d-8bf6-1205735e42c4`, Ubuntu kernel
  `5.13.0-30-generic`, target CPU 2, controller CPU 3, no turbo, and 1.8 GHz cap.
- Seeded plan kind `session-a`, cohort seed `2026092740`, 1,020 rows balanced
  across 17 families and three intensities, with 204 trace-independent raw
  captures. Plan SHA-256:
  `0724fa230f8913fc1037be11b2a929e6857f47fba52e6553bad23a5fce556863`.
- Admission requires 1,020 first-attempt executions, zero rejected attempts,
  zero lost/missing/multiplexed required sources, restored CPU policy, and a
  hash-verified off-host copy before any cleanup.
- The bounded service starts only with at least 4 GiB root free and 4 GiB
  available RAM; it stops below 3 GiB free, at 80°C, on policy drift, or after
  five minutes of collection. Systemd supplies a ten-minute hard stop and an
  independent delayed policy-restoration action.

## Frozen representation gate

All three reconstruction and VICReg encoders from R3 remain frozen. Cohort A
supplies benign nearest-centroid references; prospective cohort D supplies all
evaluation rows. VICReg must satisfy, for every seed, at least 0.93 family,
0.55 intensity, 0.93 family/intensity-stratum, and 0.58 masked-view identity
accuracy. Median thresholds are 0.94, 0.58, 0.94, and 0.60. Median VICReg must
also beat reconstruction by at least 0.03 family, 0.10 intensity, and 0.05 view
identity accuracy. Every clause must pass.

This gate confirms a foundation representation candidate, not an anomaly model.
If it passes, the next stage is paired task-post-training sample efficiency
against scratch across several held-out effect-family splits.

## Result

Pending.
