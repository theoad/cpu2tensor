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

The third launch completed after two safe pre-capture thermal rejections. The
first two attempts observed package temperature at or above the 70°C start
gate, produced no output, and left CPU policy unchanged. They also exposed that
systemd ignores `RuntimeMaxSec` for `Type=oneshot`; the unit was corrected and
committed as `f148ba4` with `Type=exec` before the successful attempt. The host
cooled to 48°C before that launch.

The prospective session completed 1,020/1,020 first-attempt executions in 48.42
seconds (21.07 executions/s), with zero rejection, loss, or required-source
failure. It retained exactly 204 planned raw captures and 1,020 derived tensors.
CPU policy restored to turbo enabled and a 4.9 GHz maximum; the independent
restore timer was removed after clean restoration.

All 1,230 files were copied off host and matched a complete remote hash list.
The hash-list SHA-256 is
`53462a78ea0734a2331f4f822f0f9508f038ecaccf1c0aea5e806b3f9b5b3468`;
the capture-manifest SHA-256 is
`002222431a58e29cd31523fbb1a44eb77e4088c74b2b2a7a14be286282bb780d`.

The locked representation gate did not fully pass:

| Objective | Family | Intensity | Stratum | Masked-view identity |
| --- | ---: | ---: | ---: | ---: |
| Reconstruction | 90.5% | 47.0% | 98.3% | 49.8% |
| VICReg | 95.9% | 60.9% | 96.9% | 56.5% |

VICReg passed every family, intensity, and stratum absolute threshold and all
three paired margins over reconstruction. Masked-view identity failed both its
60% median threshold and 58% per-seed floor (the three seeds reached 57.6%,
56.5%, and 56.1%). This metric remains far above its 0.39% chance rate and beats
reconstruction by 6.7 points, but the preregistered conjunction is **NO-GO**.

The generic structure is mostly prospective and reproducible; invariance under
the fixed masking perturbation is not strong enough. Cohort D now becomes
development data. The next bounded ablation may strengthen label-free view
invariance, but any chosen objective needs another untouched physical session.

Off-host capture:
`/Users/theoad/.cache/cpu2tensor/hardware-latent-confirmation-r5-92618e9`.
Evaluation artifact:
`/Users/theoad/.cache/cpu2tensor/hardware-latent-confirmation-r5-result`.
Evaluation report SHA-256:
`f2e3d6efa55419c17a7d2e0575cc4453c20fe8d22a4288d95e4d8a72d74658bf`.
