# Hardware futex effect R1

Accepted 2026-09-27: after repairing PEBS coordinates, measure whether the exact
physical subject exposes a repeat-stable kernel-path effect and whether the v4
masked encoder adds value to a minimal downstream transfer task. This is a
lawful sensor proxy, not a bug, corruption, or vulnerability-sensitivity gate.

## Capture

Physical `trail-x86` (`iseeyou`, i7-10510U, Linux `5.13.0-30-generic`) ran two
independent 48-row sessions on boot
`31179aea-9a43-4c5d-8bf6-1205735e42c4`. Each session contained 16 repetitions
of three fixed 5,000-syscall families:

- `futex_wake`: `FUTEX_WAKE_PRIVATE` with no waiter;
- `futex_mismatch`: `FUTEX_WAIT_PRIVATE` with a deliberately mismatched value,
  which must immediately return `EAGAIN` and never sleep;
- `getpid`: an unrelated background family retained for later diagnostics.

The capture ran target CPU 2 at a 1.8 GHz maximum with turbo disabled, controller
CPU 3, PEBS loads at period 10,000, 1,024 data pages, 8,192 PT AUX pages, and all
raw traces retained. A systemd cgroup imposed a ten-minute hard limit, the script
imposed per-session deadlines and thermal/disk/policy checks, and an independent
timer could restore the original CPU policy. The completed service used 10.21
seconds wall time and 9.86 CPU seconds. These are finite all-raw capture costs,
not sustained fuzzing throughput.

Both sessions completed every row on its first attempt with zero loss,
rejection, missing source, or multiplexed source. Session A collected 48 rows
at 26.79 executions/s, 6,120,576 PT bytes, and nine PEBS samples. Session B
collected 48 rows at 26.04 executions/s, 6,123,744 PT bytes, and ten PEBS
samples. The host policy was restored to `no_turbo=0` and a 4.9 GHz CPU-2
maximum; the service and restore timer are inactive.

The host evidence remains under
`/home/user/.cache/cpu2tensor/futex-effect-r1-bccf7a8`. A 36 MiB off-host copy
under `/Users/theoad/.cache/cpu2tensor/futex-effect-r1-bccf7a8` matched all 200
file hashes. Important SHA-256 values are:

- session A manifest: `d5864386910eba4463d6a9dc93a001e4e55efcc9f067a84cf580a17dc8417280`;
- session B manifest: `20487bc9785624606524429ab75e69eb4f30da5b80063f2d817916115a12b1a9`;
- off-host hash ledger: `d63b7684601a906ab7d14d1d2964504179acbc367e86f843ce71e010bfa37a79`.

## Independent-session sensor result

Session A alone standardized coordinates and defined one mean
`futex_wake - futex_mismatch` direction. Session B was then projected without
refitting. The 16 effects and 16 controls give 256 cross-class comparisons.

| Modality | Session-A AUROC | Session-B AUROC | Effect-direction cosine |
| --- | ---: | ---: | ---: |
| PT v4 | 1.000 | 1.000 | 0.8776 |
| PEBS v4 | 0.781 | 0.781 | 0.8849 |
| PMU | 1.000 | 1.000 | 0.9694 |
| Timing | 1.000 | 1.000 | 0.9966 |
| Fused | 1.000 | 1.000 | 0.8810 |

PT, PMU, timing, and fused directions win all 256 second-session comparisons.
This is strong evidence that the corrected representation preserves a stable
kernel-path effect. It is also an easy task: wake and mismatched wait execute
different lawful paths thousands of times.

PEBS requires narrower interpretation. All 16 wake rows have zero usable
samples in both sessions; nine of 16 mismatch rows have one sample in each.
The resulting $0.78125$ AUROC is exactly the availability/exposure separator.
It does not demonstrate transfer of PEBS site or data-address coordinates.

## Short pretraining transfer

A d128 v4 masked model pretrained for 60 steps on 51 benign rows from the prior
anchor session; reconstruction loss fell from 0.4038 to 0.1150. Three fixed
seeds then compared the same 80-step head schedule for a random encoder, frozen
pretrained encoder, and fine-tuned pretrained encoder. Training used only effect
session A. Every arm and every seed obtained 1.000 AUROC and all 256 cross-class
wins on independent session B.

The positive result validates the representation-to-head seam and shows that a
frozen pretrained embedding retains the effect. It does **not** demonstrate a
pretraining advantage: scratch reaches the same ceiling. Nor does a lawful path
classifier establish anomaly or vulnerability sensitivity.

The run used the Mac coordinator (`mac.local`, arm64, Torch 2.13.0 MPS), not the
capture host. Its fixed d128 pretraining took 3.06 seconds; each 80-step transfer
arm took 0.98--1.66 seconds. These are tiny retained-tensor development timings.
The final report, score arrays, pretrained checkpoint, and all nine arm
checkpoints are under `/Users/theoad/.cache/cpu2tensor/futex-transfer-r1-r3`.
The report records every checkpoint hash and the executed source hash. Its
SHA-256 is
`997d60f8f38ebfbe362420b21c790df92c049d490f4cb7a1b5890e35102d62d9`;
the pretraining checkpoint is
`91f2685a41d0704902bf88daa57a8ec712479c2ea4cc17529fbde74a0c25c5d8`.

## Decision

The observation repair passes this bounded sensor gate, but the detector remains
**NO-GO** for a million-execution or 24-hour campaign. The next short experiment
must be harder and independently varied: train a label-efficient head on one or
more safe, hardware-visible pathological-effect families, then test a held-out
effect family and benign intensity/session shifts at the required review budget.
The acceptance metric remains recall/top 100 at benign FPR at most $10^{-4}$;
AUROC or familiar-family accuracy alone cannot authorize scale.
