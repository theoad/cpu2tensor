# Hardware held-out effect R1

Accepted 2026-09-27: replace the same-family futex sensor proxy with three
output-matched lawful syscall pairs and test benign-only v4 pretraining on an
independent physical session. This is exceptional-path validation, not a kernel
bug or vulnerability claim.

## Physical evidence

The pairs are successful `fstat` versus `EBADF`, successful `openat`/close
versus an `ENOENT` open followed by an `EBADF` close, and a valid `/dev/zero`
read versus the same read into a `PROT_NONE` mapping returning `EFAULT`. The
open pair deliberately matches two syscalls per iteration, so its exceptional
arm is compound rather than isolated `ENOENT`. Both read arms share open,
mapping, unmapping, and close lifecycle. All six paths are finite, use 5,000
iterations, and produce identical `0` output.

Physical `trail-x86` (`iseeyou`, i7-10510U, Linux `5.13.0-30-generic`) ran two
96-row sessions on boot `31179aea-9a43-4c5d-8bf6-1205735e42c4`, target CPU 2,
controller CPU 3, turbo disabled, a 1.8 GHz CPU-2 maximum, PEBS loads at period
10,000, 1,024 data pages, 8,192 PT AUX pages, and all raw traces retained. The
bounded service completed in 14.24 seconds wall and 14.29 CPU seconds. It then
restored `no_turbo=0` and the 4.9 GHz maximum; its service and restore timer are
inactive.

Every one of 192 executions succeeded on its first attempt. There were zero
rejections, losses, missing sources, or multiplexed sources. Each session has
16 rows per family. Session A ran at 22.06 executions/s, retained 71,619,600 PT
bytes, and observed 514 PEBS samples. Session B ran at 22.05 executions/s,
retained 71,641,056 PT bytes, and observed 512 PEBS samples. These are finite
all-raw capture rates, not sustained fuzzing throughput.

Host evidence remains under
`/home/user/.cache/cpu2tensor/heldout-effect-r1-92618e9`. The 167 MiB off-host
copy at `/Users/theoad/.cache/cpu2tensor/heldout-effect-r1-92618e9` matched all
392 files. SHA-256 values are:

- session A manifest: `a0f65a27d8253fe9fe2d8f6d948f740d2bebde655bd2b66aa39fabaa363376d3`;
- session B manifest: `c0e24dfaa84ca500bc538616bc522950823448e8b2323c685d9bb82ff922008e`;
- off-host hash ledger: `0d3c8a58d3a5d13db147dce998782a2cc359da780ecec47225e9ee852ea20792`.

## Benign-only anomaly result

Three d128 models independently pretrained for 60 masked-reconstruction steps
on the prior 51-row benign anchor session A. No effect label or effect trace was
used for effect fitting or threshold selection. Each threshold is the maximum
fused anomaly score on the same 51 anchor-A rows that fit normalization and
model weights, so it is a training/resubstitution maximum—not independent
calibration. The table evaluates the independently captured effect session B
and the separate 51-row benign anchor session B.

| Seed | Benign alerts | `fstat` AUROC, alerts | `openat` AUROC, alerts | `read EFAULT` AUROC, alerts |
| --- | ---: | ---: | ---: | ---: |
| 2801 | 1/51 | 0.133, 0/16 | 0.934, 0/16 | 1.000, 16/16 |
| 2802 | 1/51 | 0.063, 0/16 | 0.926, 0/16 | 1.000, 16/16 |
| 2803 | 1/51 | 0.449, 0/16 | 0.875, 0/16 | 1.000, 16/16 |

No control row in any pair crossed the fused threshold. The read-`EFAULT`
effect is therefore repeat-stable and initialization-stable in this bounded
test. It is the first result here showing that benign-only hardware pretraining
can flag an independently recaptured memory-fault path. It remains a lawful
error path, not corruption or a security flaw.

The false positive is also stable: all three models flag `memfd-i0-s00` in
benign session B. That is 1/51, about 19,608 per million—not remotely the
required $10^{-4}$ review rate. Three initializations reuse the same observations
and do not increase the benign sample size. The model ranks the compound open error well but
its absolute score stays under threshold, while `fstat(EBADF)` is not a shared
anomaly at all. A generic notion of "syscall error" has not emerged.

For seed 2801, read-`EFAULT` fused scores are 0.672--0.720 versus 0.175--0.247
for controls. Its modality residual is dominated by PMU (1.16--1.32), with PT
also elevated; the false-positive `memfd` row has high PT/PEBS residual but PMU
only 0.036. Post-hoc, requiring both fused and PMU residuals to exceed their
anchor-A maxima gives 0/51 benign-B, 16/16 `EFAULT` effects, and 0/16 controls
for all seeds. Because that conjunction was designed after inspecting session
B, it is only a preregistered hypothesis for a fresh session—not accepted
performance evidence.

The exact formal report, all scores, and three checkpoints are at
`/Users/theoad/.cache/cpu2tensor/heldout-effect-anomaly-r1-r2`. The report pins
all imported experiment/model/feature source hashes. Report SHA-256 is
`2d0efd99db73c03562b4a196c4054c8f03b5edfab473acbd3123a1ac437da04c`.

## Supervised transfer diagnosis

An exploratory leave-one-pair-out head trained on two pairs and evaluated the
third. Same-family second-session AUROC was always 1.000, but held-out results
were seed dependent. Without broad benign negatives, the frozen pretrained
encoder was more stable than a jointly trained scratch encoder on read/open
holds, yet no positive exceeded the benign maximum. Adding all 51 benign anchor
rows as supervised negatives did not solve calibration and damaged `fstat`
transfer. These unsealed development ablations guide the next objective; they
are not promoted metrics.

## Decision

The memory-fault signal justifies continuing short detector research, but not a
million-execution or 24-hour campaign. Next, validate the preregistered
cross-modal conjunction on a fresh effect and benign session, then enlarge
independent benign calibration and learn a cross-modal anomaly head without
using the confirmation labels. Operational promotion still requires top-100
recall at benign FPR at most $10^{-4}$.
