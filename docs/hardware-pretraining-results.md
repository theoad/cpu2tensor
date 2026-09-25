# Multimodal hardware pretraining results

This log separates measured evidence from the proposed
[validation](hardware-pretraining-validation.md) and
[24-hour procedure](hardware-pretraining-scale.md). The current result qualifies
capture and a small model implementation. It does not yet establish useful
kernel anomaly detection.

## Exact subject

The capture qualification ran on `trail-x86` (`iseeyou`), an Intel Core
i7-10510U host,
with target CPU 2 and controller CPU 3. The retained subject was Linux
`7.0.0-31-generic`, boot `2ba4b210-4c31-48ce-92d5-4b306e16d26c`, microcode
`0x100`. Its BTF SHA-256 is
`c0c2d95eb84d04b2332c7a456b9d6cf4d18974ee439abd1b44c20d9f738bcd38`,
kernel-notes SHA-256 is
`764d1c9de60154c47117ba8d6889a52f50e5f79cc4b38b25fa1683da0d81011e`,
and workload SHA-256 is
`d3290444d0ce16c4060985b9297d009245fbaab34ba9c5442aa6b2de027f70bd`.
The first qualification artifact is cached as
`~/.cache/cpu2tensor/multimodal-poc/qualification-r12-v2.json`; its SHA-256 is
`655981233a19e8c4f89e30c66b6b7fb838bd78eb8c14fefab137c8889f8e59b8`.

Twelve randomized repeats of each perturbation arm used the gated `mmap`
kernel workload with 1,000 iterations:

| Arm | Median target window | Ratio to baseline |
| --- | ---: | ---: |
| No hardware events | 2.955 ms | 1.000 |
| Boundary PMUs | 2.903 ms | 0.983 |
| PT + boundary PMUs | 3.072 ms | 1.040 |
| PT + PEBS + boundary PMUs | 3.086 ms | 1.045 |

All 48 executions retained the exact output. Every requested source was present,
no perf loss was reported, and `time_enabled == time_running` for every PMU
group. Tri-modal runs had a 602,456-byte median PT trace and six exact-IP PEBS
samples at period 10,000. Every PEBS sample named the target TID, CPU 2, a
nonzero address, and a timestamp inside the shared `CLOCK_MONOTONIC_RAW`
envelope. Median source-control uncertainty was 44.103 microseconds at arm and
14.107 microseconds at stop; maxima were 106.599 and 37.474 microseconds.

## PEBS period decision

A follow-up used 5,000 iterations, six randomized repeats per arm, and three
kernel workload families on the same boot. Its six JSON artifacts and verified
hash manifest are cached below
`~/.cache/cpu2tensor/multimodal-poc/family-period-matrix-loops5000-r6/`.
The `SHA256SUMS` file hash is
`30f827aec6693578045411e4ee97d724dc51bde99d27668555b280843f8a850f`.

| Family | PEBS period | Median PT bytes | Median PEBS | PEBS/million PT bytes | Tri-modal overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openat` | 10,000 | 1,002,800 | 10 | 10.00 | +5.30% |
| `mmap` | 10,000 | 2,991,952 | 34 | 11.36 | +4.20% |
| `socketpair` | 10,000 | 5,490,176 | 45.5 | 8.31 | +4.66% |
| `openat` | 50,000 | 1,003,832 | 2 | 1.99 | +6.00% |
| `mmap` | 50,000 | 2,979,144 | 6 | 2.01 | +3.58% |
| `socketpair` | 50,000 | 5,408,592 | 9 | 1.66 | +4.46% |

All 144 executions kept their exact outputs and remained lossless and
non-multiplexed. A separate short `openat` attempt at period 50,000 produced an
empty PEBS execution. Period **10,000** is therefore frozen for the first PoC:
it gives roughly five times the sample density without a consistent overhead
penalty in this matrix. These six repeats qualify a choice; they do not establish
a universal perturbation bound.

The initial matrix inferred PEBS scheduling from source presence. Commit
`70cbfb5` subsequently pinned the precise event and made perf's own
`time_enabled` and `time_running` values part of the accepted evidence. A fresh
three-repeat `mmap` replay on the same exact subject produced six exact-IP,
nonzero-address samples in every tri-modal execution and equal nonzero scheduling
times in every case. Its artifact is cached as
`~/.cache/cpu2tensor/multimodal-poc/pebs-schedule-qualification.json`, SHA-256
`139308b7b4667548f4a4dc791b93bae7453707a4c36d4c48eaecc3932a15b1ff`.
The tri-modal median was 1.092 times its matched baseline in these three repeats;
the wider earlier 4.20--5.30% matrix and this 9.2% short replay are reported
separately rather than combined into a universal perturbation estimate.

## Real capture-to-tensor seam

Commit `2ca2754` was exercised directly on the same host and boot with
`process_kernel`, `mmap` for 1,000 iterations, and PEBS period 10,000. The exact
workload output remained `124716`. One live capture produced finite model input
with PT shape `[1,1,16,256]`, PEBS shape `[1,1,16,24]`, and PMU shape
`[1,1,1,4]`; all 16 PT segments, six PEBS time bins, and the PMU interval were
available. The acquisition report retained TID and observed CPU 2 outside the
model and marked migration verified from six PEBS samples. The cached report is
`~/.cache/cpu2tensor/multimodal-poc/feature-smoke-r1.json`, SHA-256
`3b4b75dc3004cb9cbe3ef21de12a978743abe0b7aef6aa768a0f63a74f94d722`.

This proves that the accepted capture object reaches the accepted tensor seam on
real hardware. It does not yet prove that training learns useful relationships;
that is the kernel-family experiment gate.

## Kernel-family training gate

Commit `4c0c71d` collected and sealed 204 executions of all 17 gated kernel
workload families on the same exact subject. Each workload used five times its
base loop count so that light syscall families could produce PEBS evidence at
the frozen period of 10,000. The manifest SHA-256 is
`f8e9b77a77aa8820e715e952cdc8e04ba7df97d2aa1b3aef4787404f5429cd59`;
the 1.7 GiB artifact is cached as
`~/.cache/cpu2tensor/multimodal-poc/kernel-multimodal-r1-full-scale5/`.
Its frozen training report SHA-256 is
`3121644d8c22f3b552ca3ef608c54e2a7e6be346bfcf120c2c3458b06e03659d`.
The split used 84 training executions, 42 calibration executions, 42 familiar
validation executions, and 36 executions from the wholly held-out `dup`,
`memfd`, and `pipe` families.

Collection retained 1,728,624,992 raw PT bytes and 16,960 exact-IP PEBS
samples in 190.78 seconds. All 204 admitted executions had PT, PEBS, and PMU
evidence with target-CPU attribution, equal nonzero enabled/running times, no
loss, no migration, and no output failure. Nine attempts were rejected and
retried before admission. The current manifest counts rejections but does not
retain their reason categories; that is an observability gap. Finite capture
plus Python featurization achieved only 1.069 admitted executions/s and 1.210
featurized executions/s, so this runner is evidence machinery rather than the
eventual high-rate data plane.

The 173,852-parameter fused and span-only models trained for 300 steps on MPS.
All model and baseline checkpoints reloaded bit-for-bit. The table reports
mean corrupted-to-clean anomaly-score ratios over the 78 validation executions:

| Model | Familiar alerts | Held-out alerts | PEBS swap | PMU swap | PT swap | PEBS-bound reversal | Score rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Marginal features | 0/42 | 12/36 | 1.000 | 1.000 | 1.000 | 1.000 | 99,675/s |
| PT-only PCA | 1/42 | 13/36 | 1.000 | 1.000 | 1.000 | 1.000 | 115,157/s |
| Span-only transformer | 1/42 | 13/36 | 1.290 | 1.264 | 1.264 | 1.004 | 853/s |
| Fused whole-modality transformer | 1/42 | 14/36 | 1.449 | 1.279 | 1.391 | 0.996 | 290/s |

This is the first real evidence that the model learns relationships between raw
PT, PEBS, and PMU observations that independent marginals and PT-only PCA cannot
represent. Paired fused-score increases occurred on 67/78 PEBS swaps, 59/78 PMU
swaps, and 53/78 PT swaps. Whole-modality masking strengthens all three mean swap
residuals relative to span-only training. This result still needs a stricter
control: 74/78 row rotations cross workload families, so the model may chiefly
learn family or intensity compatibility rather than a microarchitectural
invariant.

The run does **not** yet establish useful anomaly detection. All 12 benign
held-out `memfd` executions alert under every method; they are also the dominant
intensity family at roughly 30 MB PT, 312 PEBS samples, and 170 ms per execution.
Whole-modality masking does not improve held-out alert count, and the timestamp
corruption is effectively invisible. The 42-row calibration partition is too
small for an operational reviews-per-million claim and provides no session or
boot holdout. Scaling the current objective for 24 hours is therefore **NO-GO**
until trace magnitude is treated as a nuisance/conditioning variable, timing is
directly predicted from real timing evidence, and benign-family generalization
improves on a fresh multi-session retained split.

The timing failure is structural. Timing is presently an encoder input, but no
loss or anomaly residual predicts it; PEBS token position also already names its
coarse time bin. The reported bound reversal changes bounds without moving the
corresponding PEBS values, so it is a bookkeeping diagnostic rather than a
physical corruption. A read-only follow-up on the retained corpus found a median
of 16 available PEBS bins and all 16 bins in 120/204 executions, ruling out simple
sparsity as the cause. Coherent one-, four-, and eight-bin PEBS feature rotations
and reversal on the 48 fully populated validation traces also stayed at roughly
0.999--1.000 times clean. A held-out-family ridge probe could not predict relative
time bin from PEBS features ($R^2=-0.0089$, correlation $-0.0076$). The smallest
next test is therefore an explicit within-lane temporal-consistency energy trained
with coherent timestamp-shift negatives re-featurized from retained raw samples;
it must not manufacture order across CPU lanes.

That test is complete in [temporal-consistency R1](hardware-temporal-consistency-r1.md).
It adds real held-out timing sensitivity but fails localization, seed stability,
and the inference-overhead gate, so the experimental head remains out of the
accepted path.

## Throughput profile

The 204-run manifest separates 7.896 seconds of accepted workload execution
from 168.649 seconds of explicit featurization and 14.234 seconds of all other
capture/custody work. The live timer therefore attributes 88.40% of the 190.78
second run to the featurizer. It is not yet valid to attribute that cost to the
histogram itself: replaying the exact featurizer over 40 retained shards on the
same pinned `trail-x86` CPU processed 254.7 MB at 568 MB/s, and the isolated
29.9 MB `memfd` histogram reached 637 MB/s there and 1.60 GB/s on `mac.local`.
Those rates are roughly 55 and 156 times the live run's implied 10.25 MB/s. The
discrepancy points to a live-buffer/runtime interaction or an over-broad phase
timer and needs subphase instrumentation before optimization. PEBS contained
only 16,960 samples and is unlikely to explain 168.649 seconds. Raw custody
wrote 1.731 GB, fsynced each execution, then reread it for SHA-256; this belongs
in the same phase benchmark but is bounded by the 14.234-second residual as
currently timed.

The workload itself also prevents a 1,000-execution/s claim: its mean accepted
window was 38.7 ms, and even `getpid` averaged 2.44 ms. High-rate operation needs
persistent sub-millisecond fuzz targets, one long-lived per-core perf session,
continuous AUX draining, native one-pass reduction and hashing, and append-only
sharded custody. At the present 8.47 MB/execution, 1,000 executions/s would emit
8.47 GB/s per core; 100 such cores would approach 73 PB/day. The operational
design must score every execution from bounded raw memory while retaining full
raw evidence for alerts and a preregistered benign training sample, rather than
claiming durable raw custody for every high-rate execution.

The nine pre-admission retries expose another bias: requiring at least one PEBS
sample at period 10,000 preferentially retains executions that happen to receive
a sample. The next runner must admit a zero-sample PEBS interval as explicitly
unavailable when the pinned event and target CPU affinity are independently
verified, rather than resampling until positive. Suspicious cases can receive a
targeted denser PEBS replay.

## Small-model control

The initial masked bidirectional transformer has 171,143 parameters for four
PEBS and three PMU fixture features. On `mac.local` it scored 964 executions/s
on one CPU thread and 1,996 executions/s on MPS for batch 128 and one CPU lane.
Checkpoint reload reproduced scores bit-for-bit. On a deliberately correlated
synthetic fixture, whole-modality masking increased the modality-swap to clean
score ratio from 6.74 to 10.83. That supports the architectural choice but is
not evidence of hardware-signal learning. The real kernel-family result above
now supplies the first hardware-signal evidence, with the stated limits.

With the accepted 24-feature PEBS and four-feature PMU schema, the same small
configuration has 173,852 parameters. A repeated synthetic-shape scorer benchmark
on `mac.local`, batch 128 and one lane, measured 1,022 executions/s on one CPU
thread and 1,753 executions/s on MPS across ten full deterministic scoring calls.
This is a component rate, not the end-to-end fuzzing rate, and remains far below
the eventual 100,000-execution/s aggregate target.
