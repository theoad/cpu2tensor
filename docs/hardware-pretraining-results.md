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

## Fresh v2 representation and temporal gate

Commit `3719b01` was used to collect a new sealed 204-execution corpus on the
same named `trail-x86` subject. The feature contract applies `log1p` to each raw
PT byte count while retaining the original undecoded bytes. Collection admitted
204/214 attempts without PT/AUX loss, retained 1,714,632,128 PT bytes and 16,865
PEBS samples, and completed in 19.50 seconds. The capture manifest SHA-256 is
`f20cedcfb88c6155119863797b9a0be99c7896de1a14827c49a023631955e589`.

Across model seeds 17, 23, 31, and 41, both transformer variants produced zero
alerts in all 36 executions from the wholly unseen `dup`, `memfd`, and `pipe`
families. The fused model retained 2.25--2.71 times clean mean score under PT,
PEBS, and PMU row swaps. Its only recurring familiar-benign alert cluster was
`fork`, with one to three of three validation repetitions depending on seed.
This is treated as one stable novel behavior for review, not silently relabeled
normal and not counted twelve times in an operational queue.

The accepted temporal objective hides the selected token's timing input and
predicts its six bounded timing features alongside its modality value. Across
the same four seeds, PEBS-bound reversal raised fused mean score by
1.505--1.552 times clean; all four models retained zero unseen-family alerts.
For seed 41, the maximum timing residual landed on a genuinely changed PEBS
token in 78/78 validation executions. Changed tokens averaged 0.2722 timing
residual versus 0.0232 for the 22 unchanged tokens. Batched scoring of 78 rows
on the named MPS host reached 1,497 executions/s versus 1,505/s for the exact
pre-temporal model, a measured 0.6% reduction. The seed-41 report and checkpoint
SHA-256 values are `a66044e653a47c4496373f0d3da4e18dcb473cd1ba79c3ed3820a3fd2374c749`
and `d0924f7797a5dd028f5ad64af4140bc259597035b003f456975d57d56806a4de`.

The hardware evidence bundle now preserves the exact invocation and sealed raw
input, empirical tail probability, modality and timing residuals, raw PT byte
windows, decoded PEBS memory-source semantics, raw IP/address values, and
exact-boot symbol resolution when authorized symbols are supplied. Residuals
are the quantities that produced the score. No attention weight is presented as
a causal explanation, and an anomaly token will be admitted only if a blinded
intervention test improves localization beyond these faithful residuals.

The long-lived perf path reuses event descriptors and mappings while resetting
each window and publishing data/AUX tails only after successful decode. On
`trail-x86`, 1,000 steady-state `getpid` windows of 100 syscalls each sustained
8,119 complete PT+PEBS+PMU windows/s on one core, with 118.8 microsecond median,
175.6 microsecond p99, and 397.2 microsecond maximum latency after one lazy
Torch warm-up. This includes perf control, workload execution, ring copy, record
validation, and tensor construction, but excludes durable alert retention and
model inference. It clears the 1,000 windows/s/core systems gate on this named
host; aggregate accelerator scoring and rolling retention remain separate gates.

## Throughput profile

The 204-run manifest separates 7.896 seconds of accepted workload execution
from 168.649 seconds of explicit featurization and 14.234 seconds of all other
capture/custody work. The live timer therefore attributes 88.40% of the 190.78
second run to the featurizer. A later nine-execution phase-instrumented pilot
localized 5.156 of its 5.813 wall seconds (88.70%) to PT histogram construction;
PEBS and PMU features used only 3.184 ms. The cliff was workload dependent:
107 KB `getpid` traces took 0.55--0.59 ms, while both roughly 1 MB `openat` and
6 MB `memfd` traces took 0.85--0.87 seconds. This rules out the earlier
over-broad-timer hypothesis.

The cause was controller affinity, not the tensor representation or histogram
algorithm. Torch selected four intra-op workers while the process could use all
eight logical CPUs; the runner then pinned the controller to CPU 3 before those
workers were created. The workers inherited the one-CPU mask and contended when
`bincount` entered its parallel path. Retained and live-buffer tensors both took
about 864 ms under that ordering. Setting the intra-op pool to one immediately
after pinning reduced isolated 1 MB and 6 MB histograms to 1.6--1.9 ms and
8.0--8.6 ms.

A clean end-to-end rerun at revision `eadcff44b255c517b47f9b1f08ff1e81cc9f5d35`
confirmed the correction without changing capture admission: 9/9 executions
were retained on their first attempt, with the same 6 sampled and 3 verified
zero-PEBS outcomes and no source loss or multiplexing. Wall time fell from
5.813 to 0.371 seconds (15.6 times), explicit featurization from 5.310 seconds
to 39.573 ms (134 times), and PT histogram construction from 5.156 seconds to
31.655 ms (163 times). The histogram now accounts for 8.52% of wall time and
processes the aggregate 21.3 MB at 673 MB/s, consistent with retained-shard
replay. Raw serialization/fsync/hash is now 28.82% and the workload windows
30.43%, exposing the next real finite-run costs.

The workload itself also prevents a 1,000-execution/s claim: its mean accepted
window was 38.7 ms, and even `getpid` averaged 2.44 ms. High-rate operation needs
persistent sub-millisecond fuzz targets, one long-lived per-core perf session,
continuous AUX draining, native one-pass reduction and hashing, and append-only
sharded custody. At the present 8.47 MB/execution, 1,000 executions/s would emit
8.47 GB/s per core; 100 such cores would approach 73 PB/day. The operational
design must score every execution from bounded raw memory while retaining full
raw evidence for alerts and a preregistered benign training sample, rather than
claiming durable raw custody for every high-rate execution.

The full run's nine pre-admission retries exposed a selection bias: requiring at
least one PEBS sample at period 10,000 preferentially retained executions that
happened to receive a sample. The runner now admits a zero-sample PEBS interval
as explicitly unavailable only when the event is present, scheduled without
multiplexing, and the target affinity is independently exact. In the retained
nine-execution pilot, all three short `getpid` runs were admitted this way on
their first attempt, while the six longer runs carried 216 exact-IP, nonzero-
address samples on the requested CPU. The manifest reports admission reasons
and every rejected attempt; suspicious cases may still receive a separate,
denser PEBS replay.

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
