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

## Discarded AWS 5.16.10 and Dirty Pipe boundary

The AWS run below is retained as a sensor and algorithm experiment only. It is
not a qualified subject for the physical-host campaign: cloud bare metal has a
different platform, firmware, interrupt topology, NUMA and device layout from
the controlled laptop. Replacing its boot kernel could never make its learned
hardware distribution stand in for that machine. AWS may train on exported
shards, but it must not originate traces for this campaign.

The first known-CVE hardware validation ran on an AWS `c5.metal` host with an
instrumented upstream `5.16.10-cpu2tensor-dirtypipe` kernel, boot
`56bee244-ea31-4b36-b7ce-773b09d34ffd`. A fresh broad pretraining corpus retained
204/204 first-attempt executions, 320,535,776 PT bytes, and 2,942 exact usable
PEBS load samples with no censored sample. Its manifest, fused checkpoint, and
report SHA-256 values are
`ad9c03bbbd938fed2855a57f8a55ee378948031cd5e62f7b84565ac0880f6dc8`,
`aec56b1db67ded18da18ab40fd212712afd84d74eaea7602a93a8e0327626146`,
and `1f2ce7f5f3ab9e51754a8ddd8ee59d83954068172b49fa51360e95f7bf8c7c8b`.
The fused model scored 1,681 executions/s on eight named host CPUs, but its small
calibration produced 3/42 familiar and 7/36 held-out alerts and is not an
operational false-positive estimate.

The safe canary exercised CVE-2022-0847 only against caller-owned temporary
files. Every vulnerable effect execution changed all requested markers and
every matched `AAAA` arm retained identical file bytes. With 12 randomized
executions per arm at 20 loops, the frozen model alerted on all 24 but separated
the arms at only 0.528 AUROC; PT alone reached 0.625. At 200 loops, fused and PT
AUROC were 0.563 and 0.632. Adding 24 neutral same-workload executions to benign
training and six to calibration did not rescue held-out separation: 36 hidden
effects versus six neutral validations produced 0.468 AUROC. A label-aware
linear probe trained on a separate session also failed to generalize, reaching
0.479 AUROC for byte histograms and 0.417 for adjacent-byte sketches. The
20-loop and 200-loop report SHA-256 values are
`2d6296dfe62f394f123b9c653b0e4abbde7a5ca4d97993c10fdcf1faf0779372`
and `11990552e48990d21a7a7fdfc6538fd816ba4c10cf6d06435df4c2f696feb96c`.

This first subject is not a production release subject: its inherited
configuration includes
`CONFIG_UBSAN=y`, `CONFIG_DEBUG_KERNEL=y`, `CONFIG_SLUB_DEBUG=y`, and related
diagnostic behavior. The campaign attempted to establish a lawful fixed-arm
comparison by building upstream 5.16.10 and 5.16.11 with one compiler and
otherwise matched normalized configurations, disabling UBSAN, KASAN, KCSAN,
KCOV, SLUB debugging, lockdep, proving locks, and debug info. The vulnerable
kernel, config, and normalized-config SHA-256 values are
`be1a7e7aa5151f1e0c377e52b98ce21cd9a53b9aef25b53f196c09a28c72478e`,
`6c6452e32362272ff33f409fd0649e46144b666ef3d6f017d9b56b9654146a25`,
and `2f7c7026b447e8a128da45a9709a470bc9b0ce625871ed1e4660d40714ffbc55`.
The fixed values are
`200146b5bffb59f518e33b15ecf89e7947a67efb665607049fef86f17f3f9268`,
`bcc27601bdd8209b98a0a116234b5b8f35de7457782fb27411ad423177f4531d`,
and `112a332d19911d857b21e3e18876214b7c5ce748f1cdf360c618bc9668636954`.
The normalized configs differ only in the generated version comment and an
obsolete 5.16.10-only `IWLWIFI_BCAST_FILTERING` symbol. `CONFIG_DEBUG_KERNEL`
remains selected indirectly with diagnostic facilities. These images are
therefore rejected as validation subjects; being less instrumented is not the
same as being production-equivalent.

The learned subject and every validation arm must instead use an unmodified,
cryptographically identified production distribution kernel binary, modules,
initramfs, command line, microcode, and configuration. Collection may use
privileged hardware perf facilities, but it must not rebuild, patch, or enable
behavior that the production package does not ship. Audit and record the shipped
config and runtime command line for sanitizers, KCOV, lock debugging, slab
debugging, assertion-heavy modes, and diagnostic-only allocators. A capability
compiled into the signed vendor binary is part of that real subject and must not
be toggled between training and inference. If the study instead requires that
no such facility be compiled at all, choose a different off-the-shelf production
package and restart the subject; do not manufacture a custom "release-like"
binary. Known-CVE
validation should use archived vulnerable and fixed production packages from
the same distribution series and flavor. Each arm must boot, expose the
expected canary outcome, and produce comparable PT, PEBS, and PMU custody before
the matched-CVE gate can pass.

This is an expected sensor boundary, not a detected vulnerability. Both canary
arms execute the vulnerable splice/write path; one writes different bytes while
the other writes bytes already present. Kernel control flow is therefore nearly
identical, raw PT carries no data values, and the accepted PEBS event sampled
loads only. Increasing trace length or transformer capacity cannot recover an
unobserved state mutation. Commit `061f431` added strict Intel `mem-stores`
capture. A live period-1,000 smoke on this exact boot produced 898 neutral and
972 effect store samples for 20 loops; every sample had exact IP, nonzero virtual
address, and CPU 2 attribution. Store sampling exposes the missing mutation
channel, but a matched fixed build and address-provenance sideband are still
required before it can qualify CVE detection.

The campaign also found that the combined collect-and-train command retained
the collector's one-CPU affinity during training. Commit `0aa0100` restores the
caller's full CPU set on success and failure. Re-running the sealed corpus with
`--train-only` on CPUs 4--11 completed the same work promptly; this was an
orchestration defect, not a model improvement.

## Canonical physical-host R2

The first corrected run used the unchanged signed Ubuntu
`7.0.0-31-generic` package on `trail-x86`, an Intel i7-10510U laptop, boot
`2ba4b210-4c31-48ce-92d5-4b306e16d26c`, microcode `0x100`. Its subject identity
also seals the kernel version, BTF, notes, command line, CPU assignment, workload
build ID, and clean source revision `0ce6c67`. The Ubuntu production config does
ship UBSAN bounds support, SLUB debug capability, page poisoning support, and
allocation initialization; the experiment records those real vendor choices and
does not enable, disable, or rebuild any of them.

All 204 executions were admitted on their first attempt with no loss or PMU
multiplexing. They produced 342,937,312 raw PT bytes and 3,267 PEBS samples, of
which one unusable sample was retained in custody but censored from learning.
Collection sustained 32.60 sealed executions/s and 54.80 MB/s of PT on this
named host. The fused model trained and scored at 1,386 and 1,272 executions/s
respectively with four CPU threads. Its manifest, report, and checkpoint hashes
are `97ddd255200732c5232bad072d57db93fb17e451132b7b5cf851bbe42048c6e3`,
`8ea63c5f10da5a4ab7aac3274a9b3d59ca9f233f7bcd9a32bb35a093d9a2f11c`,
and `847bc1dc805d5314d9ae59b20447e2844375344c7130ece7b2ffc2ddb2b05ee3`.

The 42-row pilot calibration is too small for an operational alert-rate claim.
It yielded 1/42 familiar and 9/36 wholly held-out-family alerts. Controlled
modality swaps increased mean fused score by 1.90 times for PT, 1.90 for PEBS,
and 1.73 for PMU; timestamp misalignment increased it by only 1.09 times. A
real LLM bundle for the maximum calibration row retained the replay input, eight
exact raw PT windows, four decoded and symbolicated PEBS samples, every PMU
counter, and faithful residuals. Its SHA-256 is
`54d031afe4512eca328ccc17ce9e7d8ed6abbe6e0f69790dff2a513f6c3d3ecf`.

That bundle exposed a one-ULP alert flip when a new process used a different
Torch CPU thread count. Commit `0ce6c67` makes the CPU thread count part of the
frozen inference contract. The reproduced calibration score and threshold are
now bit-identical at `0.5289283394813538`, with zero margin and no alert. This
closes the mechanics and review-evidence PoC on the correct subject; it does not
yet pass the process-scope sensor-canary, large calibration, rolling retention,
or known-vulnerability gates required for a 24-hour campaign.

A subsequent 3,060-execution R3 on the same boot used 1,260 training, 1,008
calibration, 252 familiar-validation, and 540 whole-family-held-out executions.
It admitted every capture on the first attempt with no loss or multiplexing,
retaining 5,150,136,704 PT bytes and 49,105 PEBS samples; 42 unusable PEBS rows
were censored from learning but kept in raw custody. The fused model reduced loss
from 0.393 to 0.066. At the preregistered 1% pilot threshold it flagged 4/252
familiar and 42/540 held-out executions, while PT, PEBS, PMU, and timestamp
perturbations raised mean score by 2.66, 2.51, 2.10, and 1.32 times. At the
operational 0.1% calibration threshold it flagged 0/792 unmodified evaluation
executions while still detecting 312 PT, 270 PEBS, 114 PMU, and three timing
perturbations. Those corruptions are sensor controls, not vulnerability proxies.
The manifest, report, and checkpoint SHA-256 values are
`900b834c3d109d26fce7632f9ce9eaec02d19fa58d17e39911f51171bac51bc8`,
`93d4747e4b514c5399d8f661c2ce8ca75eeec21ac11bf04aedd83e14456c986c`,
and `798df994f3d93fbc3117a40463c93a7eb6baca926f583a550a3fe1a507176be2`.

Copying the sealed 5.25 GB artifact from `trail-x86` to the local durable cache
took 115.17 seconds, only 45.60 MB/s. That is below the run's 49.40 MB/s PT
production rate and fails the required two-times headroom. Full raw streaming is
therefore rejected. The same corpus's derived tensors occupy only 72 MB. The
runner now supports a deterministic, content- and model-independent raw sampling
fraction: it always retains derived tensors and the original raw hashes, while
deleting non-sampled benign-pretraining raw files only after their derived file
is sealed. This mode is for explicitly benign pretraining; prospective inference
must score before retention and preserve every alert's original raw window.

The first live bounded-retention smoke at commit `591d8b2` retained one
preregistered raw execution out of 204 at fraction 0.01, all 204 derived tensors,
and the complete manifest, report, and checkpoints in 7.1 MB. Collection and
training completed with no retry or loss. Attempting to create an LLM bundle for
a sampled-out execution failed explicitly with `raw evidence was not retained`;
it did not fall back to replay or fabricate decoded evidence.

## Prospective custody and production-kernel control

Commits `42bbfe4` through `52326fa` close the prospective alert-custody gap.
Each Intel PT batch now retains the exact perf data-ring records that describe
its AUX bytes. Before arming the event, the collector captures process maps,
the exact perf attributes, and root-authorized kernel symbols, modules, module
build IDs, namespaces, build IDs, ksymbols, BPF events, text patches, context
switches, and clock-identifying sideband. Boot-static kernel evidence is sealed
once and content-addressed; each execution retains its dynamic sideband and a
hash-verified reference to that exact-boot state.

Enabling the complete sideband immediately found two real collector defects.
`/sys/module/compression` is a regular file rather than a module directory, and
5,001 `PERF_RECORD_MMAP2` records overflowed the original 256 KiB data ring.
The collector now ignores non-module sysfs entries and uses a 4 MiB data ring for
this path. A fresh nine-execution physical-host run then admitted 9/9 captures
without loss or retry. One mmap-heavy execution retained 520,288 bytes of perf
records, including all 5,001 mmap2 records.

Prospective scoring at commit `52326fa` completed 9/9 physical executions at
13.96 sealed-and-scored executions/s. Every execution was scored before any raw
eviction: three model alerts retained their complete original evidence, while
six below-threshold rows were deleted only after the custody decision. Median
frozen-model scoring was 9.82 ms, median raw sealing was 7.23 ms, and steady
event-open/arm time was 5.13 ms after the exact-boot cache was populated. The
single deduplicated kernel decode state is 18,376,993 bytes; retained raw mmap
executions are about 4.0--4.14 MB each instead of repeating roughly 18 MB of
kernel metadata. Root-visible kallsyms contain real nonzero addresses and every
retained alert contains nonempty original perf records. This proves decision
ordering and custody, not the final aggregate inference rate.

A blind fixed-arm Dirty Pipe control then ran on the same unchanged signed
Ubuntu `7.0.0-31-generic` subject. All eight effect inputs and eight lawful
neutral siblings completed on their first attempt, with zero page-cache mutation
in the effect arm as expected for a fixed kernel. The report content SHA-256 is
`8b2abe383460cbf065d436f3c5d04f99ee5ea6ec39d31a90bbe73d74290dddd9`.
The intentionally tiny custody-smoke model alerted on all 16 rows and ranked the
effect below the sibling control (AUROC 0.3594); it was trained on only nine
unrelated `getpid`, `mmap`, and `openat` executions and is not a sensitivity
model. This is useful negative evidence: trigger novelty alone is not a
vulnerability detector, and each production-kernel arm needs a subject-matched
benign corpus and frozen calibration.

The vulnerable production arm selected official Ubuntu Impish
`5.13.0-30.33`, below Ubuntu's documented `5.13.0-35.40` Dirty Pipe fix. The
signed image and base module package were installed without changing the
permanent GRUB default. Their SHA-256 values are
`a822bd320e16ced00ed434793dd33f6799016862c0294a46f5ec3314eed01b75` and
`4262902743dffc76f510d924dc8f2c2a295bb18d2e3e3bee7ba54b616d569997`.
The one-shot boot reached the desktop and `uname` reported
`5.13.0-30-generic`. The built-in wired interface remained DOWN, but Wi-Fi and
USB Ethernet were restored using the exact archived extra modules and installed
firmware. The package database is clean; the extra module files are official
package contents copied into `/lib/modules`, not dpkg-managed. The vulnerable
boot ID is `31179aea-9a43-4c5d-8bf6-1205735e42c4`.

On that same physical boot, two tri-modal mmap qualification captures had PT,
exact-IP PEBS, and nonmultiplexed PMU data without loss. A first larger benign
run stopped after 1,867 executions when a transient `/proc/modules` change
tripped the decode-state guard. Commit `7e62e82` ignores volatile module
refcounts while retaining mapping-bearing changes as separate decode-state
generations. A fresh run then completed 3,060/3,060 executions without retry
or loss: 5,126,204,720 PT bytes in 107.54 seconds, or 28.45 sealed
executions/s on the i7-10510U laptop. It trained a subject-matched frozen
fused checkpoint, SHA-256
`110c2cadf89c9dc5de10adcd4b34cbb89a0237c3c0ba1a9b3ace366952787e4f`.
Its 1,008-execution calibration is only a pilot, not evidence for an operational
1,000-per-million review rate.

The first blinded vulnerable Dirty Pipe canary completed 12 effect and 12
nominally neutral executions, each on its first attempt without trace loss.
Every effect execution manifested all 20 requested page-cache mutations. Its
nominal separation was AUROC 0.6319 and 8/12 paired effect wins; at the pilot
threshold it alerted on 4/12 effect and 2/12 nominal neutral executions. The
content-hashed report is
`83c638c4246f403fbf13ca40a5694209e39ddeabfaab9c56980efc207a80bc18`.
Source review subsequently found that the nominal neutral arm also used the
splice/write path, merely writing bytes already present in the file. It could
exercise the same unauthorized page-cache write without an observable content
change. Thus this AUC is **not a valid vulnerable-versus-lawful comparison**.
The corrected neutral arm uses an ordinary read before writing to the pipe.
All 24 original raw captures and their exact decode state remain preserved as
invalid-control evidence; no sensitivity or review-rate claim follows from them.

The corrected lawful-sibling replay used the same frozen checkpoint on the same
boot. It again completed 24/24 first-attempt lossless captures and all effect
executions manifested 20/20 mutations, but the model ranked the effect *below*
the lawful sibling: AUROC 0.3681, 4/12 paired wins, and 2/12 effect versus 5/12
lawful-sibling alerts. The report content SHA-256 is
`396c2db0c3c74f6970c2f03368049e135e8697e1ffb0f7972f7bf6da8873ca32`.
This is a failed known-CVE sensitivity gate, not a capture failure. On the
vulnerable `trail-x86` i7-10510U host, a separate process-kernel precise-store
probe obtained 954 exact-IP/nonzero-address samples on 5,000 `mmap` iterations
at period 1,000, and roughly 5,900--8,300 such samples per 10-loop canary at
period 100. This qualifies store sampling as a candidate observation path, but
does not yet prove that it distinguishes the vulnerable mutation.

The full 278-test local suite passed with 75 platform skips before the
decode-generation change; its focused 31-test suite passed afterward. The
24-hour campaign remains **NO-GO** pending a better invariant-bearing signal,
independent calibration, and semantic Intel PT packet/address decoding from the
preserved same-session sideband.

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
