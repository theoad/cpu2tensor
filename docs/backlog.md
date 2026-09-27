# Work board

Updated 2026-09-27. This file owns work status. The first vertical slice is complete:
real observation, integration, packaging, and standard-IDE navigation checks pass.
Core register and memory instrumentation is checked on ARM, x86 guests, CPU and
MPS. Mixed frames, bounded multiworker collation and packed MPS uploads are now
measured. AWS/CUDA and sustained multi-host scaling remain open.
Later milestones are outcomes, not schedule promises.

## PEBS coordinate correctness: issue 28

The isolated generic fix uses retained runtime core `_text`/`_etext` bounds,
page-local data-address summaries, and an explicit v4 feature schema. Derived-v2
rows bind their anchor to their own retained decode state; legacy v3 remains
readable without mixing schemas. [PEBS anchors](pebs-anchors.md) defines the
contract. Thirty-seven feature/model/collector/seeded-plan tests pass on Mac CPU.
The physical same-boot audit is now complete: two identical 51-row seeded plans
finished first-attempt clean with zero rejected, lost, or multiplexed sources.
Median repeat cosine was 0.9916 PT, 0.9845 PEBS on the 35 rows sampled in both
sessions, and 0.99999 PMU. Sampled-only PEBS p10 was 0.8045, including a noisy
sparse-sample tail. This qualifies v4 coordinate stability for the next small
experiment, not detector sensitivity. The experiment coordinator restored CPU
policy and copied all 121 MiB off-host with production loader verification. The
change does not relabel the old raw-evicted corpus.

## Current decision: harder held-out effect transfer

The corrected v4 representation first passed the bounded
[futex sensor gate](hardware-futex-effect-r1.md), which was too easy to show a
pretraining advantage. The next [held-out effect gate](hardware-heldout-effect-r1.md)
captured 192 lawful error/control traces without loss. Three benign-only d128
models independently score `read(..., PROT_NONE)` above the training/resubstitution maximum
for 16/16 effects and 0/16 controls at 1.000 AUROC on session B. This is the first
repeat-stable pretrained anomaly signal for a hardware-visible memory-fault path.

The raw fused score was not operational: each model also flagged the same 1/51
benign `memfd` row, while open errors only ranked well and bad-`fstat` did not
transfer. The [preregistered fresh-session confirmation](hardware-effect-confirmation-r1.md)
of the fused-plus-PMU conjunction now passes unchanged for all three models:
48/48 aggregate `read_efault` detections and 0/201 aggregate ordinary-benign and
control alerts. All 83 fresh physical executions were first-attempt, lossless,
and copied off-host with every raw and derived hash verified.

This confirms a reusable hardware-visible proxy, not vulnerability sensitivity.
The subsequent [3,060-row benign pilot](hardware-effect-calibration-pilot-r1.md)
cleanly collected all rows but failed its zero-alert gate: frozen seeds 2801,
2802, and 2803 produced 1, 0, and 23 alerts. Seed 2803's failures concentrate on
22 `uname` rows across all three cohorts; the remaining `yield` row is also the
seed-2801 alert. The 30,000-row calibration is therefore blocked, and no
threshold or ensemble rule will be selected post hoc on these data.

The [paired few-shot transfer probe](hardware-fewshot-transfer-r1.md) changes the
research decision. A head trained only on `fstat` and `openat` labels transfers
to unseen `read_efault` with median AUROC 1.000 for the frozen pretrained encoder
at 1, 2, and 4 shots, versus 0.496, 0, and 0 for an identically trained random
encoder. Both pretrained arms win all three paired seeds without increasing
median benign alerts. This is the first clear sample-efficiency advantage over
scratch and evidence for emergent cross-family structure.

It is not an operational detector: the benign-calibration maximum remains above
all held-out effect scores, and benign tails are not yet controlled. The primary
next program is therefore the [Large Hardware Model scaling roadmap](large-hardware-model-roadmap.md):
measure self-supervised loss and transfer against model size, data volume,
workload diversity, modality diversity, and compute before task post-training.
Anomaly heads remain probes. Promotion still requires recall in the top 100 at
benign FPR at most $10^{-4}$; the million-run and 24-hour campaigns remain
**NO-GO**.

The [first nine-point scaling pilot](hardware-scaling-pilot-r1.md) now gives a
stable local capacity signal: held-out masked loss falls from roughly 0.146 at
d64 to 0.114 at d128 and 0.098 at d256, with parameter-loss slopes near −0.16
at all three data budgets. Data slopes are flat under the fixed 2,560-example
optimizer budget, identifying compute limitation rather than data saturation.
The [R2 compute matrix](hardware-compute-scaling-r2.md) resolves that ambiguity:
all six d128/d256 trajectories improve from 80 to 320 steps, and the 2,040-row
corpus beats 255 rows by 2.09% at d128 and 5.47% at d256. The preregistered
self-supervised scaling gate passes. The replicated cross-family transfer probe
does not: its median AUROC is at or near zero, and longer training makes it no
better. The next bounded experiment varies a generic representation objective
on the retained tensors under the same locked transfer protocol. More parameters
or physical collection wait until reconstruction scaling and reusable transfer
are no longer moving in opposite directions.

The first [R3 objective ablation](hardware-objective-ablation-r3.md) also fails
its promotion gate. Contrastive learning improves median transfer AUROC from 0
to 0.180 and wins two paired seeds, but remains below the required 0.75. VICReg
is bifurcated: one pretraining seed transfers at approximately 1.000 AUROC across
all heads, while two seeds remain near zero. Selecting that seed is forbidden.
All objectives preserve masked loss, localizing the problem to unstable latent
geometry. Next: score all retained encoders on benign-only cross-session family
and intensity retrieval, view agreement, and effective rank; keep the effect
labels sealed until a generic metric predicts reproducibility.

The [R4 benign-only geometry audit](hardware-latent-geometry-r4.md) rules out
collapse as that explanation. VICReg raises median effective rank from 4.11 to
14.95, cross-session family retrieval from about 91.9% to 96.5%, intensity
retrieval from 46.8% to 61.2%, and masked-view identity from 54.1% to 63.5%.
All three VICReg seeds are tightly grouped; the one perfect R3 effect seed is not
a generic outlier. The stable frontier therefore separates foundation learning
from task semantics. Next: prospective fresh-session confirmation of these
locked benign metrics, followed by a paired sample-efficiency comparison of
task post-training versus scratch across multiple held-out effect families.

## Closed decision: initial hardware transfer comparison

The previous sprint is closed NO-GO and its heartbeat is paused. The accepted
next slice is the [three-arm transfer comparison](hardware-transfer-r1.md):
scratch, frozen pretrained encoder plus trained head, and fine-tuned encoder/head.
The result is negative. Two corrected 40-step seeds now have complete training and evaluation
reports from physical `trail-x86`, identical per-arm row schedules, and
hash-verified off-host custody. Pretrained/fine-tuned arms remain near chance on
the second session; scratch varies from AUROC 0.424 to 0.715 across seeds. A
fixed train-session mean-difference audit fits PT/PEBS training pairs perfectly
but reverses on the second session. This establishes failure of that linear
transfer direction, not absence of every nonlinear modality signal.

Raw inspection found a generic representation defect: v3 PEBS features anchor
site and address offsets to each execution's sampled minimum. Each retained
24-row session had 16 minimum-IP identities and 24/24 distinct minimum-address
identities despite the exact same boot and workload recipe. cpu2tensor issue
[#28](https://github.com/theoad/cpu2tensor/issues/28) owns the exact-boot anchor
correction. Dirty Pipe remains a secondary logic challenge, not the primary
hardware-sensitivity gate. Next: validate a corrected feature schema on a small
two-session seeded capture and admit a separate safe corruption/pathological-
execution effect gate before recollecting pretraining data. Do not increase
capacity or promote a detector on familiar-defect training accuracy.

## Closed decision: six-hour hardware-anomaly sprint

The 102,000-execution same-subject collection finished without loss or retry,
but the frozen d128/d512 models failed the corrected Dirty Pipe canary. Capacity
alone did not help. Dirty Pipe is now a secondary logic-flaw challenge; the
primary validation will use a matched corruption or pathological-execution
effect with lawful controls. Parallel tracks are examining validation,
generic PT/PEBS/PMU observability, efficient model baselines, a separate
benign-only pretraining objective, and exact-package safe candidate provenance.
The coordinator owns the [sprint gate](hardware-anomaly-sprint.md): recall in
the top 100 of one million executions, benign FPR at most `1e-4` (target
`5e-5` for headroom), independent calibration/session holdout, complete
capture custody, and bounded GPU resources. Review by 2026-09-26 06:20 UTC.
The 24-hour campaign remains **NO-GO** pending measured sensitivity.

First bounded ablations are recorded in the [validation](hardware-anomaly-validation-r1.md),
[capture](hardware-anomaly-capture-r1.md), [model](hardware-anomaly-model-r1.md),
[pretraining](hardware-anomaly-pretraining-r1.md), [candidate](hardware-anomaly-candidate-r1.md),
and [nuisance](hardware-anomaly-nuisance-r1.md) notes. Compact models and a
cross-modal residual failed benign whole-family transfer; the raw audit found
PT order omitted by the model but no PT timing packets and no compelling Dirty
Pipe effect. No safe distinct exact-package physical bug trigger is admitted.
Fourteen focused new tests pass. The thirty-minute sprint monitor can launch
one bounded, disjoint next ablation when a measured blocker is identified;
none of these negative results authorizes scale by itself.

## Current hardware tracing slice

Requested 2026-09-24. [Hardware tracing](hardware-tracing.md) defines the
sample-versus-trace contract and evidence. Linux perf process and host-kernel
capture, sampled generic PMU events, Intel precise loads, raw Intel PT AUX, and
loss checks have real `trail-x86` smoke evidence. Windows WPR memory profiles
and raw ETL export have unit coverage; WPR syntax CI and physical Windows
capture remain to be checked. The Linux process backend does not yet follow
new threads. Intel PT packet decoding, Windows ETL tensor decoding, AMD/ARM
trace-specific hardware, and measured perturbation overhead remain open.

The next accepted slice is [raw hardware trace triage](raw-hardware-triage.md):
train a small non-autoregressive model once on undecoded kernel-only PT from benign
benchmarks, freeze it, and measure review-budget ranking and capture-to-score
throughput. The first code is a tensor-only raw-byte sketch and standardized
low-rank linear autoencoder. Sustained collection requires a long-lived per-CPU event, continuous
AUX draining and execution-boundary offsets; finite `PerfCapture` is only the
signal-validation path. PEBS and extra PMU events remain selective replay signals
until their perturbation and added discrimination are measured.

[R1](raw-hardware-triage-r1-results.md) reached 2,497 raw reductions/s and a
111,389/s frozen-scorer component rate on `trail-x86`, while finite perf setup
reached only 285 executions/s. Its anomaly gate failed: 7/10 familiar validation
traces and 8/8 hidden benign `mmap` traces were flagged. Next: continuous AUX
draining plus a broader family-held-out benign corpus; no CVE efficacy claim yet.

[R2](raw-hardware-triage-r2-results.md) added target-attributed kernel PT and a
retained 17-family, 8,704-execution raw corpus. The simple four-segment reducer
reached 3,290 executions/s and the frozen scorer 247,907/s on one pinned
`trail-x86` core. Familiar validation repeated at 670 reviews/million, but
`readlink` and `yield` were still wholly anomalous when hidden. Next: label-hidden
known-vulnerability sensitivity on an exact vulnerable kernel and continuous AUX
draining; larger models are not justified by benign reconstruction alone.
One minimal AWS bare-metal replication independently completed capture, offline
fit, checkpoint reload, and frozen inference, while again rejecting unseen-family
quality; its exact subject and rates are in the R2 results.

### Active multimodal pretraining PoC

Requested 2026-09-25. Within six hours, qualify simultaneous target-attributed
Intel PT, PEBS, and sparse non-multiplexed PMU observation on one exact bare-metal
subject; encode per-lane timing uncertainty without manufacturing total order;
train and freeze a small masked cross-modal model; and run blinded flow, memory,
contention, phase, swap, and timestamp canaries. Capture, model, and validation
work proceed on disjoint branches against one coordinator-owned tensor contract.

The accepted first model is a bidirectional masked predictor rather than a causal
GPT: completed executions provide both contexts, reconstruction residuals support
calibrated anomaly scores, and a first JEPA target would add collapse and teacher
choices before the signal is qualified. The PCA remains the external baseline.
The proposed [24-hour scale procedure](hardware-pretraining-scale.md) uses a
capacity ladder, immutable shards, recurrent health checks, bounded adjustments,
and final blinded evaluation; it is not authorized to launch until the PoC gates
pass.

Capture, scheduling evidence, and the real tensor seam now pass on one retained
`trail-x86` boot. Across `openat`, `mmap`, and `socketpair`, PEBS period 10,000
gave 8.31--11.36 samples per million PT bytes with exact outputs, no loss, and
4.20--5.30% tri-modal overhead in the six-repeat matrix. A later pinned-PEBS replay
proved nonzero equal perf scheduling times and an end-to-end live feature smoke
produced finite `[1,1,16,256]` PT, `[1,1,16,24]` PEBS, and `[1,1,1,4]` PMU
tensors. See [the measured results](hardware-pretraining-results.md).

Current state: the corrected canonical-host R2 corpus retained 204/204 executions
from the unchanged signed Ubuntu kernel on the physical `trail-x86` laptop, with
no retry, loss, or PMU multiplexing. It sustained 32.60 fully sealed executions/s
and 54.80 MB/s of PT; the fused model retained strong PT, PEBS, and PMU swap
response and scored 1,272 executions/s on four host threads. The LLM bundle
retains exact replay input, empirical confidence, raw PT windows, timing and
feature residuals, decoded PEBS semantics, and exact-boot symbols. Its discovery
of a one-ULP threshold flip caused by a different Torch thread count led to a
frozen inference-thread contract and a bit-identical cross-process replay.
Cloud bare metal is no longer a collector candidate because it is a different
physical subject; AWS may train only on exported immutable shards. Next: three
randomized sessions on the same physical host, rolling retained-alert custody
plus accelerator batching, and label-hidden validation on archived production
kernel packages. The 24-hour campaign remains **NO-GO** until those gates pass.
The prior isolated temporal R1 remains preserved by tag for provenance.

Prospective custody now scores before eviction and preserves every alert with
the original PT AUX bytes, perf records, dynamic maps, and a hash-verified
deduplicated exact-boot decode state. A nine-execution physical smoke retained
all three alerts and safely evicted six below-threshold rows after scoring. A
blind 8+8 Dirty Pipe control on the fixed production kernel manifested no
mutation, as expected, but an intentionally tiny smoke model could not separate
effect from neutral inputs. The archived vulnerable Ubuntu `5.13.0-30.33`
subject is now reachable over Wi-Fi and USB Ethernet. It completed 3,060 benign
tri-modal captures and subject-matched frozen training. A blind 12+12 Dirty
Pipe validation manifested all effect mutations, but its supposed neutral arm
also executed the vulnerable splice/write path with unchanged bytes. Its AUROC
0.6319 is therefore invalid as a lawful-counterfactual metric. The control is
now corrected to use an ordinary read before writing to the pipe. Next: rerun
the blind effect/lawful-sibling comparison. That replay failed: AUROC 0.3681,
2/12 effect alerts versus 5/12 lawful-sibling alerts, with 24/24 lossless
captures and 20/20 effect mutations each time. Precise kernel-store sampling
is available on the same host and is now an optional, event-identity-separated
PEBS observation path. Next: qualify simultaneous PT/store/PMU capture and
test whether richer relative-IP/address features improve this gate. The first
store-based model also failed (AUROC 0.4375, zero alerts in either arm). A closer
non-triggering splice control is now implemented and also failed the v3 gate
(AUROC 0.5069, zero alerts in either arm), despite 6.1 million precise-store
samples in its benign corpus. The laptop's 97--100°C short-run package
temperature required a fixed no-turbo
policy before sustained collection; exact subject identity now records that
policy. The user-authorized scale step is a bounded 102,000-execution benign
collection with intensity variants and a five-hour hard stop, followed by a
capacity comparison on the same immutable corpus. It tests data and model
scale, **not** a claim that the current detector finds Dirty Pipe. Full-day
expansion remains conditional on validation and bounded storage/custody.
Semantic Intel PT decode remains open. The 24-hour campaign is **NO-GO** until
one of these paths demonstrates useful sensitivity and custody.

## Completed first iteration: AArch64 observation to MPS

Demo: run an unmodified, benign AArch64 program on prescribed input in `trail-arm`;
stream basic-block events through a bounded native pipeline and endpoint transport;
consume a tensor batch on macOS MPS with no client action calls. Basic-block-only
is a first slice, not the final signal contract. Use a tiny parser/checksum target,
with a known input and independently checkable output. The selected fixture is the
C17 stdin byte checksum in `native/examples/checksum.c`.

| Card | Status | Suggested owner | Exit evidence |
| --- | --- | --- | --- |
| C2T-00 Agent/development scaffold | Done | Coordinator | Repo rules, skill, architecture, work board, environment routing validated |
| C2T-01 Dependency and semantic probe | Done | Investigator | [Probe](qemu-probe.md) records ARM/x86 features and limits; user-authorized external ARM QEMU build and smoke check passed |
| C2T-02 First shared contract and build | Done | Coordinator / reviewer | Portable CMake tests pass on Mac/Linux; editable and regular wheel client tests pass; actual native/Python navigation passed in VS Code; [IDE limitations](ide-evidence.md) recorded |
| C2T-03 Real ARM capture | Done | Capture builder | Real block entries, known ELF address, final partial batches, pthread sources, faults and explicit unsupported behavior checked |
| C2T-04 MPS consumer | Done | Consumer builder | 11 fixture tests pass on CPU/MPS, including exact address bits, retained storage and model forward/backward |
| C2T-05 Remote integration | Done | Coordinator | 9 real-worker/lifecycle checks pass, including ARM-to-MPS, slow consumer in pipe backpressure, faults, disconnects, child reaping and same-port restart |

C2T-01 must inspect full-system feasibility on the x86 host too: TCG plugin support,
multi-vCPU callback semantics, and usable operator-provided guest artifacts. It
does not need to implement a kernel adapter. Record missing dependencies and follow
the operator's provisioning instructions. The user authorized a separate development
QEMU build after the initial probe; the package must never install it. A missing dependency
does not block contract and consumer fixture work that can be done independently.

Only C2T-03 and C2T-04 are planned for parallel builders. Do not start either against
different invented versions of the batch contract. The first iteration exits only
when C2T-05 is demonstrated on the real endpoints. Small allocations/copies outside
hot paths are acceptable when explicit and bounded; record the baseline before
attempting optimization.

The shared contract and working build were settled before the capture and consumer
builders started. C2T-02 GUI verification proceeded independently and passed in
VS Code; successful integration was not substituted for that check.

## Completed signal implementation

| Card | Status | Owner | Exit evidence |
| --- | --- | --- | --- |
| C2T-06 Signal semantics and reference review | Done | Investigator | Public QEMU API probe, controlled ARM/x86 fixture, independent review; no AlphaFlow implementation copied |
| C2T-07 Native capture and wire version 2 | Done | Capture builder / coordinator | Per-vCPU baselines and deltas, memory address/size/direction, opt-in transaction values, validated framing and baseline completeness |
| C2T-08 Tensor columns | Done | Consumer builder | Owned register/memory tensors; CPU and MPS bit patterns, widths, retention and model checks |
| C2T-09 Real signal integration | Done | Coordinator | ARM and x86 guest checks, rich pthread/fault checks, separately installed wheel; [results](instrumentation-results.md) |
| C2T-10 CUDA execution | Pending device | Consumer builder | Run the existing CUDA check on an operator-provided CUDA learner |
| C2T-11 Observation throughput | Measured first optimization | Coordinator | See current iteration below: mixed frames, experimental rings and packed/collated CPU-to-device path |

The original baseline yielded one signal kind per batch. Version 0.5 adds opt-in
mixed frames and CPU collation, preserving per-vCPU event sequences and owned
columns. Register samples remain checkpoints rather than every write or final
state. [Capture performance](capture-performance.md) describes current buffering;
the original [capture audit](capture-efficiency.md) remains design history.

## Completed iteration: kernel integration and learning

| Card | Status | Evidence / remaining work |
| --- | --- | --- |
| C2T-12 Trace learning quickstart | Done | Real ARM capture, exact sample-count oracle, unique held-out inputs/features, CPU/MPS training and checkpoint reload; [results](learn-trace.md) |
| C2T-13 Stdin actions and learning | Done | Single-vCPU stop at real read, reset-by-restart, cancellation, optional Gym; real MPS imitation and REINFORCE, both 40/40 fresh evaluations; [results](stdio-example.md) |
| C2T-14 Benign kernel guest | Done | Static PID1, rootless initramfs, pinned concurrent memory workload, exact checksums and clean poweroff; [scope](kernel-examples.md) |
| C2T-15 Kernel rich capture | Done | Upstream QEMU 11 developer build; rich baselines/values and exact parallel routine on both CPUs, repeated drain fences, full boot through poweroff; [results](kernel-integration-results.md) |
| C2T-16 Kernel pretraining learner | Done | Three actual x86 kernel workers, two training and one held out, MPS loss 5.71269→4.67176; full draining and checkpoint reload; [results](kernel-pretraining.md) |
| C2T-17 Kernel Gym adapter | Done | QMP stop plus explicit all-source drain; KernelEnv/Gym; real CPU/MPS finite policy updates, exact result oracles, reset/reaping, bounded partial actions and fixed-vCPU validation; [results](kernel-gym.md) |

The `example/` directory is the user entry point; native targets stay under
`native/`, and installed Python modules stay under `python/cpu2tensor/examples/`.
The stdin challenge uses validated ordinary input. Kernel learning uses an explicit
postboot window, block features, and fixed benign syscall workloads. Rich register
and memory capture has separate correctness evidence. CUDA, AWS execution, DDP,
and optimized observation throughput remain open; do not claim GPU-bound scale.

## Following slices

| Order | Outcome | Main acceptance checks |
| --- | --- | --- |
| 2 | Core signals and portability | Register changes plus memory address/width/direction; opt-in values; ARM and x86 validation; exact widths/bit patterns and completion; CUDA consumer |
| 3 | Kernel observation | Benign x86 full-system workload with multiple vCPUs; per-source sequence/completion; no global per-event lock; non-dropping backpressure |
| 4 | Multiworker observation throughput | Same target on several operator-started workers; variable-size batches; no all-worker step barrier; values on/off; x86 throughput, peak memory, and stall evidence |
| 5 | Simple stdio interaction | Local and remote reset/step, chunk reducer, pause at input boundary, clean restart/error behavior; optional Gym wrapper; no public trajectory registry |
| 6 | Two learning examples | Fixed recurrent state and bounded attention cache on MPS/CUDA; choose README quickstart by usability and measured limits |
| 7 | Kernel interaction and public package | Benign syscall-adapter workload; operator-provisioned AWS pretraining example; wheel/install checks on supported platforms; license/dependency review; runnable README |

Start a small stdio-boundary probe as soon as C2T-01 exposes available facilities;
do not wait until slice 5 to discover that unmodified stdin blocking is unobservable.
It is an investigation, not an asynchronous environment framework. Likewise, test
the kernel capture seam early and deepen it in slice 3. Agile slicing does not defer
all difficult assumptions until the end.

## Risks to resolve through evidence

| Difficulty | Failure to avoid | Earliest useful check |
| --- | --- | --- |
| External QEMU capabilities | Missing plugin API or incompatible operator build | Inventory plus minimal capture in C2T-01/03 |
| Cross-ISA signal meaning | x86 register assumptions or a memory reread mislabeled as the access value | Controlled register/memory fixtures in slice 2 |
| Terminal and fault semantics | Silent final-event loss or claiming faulting accesses were captured | Exit/fault/partial-batch fixtures in 01/03 and slice 2 |
| Tensor ownership and dtype | Recycled backing memory, lossy addresses, or unsafe async copy lifetime | Small retained-batch/device checks in C2T-04/05 |
| Multi-vCPU progress | Artificial guest serialization, deadlock on an idle CPU, or false total ordering | Multi-source fixtures from 02; real kernel in slice 3 |
| Stdio and world pause | Mistaking output prompts for input requests or sealing before all required sources stop | Early boundary probe; slice 5 real interactions |
| Throughput | Python per-event work, excessive copies, unbounded queues, GPU starvation | Phase timings and end-to-end measurement in slice 4 |
| Packaging and provenance | Hidden dependency installation or incompatible incorporated code | First build choices and review at each incorporation; release install checks |

## Evidence and next action

- Original block-only evidence is in [first-slice results](first-slice-results.md).
  Current signal evidence is in [instrumentation results](instrumentation-results.md).
  Native core tests pass on macOS and ARM Linux. The 11 consumer tests and 9 remote
  worker/lifecycle tests pass. A separately installed wheel passes the 11 consumer
  checks from outside the checkout. No backend checks were skipped in those runs.
- 2026-09-07: Activated the first vertical-slice goal, covering C2T-01 through
  C2T-05 and a runnable observation README quickstart. Completion requires real
  ARM-to-MPS evidence, bounded ownership, slow-consumer and disconnect checks,
  and the IDE/build checks. Keep implementation and prose in simple English;
  prefer direct code over abstractions introduced for possible future use.
- C2T-01 initial inventory, 2026-09-07: `ssh trail-arm` reports Linux AArch64,
  `/usr/bin/qemu-aarch64` 8.2.2 (Ubuntu `1:8.2.2+ds-0ubuntu1.17`), and CMake,
  Ninja, C++, and Python 3 on PATH. `ssh trail-x86` reports Linux x86-64,
  `/usr/bin/qemu-system-x86_64` with the same version, plus the same build tools.
  Commands used `uname -sm`, `command -v`, and QEMU `--version`; version presence
  does not establish plugin compatibility. Neither conventional QEMU include
  directory was found in this initial listing. The x86 optional directory listing
  returned status 2; SSH and the version query succeeded.
- Local inventory: macOS arm64 has CMake, Ninja, Clang, CLion, PyCharm, and VS Code.
  The default Python is Homebrew Python 3.14; `importlib.util.find_spec` did not
  find Torch, scikit-build-core, or pytest there. A later check found an existing
  CPython 3.10/Torch 2.13 environment with working MPS, used for host-local editable
  and wheel test environments. The initial inventory installed nothing.
- 2026-09-07: User accepted the orchestration proposal and required both source
  roots to work naturally in standard IDEs. Language/layout requirements and
  first-build acceptance checks are recorded in `docs/ide.md`.
- C2T-00: skill frontmatter validated with the bundled skill validator using an
  existing Python environment with PyYAML; 10 Markdown files checked for local
  links/whitespace; UI metadata checked; `.local/environments.md` confirmed ignored.
  ARM SSH confirmed the shared repo at `/home/user/workspace/cpu2tensor`. The full
  AGPLv3 license was retrieved from GNU. These checks validate the scaffold, not
  agent behavior during implementation or runtime correctness.
- 2026-09-07: Existing AlphaFlow record, ring, staging, and build boundaries inspected
  as references. No whole-harness audit or performance validation claimed.
- Host access/build paths are in `.local/environments.md`. ARM QEMU and Mac MPS
  now have real execution evidence. CUDA and actual kernel capture remain later work.
- Final audit: real ARM-to-MPS, exact source/address semantics, bounded ownership,
  backpressure, completion/failure handling, packaging, readable client example,
  and actual VS Code navigation all have evidence. JetBrains checks failed in this
  environment; Pylance's compiled-source warning is documented without suppression.
- The user selected core instrumentation after the first goal completed. That
  implementation is now recorded above; the old goal remains complete. Next work
  is CUDA execution when a device is available and observation batching overhead.

- 2026-09-08: Kernel integration completed on `codex/kernel-integration`. Full boot
  block capture, rich postboot observations, both active vCPUs, paused-world
  actions, repeated reset/reaping, bounded action delivery, observation-only
multiworker MPS training and CPU/MPS Gym policy updates are checked. The stress
  harness now drains diagnostic output independently. See
  [kernel results](kernel-integration-results.md) for exact scope and open limits.

## Current iteration: rich-capture correctness and throughput

Requested 2026-09-08. The [iteration proposal](rich-capture-plan.md) separates
accepted scope from open decisions. Preserve simple synchronous clients and an
observation-only path. One learner device consumes one worker, multiple local
workers, or multiple remote workers. AWS execution and actual CUDA validation
are required evidence. DDP, multiple learner devices, hotplug, migration,
rebooted episodes, and external monitor control are deferred.

| Card | Status | Exit evidence |
| --- | --- | --- |
| C2T-18 Trace semantics | Initial fixes implemented and checked | [State results](state-correctness-results.md): unavailable fields filtered, exact named selections, raw system GPR/mode hook, ordered context and physical-prefix coverage; optional fresh snapshots remain C2T-22 |
| C2T-19 Rich performance baseline | Capture matrix and learner isolation measured | Original [signal costs](benchmark-capture-results.md), [30-run framing/publication matrix](performance-matrix-results.md) and [matched MPS replay plus live rich training](pipeline-performance-results.md) |
| C2T-11 Observation throughput | First optimization checked; native collector work remains | Mixed/pipe framing share 28.85%→10.04%; packed upload plus 64 KiB collation cuts matched MPS drain median 12.101→0.350 s. Rings implemented/tested but not a measured improvement; native shared-memory column pages and overlap remain open |
| C2T-20 Multiple endpoint pool | Implemented and checked | Synchronous endpoint Pool, explicit worker/source identity, bounded CPU readers and caller-thread upload; three real ARM workers train on CPU/MPS; one/two x86 kernel workers deliver rich captures to MPS |
| C2T-10 CUDA execution | Pending assigned learner | Exact rich columns, retained storage, packed upload lifetimes, forward/backward and checkpoint reload on real CUDA hardware |
| C2T-22 Optional boundary state | Mechanism pending | Fresh validated all-vCPU register acquisition at action/end boundaries, separately acknowledged from trace drain |
| C2T-21 AWS scale evidence | Pending assigned infrastructure; local pilot complete | One/two-process x86-to-MPS rich drain completes; repeated sustained scaling and actual multi-host AWS runs remain required |
| C2T-23 Native tensor notebook | Done | [Executed tutorial](tutorials/normalization.ipynb): initial three-field executable layout, real same-binary relocation across 11 locations, CPU/MPS learning curves and controls; [evidence](normalization-results.md) |
| C2T-24 Absolute observation deadline | Implemented and checked | `--max-run-ms` covers active forwarding, backpressure and post-seal exit; 17 real ARM checks and one ordinary x86 kernel boot check pass; [semantics/evidence](worker-deadlines.md) |
| C2T-25 Context-only x86 system profile | Done; independent client acceptance reported | Independent context selection, benign paging oracle/public fallback, two-vCPU Pool acceptance, 12-run cost matrix and client getpid/26-workload acceptance; [guide](context-only.md) |
| C2T-26 Repeated action-window reducer | Done | Fixed per-vCPU adjacent-block counts, raw-block opt-out, explicit ended/aborted/incomplete and overflow metadata; real two-vCPU guest proves transport-size invariance, distinct bodies and abort through KernelEnv; [contract and evidence](action-windows.md) |
| C2T-27 Continuous integration | Done; hosted verification passed | Pinned container, 95% native/Python line gates, hosted units in 14 seconds, and merge/`[TESTME]` real-QEMU systems in 8 seconds; [workflow](ci.md) |

The user agreed to explicit sampled state and attributed transactions, while
full RAM reconstruction remains deferred. Runtime fixes, probes, contributor
extension guidance and initial named-host overhead evidence are in
[state results](state-correctness-results.md). New events must state scope,
validity and cost; do not fabricate CPU attribution for worker-wide metadata.
Optional paused-world snapshots remain unavailable, and cross-page physical
coverage is deliberately partial. Opt-in mixed frames and bounded tensor collation now address small-frame overhead.
[Measured results](pipeline-performance-results.md) distinguish matched learner
improvements from single live runs and the native ring regression. No GPU-bound
claim is made. One learner device remains the supported scope.

The user then requested a notebook built around a useful new normalization tensor.
The initial user-process layout is now implemented as an opt-in worker-wide event;
general load/unload mappings and cross-worker synchronization remain future work.
No hot-path framework or async client machinery was introduced for the tutorial.
