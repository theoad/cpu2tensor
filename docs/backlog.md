# Work board

Updated 2026-09-07. This file owns work status. The first vertical slice is complete:
real observation, integration, packaging, and standard-IDE navigation checks pass.
Core register and memory instrumentation is now implemented and checked on ARM,
x86 guests, CPU and MPS. CUDA execution and throughput work remain open.
Later milestones are outcomes, not schedule promises.

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
| C2T-11 Observation throughput | Investigation complete; implementation open | Coordinator | [Capture audit](capture-efficiency.md): per-vCPU SPSC capture and owned column pages proposed; ARM trace shape counted, no timing gains claimed |

The current API yields one signal kind per batch. This keeps ownership and client
control flow simple, but switching signal kinds flushes small frames. It is a
correctness baseline, not a demonstrated high-throughput implementation. Preserve
per-vCPU event sequence when improving batching; do not impose global event order.
Checkpoint register samples do not cover every write or promise final exit state.

C2T-11 next: replace callback pipe publication with cancellable per-vCPU rings
and a fair native worker collector, then settle ordered mixed column pages before
parallel producer/consumer edits. Preserve retained-tensor ownership; upload
completion alone does not permit device-buffer reuse. Acceptance and named-host
measurement requirements are in the capture audit.

## Current iteration: small learning examples

| Card | Status | Evidence / remaining work |
| --- | --- | --- |
| C2T-12 Trace learning quickstart | Done | Real ARM capture, exact sample-count oracle, unique held-out inputs/features, CPU/MPS training and checkpoint reload; [results](learn-trace.md) |
| C2T-13 Stdin actions and learning | Done | Single-vCPU stop at real read, reset-by-restart, cancellation, optional Gym; real MPS imitation and REINFORCE, both 40/40 fresh evaluations; [results](stdio-example.md) |
| C2T-14 Benign kernel guest | Done for guest only | Static PID1 and rootless initramfs; real two-vCPU QEMU boots in observation/command modes, exact results and clean poweroff; [scope](kernel-examples.md) |
| C2T-15 Kernel rich capture | Pending dependency and implementation | QEMU 8.2 on x86 lacks register/value APIs; separate QEMU 11 build permission requested. System capture, per-vCPU tails/completion and actual kernel traces still required |
| C2T-16 Kernel pretraining learner | Done for learner; kernel run pending | Bounded concurrent endpoints, separate source pairs, full completion after update budget; real CPU/MPS learning on disjoint ARM target inputs; [results](kernel-pretraining.md) |
| C2T-17 Kernel Gym adapter | Incomplete | Benign guest command adapter exists; QMP world pause, all-source trace boundary and model/action integration still needed |

The `example/` directory is the user entry point; native targets stay under
`native/`, and installed Python modules stay under `python/cpu2tensor/examples/`.
The stdin challenge uses validated ordinary input, not an exploit target. CUDA,
AWS endpoints and rich kernel learning have not been demonstrated. Do not promote
guest-only checks into captured-kernel or training claims.

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
