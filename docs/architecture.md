# Product and architecture

## Accepted scope

`cpu2tensor` is a standalone native library and pip package. AlphaFlow will be a
client. New code is written from scratch with reviewed implementation references.
The license is AGPL-3.0-only.

The primary throughput path runs targets on supplied inputs and streams traces
to a consuming model without model-generated actions. A second path supports
cooperating adapters and Gym-compatible interaction; stdio is the default adapter.
Clients own rewards, success/failure, models, and recurrent state.

Initial targets are an unmodified AArch64 Linux process and an x86-64 Linux kernel.
The Mac development path runs the target in an operator-provided UTM Ubuntu VM
and learning on macOS MPS. CUDA is the other initial learning backend. Both fixed
recurrent state and bounded attention-cache examples are required; choose the
quickstart by observed simplicity, latency, correctness, and memory use.

Basic-block observations are sufficient initially; retain the ability to add
instruction-level observations. CPU changes, memory addresses/access widths, and
source identity are foundational. Memory values are opt-in. Raw addresses must
remain exact, with address-space context and clear ASLR semantics. Device-originated
memory changes are deferred; future asynchronous sources must fit without fake
vCPU attribution. Physical and emulated signals need distinguishable provenance.

A worker owns one target and its vCPUs. Endpoint pools support local and remote
workers and explicit device assignments. v0 clients use ordinary iteration and
simple environment operations. Restart is sufficient for reset. Interactive action
boundaries pause the world; lossless backpressure may also pause execution.

QEMU and other dependencies are operator-supplied. Validate missing/incompatible
features with actionable diagnostics. The package must not build, download,
install, or bundle QEMU. Separately authorized developer provisioning is recorded
in [environments](environments.md).

## Proposed source layout

The native language is C++20; guest-facing headers and guest fixtures use C17.
Both `native/` and `python/` must open naturally in standard IDEs. Follow the
[language and IDE contract](ide.md) for includes, source roots, package discovery,
host-local build directories, and validation. These are first-build requirements.

Create these components as the first slices need them, not as empty scaffolding.

```text
cpu2tensor/
  AGENTS.md                 essential repo rules and routing
  .agents/skills/           one initial development skill
  docs/                    architecture, workflow, backlog, evidence
  native/
    CMakeLists.txt          all native targets and policies
    include/cpu2tensor/     intentionally public native interfaces
    core/                  trace types, buffers, batch ownership
    qemu/                  capture backend and architecture adapters
    worker/                target lifecycle and optional interaction control
    transport/             bounded local/remote trace transport
    python/                native Python binding implementation
    tests/                 native correctness fixtures
    examples/              simple C17 target programs
  python/
    cpu2tensor/            pool, target, batch, environment; add modules as needed
    tests/                 Python and end-to-end tests
    examples/              observation, interaction, recurrent/attention models
  pyproject.toml           build and package metadata once packaging is implemented
```

`native/CMakeLists.txt` is independently configurable. An optional root CMake
entrypoint may only delegate to the native build. Do not
create separate native trees for each use case or device. Python device integration
starts with PyTorch facilities; add native device-specific code only after evidence
shows it is necessary. Binding/build backend selection belongs to the first build
slice and must satisfy the lean-core and external-QEMU constraints.

## Ownership boundaries

| Component | Owns | Must not assume |
| --- | --- | --- |
| Capture | Guest events, per-source order, backend capabilities | A learner, a reward, a global vCPU order |
| Worker | Target/input lifecycle and optional action adapter | Cloud provisioning or training policy |
| Transport | Framing, bounded queues, compatible schemas, completion | Host pointers/atomics valid on a remote machine |
| Batch layer | Typed arrays, offsets, trace completeness and memory ownership | Every client wants memory values or interactive control |
| Python client | Simple iteration, explicit worker/device assignment, errors | Public trajectory registries or hidden state migration |
| Learning examples | Models, state, loss, replay, reward | That transport chunk boundaries are RL steps |

Select observation-only operation at setup. Its hot path must not dispatch action
callbacks or wait for model decisions. Use one shared signal/data implementation,
with interaction layered on the worker. Start with bounded copies where needed;
remove measured copies without weakening lifetimes or advertising unverified
zero-copy behavior.

Native hot paths use preallocated per-source buffers and batched publication.
Collation must not lock independent guest basic blocks together. Multi-source
completion uses explicit progress/end state; an idle vCPU cannot stall draining
forever. Host transport order is not guest causality. Backpressure perturbs timing
and must be measured and reported. The [capture efficiency investigation](capture-efficiency.md)
proposes per-vCPU SPSC chunks and worker-owned column materialization; this
replacement is not implemented yet.

The common batch contract needs just the slice's event types, widths, optional-value
validity, source identity/sequences, completion, and lifetime rules. Add a schema
version and reject unsupported mandatory features. No general plugin registration
framework, schema language, or frozen public ABI is needed before the first demo.

## Initial reference map

Paths below are relative to an operator-supplied AlphaFlow checkout. Read the narrow
reference needed for the current task. References are not runtime dependencies.

| Question | Reference |
| --- | --- |
| Record fields and completion | `tasks/execution-exploration/experiments/reachability-v0/native/reachability/trace_record.hpp` |
| Buffer publication and layout | `tasks/execution-exploration/experiments/reachability-v0/native/reachability/spsc_ring.hpp` |
| Buffer recycling through copies | `tasks/execution-exploration/experiments/reachability-v0/native/reachability/staging_pool.hpp` |
| Native transcode and tensor structure | `tasks/execution-exploration/experiments/reachability-v0/native/src/trace_transcoder.cpp` and `tensor_writer.cpp` |
| Measurement methodology and known limits | `docs/execution-exploration/reachability-v0/results/combined-trace-buffer.md` and `trace-to-tensor-native.md` |

Initial inspection found experiment-specific register widths, an old fixed record
layout, compatibility layers, and some status/out-parameter APIs. Those are reasons
to extract contracts carefully rather than porting the tree. These observations
are not a completed correctness, performance, or license audit.

## Deferred choices

Dynamic async action batches, opaque public trajectory identities, managed latent
caches, device migration, snapshot reset, DMA capture, and delay-action budgets
remain deferred. Minimal internal sequencing and stale-response protection still
belong to correctness. Kernel support and multiworker observation-only throughput
are required milestones, not optional extensions.

## Implemented kernel boundary

The system adapter reuses the same signal frames, native tensor conversion, and
synchronous client reducer as observation-only capture. `KernelEnv` adds only
bounded guest event/result metadata and reset/step at a verified world pause.

QMP stop establishes vCPU execution quiescence. A private control thread then
flushes captured bytes under per-source cold locks and publishes the action
boundary in trace order. Release/acquire handoff protects both reading and reusing
those buffers; no register API is called by that thread. Init, idle and exit use
the same cold locks. Hot callbacks remain owned by their vCPU and acquire no
mutex. Fixed vCPU count, exclusive control channels, and restart-based episodes
are the current system adapter contract.

The observation-only path does not start that control thread or execute action
callbacks. An optional explicit start block selects a postboot capture window;
other vCPUs join at their next block entry without a scheduling barrier. See
[kernel setup](kernel-examples.md) for exact semantics and backend limits.
