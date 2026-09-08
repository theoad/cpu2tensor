# Rich capture: correctness and throughput

Proposed 2026-09-08 after the kernel integration. Status belongs to the
[work board](backlog.md). This document records the next decisions and acceptance
checks. Initial correctness fixes and the first x86 benchmark are now recorded
in [state results](state-correctness-results.md); later pipeline changes remain proposed.

## Accepted scope

Optimize observation-only traces with selected registers and memory transactions,
including opt-in values. Demonstrate one worker, several workers on one host,
and workers on several remote hosts, all feeding one learner device. Validate
CUDA on real hardware and run the remote case on AWS. Preserve CPU/MPS support,
simple synchronous clients, per-vCPU order, bounded storage, lossless backpressure,
and the existing interactive stop/drain contract.

DDP, multiple learner devices, hotplug, migration, rebooted episodes, and external
monitor control are deferred. Operators still supply infrastructure, connectivity,
dependencies and guest artifacts. AWS/CUDA endpoints have not been assigned in
this checkout. No paid infrastructure has been launched for this iteration.

## Register decisions

Separate sampling coverage from incorrect values. Block-entry deltas represent
state at a checkpoint; they cannot recover a register that changes and changes
back inside the block. Moving publication into rings does not reduce the cost of
reading every selected register at each checkpoint.

Recommended contract: validated selected-register state at each captured QEMU
translation-block entry, with explicit baseline and checkpoint identity. Checkpoint
deltas must reconstruct that selected state exactly. A trace publication fence
and a fresh register snapshot are different events. An action/end boundary must
report whether a new snapshot exists; never silently label the last block sample
as final state. The user accepted optional validated boundary snapshots. Their acquisition
mechanism is still unimplemented; the current client reports them unavailable.

The matching upstream x86 source review found:

- `eflags` is lazy during hot callbacks; its omission is accepted for now.
- `all` includes `ftag`, `fiseg`, `fioff`, `foseg`, `fooff`, and `fop` whose readers
  return constant zero. Omit or explicitly identify these as unavailable; success
  from the register-read API is insufficient validation.
- GPR readers mask upper bits outside 64-bit mode. Register interpretation needs
  execution-mode context. In segmented modes, register EIP and linear execution
  PC are different quantities.
- There is no demonstrated stale block-entry RIP on this particular system
  build: it enables PC-relative translation and updates EIP before chained
  branches. Do not generalize this to arbitrary builds or mid-instruction reads.
- QMP stop and buffer drain do not supply a new register sample. The control
  thread cannot call the vCPU-only register API. Idle callbacks can read registers,
  but stop does not acknowledge that every idle callback ran, and round-robin
  TCG uses a different wait path. Exact boundary snapshots need a supported,
  separately validated mechanism. Atexit register reads are forbidden.

Use controlled assembly and independent guest stores as oracles for selected
registers at named PCs, chained blocks, mode changes, and supported action/end
boundaries. Test write-then-revert behavior as intentionally unsampled, rather
than treating it as a dropped event. Unsupported or unavailable fields must not
become plausible zero-valued training labels.

## Memory decisions

Virtual addresses remain useful for instruction-relative access patterns, but
are insufficient for distinguishing physical storage or address spaces. Proposed
system capture adds actual-access physical mapping, RAM/MMIO classification,
validity, and paging context. Keep raw CR3 and relevant mode controls; CR3 is not
a PID. Prefer context-change records with per-source ordering over repeating all
paging registers on every memory row, once sampling coverage is proven.

The upstream public API already exposes CR0, CR3, CR4 and EFER through register
discovery. Physical-address lookup uses the recently used soft TLB, rather than
walking current page tables again. Copy the callback-scoped result immediately.
Do not substitute a later page-table walk for the mapping used by an access.

A logical access may cross two noncontiguous physical pages. One physical address
plus the original width cannot represent that mapping. Validate a segmented
representation with a controlled noncontiguous-page fixture. Until a supported
lookup for all segments is established, only the first-byte mapping is known;
coverage must be explicit. The public header also warns that physical addresses
need not uniquely identify storage across multiple QEMU address spaces. Do not
claim a universal identity from GPA alone.

Access attribution and complete memory reconstruction are separate deliverables.
Full reconstruction needs an initial state and coverage of implicit writes,
including page-table accessed/dirty updates and device writes. The inspected
x86 page-table walker performs some A/D updates with direct host atomics outside
ordinary instruction memory callbacks. Adding CR3, page tables and GPA alone
does not close that gap. The user agreed to defer full memory reconstruction and initial RAM/page-table
snapshots for this iteration; DMA capture was previously deferred.

Test the same VA in distinct address spaces, virtual aliases of one physical
page, remapping and TLB invalidation, noncontiguous cross-page accesses, RAM/MMIO
classification and concurrent vCPUs. Never infer a global memory order from
collector or network arrival order.

## Performance work

The [existing source audit](capture-efficiency.md) identifies tiny single-kind
frames, callback pipe writes, learner transcode copies and per-column synchronous
device transfers. Use this sequence:

1. Record a fixed-coverage x86 baseline before changing the runtime. Keep both
   the 0.4 behavior and corrected semantics distinguishable in comparisons.
2. Settle one mixed-page contract with per-source event order, checkpoints,
   optional fields and ownership. Then build preallocated per-vCPU SPSC chunks
   and a fair native collector. No allocation, syscall or shared event lock in
   the normal append path while capacity exists. Full buffers must wait safely
   with cancellation and collector-liveness handling.
3. Deliver owned column pages, removing tiny-frame construction and redundant
   receive/transcode copies. Bound both page bytes and queue depth. Separate local
   shared-memory capture layout from the portable remote wire format. Start with
   measured loopback transport for local clients; add another local delivery
   backend only if its remaining copies are a material bottleneck.
4. Extend Pool to multiple endpoints with ordinary iteration and internal bounded
   prefetch. Identify rows by worker and vCPU, preserve each source's sequence,
   and consume ready workers without waiting for all others. A failed endpoint
   cannot disappear as successful completion. Keep public recurrent state and
   action scheduling out of this change.
5. Batch transfers and overlap preparation/upload/compute where supported.
   CUDA pinned memory and side streams require explicit lifetime and dependency
   handling. Never recycle a device allocation merely because upload finished.
   Retained tensors must continue to own valid storage. MPS gets its own measured
   implementation; do not assume CUDA pinning/stream mechanics apply to it.

PyTorch documents the transfer and ownership constraints in its
[pinning and nonblocking guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
and [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html).
Pinning immediately before every copy is not a substitute for a measured
preparation pipeline.

## Measurement and exit evidence

Use a reproducible supplied-input kernel workload and fixed signal coverage:
no plugin, blocks only, rich values off, rich values on. Name the x86 host,
CPU allocation, QEMU/kernel/build, input, capture window and register profile.
Do not compare unlike coverage as an optimization speedup. Repeat runs and
report variation. Measure postboot steady state separately from full-boot cost.

For one worker and local/remote worker-count sweeps, report events/s by kind,
useful bytes/s, wire bytes/s, target slowdown, per-worker fairness, CPU use, peak
resident/pinned/device memory, chunk occupancy, producer stall time, and batch
delivery latency. Instrument phase timings with an explicit overhead check.
Use byte limits as well as batch counts to keep variable-width captures bounded.
Reserve benchmark hosts; do not run competing jobs on the same allocation.

Separate capture-to-discard, transport-to-CPU, device upload and complete training
measurements. The rich learning case must consume register and memory features;
the existing block-only pretraining learner discards those batches and therefore
cannot establish useful rich-training throughput. Use exact fixture/oracle checks
before tensor feature transforms, then real forward/backward and checkpoint reload.

Compare live training with the same workload/model replayed from prepared device
data to identify learner starvation. Report learner waiting and utilization;
do not promise that any single target can saturate any GPU. Set numerical
throughput/latency targets after the baseline and assigned hardware are known.

CUDA acceptance includes exact widths/address bits/value bytes, retained CPU and
device storage, nondefault-stream dependencies, slow consumers, cancellation,
bounded memory and repeated real model updates. Run separately installed wheel
checks; a skipped CUDA test is still missing evidence.

AWS acceptance requires actual separate worker hosts and an identified CUDA
learner. Record the network path, instance types and sustained limits. EC2
[network bandwidth](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-network-bandwidth.html)
can have baseline/burst and per-flow constraints; advertised peak bandwidth is
not a sustained rich-capture result. Include transfer costs in any cost report.

## Source-review anchors

Read-only review on `trail-x86` used the exact upstream source/build recorded in
[kernel dependency evidence](kernel-qemu-build.md). These are source findings,
not new runtime probes. Paths below are relative to that QEMU source root.

| Question | Source |
| --- | --- |
| Control registers, constant FP fields, mode masking | `target/i386/gdbstub.c:122`, `:186`, `:202` |
| Lazy flags and PC synchronization | `target/i386/tcg/tcg-cpu.c:32`, `:70` |
| PC-relative enablement and chained branches | `target/i386/cpu.c:9905`; `target/i386/tcg/translate.c:2039` |
| Physical lookup and handle lifetime | `plugins/api-system.c:49`; `include/plugins/qemu-plugin.h:692` |
| TLB lookup and split-page access | `accel/tcg/cputlb.c:1578`, `:1729`, `:2336`, `:2755`; `accel/tcg/ldst_common.c.inc:175` |
| Register callback context and idle permission | `plugins/api.c:455`; `plugins/core.c:628` |
| MTTCG versus round-robin idle paths | `accel/tcg/tcg-accel-ops-mttcg.c:88`; `accel/tcg/tcg-accel-ops-rr.c:108` |
| Implicit page-table reads and A/D updates | `target/i386/tcg/system/excp_helper.c:88`, `:128` |
