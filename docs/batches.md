# First batch contract

This document records the original version 1 block-only format. The current
implementation uses the [version 2 instrumentation contract](instrumentation.md).
Version 1 streams are rejected so old clients cannot silently miss new signals.

This is the shared contract for the first implementation. It describes block
entry callbacks, not proof that every instruction in a block completed. See the
[QEMU probe](qemu-probe.md) for capture and shutdown limits.

One worker runs one target process. Each vCPU has its own sequence starting at
zero. A batch contains consecutive entries from one vCPU. Arrival order between
vCPUs is transport order, not guest memory order. The first version supports up
to 256 vCPU indices, numbered 0 through 255, and rejects larger indices. Forked
address spaces and reuse of an exited vCPU index need a later contract.

## Wire format

All integers are little endian. Do not send native C++ structs. Every frame has
a 32-byte header followed by zero or more 64-bit block start addresses:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | uint32 | Magic `0x31543243` (bytes `C2T1`) |
| 4 | uint16 | Version, currently 1 |
| 6 | uint16 | Frame kind |
| 8 | uint32 | vCPU index |
| 12 | uint32 | Address count, at most 256 |
| 16 | uint64 | First sequence, or next sequence at source end |
| 24 | uint64 | Detail, defined by kind |

Kinds are Hello=1, Blocks=2, SourceEnd=3, Complete=4, Error=5.
Hello comes first: all fields except detail are zero; detail is 1 for AArch64,
2 for x86-64. Blocks has a nonzero count and zero detail. SourceEnd has zero
count/detail and the exact next sequence for that source, including an empty
source. Complete has zero source/count/sequence and detail is the target's normal
exit code (0 through 255). Error has zero source/count/sequence and nonzero detail:
1=capture failed, 2=target killed, 3=unsupported target behavior, 4=transport failed.
Reject unknown versions, kinds, details, fields, sequence gaps, and duplicate ends.
End-of-socket without Complete or Error is an incomplete trace, even on a frame
boundary. Complete requires every seen source to have ended. A nonzero target
exit code is reported to the client after preceding batches have been consumed.

The worker and plugin use the same frames internally. The plugin's Complete
only seals callback capture; the worker waits for the child and replaces its
detail with the actual exit code. The worker must not forward Complete before
observing child termination. Kill, transport loss, or a missing seal cannot become
successful completion. Capture completion is subject to documented QEMU coverage
limits, not a claim that instrumentation observes execution after QEMU disables it.

## Buffers and tensors

Each producer has a preallocated buffer. A full or final partial batch is published
as one pipe write smaller than Linux PIPE_BUF. This serializes batch publication,
not individual events. Independent vCPUs do not acquire a shared per-event lock.
The pipe and socket are bounded. A slow consumer can block publication and pause
execution; it must never silently drop entries. This changes timing.

The receiver normally validates one frame at a time in native code. For the
context-only mixed profile it can validate up to 32 complete frames together and
return one owned column set per source. It never waits for a partial successor to
fill a group. Control frames end the group, including SourceEnd and Complete.
Other profiles retain their wire-frame ownership and publication behavior.
Neither path creates Python objects per event. Tensors own their storage;
retaining a batch cannot expose a later recycled receive buffer. Client retention
is explicit user-owned memory; the pool does not retain consumed batches.

Addresses use signed int64 tensor storage to preserve all 64 bits on CPU and MPS.
Values with bit 63 set appear negative; unsigned address = signed value modulo
2**64. Do not cast raw addresses to floating point. Device transfer is synchronous
in this slice so lifetime is explicit. Models may derive their own features.

## Client boundary

Use ordinary iteration with endpoint strings and an explicit device. The first
slice connects to one operator-started worker per pool. The target command and
input are set on that worker; connecting starts its run. No SSH credentials,
installation, cloud provisioning, action requests, or model code belongs in the
package. Closing the client cancels its run and releases connections.

Multiworker scheduling, interaction, additional signals, and asynchronous device
copies are later work. Keep framing and capture independent of those policies.
