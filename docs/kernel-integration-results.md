# Kernel integration results

Checked 2026-09-07/08. Code/build identity is recorded in
[kernel-source.sha256](kernel-source.sha256). This is the 0.4.0 kernel integration;
prior user-process results remain separate historical records.

## Actual paths checked

The worker host was `trail-x86`, Ubuntu 24.04.2 x86-64. It used unmodified upstream
QEMU 11.0.3 with matching API 6 headers, TCG with two vCPUs, 256 MiB RAM, the
operator's Linux 6.9.0-dirty kernel, and the static benign PID1/initramfs. The
[dependency review](kernel-qemu-build.md) records signed-source provenance,
register limitations, and independent raw trace probes.

The learner was Apple M2, macOS 15.7.3, Python 3.10.12, PyTorch 2.13.0. Separate
CPU and MPS kernel Gym runs each checked getpid/memory/pipe/parallel results,
performed four finite policy-gradient updates, changed weights, and completed
through `quit`. Both checkpoints reload; see [the Gym example](kernel-gym.md).

The [observation-only trainer](kernel-pretraining.md#real-kernel-evidence) ran two
training workers and one held-out worker on the x86 host. MPS held-out loss fell
from 5.71269 to 4.67176 after 100 updates. All three seeded kernel traces completed
and the saved weights reproduced metrics when restored on CPU. These endpoints
were separate processes on one remote host, not AWS instances.

## Capture and lifecycle evidence

- Rich capture includes selected-register baselines/deltas, memory addresses,
  widths/directions, and actual transaction values. The independent raw probe
  verified the first 16 byte stores of each pinned parallel memory routine,
  both checksums, and contiguous per-source sequences.
- Each raw rich probe exercised six stop/drain/resume fences; a second run held
  every fence for at least 50 ms and rejected any post-fence data. Both passed.
- The Python client found the exact parallel routine entry on both source 0 and
  source 1, verified 4096-byte checksums 520419/522544, retained independent CPU
  storage, transferred a retained value tensor to MPS, and ran backward.
- Full boot without a start marker completed through poweroff: 482,583,624 block
  entries, first PC `0xfffffff0`, kernel addresses and the parallel routine on
  both vCPUs. This check used block-only capture. Rich full-boot throughput was
  not measured; rich correctness checks selected a postboot window.
- Real tests cover slow-consumer pipe backpressure without sequence gaps, reset
  while paused and midstream with old children reaped, a partial-action deadline
  that is not extended by later bytes, an unreached start marker failing capture,
  and rejection of spare hotplug slots in kernel action mode.

Seven real kernel checks passed across the full suite and focused follow-ups.
Ten kernel socket/decoder/Gym fixtures pass. The existing ARM observation/stdin
regressions passed, including the separately selected x86 user guest register
check. Portable native CTest passed on macOS, ARM Linux, and x86 Linux.
The regular 0.4.0 wheel was installed in a separate Mac interpreter and tested
outside the source checkout: 61 passed, one CUDA-unavailable skip. Wheel SHA-256:
`c7ca1959e669702e0c4ae1252d336e3bb85a2b49f84c175bd2987b3f657ad406`.

An early stress attempt used an undrained diagnostic stdout pipe; enough boot
output could stall that fixture. The harness now stores both diagnostic streams
independently of trace consumption. Full boot and the larger rich workload were
rerun successfully. No throughput claim is inferred from their durations.

## Deliberate limits

The current rich path emits many small single-signal frames and constructs tensors
per frame. It is not GPU-bound capture and has no measured throughput improvement.
Per-vCPU rings, mixed column batches and copy/compute overlap remain the next
performance work. Large captures need suitable consumer and worker timeouts.

Upstream system TCG exposes stale lazy x86 `eflags` at hot callbacks; the plugin
omits that field with a visible diagnostic. Selected schemas are authoritative.
Register checkpoints are not every register write or guaranteed final state.
Memory addresses are guest virtual addresses; physical/PID/CR3 context and
DMA/device-originated writes are not yet columns. Arrival order never claims a
global memory order, and measurement/backpressure can perturb timing.

Kernel actions require a fixed vCPU count, exclusive worker-owned QMP/serial,
and reset by restart. Hotplug, migration, rebooted episodes and external monitor
controllers are unsupported. AWS execution, CUDA validation, DDP, and multiple
learner devices remain open. None of these dependencies is installed by the package.

Evidence artifacts are in `~/.cache/cpu2tensor/kernel-integration/` on the Mac and
x86 host, plus `kernel-drain-probe/` and `kernel-gym-check/` documented above.
