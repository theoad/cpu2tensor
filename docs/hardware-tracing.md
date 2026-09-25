# Hardware tracing

Hardware capture is an alternative to QEMU instrumentation when guest execution
under TCG would change the behavior being measured. It still has measurement
cost: counter interrupts, trace buffers, and the stop operation can perturb a
workload. Compare an untraced baseline with the exact signal and sample period
on the same host before drawing timing or race conclusions. A sampled event is
not a complete execution trace.

## Linux perf

`PerfCapture` opens kernel perf events and maps their data and AUX rings into
memory. Python only reads and tensorizes them after `stop()` disables every
event. The package writes no capture data to disk. There is no QEMU dependency.

```python
from cpu2tensor import HardwareConfig, PerfCapture

# The target is already running but waiting for its input.
with PerfCapture(HardwareConfig("process", pid=target.pid,
                                signal="memory_loads")) as capture:
    target.stdin.write(b"start\n")
    target.stdin.flush()
    target.wait()
    batches = capture.stop()

for batch in batches:
    print(batch.source, batch.ip.shape, batch.address.shape)
```

`scope="process"` attaches to threads present at entry and excludes kernel code.
`scope="process_kernel"` attaches in the same way but excludes user code, retaining
only kernel execution attributed to those threads. New threads and child processes
are not followed in either mode. Start capture while the target is waiting if
startup events matter. `scope="kernel"` uses `cpus=(...)` and samples the **host**
kernel on those CPUs, including other processes that run there. It does not by
itself expose a VM guest's kernel. The caller controls target lifetime and must
select a short enough capture window for the configured rings.

`cycles` and `instructions` produce sampled IP, PID, TID, CPU, perf timestamp,
and sample period tensors. Those generic PMU signals can work on Intel, AMD,
and AArch64 where perf exposes them. `memory_loads` and `memory_stores` use the
checked Intel `mem-loads` and `mem-stores` aliases: sampled virtual address,
weight, and data-source bits. Neither records every memory operation, a memory
value, or a page-table translation. `exact_ip` identifies samples for which perf asserts an exact IP;
the client must not assume all samples are exact. Addresses are signed `int64`
tensor bit patterns; reinterpret negative values as unsigned addresses when
needed.

`intel_pt` returns raw packet bytes in `trace_bytes`, with loss/gap checks on
the perf AUX records. It does not yet decode branches or instructions to IP
columns. A PT capture is only considered complete for the bounded bytes that
fit the ring. An overflow, reported loss, or unexpected record layout raises
`HardwareTraceLost` or `HardwareCaptureError` instead of returning a plausible
looking complete batch. Default data and AUX rings use 64 and 128 pages per source;
larger windows may need more pages and an operator-adjusted perf memory limit.

For a host-kernel sample:

```python
from cpu2tensor import HardwareConfig, PerfCapture

with PerfCapture(HardwareConfig("kernel", cpus=(0, 1),
                                signal="instructions", period=1_000_000)) as capture:
    run_workload()
    batches = capture.stop()
```

Each CPU is a separate source. `stop()` disables the descriptors in sequence;
it does not claim an atomic all-CPU boundary or a total memory order. The
recorded perf timestamps can align sampled observations, subject to each
source's clock and perf semantics. Target memory writes and DMA are not inferred
from sampled load addresses.

## Windows ETW / WPR

`WprCapture` generates a custom WPR hardware profile and starts it in WPR's
**memory** logging mode. `cycles` and `instructions` request PMU overflow
samples; `processor_trace` requests WPR processor-trace packets on sampled
profile events and can choose `User`, `Kernel`, or `UserKernel` code mode.
WPR's available sources should be checked with `wpr -pmcsources` on the actual
machine. WPR's processor-trace facility is not a portable promise of Intel PT
on every Windows CPU.

```python
from cpu2tensor import WprCapture, WprConfig

with WprCapture(WprConfig(signal="processor_trace",
                          code_mode="Kernel")) as capture:
    run_and_finish_target()
    raw_etl = capture.stop()

# The target is no longer running; saving this ETL cannot add disk I/O to it.
with open("capture.etl", "wb") as file:
    file.write(raw_etl.etl_bytes.numpy().tobytes())
```

WPR writes the ETL on `stop()`, so the measured target must be finished or
quiescent before that call. The Python result is a raw ETL byte tensor; ETL
events and processor packets are **not yet decoded** into the Linux sample
columns. PMU profiles collect system-wide user and kernel samples, with PID
context in the ETL for later filtering. WPR's memory buffers are circular and
may overwrite old events; `WprBatch.complete` is always `False` until a real
ETL loss/window validator exists. Do not use this backend where complete
training traces are required.

The profile shape has unit tests and a Windows WPR syntax check in CI. An
actual Windows hardware capture, processor-trace availability, ETL decoding,
and a measured perturbation baseline still need a physical Windows host.

## Evidence and next work

The local `HWTracing` project is a possible future collector for this same
physical host. It aims to configure PT, PEBS, and PMU capture before the OS
starts and record boot execution independently of Linux perf. Its removable
media currently supplies the next workload; the host's USB Ethernet connection
could later support trace transfer and episode control. This has not been
integrated or measured here. The current kernel pretraining subject uses perf's
mapping and symbol sideband for running-kernel evidence.

On 2026-09-24, the `trail-x86` Linux x86-64 host exposed `cpu` and `intel_pt`
PMUs and an Intel `mem-loads` alias. Real bounded user-process captures
returned cycle samples, instruction samples, precise memory-load samples, and
branch-enabled PT AUX bytes. Host-kernel captures on CPU 0 returned cycle
samples, instruction samples, and branch-enabled PT bytes. A CPU-0 syscall
workload also returned 382
kernel memory-load samples, all with exact-IP flags and nonzero addresses;
the brief idle CPU-0 window before it had zero samples. These are backend smoke
checks, not throughput or low-taint
measurements. No AMD, ARM CoreSight, Windows, or guest-kernel PMU path has been
validated yet. PT decoding and Windows ETL tensor columns are required before
those streams are useful as direct model inputs.

A first matched overhead check on `trail-x86` (Intel i7-10510U, Linux
7.0.0-31-generic) ran a C17 volatile arithmetic loop for 100 million iterations,
compiled with `cc -O2`. The target was pinned to CPU 2, the controller to CPU 3.
Twelve runs per mode were randomized; timing covered release of the target's
stdin gate through process exit, excluding tensorization after exit. The baseline
median was 160.679 ms; cycle sampling at a 1,000,000 period was 161.293 ms;
precise memory loads at a 100,000 period were 160.738 ms. The cycle runs
produced 9,192 samples in total and memory loads 144 samples.

A separate matched Intel PT check on the same host used 20 million unpredictable
branches, an 8 MiB AUX ring, and 12 randomized runs per mode. The baseline
median was 56.870 ms versus 57.406 ms with PT; each traced run produced about
6.86 MiB of branch-enabled packets. A 256 KiB AUX ring reported loss even for
one million branches, as required. On the host-kernel path, a 50 ms syscall
window produced 5.59 MiB of PT packets; a 300 ms window exceeded an 8 MiB AUX
ring and raised `HardwareTraceLost`. Sustained PT collection will need a
memory-only draining path. These short-window results do not establish
negligible overhead for sustained PT streaming, kernel workloads, other CPUs,
or timing-sensitive races.

The Linux ring layout and loss semantics follow the
[perf ring buffer documentation](https://docs.kernel.org/userspace-api/perf_ring_buffer.html)
and [perf_event_open manual](https://man7.org/linux/man-pages/man2/perf_event_open.2.html).
The Windows profile uses the [WPR profile schema](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wprcontrolprofiles-schema),
[PMU profile guidance](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/recording-pmu-events),
and [memory logging contract](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/logging-mode).
