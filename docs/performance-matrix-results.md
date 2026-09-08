# Capture framing and publication on trail-x86

Measured on 2026-09-08 using the [capture benchmark](benchmark-capture.md).
All **30 runs passed**: three repetitions of ten cases, including no-plugin and
block-only controls, plus all four combinations of legacy/mixed framing and
pipe/ring publication with memory values off and on. No repetitions were discarded.

Mixed framing substantially reduced framing overhead. Its complete-run timing
ranges overlap those of the existing pipe/legacy path, so this matrix establishes
no robust end-to-end speedup. Ring publication with legacy framing regressed:
its median elapsed time was 1.92 times pipe/legacy with values off and 1.55 times
with values on. Guest event counts also changed; these are complete-run outcomes,
not isolated per-event costs.

## Configuration and coverage

The host was `trail-x86` / `iseeyou`, Linux `7.0.0-30-generic` x86-64 with eight
logical CPUs and Python 3.12.3. There was no affinity restriction; the existing
`powersave` governor was unchanged. Initial load averages were 0.52/2.61/1.56.
Cases ran serially with the starting case rotated between repetitions.

Every case used the same optional-hook QEMU 11.0.3 binary, x86-64 Linux
`6.9.0-dirty` kernel and initramfs, two MTTCG vCPUs, 256 MiB RAM, and no network
device. The guest performed the fixed benign getpid, memory, pipe and parallel
workload with seed 17 and configured memory size 64 bytes. Rich cases captured
general-register changes, memory transactions and address context. Memory values
were the only signal difference between each off/on pair.

All instrumented cases used the same start-only capture window at `0x402670`.
There was no stop-PC filter. Timings include launch, boot, postboot capture,
local pipe draining and guest poweroff. Success required the prescribed guest
completion marker, QEMU exit zero and continuous, sealed trace streams from both
sources. The runner discarded trace payloads after framing/progress validation;
architectural payload correctness has separate integration evidence.

## Time and process resources

Wall time shows median [minimum–maximum]. CPU seconds and context-switch counts
are medians. QEMU CPU seconds sum its measured user and system time, including
the collector thread when enabled. Drain CPU seconds measure the local Python
sink process. CPU time is not wall time and these columns must not be added to
derive elapsed time. Context switches are QEMU's voluntary/involuntary counts.

| Publication / framing | Values | Wall seconds [min–max] | QEMU CPU seconds | Drain CPU seconds | Context switches V / I |
| --- | --- | --- | --- | --- | --- |
| No plugin | — | 3.706 [3.514–3.744] | 4.116 | 0.001 | 10,169 / 20 |
| Pipe / legacy, blocks only | — | 5.448 [4.961–5.583] | 6.102 | 0.012 | 13,740 / 26 |
| Pipe / legacy | Off | 10.168 [9.788–11.084] | 10.747 | 3.122 | 28,428 / 47 |
| Pipe / legacy | On | 10.309 [10.046–10.595] | 11.097 | 3.078 | 25,751 / 58 |
| Pipe / mixed | Off | 9.761 [9.390–10.443] | 9.423 | 2.483 | 24,884 / 50 |
| Pipe / mixed | On | 10.039 [9.878–10.533] | 9.658 | 2.796 | 26,220 / 51 |
| Ring / legacy | Off | 19.511 [18.631–20.671] | 16.134 | 7.504 | 727,097 / 32 |
| Ring / legacy | On | 15.951 [15.784–18.227] | 13.774 | 4.808 | 531,364 / 38 |
| Ring / mixed | Off | 10.492 [10.090–10.723] | 10.620 | 2.511 | 244,233 / 52 |
| Ring / mixed | On | 10.249 [9.749–10.605] | 10.201 | 2.671 | 244,851 / 31 |

## Framing and producer waits

The following columns are medians except the event-count ranges. Events include
blocks, register changes, memory transactions and address-context changes.
Framing share is computed for each run as
`(32 × frame_count + 8 × mixed_run_count) / trace_bytes`, then reduced by median.
It includes the eight-byte headers inside mixed frames; counting only outer
frame headers would understate mixed framing overhead.

| Publication / framing | Values | Events, millions [min–max] | Frames | Trace MiB | Framing share |
| --- | --- | --- | --- | --- | --- |
| Pipe / legacy, blocks only | — | 0.284–0.348 | 1,169 | 2.27 | 1.57% |
| Pipe / legacy | Off | 4.458–5.313 | 2,120,389 | 198.32 | 32.83% |
| Pipe / legacy | On | 4.197–4.468 | 1,949,563 | 206.56 | 28.85% |
| Pipe / mixed | Off | 4.013–5.091 | 34,075 | 132.51 | 11.65% |
| Pipe / mixed | On | 4.033–4.978 | 44,114 | 171.31 | 10.04% |
| Ring / legacy | Off | 6.203–7.149 | 3,075,933 | 288.30 | 32.56% |
| Ring / legacy | On | 3.952–5.964 | 1,909,171 | 204.60 | 28.48% |
| Ring / mixed | Off | 2.614–3.949 | 29,154 | 113.34 | 11.68% |
| Ring / mixed | On | 2.626–3.952 | 37,511 | 145.67 | 10.09% |

The plugin's stderr records these `full_waits` counters:

| Ring framing | Values | Full-queue retries, median [min–max] |
| --- | --- | --- |
| Legacy | Off | 393,568 [366,347–418,535] |
| Legacy | On | 244,281 [227,807–350,903] |
| Mixed | Off | 73,119 [49,241–81,466] |
| Mixed | On | 75,965 [53,433–78,852] |

These are failed enqueue attempts, not distinct stalls or measured stall duration.
Repeated polling can count the same full queue multiple times. Neither multiplying
the count by the requested sleep interval nor interpreting context switches as
queue stalls yields measured backpressure time. This matrix has no queue-occupancy
or time-in-backpressure measurement.

The same prescribed guest can execute different amounts of kernel work under
natural multi-vCPU scheduling and changed instrumentation overhead. Continuous
source sequences do not imply identical event counts across runs. Consequently,
trace-byte totals and elapsed times must be read with their observed event counts;
this is not a replay of an identical event stream.

## Source identity and limits

Every instrumented case used the same plugin binary, SHA-256
`c0e8f8e184d713580c8f85c2c1f602797525141bc456373816ad395990c67bcd`.
Its staged source root was
`/home/user/.cache/cpu2tensor/performance/source`; the recorded `plugin.cpp` hash is
`c7654f451c27ce6eb576385a12d4fa150530f3c0d959c559bf114e20ab84c447`.
The results also record hashes for FrameRing, trace core, bindings, native decoder,
benchmark runner, QEMU, kernel, initramfs and guest executable. Pipe/legacy here is
a setting of this new implementation, not an older release binary.

The Mac copies of `results.json`, `manifest.json` and all per-run stdout/stderr
logs are under `~/.cache/cpu2tensor/performance-benchmarks/matrix-001/`.
Manifest SHA-256:
`9e3783cc05899226e5ae1e11c6713855dabde78c3bb3f95e04c2a1712d2ac80c`.
These artifacts preserve all 30 measurements and exact commands.

Collector scan-range tuning is a separate experiment and is not represented here.
This matrix does not compare ASLR handling or different shutdown-tail coverage.
It includes no remote learner transport, tensor materialization, accelerator
transfer, model training, or evidence of GPU-bound execution.


## Follow-up scan-range experiment

A separate six-run paired experiment changed only the collector's scan range to
initialized CPU slots. Rich-values ring/legacy median wall time was 18.965 seconds
for the original collector and 20.294 seconds for the change. Full-queue retries
and context switches did not improve consistently. Natural event counts also
varied. This established no speedup, so the extra state was not retained.
The original matrix binaries remain unchanged. Paired commands, source hashes
and results are in `~/.cache/cpu2tensor/performance-tuned/paired-001` on both hosts.
