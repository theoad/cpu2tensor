# Concurrent context-only decode investigation

Issue 7 is not ready to close. A bounded 1/4/16 replay reproduced the shared
interpreter ceiling, and a targeted per-frame GIL-release candidate made threaded
scaling worse on the x86 performance host. That implementation was removed. A
new bounded multi-frame candidate passes local exactness and latency checks. It
still needs the same real x86 matrix before making a scaling claim.

## Measured boundary

The baseline `Pool._read` received one frame and called `_native.decode`. The
native call validated its connection-local `Stream`, allocated Python byte arrays
and dictionaries, converted little-endian rows, and returned one Python object
graph per wire frame. `Pool._batch` then created tensor views and batch objects.
These steps repeated even when a positive `batch_bytes` later combined tensors.

The rejected candidate allocated owned output buffers under the GIL, then released
the GIL once per mixed block/context frame while `Stream::accept` validated source
progress and native code filled the buffers. The input remained pinned and each
Pool retained one owning reader and one independent Stream. Exact rows, source
sequences, retained storage, lossless queues and incomplete errors passed, but
785 release/reacquire cycles per kernel trace created harmful contention.

Header parsing, Python allocation, dictionary/tuple publication,
`torch.frombuffer`, batch construction, optional collation and device transfer
remained serialized. A larger GIL-free region therefore needs to span several
wire frames and publish fewer Python objects; adding more short release regions
is not a supported optimization.

## Bounded multi-frame candidate

For a Hello that enables mixed AddressContext while leaving register and memory
signals disabled, `Pool` now reads ahead only while a complete next frame is
already buffered. It peeks at the full 32-byte header and its validated bounded
payload before consuming it. A partial successor therefore cannot delay a ready
batch. SourceEnd, Complete, errors and all other control frames are barriers.

At most 32 mixed frames, 131,072 wire bytes, enter one native call. Native
validation spans the group with one GIL release. The decoder then publishes one
owned block/context column set per source, with explicit original sequence values.
Different sources are never merged and retain no promised total order. Invalid
frame validation returns every earlier accepted group before the client raises;
disconnects discovered during read-ahead are deferred the same way. Legacy and
rich profiles continue through the one-frame decoder. The public `Pool.read()`
API is unchanged and synchronous.

## Reproduction

`test_concurrent_decode.py` runs 1, 4 and 16 independent public Pools in bounded
threads. Every endpoint carries two source streams. It checks every source
sequence, exact block count and address sum, context count, retained storage and
completion. A second case truncates one of four concurrent endpoints and requires
the other three exact results plus an explicit `Incomplete trace` error.

The native replay needs no Torch installation and accepts an operator-built
decoder plus a recorded wire trace:

```sh
python3 python/tests/replay_native_decode.py \
  --module /path/to/_native.so --capture /path/to/context-only.trace \
  --revision REVISION --workers 1 4 16 --iterations 100 --repetitions 5 \
  --group-frames 32
```

It creates a new Stream per trace replay. Native validation rejects missing or
repeated source positions and incomplete completion. Every measured replay also
compares every returned header and column byte with a canonical decode. The
`--check-last` diagnostic keeps validation on every replay but compares returned
payload bytes only on each worker's final replay.

## Real x86 result

Measured on 2026-09-11 on `trail-x86` / `iseeyou`, an Intel Core i7-10510U with
four cores/eight logical CPUs, Linux `7.0.0-30-generic`, Python 3.12.3 and the
`powersave` governor. No QEMU or cpu2tensor job was present before or after the
matrix. Initial loads were 0.55/0.87/0.73 for the candidate and 2.29/1.28/0.88
for the later baseline run; no affinity or frequency setting was changed.

The current plugin was built from archived revision `3935da2`; later candidate
revisions changed only this benchmark. Its SHA-256 was
`10aa6b741267b95b60b3fc31ecc1045fb29ad220d9e06b5a043c1af00fc1d5db`.
It captured the fixed two-vCPU Linux four-action workload with context on,
registers/memory/values off, mixed/pipe publication, and the existing start/stop
PCs. QEMU exited zero and printed the exact four-step success marker.

The trace is 1,617,448 bytes and 785 frames, SHA-256
`10077444d58da7a6e8cf1114f2845ac8e76c5ce47b2b0a38d85fb1ad759a4de2`.
It contains 198,208 events. Source 0 ends at sequence 101,689 with 101,686
blocks and three contexts; source 1 ends at 96,519 with 96,516 blocks and three
contexts. All positions from zero through each end minus one occur exactly once.

Each matrix cell replays that complete trace 100 times per worker for five
repetitions. Executor creation and spawned-process startup are included. Rates are
the median total rows divided by wall time.

| Isolation | Workers | Baseline rows/s | GIL-release candidate rows/s | Candidate wall seconds |
| --- | ---: | ---: | ---: | ---: |
| threads | 1 | 89.04 million | 128.13 million | 0.1547 |
| threads | 4 | 95.20 million | 61.87 million | 1.2815 |
| threads | 16 | 90.06 million | 49.27 million | 6.4366 |
| processes | 1 | 68.97 million | 90.42 million | 0.2192 |
| processes | 4 | 171.72 million | 185.60 million | 0.4272 |
| processes | 16 | 163.01 million | 167.80 million | 1.8900 |

The baseline decoder SHA-256 was
`6594c7af98f919d04e99987bfb9db392cbc7d3adc5c334ec8f78ae3dd7a0e606`;
the candidate was
`ab0546a0561626c0bbf6587fdd5601c2a36a2f786f69003b0732d48cde8c2aff`.
Raw JSON, environment text, capture output and diagnostics remain under
`/home/user/.cache/cpu2tensor/issue-7-{de41d46,f5fd3d1,3935da2}/`.

Thread worker CPU divided by wall time was about 1.0× for the baseline at every
width. The candidate reached only 1.56× at four and 1.60× at sixteen while taking
substantially longer. Processes reached 3.02× and 6.16× respectively. A separate
`--check-last` run still held threaded throughput near 37–39 million rows/s while
four processes reached 93.8 million, so per-replay equality was not the primary
ceiling. The remaining per-frame Python call and object-publication boundary is
the measured bottleneck.

## Local candidate evidence

The amd64 CI image ran under emulation on the Apple Silicon development Mac. Its
deterministic trace contained 132 wire frames and 32,768 exact rows. Grouped and
legacy decode produced the same per-row SHA-256,
`b05a5cceccae3f40fb2c94f4cc17ca806165975d346b80292f82d2ee85974b1e`.
Each cell used 50 decodes per worker and three repetitions with final-payload
checking. This is implementation evidence, not an x86 performance result.

| Threads | Legacy rows/s | 32-frame group rows/s |
| ---: | ---: | ---: |
| 1 | 100.75 million | 162.10 million |
| 4 | 112.91 million | 112.27 million |
| 16 | 113.60 million | 141.00 million |

The public TCP `Pool` replay used 256 frames per source and three repetitions in
the same emulated image. Its median forward-progress rates were 8.70, 3.19 and
1.83 million exact rows/s at 1, 4 and 16 Pools. This includes loopback transport,
Python batch objects, `torch.frombuffer` and digest consumption, so it identifies
remaining client-side serialization rather than a native-decode scaling result.

Focused tests also run independent 1/4/16 public Pools, preserve every source
position and retained tensor, expose a truncated endpoint while peers complete,
return accepted rows before a later validation error, and hold a successor at 31
header bytes while requiring the ready group immediately. The live x86 trace
matrix must still show better 4/16 threaded throughput without a material
one-worker regression before issue 7 closes.
