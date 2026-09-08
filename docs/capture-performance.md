# Capture performance

Keep coverage fixed when comparing speed. The earlier
[capture baseline](benchmark-capture-results.md) measured signal cost under the
same QEMU build; the following opt-in paths let us isolate batching from callback
publication. Defaults remain `--batching legacy --publication pipe`.

```sh
cpu2tensor-worker ... --batching mixed --publication pipe -- TARGET
```

These flags apply to user and system workers. No QEMU build or dependency
installation is part of package setup. Use [an absolute worker deadline](worker-deadlines.md)
when a finite observation run must stop even while it is producing data.

## Mixed frames

Hello bit 18 advertises Mixed=14 frames. The existing version 2 header keeps the
vCPU source, first event sequence, total event count (at most 256) and payload
length (at most 4064 bytes). A frame contains flat consecutive runs:

| Field | Wire type |
| --- | --- |
| Event kind: Blocks=2, Registers=7, Memory=8, AddressContext=12 | uint16 |
| Row count | uint16 |
| Run payload length | uint32 |
| Existing records for this kind | bytes |

All integers are little endian. Runs cannot nest. Schema, layout and control
records remain separate frames. The decoder validates every run and its original
source sequence before constructing columns. Count both 32-byte outer headers
and 8-byte run headers when measuring framing overhead.

One Python Batch can contain several tables. Use `batch.block_sequences` for
blocks and `table.sequences` for register, memory and context rows. Their union
is the original consecutive event range; a single column table can be sparse.
Memory context references still name the original context event on the same
worker and vCPU. Legacy frames retain their existing dense sequence convention.

The decoder gathers records into bounded native scratch space and then creates
owned columns. CPU tensors use `frombuffer` over those columns. Record gathering,
wire-to-column conversion and device copies still exist. Rich accelerator batches
use one aligned CPU staging buffer and one upload, then expose typed views into
that owned device allocation. This adds a CPU staging copy to reduce small device
transfers. CPU columns remain `frombuffer` views; a lone block column uploads
directly. Retaining any device column keeps its entire batch allocation alive. Mixed frames reduce
small dispatches; they do not provide end-to-end zero-copy capture.

## Per-vCPU publication rings

The [x86 matrix](performance-matrix-results.md) found that this first ring
implementation regresses legacy framing and does not reliably improve mixed
framing. Keep it experimental; mixed/pipe is the measured starting point.

`--publication ring` gives each source an eight-slot SPSC ring. Each slot holds
one frame of at most 4096 bytes. A normal enqueue copies a frame and publishes
its cursor using release/acquire atomics, without a pipe syscall or shared event
lock. Full rings wait without dropping data. Waiting changes guest timing, as
does instrumentation itself; no physical noninterference claim is possible.

A plugin collector thread visits sources in rotating order and sends at most one
frame from each source per pass through the existing worker pipe. It releases a
slot only after the write finishes. SourceEnd and action/drain acknowledgements
wait for earlier source frames to publish. Worker cancellation terminates QEMU;
a broken collector pipe cannot turn lost data into successful completion.

Rings occupy 34,048 bytes per supported source, including padded cursors/slots.
The static 256-source array reserves about 8.3 MiB for rings even when only a few
CPUs are active; physical pages are touched on use. This first collector remains
inside QEMU. Shared-memory worker collection, larger owned column pages, and
copy/compute overlap are still later optimizations.

Exit diagnostics report frame count and `full_waits`. The latter counts failed
enqueue attempts followed by a requested 50 microsecond sleep. It is neither
measured stall duration nor the number of distinct stall episodes.

## Multiple endpoints

```python
from cpu2tensor import Pool

with Pool(endpoints, device="mps", batch_bytes=65536, timeout=120) as pool:
    for batch in pool.read():
        cpu = (batch.worker, batch.source)
        consume(cpu, batch)
```

Endpoints are supplied and started by the operator. `worker` is the endpoint's
zero-based index in this Pool; it stays stable for the run. Do not combine state
using the vCPU number alone. Initial executable-layout metadata also carries its
worker index and has no vCPU source.

Each internal reader keeps at most one ready CPU batch and one in-progress CPU
batch. With a 4096-byte wire frame bound, both row width and queue depth are
bounded; decoded columns can be larger than the wire records. Client-retained
batches are additional client-owned memory. A reader's failure fails the Pool
with its endpoint/index and underlying cause. Device transfers happen on the
consumer thread, including MPS synchronization. There is no device state registry,
all-worker barrier or public async API.

Closing cancels sockets and wakes blocked readers. OS hostname resolution itself
cannot be interrupted; numeric addresses avoid that DNS cancellation limitation.
Workers may be local or remote. This is one learner device per Pool; DDP and
multiple learner devices remain deferred.

## Bounded tensor collation

`batch_bytes=65536` groups decoded CPU frames before the device upload. The
default is zero, which preserves worker frame boundaries. Each worker groups
rows by vCPU; it never merges different sources. Original per-table sequences
survive collation. Layout metadata is forwarded immediately; SourceEnd and
successful completion flush pending tails. This is for observation-only Pool
consumption, not an additional action boundary.

The target accepts up to 4 MiB. Pending input columns per worker are bounded by
twice the target plus one incoming frame, with independent limits of 256 frames
per source and 512 pending frames per worker. Concatenation, generated sequences,
register-row padding, output buffers and Python object overhead are additional.
The endpoint reader's ready and in-progress batches are additional too. These
limits bound buffering without claiming that `batch_bytes` is a process-memory
cap. A quiet source may wait for its next event, another source's budget pressure,
or completion; choose zero when frame delivery latency matters more than upload
overhead. Incomplete streams raise an error and may leave buffered rows unyielded.

Collation adds CPU copies. On the measured rich trace those copies substantially
reduce MPS dispatch overhead; see [pipeline results](pipeline-performance-results.md).

## Evidence

Native ring tests include a million variable-size frames and a ThreadSanitizer
run. Real ARM checks cover mixed/ring rich capture, MPS tensor retention, two stdin
boundaries, full-ring backpressure, disconnect/reaping and missing stop markers.
The x86 kernel window checks have [separate acceptance evidence](kernel-window-results.md).
These checks establish tested behavior, not a throughput gain.

The same-coverage x86 matrix compares all four combinations of publication and
batching with memory values off/on, plus vanilla and block-only controls. Actual
AWS execution and CUDA validation require operator-provided endpoints/devices;
neither is inferred from a successful local test or a skipped CUDA test.
