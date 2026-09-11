# Bounded observation reduction

Long kernel runs can execute billions of basic blocks. A raw block stream is the
right input when occurrence order matters, but writing every event to a capture
file is impractical when a model needs only transition counts and address-space
changes. The observation reducer performs that projection inside the QEMU plugin.
It does not create `capture.bin` or another package-owned trace file.

Enable the reducer on an observation-only x86 system worker:

```text
--system on
--registers none
--memory off
--context on
--blocks off
--reducer block-transitions
--transition-capacity 4096
--batching legacy
```

The same options work with a direct system run or `--kernel-protocol on`. They do
not enable the interactive kernel adapter. `--start-pc` and `--stop-pc` can bound
the execution interval. QEMU, the guest and any remote connection remain operator
supplied.

The public client stays synchronous:

```python
from cpu2tensor import Pool

with Pool(["tcp://worker:9000"]) as pool:
    for batch in pool.read():
        if batch.observation_transitions is not None:
            counts = batch.observation_transitions.counts
        if batch.observation_context is not None:
            positions = batch.observation_context.block_positions
        if batch.observation_summary is not None:
            print(batch.worker, batch.source, batch.observation_summary.blocks)
```

Multiple endpoints use the same `Pool`; `(batch.worker, batch.source)` identifies
one vCPU. There is no all-worker barrier and no action, trajectory, or async API.

## Information retained

Each vCPU processes blocks in its natural callback order. Its first processed
block has position zero and no incoming transition. Every later block increments
the exact `transitions` total and updates the count for its adjacent
`(previous, current)` pair. The fixed hash table retains at most
`transition_capacity` distinct pairs per vCPU. `transition_overflow` counts every
executed transition whose pair could not fit. Counts for retained pairs remain
exact.

Transition rows are aggregates. Their row order is table order and they consume
no raw source-event sequence. They do not preserve the order in which individual
transitions occurred and must not be treated as a contiguous trace.

Paging and execution context is checked at every processed block. The reducer
retains the first `transition_capacity` changes per vCPU. Every retained row carries
the exact zero-based block position where the change was observed. Later changes
still update the plugin's current context and increment `context_overflow`, but do
not produce rows. `context_changes`, `retained_contexts`, and `context_overflow`
therefore account for the complete run. Context fields and `known` bits have the
same CR3, mode, and availability meaning as ordinary `AddressContext` rows.

`ObservationSummary` is per vCPU. It reports the configured source count, both
capacities, exact block/transition totals, retained distinct transitions, and both
overflow counts. `summary.exact` is true only when neither projection overflowed.
Exact means exact for this declared projection; raw occurrence order was still
discarded.

## Completion and bounds

`Pool` withholds summaries until it validates the worker's successful Complete
frame. A disconnect, deadline, capture failure, killed target, or nonzero target
exit raises the existing terminal exception and publishes no summary, even if the
plugin had already emitted its final aggregate. As with every Pool run, clients
must exhaust the iterator before treating retained rows as a completed trace.

For capacity $C$, each vCPU emits at most $C$ context rows and $C$ transition rows.
Their payload bound is $96C$ bytes, plus 32-byte frame headers, one 72-byte summary
payload, and source completion. Context frames hold at most 56 rows and transition
frames at most 169 rows, so the complete wire bound is directly calculable from
$C$ and the fixed vCPU count. The plugin allocates its transition tables once;
neither resident storage nor package-owned disk grows with executed block count.
Socket and pipe backpressure remain lossless for the retained projection.

The reducer deliberately rejects raw blocks, raw context, registers, memory,
mixed batching, interactive control, and variable hotplug source counts. Those
signals would violate this whole-observation bound. Use ordinary capture when a
model needs them.

QEMU can invoke a registered plugin atexit callback after C++ destructors for the
plugin shared object. The reducer therefore uses explicitly constructed
process-lifetime storage: normal execution callbacks stop first, the QEMU atexit
callback reads and publishes the fixed tables, and the operating system reclaims
them when QEMU exits. A compile-time check keeps every other source object read by
that callback trivially destructible. Ordinary kernel control also retains the
previous always-live idle reducer behavior when reduction is disabled.

## Local evidence

The native stress fixture processes one million blocks on each of two sources.
One source has four repeating transitions; its reduced counts equal the raw
prefix exactly. The other presents a new pair at every step; eight rows remain
resident and every later transition is reflected in overflow. Protocol fixtures
also compare a 100,000-block raw sequence with public tensor counts, cap ten
context changes at four retained rows, reject raw/malformed frames, withhold a
summary on truncation, and combine independent workers through `Pool`.

This is deterministic functional and structural-bound evidence.

For a local structural check on the macOS Apple Silicon coordinator, the native
fixture processed those two million blocks with a 1,064,960-byte reported peak
resident set. It created no files in an initially empty temporary directory.
The measurement built only `transition_window_test`, ran it with that empty
directory as its working directory, redirected `/usr/bin/time -l` diagnostics to
a file outside that directory, then checked the directory with `find` and `du`:

```sh
root="$(mktemp -d /tmp/cpu2tensor-bound.XXXXXX)"
mkdir "$root/empty"
cmake -S native -B "$root/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$root/build" --target transition_window_test --parallel 2
(cd "$root/empty" && /usr/bin/time -l "$root/build/transition_window_test") \
  2>"$root/time.txt"
find "$root/empty" -type f -print
du -sk "$root/empty"
```

`find` printed no files and `du` reported zero KiB. This is not an x86 throughput
result and is not compared with another ISA.

## x86 kernel acceptance

Candidate `74b5c581c19c2b499c0e0d150e9f4af51e38f0b8` passed the real-kernel
acceptance on the named `iseeyou` x86-64 host. The guest was x86-64 Linux
6.9.0-dirty under `tcg,thread=multi`, with two vCPUs, 256 MiB RAM, the patched
operator QEMU 11.0.3 build, a 4,096-row capacity per vCPU, and the maximum benign
guest workload of 65,536 bytes. The coordinator measured from remote worker
launch through guest shutdown and complete client consumption:

| Result | Measurement |
| --- | ---: |
| Wall time | 8.176 s |
| Processed blocks | 1,117,787 |
| Processed transitions | 1,117,785 |
| Tensor column bytes | 196,896 |
| Worker/client wire bytes | 198,928 |
| Sampled peak worker plus QEMU process-tree RSS | 203,264 KiB |
| Package-created capture files | 0 files, 0 bytes |

RSS was sampled every 50 ms and summed across the measurement wrapper, worker,
and QEMU descendants. Wire bytes were calculated from every validated frame's
payload and its 32-byte header, including Hello, summaries, source ends, and
Complete. The isolated remote working directory remained empty; its ext4
directory metadata occupied 4 KiB, while contained file bytes were zero.

vCPU 0 processed 848,327 blocks and vCPU 1 processed 269,460. Each retained the
configured 4,096 distinct transition rows; their explicit occurrence overflow
counts were 218,288 and 21,295. They retained one and three positioned context
changes respectively, with no context overflow. The exact remote test passed in
8.77 seconds. A separate ordinary KernelEnv reset/reaping test passed in 19.20
seconds with no reducer, covering the shared process-lifetime object's idle path.

An unrelated unleased process was found on the host before acceptance. All
timing and RSS collected before the clean 2026-09-11T12:25Z audit were discarded.
The recorded run followed clean preflights, and the final audit found no worker,
QEMU, or debugger process before the host reservation was released.
