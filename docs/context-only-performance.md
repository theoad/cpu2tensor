# Context-only capture cost on trail-x86

Measured on 2026-09-08 with the [capture benchmark](benchmark-capture.md).
Adding address-context observations without general registers or memory worked
in every measured run. This small experiment did not resolve an added wall-time
cost relative to block-only capture; it does not establish that context sampling
is free. There is no tensor preparation, network transfer or learner in this measurement.

## Results

All **12 runs passed**: four profiles, three repetitions each, with the starting
profile rotated between repetitions. No repeats were discarded. All nine
instrumented runs completed both vCPU streams with continuous source sequences.
Every run required QEMU exit zero and the exact guest four-step success marker.
The outer timeout was 120 seconds; none expired.

| Profile | Wall seconds, median [min–max] | Relative to vanilla | Total events/s | Trace MiB/s |
| --- | --- | --- | --- | --- |
| vanilla | 3.674 [3.483–3.816] | 1.00× | — | — |
| blocks-only | 5.362 [5.198–5.498] | 1.46× | 32,502 | 0.253 |
| blocks-context | 5.349 [5.342–5.503] | 1.46× | 29,178 | 0.227 |
| rich-values | 9.718 [9.562–10.702] | 2.64× | 450,157 | 17.418 |

`blocks-only` uses `registers=none,memory=off,values=off,context=auto`.
`blocks-context` changes only `context=on`. It emitted **six context rows per run**,
with no register or memory rows. `rich-values` uses general registers, memory
and transaction values, with automatic context capture. Every instrumented
profile uses mixed batching and direct pipe publication.

Per-run count ranges and median individual-signal rates:

| Profile | Blocks [min–max] | Context rows [min–max] | Blocks/s | Context rows/s |
| --- | --- | --- | --- | --- |
| blocks-only | 163,907–178,697 | 0–0 | 32,502 | 0.000 |
| blocks-context | 154,039–172,406 | 6–6 | 29,177 | 1.122 |
| rich-values | 552,615–827,857 | 7–12 | 73,781 | 1.121 |

The rich profile additionally emitted 1,540,358–2,304,005 register rows and
1,284,013–1,904,199 memory rows per run. All rates are medians of per-run rates.

| Profile | Median trace MiB | QEMU CPU seconds | Drain CPU seconds | QEMU peak RSS MiB | Framing share |
| --- | --- | --- | --- | --- | --- |
| vanilla | — | 3.953 | 0.001 | 187.3 | — |
| blocks-only | 1.369 | 6.027 | 0.011 | 197.6 | 1.95% |
| blocks-context | 1.213 | 5.977 | 0.012 | 195.7 | 1.95% |
| rich-values | 169.261 | 9.454 | 2.714 | 224.0 | 10.00% |

CPU seconds sum user and system time. Framing includes both 32-byte outer frame
headers and eight-byte mixed-run headers. The sink retains bounded input data,
checks framing/progress/completion and discards payloads; it does not validate
architectural row contents. Separate real-capture checks cover those semantics.

## Scope and interpretation

The host was `trail-x86` (`iseeyou`), Intel Core i7-10510U, four cores/eight
hardware threads, Linux `7.0.0-30-generic` x86-64, Python 3.12.3. No competing
QEMU process was present at launch. Initial load averages were 0.21/0.12/0.49.
The benchmark applied no CPU affinity or frequency changes. Ordinary host
services were not isolated.

Every case used the same optional-hook QEMU 11.0.3 binary, Linux
`6.9.0-dirty` kernel and original kernel-example initramfs, two TCG vCPUs,
256 MiB RAM and no guest network device. The fixed benign getpid, memory, pipe
and parallel-memory workload used seed 17 and configured size 64 bytes.

Capture begins at `cpu2tensor_capture_begin` (`0x402670`) and stops at
`poweroff` (`0x4023c0`), verified against the matching `kernel_init` executable.
Wall time includes launch, boot, the captured four-action window, draining and
unobserved shutdown. Event rates divide window events by that full duration;
they are not steady-state callback rates. This bounded window differs from
the older start-only performance matrix, so its totals should not be compared
as if coverage were unchanged.

Kernel scheduling, timers and wait loops react to instrumentation costs.
Block counts therefore vary despite identical target inputs. The overlapping
block-only/context timing ranges and different event volumes cannot establish
an isolated getter cost or a speedup. Smaller total trace bytes for the context
profile reflect fewer observed blocks, not compression from adding a signal.
Context sampling still executes at checkpoints even when unchanged state emits
no new row. These results use the raw-state hook and do not measure the upstream
QEMU public-register fallback.

## Reproduction and artifacts

On `trail-x86`, exact commands, all raw repetitions, source/binary identities
and guest diagnostics are under
`/home/user/.cache/cpu2tensor/context-only/matrix-001/`. The Mac copy is under
`/Users/theoad/.cache/cpu2tensor/context-only/matrix-001/`.

```sh
PYTHONPATH=/home/user/.cache/cpu2tensor/performance-benchmarks/python \
python3 -m cpu2tensor.examples.benchmark_capture \
  /home/user/.cache/cpu2tensor/context-only/manifest.json \
  --output /home/user/.cache/cpu2tensor/context-only/matrix-NEW
```

The source snapshot and plugin build are in the same root’s `source/` and
`build/` directories. Original QEMU and earlier benchmark artifacts were
preserved. The existing benchmark environment was reused without installing
Torch or rebuilding its native decoder; this profile does not change the wire ABI.

Measured SHA-256 identities:

```text
qemu-system-x86_64: 2b74dee41743b0fc390fbfb489d4bfafdf800b07bf54c748c41b955122c76b6e
libcpu2tensor_plugin.so: 8c68053ecfe4085305cb0eed8fc3dc63c06fd9bc5dcf6bd47be2a496af0306c7
vmlinuz-6.9.0-dirty: fb23e59b7ee1715730d726d1aab65775a2637fa1a73810643268ac76c30bc140
initramfs.cpio.gz: 224805fe9d4c5f92a587423ee3cd8a23bd2901abe18cf4569d1f0a3f368c73e6
kernel_init: cab2a45a87f49ac9f7f43b4ce19f8bf08d96e02571f0c6068eb3954681222c84
```
