# Rich capture overhead on trail-x86

Measured on 2026-09-08 with the [capture benchmark](benchmark-capture.md).
This compares signal configurations under the same QEMU build. It is not a
measurement of a faster transport implementation, multiworker scaling, tensor
preparation, accelerator transfer or training.

## Results

All **30 measured runs passed**: ten cases, three repetitions each, a 120-second
per-run timeout, and no failed or discarded repetitions. The separate successful
vanilla pilot took 3.488 seconds and does not enter the table. Cases ran serially
with a rotated starting case on each repetition. No automatic warmup, CPU pinning,
frequency change or outlier removal was used.

On this host and workload, rich capture with values had a median complete-run
cost of **2.83× vanilla QEMU**. The tested individual-register profiles showed no
orders-of-magnitude slowdown; their timing ranges overlap substantially. This is
bounded evidence for these profiles, not a guarantee about every register,
workload or QEMU build.

| Profile | Median seconds [min–max] | Slowdown | Median events/s | Median MiB/s |
| --- | --- | --- | --- | --- |
| vanilla | 3.656 [3.550–3.852] | 1.00× | — | — |
| blocks | 5.657 [5.526–5.777] | 1.55× | 55,040 | 0.43 |
| rip | 7.411 [7.046–7.816] | 2.03× | 220,503 | 10.07 |
| rax | 7.490 [7.009–7.690] | 2.05× | 252,584 | 10.75 |
| cr8 | 7.309 [7.232–7.605] | 2.00× | 206,462 | 9.43 |
| xmm0 | 7.759 [7.088–7.849] | 2.12× | 224,066 | 10.23 |
| mxcsr | 7.452 [7.223–7.646] | 2.04× | 204,346 | 9.34 |
| general | 7.364 [7.148–7.506] | 2.01× | 340,830 | 11.93 |
| rich | 10.040 [9.636–10.335] | 2.75× | 407,641 | 17.65 |
| rich-values | 10.343 [10.215–10.715] | 2.83× | 420,843 | 20.64 |

`blocks` disables register and memory capture. Individual and `general` register
profiles disable memory. `rich` adds memory transactions, physical-prefix mapping
and context references; `rich-values` additionally records transaction values.
Event counts combine blocks, register deltas, memory transactions and context
changes when enabled. Throughput includes boot time, as described below.

Selected median resource measurements:

| Profile | QEMU CPU seconds | Drain CPU seconds | QEMU peak RSS MiB | Trace header share |
| --- | --- | --- | --- | --- |
| vanilla | 4.072 | 0.001 | 191.1 | — |
| blocks | 6.171 | 0.014 | 196.0 | 1.57% |
| rip | 7.621 | 1.931 | 196.8 | 66.64% |
| general | 8.160 | 1.813 | 196.4 | 46.18% |
| rich | 11.026 | 2.547 | 221.2 | 32.82% |
| rich-values | 11.423 | 2.720 | 221.7 | 28.86% |

CPU seconds sum user and system time; they are not wall time. Header share is
`32 × frame_count / trace_bytes`, computed per run and then reduced by median.
The RIP profile's two-thirds header share is a concrete batching problem:
frequent register/block alternation produces many tiny frames. Rich captures
also consume appreciable CPU in the Python frame sink. These observations support
larger mixed batches and a native collector; they do not measure the gains those
changes would deliver.

## Host and workload

The host was `trail-x86` (`iseeyou`), an Intel Core i7-10510U with four physical
cores and eight hardware threads, running Linux `7.0.0-30-generic` x86-64 and
Python 3.12.3. Processes could use CPUs 0–7 without an affinity restriction. The
existing `powersave` CPU-frequency governor remained unchanged. Initial load
was 0.00/0.01/0.00; no competing QEMU workload was present. Host use was reserved
within this development task, without claiming isolation from every host service.

Every case booted the same x86-64 Linux `6.9.0-dirty` kernel and existing benign
kernel-example initramfs with two TCG vCPUs, 256 MiB RAM and no network device.
The guest ran its fixed getpid, memory, pipe and parallel-memory actions with
seed 17 and a 64-byte configured memory workload. The pipe workload retains its
fixed 256-byte size. Successful runs required the exact guest marker
`C2T {"event":"complete","steps":4,"ok":true}`, QEMU exit status zero, and—for
instrumented cases—a complete trace with continuous per-source sequences.

The QEMU binary was the optional `x86-state-v1` developer build, including the
MMIO-classification correction. The no-plugin baseline used **that same binary**;
these results do not compare the patched dependency with upstream QEMU. Exact
system-register capture requires that hook. The installed unmodified upstream
binary was not replaced.

All instrumented cases started recording at `cpu2tensor_capture_begin`, address
`0x402670`, verified against the `kernel_init` binary associated with this
initramfs. QEMU still generates and invokes installed callbacks during boot,
before the plugin's capture window opens. Wall time includes launch, boot,
postboot capture, pipe draining and guest poweroff. Events/s divide postboot
observed events by that entire duration; they are not steady-state callback rates.

The runner used a fresh Release native decoder built from the current source.
Its native contract test and all 15 benchmark-runner fixture tests passed on this
host before measurement. Torch was neither required nor installed for the run.
Trace payloads were discarded after header/progress checks; stdout and stderr
were written to host-local files. Architectural payload correctness has separate
probe and runtime evidence.

## Reproduction and artifacts

Artifacts are under `/home/user/.cache/cpu2tensor/state-benchmarks` on `trail-x86`:

- `manifest.json`: exact ten-case commands, signal descriptions and dependencies.
- `matrix-001/manifest.json` and `matrix-001/results.json`: measured manifest,
  ordered raw repetitions, counters and summaries.
- `matrix-001/run-*/stdout.log` and `stderr.log`: per-run guest output and diagnostics.
- `host.json`: CPU model/topology, memory, affinity and frequency governors.
- `source-hashes.json`: source-file SHA-256 identities of the benchmark snapshot.
- `pilot/`: a separate one-run vanilla pilot, excluded from matrix statistics.

The checkout base was `dc0a9586a70a9259140351cb6e23999e990902f2`, with the current
correctness changes applied. File hashes identify the measured dirty snapshot;
the base commit alone does not reproduce this run. The result JSON also hashes
the runner, native decoder, QEMU, plugin, kernel, initramfs, guest executable and
plugin source. Builds and captures stayed outside the shared source mount.

```sh
PYTHONPATH=/home/user/.cache/cpu2tensor/state-benchmarks/source/python \
python3 -m cpu2tensor.examples.benchmark_capture \
  /home/user/.cache/cpu2tensor/state-benchmarks/manifest.json \
  --output /home/user/.cache/cpu2tensor/state-benchmarks/matrix-NEW
```

Use a new output directory. `PYTHONPATH` selects one ordinary package root for
this development installation; there are no source-level import-path edits.

## Interpreting signal costs

Each individual-register profile automatically includes RIP. Register-enabled
system captures also emit the new address-context signal. Therefore `rip` versus
`blocks` changes more than one byte column; `rax`, `cr8`, `xmm0` and `mxcsr` compare
against the RIP/context profile. Raw hooked registers are sampled in bulk, while
additional public-register readers can have different costs. These comparisons
measure practical profile costs, not isolated getter instruction counts.

The register sampler still reads selected values when no delta is emitted. The
profile comparison therefore helps find unexpectedly expensive selections, but
three repetitions of one boot workload cannot establish a universal upper bound
on a register's cost. Profiling and a longer controlled hot loop are needed to
attribute smaller differences.

The guest's kernel timers, scheduling and wait loops react to execution speed.
Identical workload inputs consequently produced different block/event counts
across runs and signal configurations. Recording preserves that behavior rather
than forcing a common schedule. A larger event rate does not by itself mean a
faster engine: the observed workload and amount of emitted data also changed.

The current producer still publishes many small frames through one pipe, and the
Python sink performs work per frame. Drain CPU usage measures that reader's cost;
it is not time spent creating tensors. Future ring/column-page work needs a
separate comparison holding coverage fixed and reporting the changed event volume
and target behavior. None of these results establish GPU saturation or remote
worker scaling.

## Measured binary identities

SHA-256 values from the retained result JSON:

```text
QEMU       2b74dee41743b0fc390fbfb489d4bfafdf800b07bf54c748c41b955122c76b6e
plugin     0c1f6c456fe76d9cdde2058c62bf0f14a4dee9d0940719b2e794c58842e46bd2
kernel     fb23e59b7ee1715730d726d1aab65775a2637fa1a73810643268ac76c30bc140
initramfs  224805fe9d4c5f92a587423ee3cd8a23bd2901abe18cf4569d1f0a3f368c73e6
kernel_init cab2a45a87f49ac9f7f43b4ce19f8bf08d96e02571f0c6068eb3954681222c84
```

Small copies of `results.json` and `manifest.json` are also retained on the Mac
under `/Users/theoad/.cache/cpu2tensor/state-benchmarks/`. Full diagnostic artifacts
remain on the benchmark host. See [the state-hook evidence](qemu-state-hook.md)
for the dependency change and [the mapping probe](system-memory-probe.md) for
architectural observations; a successful timing run does not replace those tests.
