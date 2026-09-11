# Concurrent context-only decode

Independent context-only Pools can validate and materialize mixed block/context
frames concurrently in one Python process. The public API remains ordinary
synchronous iteration; callers may use their own bounded threads or pass several
endpoints to one `Pool`.

## Lock and ownership boundary

For a mixed frame containing only block and address-context runs, the native
decoder first inspects bounded run metadata and allocates the returned Python
byte arrays while holding the GIL. It then releases the GIL once to validate the
frame against that connection's `Stream`, convert little-endian rows and fill
the private output columns. The input stays pinned through `Py_buffer`; the
output dictionary strongly owns every byte array. The decoder reacquires the GIL
before raising an error or returning any object.

Each `Pool` still has exactly one reader and one native `Stream`. Separate Pools
never share sequence state. Socket reads, one-ready-batch queues, source-end
flushes, caller-owned tensors, cancellation and incomplete-trace errors are
unchanged. A single Stream is not a concurrent work queue.

Header parsing, output allocation, dictionary assembly, `torch.frombuffer`,
batch construction, optional collation and device transfer still use their
existing Python boundary. Rich mixed frames with register or memory rows use the
general decoder. This change removes the GIL only from the native work identified
in the reported context-only scale cliff; it does not add a public asynchronous
API or claim that the complete Python consumer scales linearly.

## Deterministic replay

`test_concurrent_decode.py` runs 1, 4 and 16 independent public Pools in bounded
threads. Every endpoint carries two source streams. The test checks every source
sequence, exact block count and address sum, context count, retained storage and
completion. A second case truncates one of four concurrent endpoints and requires
the other three exact results plus an explicit `Incomplete trace` error.

The longer replay can be run from a built test environment:

```sh
PYTHONPATH=python/tests python python/tests/replay_concurrent_decode.py \
  --workers 1 4 16 --frames-per-source 256 --repetitions 7
```

On 2026-09-11, `mac.local` (Apple Silicon, eight logical CPUs) ran the pinned
amd64 Linux CI image `sha256:3aeb6e6e5177` under Docker emulation. The unchanged
baseline was `de41d46`. Each worker replayed 131,072 ordered events. Wall time is
median [minimum–maximum] over seven runs.

| Pools | Baseline seconds | Candidate seconds | Candidate aggregate events/s |
| ---: | ---: | ---: | ---: |
| 1 | 0.01859 [0.01833–0.02734] | 0.01858 [0.01755–0.02655] | 7.05 million |
| 4 | 0.20598 [0.20498–0.31927] | 0.21669 [0.21237–0.39122] | 2.42 million |
| 16 | 0.89887 [0.85923–0.94212] | 0.94401 [0.93238–1.03713] | 2.22 million |

The candidate did not improve this replay. Variability and emulation make this a
correctness/forward-progress result, not evidence of a throughput improvement.

The issue's matched x86-64 kernel thread/process experiment remains necessary.
It should reuse the sealed context-only worker, plugin, QEMU, kernel, initramfs,
40 public inputs and deadline, and record accepted traces, exact sequence custody,
wall time, trace bytes, process CPU and pressure. No speedup or resolved scale
limit should be claimed until that check passes on the x86 performance host.
