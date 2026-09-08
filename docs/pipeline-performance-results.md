# Rich tensor pipeline measurements

Measured on 2026-09-08. Packing rich columns into one upload and then collating
small CPU batches removed a large MPS dispatch cost. This is a measured learner
improvement, not evidence that capture or training is GPU-bound.

## Identical recorded events

The learner was an Apple M2 Mac running macOS 15.7.3, Python 3.10.12 and Torch
2.13.0 with MPS. The input was one actual benign getpid kernel capture from
[the window acceptance run](kernel-window-results.md): 497,189 events from two
vCPUs, comprising 92,615 blocks, 247,641 register rows, 156,931 memory transactions
and two context rows. Its 18,910,736 wire bytes have SHA-256
`172eb95af04a21e2bb79eb5b5221ffed0340c547d14666ff0345e065c6f34634`.

The replay runner feeds this file through a bounded loopback sender and the public
Pool. Every variant consumes exactly those events and validates completion.
The sender runs in the learner process; its CPU cost is included. QEMU does not
run during replay. These are three separate-process repeats per variant, executed
before/after, without warmup removal or discarded outliers.

| MPS drain path | Seconds, all three repeats | Median seconds | Output batches |
| --- | --- | --- | --- |
| Separate upload for each column | 12.364, 11.656, 12.101 | 12.101 | 4,640 |
| One packed upload per batch | 1.660, 1.731, 1.698 | 1.698 | 4,640 |
| Packed upload, 64 KiB CPU collation | 0.363, 0.350, 0.345 | 0.350 | 388 |

Packing alone is about 7.1 times faster here; packing plus collation is about
34.5 times faster than the original upload path. This is drain throughput, with
all rich columns uploaded but no policy or training action. CPU `frombuffer`
views retain their owned columns. Packing and collation add bounded CPU copies;
neither claims zero-copy transfer to an accelerator.

Peak learner RSS was roughly 208–210 MB in the packed/collated repeats. With
collation the largest logical batch was about 71.5 KB and sampled MPS allocation
about 71.7 KB. RSS includes process/library overhead, and sampled MPS allocation
can miss transient peaks. Holding a tensor keeps its whole packed allocation alive.

A single two-endpoint replay check consumed the same capture twice: 994,378 events
in 776 batches in 0.969 seconds. All four `(worker, source)` streams completed.
This is a concurrency check, not repeated remote scaling evidence.

## Live x86 workers to the MPS learner

Three exploratory runs used `trail-x86` (Intel Core i7-10510U, four cores/eight
threads), the same M2 learner, mixed framing, pipe publication and 64 KiB CPU
collation. Workers ran the benign four-action Linux 6.9.0-dirty example with two
MTTCG vCPUs each, 256 MiB guest RAM, seed 17 and 64-byte memory work. The capture
started at `0x402670` and stopped before `0x4023c0` in the matching original guest
binary. The native build and patched QEMU are the ones identified in the
[capture matrix](performance-matrix-results.md) and [window build record](kernel-window-results.md).
Workers were remote from the learner; the two-worker case used two processes on
one x86 host. No competing QEMU workload ran in this reserved slot.

| Mode | Workers | Events | Seconds | Events/s | Learner CPU seconds |
| --- | --- | --- | --- | --- | --- |
| Drain | 1 | 6,712,869 | 23.577 | 284,724 | 4.287 |
| Drain | 2 | 12,292,292 | 28.540 | 430,709 | 13.745 |
| Summary training | 1 | 11,165,101 | 64.519 | 173,052 | 10.739 |

Each guest reported all four actions successful, reached the stop marker, exited
zero, and sealed both vCPU streams. The runner retained final logs and reaped all
workers. Timing covers Pool consumption through trace completion; workers are
started before it. Final guest checks and cleanup are outside this timing;
model/checkpoint validation has a separate timing field.
These are individual runs, not a repeated scaling estimate. Natural scheduling
and backpressure change kernel event counts, so the event-rate ratio is not a
claim that two workers halve target runtime. Trace counts are not matched across
these live runs.

The training run formed 9,258 summaries and made 145 finite updates. Its first
loss was 0.014868 and last loss 0.000327; weights changed and checkpoint reload
matched exactly. The 112-feature summary reconstruction task exercises register,
memory, block and context tensors; it is not held-out task accuracy. Feature
preparation took 47.835 seconds, model updates 0.750 seconds, and requesting
batches 15.648 seconds. This example is dominated by feature operations and
synchronization, not model compute. Peak RSS was 411 MB; sampled MPS allocation
was 3.21 MB. DDP and multiple learner devices are outside this iteration.

An earlier two-worker packed-but-uncollated pilot was manually interrupted after
about six minutes while both guests were still running their parallel action.
It is retained as incomplete, with no invented completion rate. Other early
one-worker pilots had different input size or possible host overlap and are
excluded from the table. Worker/socket timeouts bound inactive operations;
`--max-seconds` now separately checks a soft consumption budget between batches.

## Reproduction and remaining work

Use the [pipeline runner](benchmark-pipeline.md). Run workers with
`--batching mixed --publication pipe`, then consume with
`Pool(endpoints, device="mps", batch_bytes=65536)`. The ring path remains
experimental because [its native matrix regressed](performance-matrix-results.md).
The simple synchronous API, worker/source identity, bounded buffering and
[ownership limits](capture-performance.md) apply equally to supplied local and
remote endpoints.

Raw metrics/logs are under `~/.cache/cpu2tensor/performance/replay/` and
`~/.cache/cpu2tensor/performance/live/` on the Mac. The repeated replay directories
are `before-repeat-{1,2,3}`, `packed-repeat-{1,2,3}` and
`collated-repeat-{1,2,3}`. Live directories are `mps-collated-one`,
`mps-collated-two` and `mps-collated-train`. Each contains exact measurements;
the live directories also retain worker arguments and shutdown logs.

The original upload source is frozen in
`performance/before-upload/python.tar`; a separate installed interpreter avoids
the editable import hook overriding that source. Measured runner sources and
hashes are frozen under
`performance/replay/measured-source/before-failure-fixes-20260908T133748Z`.
Subsequent changes only extend failure reporting and delay success publication
until wrapper validation/cleanup; successful timing boundaries are unchanged.
The final source manifest is [performance-source.sha256](performance-source.sha256).

Repeated sustained scaling across distinct remote hosts, AWS execution and real
CUDA validation remain open pending assigned infrastructure. CPU/MPS validation
cannot substitute for CUDA. Fresh all-vCPU boundary snapshots, shared-memory
column publication and copy/compute overlap are also still unavailable. Next
performance work should measure feature aggregation cost and queue occupancy,
then improve the native collector against matched coverage and named-host runs.
