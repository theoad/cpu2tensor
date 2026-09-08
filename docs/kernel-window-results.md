# One-shot kernel observation window

Checked on 2026-09-08 against the fixed benign AlphaFlow `getpid` workload from
[cpu2tensor issue 1](https://github.com/theoad/cpu2tensor/issues/1).
The supplied input is the ASCII string `39`; this is ordinary syscall execution.

`--start-pc 0x404f40 --stop-pc 0x404f50` captures the requested window and
continues running QEMU to normal exit. Stop is one-shot. A requested marker that
is never reached makes capture incomplete. The stop block is excluded; other
CPUs' already admitted callbacks can finish. This is not a world-state snapshot.
See the [signal contract](instrumentation.md) for admission semantics.

## Actual kernel acceptance

Host: `trail-x86`, Intel Core i7-10510U, Linux x86-64. Learner: Apple M2 Mac,
CPU tensors. Both runs used two TCG vCPUs, 512 MiB RAM, the same optional
`x86-state-v1` QEMU build and the same supplied guest archive. Each ran alone
before the later separate throughput matrix.

| Capture | Block rows | Register rows | Memory rows | Context rows | Tensor batches | Wire bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blocks, legacy/pipe | 13,963 | 0 | 0 | 0 | 55 | 113,592 |
| Rich values, mixed/ring | 92,615 | 247,641 | 156,931 | 2 | 4,640 | 18,910,736 |

Both captures passed native payload validation and continuous per-vCPU sequences,
ended sources 0 and 1, and completed with worker exit zero. The guest reported:

```text
AGENT: accepted bytes=2 fnv1a64=07ff8707b4c014a1
AGENT: candidate-exit status=0
AGENT: verifier-boundary
```

The start marker appeared once. The stop marker, known later abort marker and
reboot function PCs were absent from every captured block/register/memory PC
column. This is a controlled-symbol check, not a proof covering every possible
shutdown path. Rich capture included full selected-register bytes, memory values
and paging context. All 26 retained rich tensor columns remained equal to their
saved copies after completion, and a model backward pass produced finite gradients.

Elapsed client consumption was 34.14 seconds for blocks and 54.81 seconds for
rich capture, including boot and the continuing guest shutdown. Those are single
acceptance runs with different signal coverage and different naturally scheduled
event counts. They are **not a throughput comparison**. In particular, excluding
shutdown-tail observations must not be described as making each captured event
faster. The rich ring reported 45,096 full-enqueue retries, not measured stall time.

## Identity and reproduction

Artifacts are under `~/.cache/cpu2tensor/performance/acceptance/{block,rich}` on the
Mac. Each contains `command.json`, `capture.bin`, `result.json`, guest serial output
and worker diagnostics. Remote serial logs use the same cache prefix on `trail-x86`.
The guest files live in `~/.cache/cpu2tensor/alphaflow-fixed-subject-r1` on that host.

| Artifact | SHA-256 |
| --- | --- |
| Linux 5.15.25 bzImage | `1da70cd0aa9b44c8b62c5345989382b461cd96464eb4f54d09970337fd3da5d0` |
| evaluation.cpio.gz | `0389157b1137c6c06aba14099573e351e4b473f27f218457d8ec2a732cc3b2a9` |
| Embedded /init | `dd7841769e699c259841a573aa50447b0e71cb50a030251072e67e9a39123f00` |
| QEMU executable | `2b74dee41743b0fc390fbfb489d4bfafdf800b07bf54c748c41b955122c76b6e` |
| Plugin used in these captures | `c0e8f8e184d713580c8f85c2c1f602797525141bc456373816ad395990c67bcd` |

Use `python/tests/capture_kernel_window.py --help` with those operator-provided
paths and a new output directory. The script saves the exact launch command and
checks the fixed guest's markers. These addresses are specific to the embedded
`/init`; a separate rc2 executable in the artifact directory has different symbols
and must not supply the capture markers for this archive.

The issue's original start-only invocation was not fully retained, so no exact
before/after ratio is claimed against its reported 19-million-block shutdown tail.
The [performance matrix](performance-matrix-results.md) uses unchanged capture
coverage and an independently identified guest instead.

## Additional boundary checks

The small AArch64 `window_target` checks stop-before-start, later starts after
stop, missing start/stop, rich mixed decoding, retained MPS storage and explicit
failure. Eight real worker checks passed after integration. Ring slow-consumer
and disconnect/reaping checks plus a mixed/ring two-read stdin check also passed.
These tests use exact fixture symbols and output checksums. They do not add
paused-world register snapshots, hotplug or migration support.


## Client follow-up: build identity and deadlines

The older `state-correctness/build` directory intentionally preserves the previous
runtime and rejects newer worker options. The compatible directory is
`/home/user/.cache/cpu2tensor/performance/build` on `trail-x86`, with source under
`performance/source`. Worker SHA-256 is
`32e4c28bf1752c8404e4fb33a8ba5b5be18673b715e116c13279d45808aa763c`;
the plugin identity is in the table above.

A subsequent client run reached stop but reported a candidate timeout under its
2-second guest deadline. That is not successful getpid execution. Trace completion
and target success are separate checks; the acceptance script now requires the
candidate-exit and verifier markers explicitly. Ring retries, client backpressure
and competing host load can change guest timing. Raising the worker socket/pipe
operation timeout does not change the guest's deadline or create a total-run cap.
See [the issue follow-up](https://github.com/theoad/cpu2tensor/issues/1#issuecomment-5585778145)
for the exact compatible source hashes and retained failure context. No timing
independence or universal deadline guarantee is claimed.
