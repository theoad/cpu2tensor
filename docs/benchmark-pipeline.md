# Measure endpoint-to-tensor cost

Use this after the [capture-to-discard benchmark](benchmark-capture.md). The two
measurements answer different questions: the first isolates QEMU instrumentation
and local draining; this one includes worker transport, native decoding, tensor
construction and device transfer through the real public Pool.

Start finite observation-only workers yourself. Keep guest binaries, inputs,
register selection, memory-value settings and capture window fixed across runs.
Use a new output directory for every run:

```sh
python -m cpu2tensor.examples.benchmark_pipeline \
  --endpoint tcp://WORKER_A:9000 --capture-host worker-a \
  --endpoint tcp://WORKER_B:9000 --capture-host worker-b \
  --device mps --batch-bytes 65536 --mode drain --timeout 120 --output results/drain-01
```

One endpoint measures one worker. Several endpoints can refer to separate worker
processes on one host, or different hosts. A host label records provenance; it
is not a connection or provisioning instruction. Worker identity is the endpoint
index, and every vCPU remains scoped to that worker.

`--batch-bytes` is an opt-in target for bounded CPU collation before upload;
zero preserves worker frames. See [buffering and copy costs](capture-performance.md).

`--mode drain` constructs and uploads every requested tensor column, then releases
the batch. `--mode train --group-size 64` additionally makes small histograms from
blocks, register bytes, memory addresses/sizes/values, context and layout when
present. Register and memory-value padding is masked. It trains a small summary
reconstruction network in bounded groups and verifies changed finite weights and
an exact checkpoint reload. This exercises the full rich path; its loss is not
held-out task accuracy or evidence of useful representation learning.

The measured report includes:

- Rows by signal and `(worker, source)`, complete-worker count and event throughput.
- Time requesting batches, forming features, updating the model and accounting.
- Maximum logical tensor bytes in a batch, process peak RSS and device allocation.
- Device, library versions, endpoint/host labels, model configuration and validation.

Time requesting a batch includes waiting, decoding and device upload; it is not
pure network time. Per-phase device fences make timings explicit but inhibit
overlap. RSS is a process-lifetime high-water mark. CUDA allocation uses its peak
counter; MPS allocation is sampled and can miss transient peaks. Neither measure
is device utilization. Queues, allocator caches and client-retained batches are
not included in logical batch bytes.

`--max-seconds` gives the measurement a separate consumption budget, checked
between batches. A blocked socket operation can still take its configured
`--timeout`; this is not a hard real-time deadline. The worker's own timeout is
also an operation timeout, not a total target-runtime limit. Failed or interrupted
measurements write `failure.json` with partial counters and never a successful
completion report.

Repeat measurements and report variation. Several local workers may oversubscribe
a host; a higher event rate with different naturally scheduled event counts is
not automatically a shorter target run. A remote topology check on unlike host
ISAs cannot establish same-host scaling. AWS and CUDA require actual assigned
hardware; a CPU/MPS result does not validate either.

For learner-only investigation, `python/tests/replay_pipeline.py` replays a saved
wire capture through bounded loopback senders. `--legacy` expands mixed runs into
legacy frames while preserving all captured event bytes and source sequences.
It compares framing/tensor overhead with identical events, without rerunning QEMU.
Sender threads live in the measured Python process, so their CPU cost is included.
This is an isolation experiment, not real remote-worker throughput.

`python/tests/run_kernel_pipeline.py` runs the benign four-action kernel example
using the existing remote test environment variables and explicitly supplied
start/stop symbols. Never reuse symbol addresses with a different guest archive.
