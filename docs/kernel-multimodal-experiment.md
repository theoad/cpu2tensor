# Kernel-only multimodal experiment

`cpu2tensor.examples.hardware_multimodal_experiment` is the first real
dataset/train/evaluate runner for the masked PT, PEBS, and PMU model. It admits
only the benign families compiled into `hardware_kernel_workload`. Collection is
always `scope=process_kernel`, uses raw undecoded Intel PT, fixes the PEBS period
at 10,000, and requires distinct pinned target and controller CPUs. The
process-scope validation canaries and CVE triggers are not accepted inputs.
The default 32 MiB PT AUX ring is intentionally conservative for the 5,000-loop
families; any reported loss still rejects the execution.

## Build and collect on `trail-x86`

Build the gated workload from the exact source revision that will be recorded in
the manifest:

```bash
cmake -S native -B build/hardware-multimodal -G Ninja \
  -DCPU2TENSOR_BUILD_HARDWARE_EXAMPLE=ON
cmake --build build/hardware-multimodal --target hardware_kernel_workload
python -m cpu2tensor.examples.hardware_multimodal_experiment \
  ~/.cache/cpu2tensor/kernel-multimodal-pilot \
  --binary build/hardware-multimodal/hardware_kernel_workload \
  --target-cpu 2 --controller-cpu 3 --collect-only
```

The default pilot collects 12 executions of each of the 17 families. Six rows
per familiar family train, three calibrate, and three provide familiar-family
validation; three whole families are randomly held out. Increase these counts
explicitly for a larger run. An execution is never divided across partitions.

Each accepted execution produces a hash-addressed raw custody shard and a derived
tensor shard. `capture-manifest.json` records the exact workload output as base64
and SHA-256, workload and Python-module hashes, source revision and dirty-diff
hash, ELF build ID when available, host/kernel/boot/microcode identities, perf
event settings, schedule, split, lane attribution, timing, loss, missing modality,
and censored-token counts. A raw shard retains raw PT bytes, every PEBS tensor,
boundary PMU values and scheduling times, status, and capture envelopes. A final
manifest is published only after all shards are fsynced and hashed. Lost or
multiplexed captures are rejected and may be retried; they never enter training.

Admission does not require a positive PEBS sample. A zero-sample execution is
retained only when the PEBS event was requested, available, scheduled for a
nonzero interval without multiplexing, and the gated target's scheduler affinity
was exactly the requested target CPU before release. Its PEBS tokens remain
explicitly unavailable. When samples are present, every one must retain exact-IP,
a nonzero address, and the requested CPU. The manifest distinguishes sampled and
zero-sample admissions and counts every retry reason; it therefore does not hide
the sampling distribution by retrying until PEBS happens to fire.

Per-execution monotonic phase costs separate launch through `READY`, event open
and arm, workload execution, stop/drain/decode, PT histogram construction,
combined PEBS and PMU feature construction, raw serialization/fsync/hash, and
derived sealing. The collection summary reports total, median, and maximum time
for the same phases, plus wall time not covered by them (including capture
context close and loop/manifest overhead). These timers diagnose the finite
evidence runner; they do not change the perf capture envelope or imply a
high-rate production data path.

### Zero-sample admission timing pilot

The smallest valid three-family matrix ran on `iseeyou` (Intel Core i7-10510U,
x86-64, Linux `7.0.0-31-generic`, boot
`2ba4b210-4c31-48ce-92d5-4b306e16d26c`) from clean revision
`555315f52f67b7a5cef56f9c1075a53a650f7d4d`. It used target CPU 2,
controller CPU 3, PEBS period 10,000, the default 32 MiB AUX allocation, loop
scale 1, and three repetitions each of `getpid`, `openat`, and `memfd`. This is a
pilot-only diagnostic, not a throughput or operational-alert claim.

All 9 executions were admitted on their first attempt with no source loss or
multiplexing: 6 had PEBS samples and 3 `getpid` executions had zero samples.
Those three retained unavailable PEBS tokens and recorded exact target affinity
`[2]`; they were not retried into the positive-sample population. The 216
observed samples were all exact-IP, nonzero-address, and CPU-2 attributed.
Collection took 5.813 s for 21,397,296 raw PT bytes. Custody reload revalidated
all 9 rows. The sealed manifest content hash is
`fb07998ea7c08db1ece0d252c98ed63c45d3d79628810079b63ec4bab0d59081`.

| Monotonic phase | Total | Median/execution | Collection wall |
| --- | ---: | ---: | ---: |
| Launch/READY | 21.554 ms | 1.983 ms | 0.37% |
| Event open/arm | 81.603 ms | 10.401 ms | 1.40% |
| Workload | 114.697 ms | 4.261 ms | 1.97% |
| Stop/drain/decode | 13.759 ms | 0.729 ms | 0.24% |
| PT histogram | 5,156.467 ms | 853.985 ms | 88.70% |
| PEBS and PMU features | 3.184 ms | 0.348 ms | 0.05% |
| Raw serialization/fsync/hash | 207.449 ms | 18.576 ms | 3.57% |
| Derived sealing | 53.659 ms | 6.806 ms | 0.92% |

The separately reported unaccounted wall time was 160.842 ms (2.77%), which
keeps capture-context close and remaining loop/manifest overhead visible rather
than assigning it to a named phase. The artifact and preserved log are on the
measurement host at
`~/.cache/cpu2tensor/kernel-multimodal-admission-timing/pilot-555315f` and
`~/.cache/cpu2tensor/kernel-multimodal-admission-timing/pilot-555315f.log`;
their manifest-file and log SHA-256 values are respectively
`f05f129fefd02df48b0fb4cd4a1c2552e4f6b77f1804ff8c2cc9a9f1b19e6b80` and
`9152e871cb4502a6e63f52cd460502ae963579e7c547733052b40640d4f7b5ed`.

The histogram cliff was an affinity-order bug. Torch selected four intra-op
workers while the controller still had CPUs 0--7 available; the runner then
pinned itself to CPU 3 before the pool was created, so all four workers inherited
one CPU and contended in `bincount`. The diagnostic used Python 3.12.3 and Torch
2.14.0+cpu with its OpenMP backend and no thread environment overrides. Exact
retained and live-buffer microbenchmarks
both reproduced roughly 864 ms for 1 MB and 6 MB inputs. Configuring one intra-op
worker immediately after the one-CPU pin reduced them to 1.6--1.9 ms and
8.0--8.6 ms; warming the pool before pinning was also fast but violates the
controller ownership design.

A fresh run of the same matrix from clean revision
`eadcff44b255c517b47f9b1f08ff1e81cc9f5d35` retained all 9 rows on their first
attempt with the same 6 sampled and 3 verified zero-PEBS outcomes. It took
0.3715 s (24.23 executions/s), including 39.573 ms of featurization and
31.655 ms of PT histogram construction. The latter processed 21,302,272 bytes
at 673 MB/s and used 8.52% of wall time, versus 88.70% before the fix. Workload
windows and raw serialization/fsync/hash now use 30.43% and 28.82% respectively.
All 9 custody rows revalidated. The manifest content, manifest file, and log
SHA-256 values are respectively
`adc8f886e49b0b898422e701380b76642f3aedcbd1d92cbd18f1ac937a4e5984`,
`4be4c62f7fad9049a4ba5464562c46e0e0966f358c0d834553879a475c4dc552`, and
`7c1a4d47de968fcf5d6b70a5a9b557e7782a1dd43c3f5d0a629e66e7c7c32d3f`.
The retained artifact and log are under
`~/.cache/cpu2tensor/kernel-multimodal-pin-fix/` on `trail-x86`.

## Transfer and train on macOS

Copy the entire artifact directory with a hash-preserving tool, then run:

```bash
python -m cpu2tensor.examples.hardware_multimodal_experiment \
  ~/.cache/cpu2tensor/kernel-multimodal-pilot --train-only --device mps
```

Train-only verifies every raw and derived shard hash and the frozen event
contract before loading tensors. It trains two models with the same architecture:
the fused model receives contiguous-span and whole-modality masks, while the
span-only ablation disables whole-modality masks. Normalization and each anomaly
threshold are fitted only from their training and calibration partitions. Both
models are frozen before evaluation. Independently fitted marginal-feature and
PT-only PCA baselines use the same partitions and frozen calibration rule.

`report.json` records losses, collection, featurization, training and scoring throughput, bit-exact
checkpoint reload, familiar and held-out-family alert rates, worst familiar and
held-out families, clean-versus-modality-swap and PEBS-timestamp-misalignment
sensitivity, and loss/missing/censored counts. The current timestamp diagnostic
reverses input bounds without moving PEBS values; it is intentionally retained
as a negative control and is not a physical timing canary. The default threshold and tiny
calibration partition are **pilot-only**. They must not be reported as a
prospective operational false-alert rate. In particular, a 1,000-per-million
claim requires at least 100,000 independent calibration executions and a separate
independent benign test partition.

This runner makes no CVE or vulnerability-detection claim and must not be used to
launch a CVE trigger.
