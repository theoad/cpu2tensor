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
