# Raw hardware trace triage

The production direction is hardware observation rather than QEMU instrumentation.
The fast path observes one exact kernel build, boot, microcode revision and machine
with Intel PT. It learns benign execution from benchmark workloads once, freezes
the model, and scores later fuzz executions without updating it. Decoding and deep
analysis happen only after a rare execution crosses a calibrated review threshold.

The first model consumes raw PT bytes. It does not parse packets, symbolize code,
or reconstruct branches. A tensor-only reducer computes coarse segment byte
frequencies, hashed adjacent-byte frequencies and trace length. A standardized
low-rank linear autoencoder (PCA) learns a latent representation once. Inference
computes residual energy without running a decoder or any iterative training. This
is an intentionally cheap baseline, not evidence that byte sketches are sufficient
for vulnerability detection.

## Scaling contract

Opening, mapping, stopping and destroying a perf event for every test cannot meet
native-fuzzer rates. The intended sustained path is one long-lived kernel-only PT
event per isolated fuzzing CPU, a continuously drained AUX ring, and monotonically
recorded byte offsets at execution boundaries. The scorer consumes batches of
completed byte ranges. It must report trace loss and must not silently truncate or
invent a cross-CPU order.

PEBS precise loads and additional PMU events are initially replay signals for the
small set of PT anomalies. This keeps counter interrupts and multiplexing out of the
highest-rate path. A later experiment may admit a sparse always-on counter only if
matched measurements show useful discrimination and acceptable perturbation.

At 1,000 executions per second on 100 cores, the system must triage 100,000
executions per second. A review rate of one per 100,000 executions is about one
deep-analysis request per second, not a license to assume a particular dollar cost:
the actual trace length, model, prompt and provider determine cost. The threshold
must be derived from benign held-out false-review measurements and an explicit
daily review budget.

## First gates

1. Collect loss-free kernel-only PT from several benign benchmark families on one
   named Intel host and one unchanged boot.
2. Train the raw-byte baseline on benchmark traces and freeze it before scoring
   held-out benign families and deliberately hidden known-vulnerability executions.
3. Measure bytes per execution, sketch and model throughput, peak memory, anomaly
   score stability, false reviews per million executions, and recall at a fixed
   review budget.
4. Implement continuous AUX draining and execution offsets only after the finite
   experiment establishes a useful signal. Measure the complete capture-to-score
   path; offline model throughput alone is not a fuzzer-throughput claim.

The kernel build, boot identity, CPU model, microcode, perf configuration, benchmark
manifest and model checkpoint are part of the learned subject. Rebooting, changing
the kernel or changing hardware invalidates the calibration unless a separate
experiment proves otherwise.

The first measured gate is recorded in [R1 results](raw-hardware-triage-r1-results.md).
It validates the mechanics and rejects the first anomaly calibration.
