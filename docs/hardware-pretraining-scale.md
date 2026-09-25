# Twenty-four-hour hardware pretraining procedure

This is the proposed operational procedure for a 24-hour frozen-subject
pretraining campaign. It is not yet execution evidence. The six-hour multimodal
PoC must qualify the capture contract, perturbation, model, calibration, and
checkpoint behavior before this procedure becomes runnable.

## Scientific objective

Train a sufficiently large masked multimodal model on one exact kernel build,
boot, microcode revision, and bare-metal machine. Intel PT, PEBS, and sparse PMU
counters are learned jointly. The campaign tests whether additional data and
capacity improve reusable latent knowledge rather than merely memorizing workload
families or instrumentation artifacts.

The breakthrough criterion is not low reconstruction loss. At one threshold
frozen from benign calibration, the final model must improve over the PCA and
small-transformer baselines on all of:

- unfamiliar benign-family alert rate;
- cross-modal matching and deliberately misaligned-modality detection;
- blinded flow, memory, contention, and phase canaries;
- localization of the responsible modality and time region;
- a matched vulnerable/fixed by trigger/sibling CVE interaction, or a reproducible
  unresolved anomaly with independent semantic evidence.

An unexplained alert is never silently relabeled as a false positive.

## Resource topology

The collector and trainer are separate failure domains. A collector interruption
changes the learned subject because it changes the boot; a trainer interruption
does not, provided its checkpoint and input manifests are durable.

Preferred topology:

1. One on-demand `c5.metal` collector in `us-east-1a`, with the exact qualified
   AMI, kernel, CPU topology, and capture revision. Spot is inappropriate for the
   collector unless a reboot is explicitly admitted as a separate subject.
2. One Spot `p5.4xlarge` trainer in the same Availability Zone. AWS documents one
   NVIDIA H100 with 80 GiB HBM3, 16 vCPUs, 256 GiB host RAM, and 3.84 TB local
   NVMe for this type. The current account has 16 P-Spot vCPUs, exactly one such
   instance, while P on-demand quota is zero.
3. A dedicated encrypted, public-blocked, versioned S3 prefix for immutable
   manifests, sealed shards, checkpoints, and metrics. NVMe is a cache, never the
   sole copy.

As observed on 2026-09-25, `p5.4xlarge` Spot was about $2.60/hour and `c5.metal`
Spot about $1.03/hour in `us-east-1`; Spot prices and capacity are not a launch
guarantee. Query price, quota, capacity, and the on-demand collector rate again
before approval. The local `trail-x86` collector is preferable only if a 1 GiB
transfer probe sustains twice the measured producer rate and the host can remain
powered, thermally stable, and uncontended for 24 hours.

Normal EC2 virtual machines are not collector substitutes unless Intel PT, real
PEBS, non-multiplexed PMUs, target attribution, clock conversion, and CPU pinning
all pass on that exact instance. Even then they define a different virtualized
subject.

## Data and capacity ladder

One finite-capture lane has already produced roughly 250 executions/s and 10 MB/s
of raw PT. A 24-hour single-lane campaign therefore has an evidence-based order of
magnitude of 20 million executions and 0.8--1 TB raw PT before PEBS and metadata.
The launch plan must use measurements from the qualified tri-modal collector,
not these PT-only estimates, to set disk and S3 bounds.

Use the first two hours as a preregistered capacity ladder rather than committing
the full day to an arbitrary model:

| Candidate | Purpose |
| --- | --- |
| Approximately 1M parameters | PoC continuity and pipeline control |
| Approximately 30M parameters | First serious representation baseline |
| Approximately 120M parameters | Large single-H100 candidate |

Each candidate receives the same early shards, mask schedule, number of target
tokens, and allowed validation set. Select the largest model that fits the memory
and throughput envelope and demonstrates a justified validation or scaling-curve
gain. Do not select on the blinded final canaries or CVE labels. Continue the
selected model for the remaining 22 hours with multiple independently sampled
mask views. Architecture, signal schema, and subject identity freeze after hour 2.

## Immutable data path

The collector writes 128--256 MiB `.partial` shards, then fsyncs, hashes, atomically
renames, and publishes a ready manifest last. A manifest contains kernel, BTF,
microcode, boot, CPU topology, workload/input, binary, perf-event encoding,
sample-period, clock conversion, enable order, boundary, loss, output-oracle, and
source-revision identities. The trainer consumes only ready shards and records the
exact ordered manifest set in every checkpoint.

Raw data stays on the collector until a second hash-verified copy exists. Derived
tensors are reproducible artifacts, not replacements for raw custody. Any PEBS
address material is encrypted and access controlled.

## Health and adjustment loop

Checks are recurrent and machine-readable:

| Interval | Checks |
| --- | --- |
| 1 minute | process liveness, disk, RAM, GPU memory, collector/trainer backlog |
| 5 minutes | capture loss, PEBS density, PMU `time_running/time_enabled`, CPU migration, target oracle |
| 15 minutes | finite loss, gradient norm, parameter norm, throughput, GPU utilization, NaN/Inf, checkpoint and S3 hash |
| 30 minutes | allowed validation loss by modality, cross-modal retrieval, modality-swap residual, score distribution drift |
| 2 hours | scaling curve, familiar and unfamiliar allowed-benign alerts, canary-development set, resource/spend forecast |

Allowed automatic adjustments are bounded:

- If the GPU is starved while sealed shards exist, increase loader workers,
  prefetch, or batch size without changing sample selection.
- On OOM, reduce microbatch and increase gradient accumulation; do not shrink the
  model after the capacity decision.
- On NaN/Inf, restore the last clean checkpoint and halve the learning rate once.
  A second occurrence aborts that lineage.
- If fresh data is temporarily unavailable, reuse sealed shards with new masks;
  never admit incomplete capture or relax loss checks.
- Spot interruption restores the latest verified S3 checkpoint on a replacement
  trainer. Collector interruption ends the subject; it does not silently resume
  after reboot.

Any optimization, mask-rate, or loss-weight adjustment creates a new lineage with
the reason and prior checkpoint recorded. Final blinded data is never used for an
adjustment.

## Abort and completion gates

Abort for unexplained PT/data loss, PMU multiplexing, sustained PEBS dropout,
invalid or migrating timing lanes, workload-oracle changes, subject reboot, bad
hashes, unbounded score drift, or two failed numerical recoveries. Resource hard
stops terminate both instances after 26 hours even if orchestration fails.

At completion, freeze and hash the last and best allowed-validation checkpoints,
model and optimizer configuration, source revision, input manifests, health log,
adjustment lineage, and cost record. Only then unblind the final canaries and CVE
conditions. Preserve suspicious raw trajectories before teardown.

## Recommendation gate

Proceed with the 24-hour campaign only if the six-hour PoC demonstrates all three
modalities with acceptable perturbation, explicit timing uncertainty, deterministic
checkpoint reload, useful cross-modal prediction beyond marginal baselines, and a
frozen threshold that meets the familiar-benign alert budget. Otherwise spend the
24 hours fixing the measured blocker rather than scaling an unqualified signal.

AWS instance specifications: [P5 instances](https://aws.amazon.com/ec2/instance-types/p5/).

