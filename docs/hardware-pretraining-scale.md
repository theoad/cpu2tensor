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
- kernel-only whole-family holdouts and process-scope flow, memory, contention,
  and phase sensor canaries reported as separate gates;
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
3. A dedicated S3 bucket with default encryption, public-access blocking,
   versioning, and Object Lock for immutable manifests, sealed shards,
   checkpoints, and metrics. Those controls are bucket-level; a prefix in an
   unrelated bucket is insufficient. NVMe is a cache, never the sole copy.

As observed through the AWS APIs on 2026-09-25, `p5.4xlarge` Spot was about
$2.60/hour and shared-tenancy Linux `c5.metal` on demand was $4.08/hour in
`us-east-1`. That is approximately $160.24 for 24 hours of compute, or $173.59
at the 26-hour hard stop, before block/object storage and taxes. Prices and Spot
capacity are not a launch guarantee. Query price, quota, capacity, and storage
rates again before approval, and require a separately approved campaign ceiling.
At the observed rates, a 2.5 TiB `gp3` collector volume costs approximately
$7.40 for 26 hours and retaining 2.2 TiB in S3 Standard for one month costs
approximately $51.80. Use a provisional $250 campaign ceiling, excluding taxes
and longer retention, only after explicit approval. Spot placement scored 1/10
in every offered `us-east-1` P5 Availability Zone during the audit, so replacement
capacity cannot be assumed.
The local `trail-x86` collector is preferable only if a 1 GiB
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

Provision a 2.5 TiB collector volume with a 2 TiB raw-data hard cap and a 2.2 TiB
S3 campaign budget. Require sustained verified upload above twice measured
production and at least 50 MiB/s. Use 128--256 MiB shards. Derived tensors may
remain reproducible local artifacts rather than duplicating another full corpus.

The real 204-execution tri-modal runner produced 8.47 MB of PT per execution and
spent 88.40% of its wall time in the current Python histogram path. Persisting
that volume at the desired eventual aggregate rate is impossible: 100,000
executions/s would approach 847 GB/s and 73 PB/day. The high-rate design must
therefore keep long-lived per-core perf sessions, continuously drain and score
bounded raw windows, and retain full raw evidence for every alert plus a
preregistered benign training sample. Every execution still receives a score,
boundary metadata, integrity/loss status, and a compact derived record. Any raw
window overwritten before its retention decision, or intersecting loss, is
censored rather than published as valid. This rolling policy must be qualified
before replacing the all-raw PoC custody rule.

Use the first two hours as a preregistered capacity ladder rather than committing
the full day to an arbitrary model:

| Candidate | Configuration | Actual parameters |
| --- | --- | ---: |
| PoC control | $d=128$, 8 heads, FFN 512, local 2, cross-CPU 2 | 867,335 |
| Serious baseline | $d=512$, 8 heads, FFN 2,048, local 7, cross-CPU 2 | 28,667,655 |
| Large candidate | $d=768$, 12 heads, FFN 3,072, local 12, cross-CPU 5 | 120,937,991 |

Each candidate receives the same early shards, mask schedule, effective batch,
target-token count, optimizer schedule, and allowed validation set. A candidate
is eligible only after five consecutive finite five-minute windows, peak HBM at
or below 60 GiB, and a projection that leaves at least 5.5 of the remaining 22
hours for validation and checkpointing. It must improve the preregistered
allowed-validation composite by at least 5% over the next smaller model with a
session-block-bootstrap 95% interval excluding zero, while no modality metric
regresses more than 2%. Select the largest eligible candidate. If 30M does not
beat 1M, retain 1M only as the control; if 1M does not beat marginal and PT-only
baselines, abort. Do not select on blinded final canaries or CVE labels. Continue
the winner for the remaining time with at most two independently sampled mask
views. Architecture, signal schema, and subject identity freeze after hour 2.

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

`cpu2tensor.examples.hardware_pretraining_health` implements the version-1
record validator and pure bounded decision policy described here. Its decision
function has no cloud, checkpoint, provisioning, or background side effects;
the campaign runner remains responsible for producing records and executing the
returned action. Tested policy code is necessary procedure machinery, not proof
that a campaign is healthy.

Append a versioned JSONL record with `run_id`, subject hash, lineage, UTC time,
cadence, status, reasons, and the cadence-specific scalars below:

| Interval | Checks |
| --- | --- |
| 1 minute | liveness and boot hash; disk/RAM/HBM; sealed/uploaded/backlog bytes; oldest unverified shard; GPU utilization and data wait; spend/end forecast |
| 5 minutes | executions/rejections; PT/AUX and PEBS loss; family-normalized PEBS density; PEBS and PMU running ratios; migration; exact target oracle |
| 15 minutes | step, unique executions/views, replay ratio, per-modality loss, gradient/parameter norm, nonfinite count, tokens/s, checkpoint age/hash/remote verification |
| 30 minutes | allowed-validation modality losses, retrieval, swap residual, per-family alerts, preregistered input drift |
| 2 hours | capacity projection, familiar/unfamiliar allowed-benign alerts, development canaries, storage and spend forecasts |

Allowed automatic adjustments are bounded:

- If the GPU is starved while sealed shards exist, add at most one loader worker
  per 15 minutes up to eight and double prefetch only up to eight. Do not change
  sample order.
- On the first OOM, halve microbatch once and increase accumulation to preserve
  the effective batch and schedule; do not shrink the model after selection.
- On NaN/Inf, restore the last clean checkpoint and halve the learning rate once.
  A second occurrence aborts that lineage.
- If fresh data is temporarily unavailable, reuse training shards with new masks
  only up to the preregistered two-view maximum; never admit incomplete capture
  or alter sample order.
- Spot interruption restores the latest verified S3 checkpoint on a replacement
  trainer within 30 minutes or aborts. Collector interruption ends the subject;
  it does not silently resume after reboot.

Any optimization, mask-rate, or loss-weight adjustment creates a new lineage with
the reason and prior checkpoint recorded. Final blinded data is never used for an
adjustment.

## Abort and completion gates

Abort on any admitted PT/PEBS loss or bad AUX flag, PEBS or PMU running ratio
below one, oracle failure, CPU migration, subject/boot/hash change, corrupt shard
or checkpoint, less than two hours of forecast disk runway, remote checkpoint
age over 30 minutes, a second numerical recovery, spend forecast above the
approved ceiling, or PEBS density below 50% of its qualified family baseline for
two consecutive windows. Input drift uses preregistered bounds; training-score
movement alone is not an abort. Resource hard stops terminate both instances
after 26 hours even if orchestration fails.

Write a local atomic checkpoint every five minutes and a full immutable S3
checkpoint every 15 minutes, immediately before an adjustment, and on Spot
interruption notice. Include model, optimizer, scaler, scheduler, RNG states,
data cursor, ordered manifest hash, subject hash, lineage, and counters. Budget
roughly 180--200 GB for retained versions of a 120M run.

At completion, freeze and hash the last and best allowed-validation checkpoints,
model and optimizer configuration, source revision, input manifests, health log,
adjustment lineage, and cost record. Only then unblind the final canaries and CVE
conditions. Preserve suspicious raw trajectories before teardown.

## Recommendation gate

Proceed only if the PoC demonstrates all three modalities with acceptable
perturbation, explicit timing uncertainty, deterministic checkpoint reload,
useful cross-modal prediction beyond marginal baselines, and a frozen threshold
that meets the familiar-benign alert budget. The September 25 kernel-family run
passes capture integrity, checkpoint determinism, and cross-modal swap learning,
but fails timing sensitivity and unfamiliar-benign generalization; its finite
data path also reaches only 1.069 executions/s. The campaign is therefore
**NO-GO** in its current form. First fix those measured blockers and reproduce
the result on a fresh whole-family split. Then repeat the tri-modal gate for at
least three randomized sessions on the exact `c5.metal` boot: zero PT/AUX or
PEBS loss, nonzero equal PEBS and PMU running/enabled times, exact-IP nonzero
PEBS addresses, stable attribution, and recorded AMI/kernel/microcode/topology/
capture hashes. The current `trail-x86` result does not qualify AWS. Spend the
24 hours fixing a failed gate rather than scaling it.

AWS instance specifications: [P5 instances](https://aws.amazon.com/ec2/instance-types/p5/).
