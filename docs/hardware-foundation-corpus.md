# Intel x86-64 foundation corpus

The first autoresearch loop requires 64 GB of diverse raw Intel PT, PEBS, PMU,
perf sideband, and exact workload evidence from one physical Intel x86-64
subject. Byte count is necessary but not sufficient: the release is rejected if
loss, workload imbalance, provenance gaps, session leakage, or collector identity
can explain its evaluation results.

## Collection ladder

1. Qualify `process_user_kernel` capture on the exact host with matched
   user-only and kernel-only controls.
2. Seal an 8 GB pilot spanning every planned software family and at least two
   sessions. Audit coverage, first-attempt admission, repeat noise, phase and
   family balance, input effectiveness, and split leakage.
3. Freeze the accepted plan and collect nested 8, 32, and 64 GB subsets. New
   bytes may expand repetitions and inputs but cannot repair evaluation splits
   after model results are visible.
4. Publish the 64 GB release only after independent manifest and object-hash
   verification from a second machine.

Application holdout is whole-application. For applications deliberately seen in
pretraining, training, calibration, and familiar-validation inputs are separate
content-addressed fixture pools. The second session replays only the
familiar-validation pool and is explicitly marked as a matched nuisance arm.
Shared interpreter scripts and fixed inspected binaries are support artifacts,
not varying inputs; they remain hash-preserved in custody. Long traces are
windowed for training but remain grouped by execution and split.

## Remote storage

The private source of truth is
`s3://alphaflow-hardware-corpus-403339561360-us-east-1`. Public access is
blocked, default SSE-S3 encryption and versioning are enabled, and incomplete
multipart uploads expire after seven days. The x86 host stages at most one
closed shard; a coordinator verifies the uploaded checksum before removing that
staging copy.

```text
objects/raw/sha256/<prefix>/<digest>.tar
releases/<corpus-id>/manifest.json
releases/<corpus-id>/index.jsonl
views/<corpus-id>/<view-schema>/sha256/<prefix>/<digest>.safetensors
views/<corpus-id>/<view-schema>/manifest.json
```

Raw objects are immutable, uncompressed tar shards targeted at 256 MiB. Each
execution member contains its original PT AUX bytes, perf records, PEBS samples,
boundary PMU counters, decode sideband reference, invocation, output, capture
envelope, and subject identity. An index records the tar data offset and length,
so readers may use HTTP/S3 range reads; sequential trainers instead download and
shuffle whole shards. Object names are SHA-256 content addresses, and release
manifests are append-only commit records.

Training views are separately versioned 128--512 MiB `safetensors` shards with
fixed-size windows and columnar metadata. They are cheap to stream, cache, memory
map, and sample with multiple data-loader workers. A view retains execution ID,
raw object/hash/offset, window offset, application/family/split/session, privilege
policy, missingness, and temporal alignment. Changing tokenization creates a new
view; it never rewrites raw custody.

AWS trainers receive a read-only instance role. External trainers such as
RunPod receive bounded temporary read-only credentials or presigned manifests;
no long-lived write key is embedded in an image. A trainer caches shards on
local NVMe and samples in two stages: select the requested family/split stratum,
then shuffle shards and windows within it. This prevents high-byte-rate programs
from defining the training distribution.

AWS training hosts may mount the release prefix with Mountpoint for Amazon S3;
RunPod and other hosts may use an S3-compatible ranged reader or `rclone` with
temporary read-only credentials. Mounting is optional: the raw index gives the
exact byte range and SHA-256 of every tar member, and training views are already
independently sampleable objects.

The independent release gate is executable rather than a prose checklist:

```bash
python -m cpu2tensor.examples.hardware_foundation_audit plan.json \
  session-a.json session-b.json \
  --bucket alphaflow-hardware-corpus-403339561360-us-east-1
```

It revalidates the plan's content splits, complete execution set, subject and
plan identity, per-application PT quota, every S3 checksum/version, and one
indexed byte-range member from each session.

## Current qualification evidence

The exact `80f0506` collector calibration on `iseeyou` admitted 164/164
executions from 41 applications across both user and kernel execution. It
captured 812,863,424 PT bytes and 3,204 PEBS rows, with zero inexact or
zero-address PEBS rows, zero rejected applications, no source loss, and no PMU
multiplexing. The four raw shards total 817,264,640 bytes. Collection took 39.2
seconds under the sealed non-turbo 1.8 GHz CPU-2 policy; the wrapper restored
the original turbo-enabled 4.9 GHz ceiling before publishing the manifest.

An earlier transport smoke published 14 content-addressed shards plus one-time
decode state and custody objects. An independent S3 range read of one indexed
107,023-byte execution matched its member SHA-256 exactly. The calibration that
first exposed the CPU-policy restoration-order bug is retained as failed
evidence but is not a release and cannot enter training.

The first preregistered 8 GB pilot was stopped after its first three remote
shards because the audit found that train and evaluation rows reused identical
fixture content. Those objects have no release record and are inadmissible. The
failed prefix and custody evidence are retained; its final partial shard was
deleted to reclaim bounded host staging space.

The replacement generator creates four deterministic, content-disjoint fixture
pools: training, calibration, familiar validation, and held-out application.
It records every varying-input digest in the plan and refuses content overlap
even when identical bytes are hidden under different paths. Session B may match
only the familiar-validation pool. Counts remain inverse-weighted by calibration
median so each application contributes approximately 100 MiB per session.
Collection backpressures when two 256 MiB shards await verified upload, stops
below 1 GiB free disk or above 85 C, and never exposes the final manifest until
CPU policy restoration.

## Admission and release gates

- exact stock kernel, boot, CPU family/model/stepping, microcode, topology,
  frequency policy, capture code, event encodings, executable and input hashes;
- both user and kernel PT admitted under one explicit privilege policy;
- no PT/perf loss, AUX gaps, PMU multiplexing, missing expected output, or
  undeclared retry;
- PEBS zero-sample windows are explicit rather than retried into a biased set;
- every accepted execution has a complete raw record before sharding;
- family and application byte quotas, execution counts, duration, PT bytes,
  PEBS counts, PMU distributions, and privilege coverage are reported;
- matched repeats and independent sessions quantify acquisition noise;
- before the pilot result was visible, the statistical gate was fixed at at
  least 100 repeat groups and 100 matched-session pairs, median repeat relative
  MAD no greater than 2% for instructions, 10% for PT bytes, and 20% for
  cycles, median matched-session shift no greater than 5% for instructions and
  15% for PT bytes, and application-signal/repeat-noise at least 5 for three of
  the four PT/instruction/cycle/reference-cycle metrics; PEBS must be at least
  95% usable and cover both user and kernel privilege in at least 80% of
  applications;
- whole-application/session split audit has zero collisions by content and
  provenance hashes, except the explicitly matched familiar-validation/session
  nuisance arm;
- every local shard, S3 object, release index, and manifest hash verifies from
  an independent reader before release.

The 8 GB pilot may expose new gates. It cannot weaken these gates merely to make
the 64 GB target complete.
