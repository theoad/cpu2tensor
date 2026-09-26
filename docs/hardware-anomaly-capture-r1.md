# Hardware anomaly capture R1: what the retained sensors still say

This is an offline observation audit, not a new detector or a Dirty Pipe gate.
The physical `trail-x86` i7-10510U capture and its 102,000-execution manifest are
immutable. No new host capture, PMU reconfiguration, or target instrumentation
was performed for this audit. The new
`cpu2tensor.examples.hardware_invariant_capture_probe` reads only the 525
preregistered raw audit shards and the separately archived 12-pair canary. It
checks each shard against its recorded SHA-256 before inspection.

## Observation bottleneck

The v3 model sees 16 `log1p` PT byte histograms, not PT packets, branch targets,
or adjacency. A within-segment byte permutation leaves its PT input unchanged;
the unit test demonstrates that an exact byte-bigram count changes. The model
does not receive the retained PT AUX record layout, dynamic maps, or exact-boot
decode state. Those remain in custody for later decoding, but the current raw
reducer does not use them.

PEBS samples have `CLOCK_MONOTONIC_RAW` timestamps, exact-IP bits, IPs, virtual
addresses, latency, and data-source fields. V3 folds them into 16 time bins,
mean/std offsets and small sketches, including 32 bins for an IP/address joint.
Sample order, rare precise sites, repeated site-address pairs, and transitions
within each bin are lost. Its IP anchor is each execution's minimum observed IP,
so a single newly sampled low address can remap all later relative-site hashes.
PMU instruction, cycle, and reference-cycle deltas cover the whole enabled
interval; rates and IPC cannot localize a short counter burst. None of these
observations proves a memory write occurred at an unsampled address.

The timing defect is more fundamental than feature width. PEBS timestamps map
to time bins, whereas **every PT byte segment is assigned the same full capture
envelope**. PT byte order is not a clock. PMU has only an enabled interval and
is also assigned an envelope. Thus the present tensor cannot establish that a
particular PT path preceded a particular sampled store, nor can it align a PMU
change to that path. Per-lane CPU identity and perf scheduling times are
preserved; no cross-CPU total order may be inferred. A future packet decoder
must first establish which PT timing packets and AUX boundaries are actually
usable on the exact boot, then propagate interval uncertainty and gaps rather
than interpolating timestamps from byte offsets.

## Fixed generic probe and benign audit

Before reading canary labels, the probe fixed three descriptors: a whole-trace
PT byte count, an exact 65,536-bin byte-bigram count that excludes the 15 v3
segment boundaries, and a 2,048-bin PEBS IP-page-offset/address-page-offset
joint count. The latter is relocation-tolerant but aliases sites on different
pages and discards PEBS time. It is only a collision/stability probe, not a
replacement timing representation. The exact-boot raw IP plus page offset is
used to count distinct observed pairs. The 32-bin comparison is a
whole-execution collision stress test, **not** the exact v3 per-time-bin
collision count. No CVE label chose bins, normalization, or thresholds.

On `mac.local` (Apple Silicon, Python 3.12, Torch 2.14 CPU), the manifest SHA-256
was `0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`.
All 525 retained raw hashes verified: 284 training, 117 calibration, 35
familiar validation, and 89 whole-family holdout. They contain 572,193,760 PT
bytes and 512,201 precise-store samples. Median exact-IP fraction is 1.0;
sample timestamps span 81.6% of the capture envelope. The median shard has
373 distinct exact-boot IP/address-page-offset pairs. When collapsed over an
entire execution, median distinct-pair collisions are 341 for 32 bins and 31
for the relocation-tolerant 2,048-bin sketch. The PT bigram vector has median
3,995 occupied entries.

Cosine similarity is descriptive here, not a trained score. Each same-family,
same-loop comparison uses two retained executions; cross-family comparisons
match loop count. Early/late compares first and last retained repetition within
each family/loop group. There is only one physical session and boot, so
early/late is **not** cross-session replication.

| Median cosine | PT histogram | PT bigrams | PEBS joint |
| --- | ---: | ---: | ---: |
| Same family and loop count, 51 pairs | 0.999853 | 0.999364 | 0.606494 |
| First/last retained in same group, 51 pairs | 0.999663 | 0.999172 | 0.616959 |
| Same family, lowest/highest loop intensity, 17 pairs | 0.997472 | 0.996443 | 0.601668 |
| Different family, same loop count, 48 pairs | 0.905906 | 0.751562 | 0.161769 |

PT adjacency therefore carries stable benign-family information beyond the
histogram, even across the sampled intensity variants. It does **not** yet show
an invariant shared across different families, sessions, or boots. The coarse
PEBS joint is much less repeatable even within a matched group, so increasing
its bin count alone is unlikely to be a safe anomaly score. The audit's 0.5%
content-independent raw selection is suitable for representation diagnosis,
not for fitting and calibrating a new model across all 102,000 executions:
101,475 raw shards were intentionally evicted after the original derivation.

## Frozen post-choice canary readout

Only after the descriptors and benign comparisons above were fixed, the same
probe read the archived `canary-102k-d128` raw files, verifying all 24 hashes
against report SHA-256
`fca22177df0926f1a14a6caf34f3f2bbb865bc431b6acfec3fd050b47fdcfc79`.
It used the report's anonymous pair IDs and computed no learned weights or
alert threshold. Twelve effect/near-miss pairs have median effect-to-neutral
cosine 0.999910 for PT histogram, 0.999625 for PT bigrams, and 0.564823 for
PEBS joint. Median adjacent same-arm similarities are 0.999933, 0.999685, and
0.547583 respectively. In cosine-distance terms, the cross-arm/same-arm ratio
is roughly 1.33 for histogram, 1.19 for bigrams, and 0.96 for PEBS joint. The
candidate order feature does not reveal a compelling defect-specific effect.
Median PT bytes are 436,640 in the effect arm and 438,224 in the near miss;
median store samples are 1,081.5 and 1,089. These small aggregate differences
also cannot establish causal separation. The canary is a known, previously
inspected development case, so this post-choice readout is exploratory and
not independent blinded validation. No Dirty Pipe-sensitive instrumentation
or trigger-specific feature was added.

## Throughput, memory, and capture quality

The 102,000-execution `trail-x86` production-subject collection sealed all rows
on the first attempt in 7,805.9 s: 13.07 executions/s and 15.42 MB/s PT on
that physical i7-10510U under its no-turbo policy. It recorded 120.34 GB PT
and 105,724,981 PEBS samples, with 30 censored samples, zero loss, zero missing
or multiplexed sources, and zero retries. Per-execution median finite-runner
phases included 6.55 ms event open/arm, 4.45 ms stop/drain/decode, 2.89 ms PT
histogram, 6.30 ms PEBS/PMU features, and 15.02 ms raw serialize/fsync/hash.
These phases overlap neither a matched untraced baseline nor a continuous AUX
collector. Consequently **capture perturbation overhead on this exact subject
is unmeasured**. The earlier +4.20--5.30% tri-modal matrix used a different
configuration and cannot be transplanted to period-1,000 precise stores.

The offline probe on `mac.local` reduced the 525 hashed/loaded audit rows at
129.0 rows/s in 4.07 s of descriptor time; hash verification, loading, and
reduction together took 5.19 s, or 101.1 rows/s. Peak process RSS was 1.61 GB,
mostly a diagnostic implementation that materializes the 170 MB JSON manifest
and keeps every dense vector. This is **not** a streaming collector memory
bound, and Mac throughput is not a cross-ISA capture comparison. Per execution,
the current three dense `int64` vectors require 542,720 bytes, or about 543 GB
per million executions before file/container overhead. A proposed sparse
bigram encoding at the observed median occupancy, with 4-byte indices and
4-byte counts plus `float32` histogram/joint vectors, would be about 41 kB per
execution or 41 GB per million; this is a projection, not a measured file size.
The original average raw PT volume alone projects to about 1.20 TB per million
executions before PEBS, PMU, sideband, and archive overhead.

## Next capture/representation gate

Do not increase the 24-hour collection scale. Decode PT on a small offline
audit/LLM-evidence subset first, using exact-session maps/build IDs, to learn
which packet-order and timing evidence exists and how AUX gaps constrain it.
That diagnostic decode is **not** a requirement to decode every production
execution. The production learner should initially train and score on fast
raw-packet or byte-order descriptors, with PEBS precise-site/address
transitions and explicit within-lane timing uncertainty; PMU remains interval
context. A full decoded branch/IP path per execution is a separate hypothesis,
admissible only if measured capture-to-score throughput and storage justify
changing the raw-trace contract. Qualify stability across new randomized
sessions, boots, benign families, and loop intensities before freezing a feature
schema and benign calibration. Only then run a new label-hidden
known-vulnerability comparison. A bounded same-subject untraced-versus-tri-modal
matrix is still needed to quantify period-1,000 store capture perturbation; no
live-host check was authorized or performed in this R1 audit.
