# Stable PEBS coordinates

The v4 multimodal feature schema uses the captured kernel's runtime `_text`
address as the origin for core instruction pointers. An additional low sample
cannot change the coordinates of sites already observed. Parse the retained
sideband once per kernel state, outside the per-execution feature loop:

```python
from cpu2tensor.examples.hardware_multimodal_features import (
    KernelTextAnchor,
    featurize_hardware_capture,
)

anchor = KernelTextAnchor.from_sideband(captured.decode_sideband)
features = featurize_hardware_capture(captured.batches, kernel_anchor=anchor)
```

`_text <= ip < _etext` defines core text. Signed tensor values are interpreted
as unsigned 64-bit addresses before testing these bounds. Restricted kallsyms
with zero addresses, missing bounds, or duplicate core symbols fail explicitly.
The kernel-state hash identifies retained boot/module/symbol state; it is custody
metadata and does not enter the learned tensor. Stable offsets remove a uniform
kernel relocation, but do not align different kernel builds.

Module, JIT, and other non-core IPs contribute to `noncore_ip_fraction`. They do
not receive invented core coordinates or contribute to core site/joint hashes.
Their sample counts, latency, data source, and data-address summaries remain.
Module-relative coordinates require a future mapping/build-identity contract.

Data addresses describe 4 KiB page offsets, 64-byte cache-line occupancy, and
unique-page occupancy within each timestamp bucket. They have no sampled minimum
origin and omit distances between virtual mappings. These constants describe
this Intel PEBS representation, not a portable claim about every architecture's
page or cache-line size. Independent page relocation preserves these features
when within-page offsets and page/cache-line equality are preserved. Changing
allocator layout, access frequency, or which samples arrive can still change
them; hardware sampling itself remains nondeterministic.

There are 140 PEBS fields per token in
`cpu2tensor-hardware-multimodal-features-v4`. Core hashes normalize by all samples,
so their sum reports the core fraction. Core offset moments normalize by core
samples only; a zero core fraction disambiguates empty core moments. This is a
hashed summary, not lossless site identity. PT and PMU representations are
unchanged.

Derived artifacts use `cpu2tensor-kernel-multimodal-derived-v2`. Each row seals
its anchor and decode-state reference; its manifest entry and retained raw
sideband must agree. The loader verifies state-file hashes and reconstructs
bounds from retained symbols. Evicted raw rows keep their execution-specific
reference. Retained raw tensor storage is memory mapped for this metadata check.

Legacy v1/v2/v3 feature artifacts remain readable under derived-v1. A dataset
must have one schema and its declared width; mixed rows fail. Old v3 tensors
cannot be relabeled v4: conversion requires raw samples and the matching retained
sideband. In particular, a mostly raw-evicted corpus is not upgradeable in place.
Frozen model metadata retains the schema on which the model actually trained,
and live v4 collection rejects legacy checkpoints.

## Evidence

Generic fixture checks cover low-sample insertion, uniform core relocation,
independent data-page relocation, non-core tagging, signed kernel IPs, legacy
readability, mixed-schema rejection, changed bounds, and swapped state references.
Physical two-session stability and featurization cost are separate checks; fixture
invariance alone establishes neither sampling repeatability nor learning quality.

### Physical same-boot repeat audit

On 2026-09-27, physical `trail-x86` (`iseeyou`, i7-10510U, Linux
`5.13.0-30-generic`) ran two separately captured sessions from one immutable
51-row plan. Both used boot `31179aea-9a43-4c5d-8bf6-1205735e42c4`, target CPU 2,
controller CPU 3, turbo disabled, a 1.8 GHz CPU-2 maximum, PEBS loads at period
10,000, 1,024 data pages, and 8,192 PT AUX pages. The freshly rebuilt seeded
workload had SHA-256
`75a291eacb057854822d1e115b4775c5a8da1f720e07d6522421e5def7b1fc8a`;
its source hash was
`1b17785c5a7cbbde0867d0cf94358e4a398b02b833edda4eb6777b8f71bec271`.

Both sessions completed 51/51 executions on their first attempt with zero
capture rejection, loss, or multiplexing. Every planned input seed was recorded,
all raw executions were retained, and production `load_dataset` verified every
raw, derived, manifest, and decode-state hash. Session A collected 51 rows in
2.831 seconds and session B in 2.830 seconds. Featurization alone processed
105.90 and 105.65 executions/s. These are small all-raw-retained smoke rates,
not sustained collection throughput or a v3/v4 performance comparison.

The audit fixed no parameters on session B and fit no classifier. It compared
identical execution IDs, families, loop counts, invocation seeds, and stdin
hashes. Both sessions reconstructed the same runtime core-text anchor:
`_text=0xffffffffb1c00000`, `_etext=0xffffffffb2c02507`, kernel-state SHA-256
`92e07df0248353ec0d82d479f819db962adac0ecc4777da09a3f9837c16812aa`.

| Repeat metric across 51 matched rows | Median | p10 | Minimum |
| --- | ---: | ---: | ---: |
| PT tensor cosine | 0.99164 | 0.98497 | 0.97932 |
| PEBS tensor cosine, 35 rows sampled in both sessions | 0.98447 | 0.80446 | 0.00000 |
| PMU tensor cosine | 0.999991 | 0.999514 | 0.999104 |
| Timing tensor cosine | 1.000000 | 0.85607 | 0.45734 |
| Relative PT-byte difference | 0.00371 | 0.00128 | 0.00022 |
| Relative elapsed-time difference | 0.00862 | 0.00172 | 0.00010 |

Thirty-five matched rows had PEBS samples in both sessions, fifteen had none in
either, and one crossed the zero-sample boundary. The PEBS cosine distribution
above excludes both-empty rows and the mismatched row. Its sparse-sample tail is
real: one row with one usable sample in each session still had cosine zero, and
several other sampled rows had low similarity. The result passes the provisional
median repeat-stability target while exposing rather than retrying away PEBS
sampling noise. The audit also verified the plan file and every declared row,
rejected raw-capture reuse, and observed zero raw SHA-256 overlap between the two
sessions. It validates the v4 coordinate contract on this benign slice only; it
does not validate anomaly or vulnerability detection.

The full 121 MiB evidence copy is retained at
`/Users/theoad/.cache/cpu2tensor/pebs-anchor-v4-two-session-20260927-4aa81da`.
Important hashes are:

- plan: `65ea9bc3a582569a19f1f03d88c0d10ea0b933c86846d0e7679ddae73f7fd0db`;
- session A manifest: `832cf42708f0348817b250c4e793bc235122b6af8151f973cf6646cf6cc8ed59`;
- session B manifest: `8e54ad9a0bff2edaa103591f79419e9dcad2925142b46a1d1b27be47cef9fbf1`;
- repeat audit: `27b787b9e44880282d0ba92642b2c92b51db64970e91ddd11c0f28cd659a256d`;
- snapshotted audit source: `256904c13bdeff56e59f3f10458c00bea9d3c8b930e90e81da5abeef54e764a7`.

After both sessions, `no_turbo=0`, CPU-2 maximum was restored to 4.9 GHz, both
independent restoration timers were inactive, and no owned collection service
remained active.
