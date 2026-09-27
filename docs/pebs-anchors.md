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
