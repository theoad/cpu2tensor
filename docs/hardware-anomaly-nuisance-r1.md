# Nuisance and raw-packet sideband R1

This is a read-only inventory of the **525 preregistered raw audit shards** from
the immutable 102,000-execution physical `trail-x86` i7-10510U collection. It
does not read canary labels, fit a model, decode full PT branches, or operate the
laptop. The new
`cpu2tensor.examples.hardware_nuisance_sideband_probe_r1` verifies every raw
SHA-256 and both exact-session decode-state SHA-256s against the sealed manifest
`0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`.
Both decode states belong to the same recorded boot but differ in module-state
custody; the probe rejects a shard pointing to an unlisted state.

## What the sideband actually supplies

| Field | Retained evidence | Safe use / missing fact |
| --- | --- | --- |
| Kernel code relocation | Exact-boot `kallsyms` bytes, runtime `_text=0xffffffffb1c00000`, `_etext=0xffffffffb2c02507` in both decode states | 512,156/512,201 sampled PEBS IPs fall inside these bounds. Core-text IP minus runtime `_text` is a valid **same-boot relative site**. No unrelocated `_text`/verified `System.map` or vmlinux image is retained, so a numeric KASLR slide cannot be independently recovered from these shards. |
| Code identity | Kernel BTF/notes/cmdline hashes in subject manifest; exact-boot symbols, module list, and 171--172 module build IDs across the two decode states | Supports subject matching and symbol lookup. Hashes alone do not provide unrelocated link addresses; two module-state snapshots differ without another boot. |
| User mappings | Per-shard `/proc/<pid>/maps` snapshot, median 2,402 bytes; PT data-ring sideband is retained | Can identify saved user mappings and later sideband changes. It is not a kernel heap/page-cache object map. |
| PEBS stores | Raw timestamp, exact IP, virtual address, raw `perf_mem_data_src`, weight, CPU/TID, period, and perf record bytes | All 512,201 audited samples have exact IP and nonzero address; median 34 unique virtual pages per shard. Addresses are virtual, not physical or object identity. Sampling cannot prove that every store was observed. |
| PEBS data source / latency | Two raw values: `0x05080144` (510,090, decoded L1 store hit) and `0x05080184` (2,111, decoded L1 store miss); every raw weight is zero | The retained store stream has some L1 hit/miss information but **no usable weight/latency variation** in this audit. A model must not be promised a store-latency signal here. Decode follows the project's lossless `perf_mem_data_src` bitfield decoder. |
| Lanes and time | One TID lane per audited shard, PEBS CPU 2, `CLOCK_MONOTONIC_RAW`; PEBS and PMU scheduling times equal | Zero recorded PEBS timestamp inversions or ties. This says nothing about other CPUs' execution or a total memory order; the multi-lane rule is structural and unit-tested, not demonstrated by this single-lane corpus. |
| PT raw order | 572,193,760 AUX bytes, 17,806,160 saved PT perf-record bytes, 279,580 candidate 16-byte PSB signatures | PSB-delimited **ordinal raw-byte spans** can be represented without branch decoding. The signature scan is not a full packet parser, packet-count proof, timestamp, or instruction count. Perf/AUX gap checks stay authoritative for completeness. |

The saved 120-byte `perf_event_attr` has `config=0x2001`: PT control and branch
bits are set, while TSC, MTC, and CYC request bits are clear in all 525 shards.
The [Linux perf Intel PT documentation](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-intel-pt.txt)
maps MTC, TSC, and CYC to config bits 9, 10, and 1 respectively. [Intel's PT
specification](https://cdrdv2-public.intel.com/868136/252046-081-sdm-change-document.pdf)
says a TSC packet requires `TSCEn` and, even when present, bounds neighboring
packet order rather than timestamping a control-flow instruction exactly.
An incidental `0x19` byte inside raw AUX is not evidence of a TSC packet. With
the requested options here, **fine PT-to-PEBS timing is unidentifiable**; the
capture envelope alone bounds PT as a whole. We did not run a host sysfs
capability check, so this finding does not say the laptop is incapable of
timing-enabled PT. A future timing-enabled mode must first measure its added
capture perturbation and throughput on the controlled physical subject.

## Minimal representable grammar

For each source lane independently, the current custody can represent:

```text
lane(TID, sampled CPU)
  PT: ordered raw AUX bytes -> candidate PSB-bounded ordinal spans; no clock
  PEBS: timestamped sampled-store records (IP, virtual address, source, weight)
  PMU: instruction/cycle/reference-cycle deltas over one enabled interval
  envelope: arm/stop CLOCK_MONOTONIC_RAW bounds and sensor completeness
```

The PT spans cannot be interleaved with PEBS samples by wall-clock time, nor
can separate lanes be merged into one guest memory order. Equal PEBS times
would also remain unordered. The saved maps and symbols can annotate sampled
IP provenance without changing that partial-order contract. A raw-packet
learner may condition on build/boot ID, runtime core-text anchor, event period,
CPU/lane, no-turbo policy, and input intensity as *separate nuisance context*;
it should retain raw IP and raw virtual address alongside any relative forms.
No address normalization should silently erase a rare code site or invent the
kernel data object's identity.

## Reproduction and next gate

From an interpreter that resolves this checkout's `cpu2tensor` package:

```bash
python3.12 -m cpu2tensor.examples.hardware_nuisance_sideband_probe_r1 \
  /path/to/store-v3-scale100k-r2
python3.12 -m pytest -q \
  python/tests/test_hardware_nuisance_sideband_probe_r1.py
```

The focused synthetic checks cover missing link-time anchor, PT config without
timing despite a naked `0x19` byte, separate overlapping lanes, PSB candidates,
timestamp ties, source identity, and multiplex rejection. The 525-shard audit
ran on `mac.local` using Python 3.12/Torch 2.14 CPU; it is a metadata check,
not a capture or scoring throughput measurement. Three focused tests passed.

Next, use the retained exact-session sideband to audit a *small* offline
packet-decode/evidence subset for true PSB validity and symbol coverage. Keep
production training/scoring on fast raw-packet or byte-order descriptors until
a measured alternative justifies full decode. If timing is needed for the
prospective learner, design a separate bounded, same-subject, timing-enabled PT
capture/overhead check before collecting more training data. Do not infer PT
time from AUX byte position or treat this one boot as KASLR generalization.
