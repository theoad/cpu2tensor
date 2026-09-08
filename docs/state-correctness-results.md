# State correctness: first fixes

Checked 2026-09-08 for the 0.5.0 development slice. This separates trustworthy
sampled CPU state, attributed CPU memory transactions, and publication fences.
It does not promise instruction-by-instruction state or full RAM reconstruction.

## Implemented behavior

- x86 `all` omits constant-zero floating-point GDB fields with visible diagnostics.
  Lazy hot-callback `eflags` remains omitted for system guests. Explicit requests
  for unavailable fields fail instead of returning plausible zero labels.
- Exact named register selections use `--registers rax:xmm0`, with PC included.
  The [optional developer QEMU hook](qemu-state-hook.md) supplies raw system GPR
  storage, CS base and execution width. Exact system register capture requires
  that capability; block-only and public-memory capture remain available without it.
- `Batch.context` carries ordered x86 CR0/CR3/CR4/EFER plus available CS base and
  execution width. Context is checked in memory callbacks, rather than assuming
  a block-entry cache remains valid inside every instruction helper. Only changes
  are emitted. The hook uses a small bulk read; the public fallback uses four
  register reads and marks mode/CS base unavailable.
- System memory rows retain VA, size, direction and opt-in transaction values,
  and add a physical prefix, coverage, QEMU dispatch flags and a checked reference
  to the current context on the same source. Cross-page tails remain explicitly
  unmapped. QEMU's dispatch flag is not a universal RAM/device classification.
- Selected register checkpoints can reconstruct the sampled raw state. The
  execution width distinguishes stored bits from those accessible to current code.
  Linear event PCs remain distinct from raw EIP/RIP offsets in segmented modes.
- Kernel action fences still mean stopped execution plus drained observations.
  Gym info explicitly reports `boundary_register_snapshot="unavailable"`.
  No stale block sample is relabeled as a newly acquired boundary snapshot.

See [the signal contract](instrumentation.md) for exact layouts/validity and
[the contributor recipe](adding-a-signal.md) for the native-to-tensor path.
The subsequent [normalization tutorial](normalization-results.md) implements
initial executable-layout metadata. General dynamic load maps remain unimplemented.

## Direct system oracles

On `trail-x86`, the standalone paging guest exercises distinct CR3 roots, virtual
aliases, noncontiguous physical pages, INVLPG remapping, paging control changes,
and a read-only APIC access. It then transitions through 16/32/64-bit execution
and back into a compatibility segment with nonzero GPR high bits.

The [independent probe](system-memory-probe.md) first exposed the upstream I/O
flag bug. The [hook evidence](qemu-state-hook.md) records its correction and
raw-state checks. The unmodified installed QEMU is preserved; the package never
installs or bundles the developer build.

The actual cpu2tensor plugin then captured the same guest with `all` and
`rax:r8`. Both streams were replayed through the public Pool client on the Mac:

- All ten selected transactions matched exact values, VA, physical prefixes,
  CR3 roots and CR0/CR4/EFER transitions.
- Every cross-page four-byte transaction reported two known mapped bytes; the
  other two value bytes remained available without an invented mapping.
- The corrected first-address dispatch flag identified the APIC I/O path.
- Contexts covered widths 16, 32 and 64. RAX `0x1122334455667788` and R8
  `0x8877665544332211` remained intact in compatibility mode.
- The selected schema was exactly RAX/R8/RIP. Unavailable fields were absent
  from `all`; an unknown explicit register name failed capture.

Four public-client checks passed: the two raw-state profiles, upstream memory-only
fallback with unavailable mode/dispatch, and a precise missing-hook error for
upstream system register capture. The boot fixture uses isa-debug-exit
and intentionally exits QEMU with status 33. This is a direct plugin/decoder/client
check with that independently checked exit status, not a claim that an ordinary
worker treats exit 33 as a successful episode.

Artifacts on the x86 host are under `~/.cache/cpu2tensor/state-correctness/final-captures/`:
the five named profiles each have trace, stdout/stderr and checked launch
provenance. They are copied to the same cache subdirectory on the Mac. Recreate
them with `python/tests/capture_system_state.py --help`, then point
`CPU2TENSOR_STATE_CAPTURES` at the output to run `test_system_state_capture.py`.
The independent native probe lives in `state-correctness-probe/`.

## Ownership, validation and portability

New CPU/MPS tests retain context/mapping/value columns, check exact bytes, and
run model backward. Malformed-stream tests reject missing or stale context,
cross-source references, false contiguous-page coverage, invalid availability,
physical-address overflow and raw register/block data without initial context.
CUDA remains unvalidated: its tests explicitly skip on this Mac.

The final regular 0.5.0 wheel was installed in a separate environment and tested
from outside the checkout: **87 passed, two CUDA skips**. This includes the four
real-capture replays, fixture validation, CPU/MPS ownership and learning checks,
action reducers and benchmark fixtures. Both package and native extension imports
resolved to that environment's `site-packages`.

```text
cpu2tensor-0.5.0-cp310-cp310-macosx_15_0_arm64.whl
SHA256 11c85fe790054a43dfdb7ac1b190abe03aac2d2ad7550e97d44755994915fa2b
```

The [final source manifest](state-source.sha256) identifies implementation,
build and test sources. It is separate from the benchmark's measured snapshot.

The first focused suite passed 55 tests with two CUDA skips. The ARM native build
and CTest passed. The existing remote ARM/user-stdin suite passed 17 tests; its
x86-user `all` check initially failed because it expected empty diagnostics.
After requiring the six intended omission diagnostics and absent schema fields,
that focused check passed too (18 real user-process checks across those runs).
The native core CTest also passed on the Mac and x86 benchmark build.
The final native builds and CTest passed on all three hosts. Legacy system x86
register streams without context now fail explicitly instead of being relabeled
as raw state.

Three real remote kernel checks passed: paused-world/reset lifecycle,
observation-only backpressure with both sources, and rich parallel actions with
exact guest checksums, attributed memory and retained MPS tensors/model backward.
The rich check uses the explicit 300-second guest deadline described below; its
155-second test duration is not a throughput benchmark.

## Performance evidence and limits

The new [benchmark runner](benchmark-capture.md) reports repeated signal costs,
provenance, complete-run status and resource counters without constructing
learner tensors. Its 15 fixtures passed on both the Mac and x86 host.

[Thirty real runs](benchmark-capture-results.md) on `trail-x86` all completed.
Whole-run median overhead was 1.55x for block capture, 2.01x for general-register
capture without memory, and 2.83x for rich capture with values. These include
kernel boot and local pipe draining; they are not steady-state, remote-learner,
CUDA or multiworker measurements. The measured plugin/build identity is in that
report; later cold exit-PC attribution and example deadline changes are separate.

An interactive rich remote test hit the benign guest's default 30-second parallel
work deadline under slow per-frame tensor consumption. This was an actual workload
failure, correctly surfaced by the worker. The example now accepts an explicit
`cpu2tensor.parallel_timeout=SECONDS` (1..3600, default still 30). Extending that
budget does not optimize capture or make guest clocks insensitive to backpressure.
The rich parallel test passed after configuring a 300-second guest budget.
Small single-kind frames, callback pipe writes and per-column tensor/device
work remain the next throughput work. No GPU-bound or large-scale claim is made.

## Still open

Fresh optional action/end register snapshots need a supported all-vCPU acquisition
and acknowledgement mechanism. Complete physical segments beyond the supported
prefix, general storage/address-space identity, load-map events, implicit MMU/DMA
writes and full memory reconstruction are not implemented. AWS/CUDA infrastructure
is unassigned. Per-vCPU rings, mixed column pages, efficient multiple-endpoint Pool
iteration and device overlap remain on the work board; the benchmark now makes
their effect measurable.
