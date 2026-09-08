# Executable layout notebook: evidence

Checked 2026-09-08. The [executed notebook](tutorials/normalization.ipynb) replaces
the prose-only native-to-tensor recipe. It includes a three-field native event,
its wire/ownership contract, a recorded real dataset, training curves, negative
controls and fresh-capture commands. Python helpers remain in the normal package;
the native target remains in the single CMake project.

## Capture

Worker: `trail-arm`, AArch64 Ubuntu in UTM. Learner: macOS Apple M2,
PyTorch 2.13.0. The existing operator QEMU 11.0.3 build was used; no QEMU build or
installation was needed. The target is one freestanding static PIE with no ELF
interpreter or relocation entries. SHA-256:

```text
3fcda5705abe3207bf715a5108b73098b0ae2db1ea05cb83eb277ee53c2abe0e
```

Eleven prescribed `-R` settings produced eleven distinct measured guest code
locations. This is controlled relocation of the same binary, not random ASLR.
The first seven settings were assigned to training and the last four to testing
before capture. Each launch consumed 128 balanced ASCII digits, with shuffled
order from seeds 103 through 113: 1,408 samples, 896 train and 512 held out.

Each run emitted one layout and 1,156 actual block events. Features contain the
block immediately before the common sample marker. The reducer has no input or
label argument; labels are attached afterward. Every selected PC was checked
against independent ELF path-symbol offsets and the input, every echoed output
matched, and all eleven workers exited zero. All PCs were inside the reported
static code span. The probe also checked repeated locations, invalid-input exit,
and a dynamic binary whose interpreter entry lies outside the main code span.

[Recorded captures](../example/normalization/data/captures.json) include exact
worker arguments, dependency hashes, layout tensors, captured features, inputs as
labels and oracle/completion results. Raw probe and collection artifacts are at
`~/.cache/cpu2tensor/load-tutorial/` on ARM and
`~/.cache/cpu2tensor/layout-tutorial/` on the Mac. These are actual worker → TCP →
native decoder → Pool observations, not shifted synthetic traces.

## Learning

CPU and MPS each ran all four comparisons for 80 epochs with fixed seeds
7, 17 and 29. Raw and normalized models used the same capacity, initialization
and optimizer budget. Vocabulary fitting uses training rows only. Final
checkpoints were saved and reloaded with identical predictions.

| Comparison | Held-out accuracy, all three seeds, CPU and MPS |
| --- | --- |
| Exact raw-address vocabulary | 25%, 25%, 25% |
| Integer-normalized position embedding | 100%, 100%, 100% |
| Metadata-only vocabulary | 25%, 25%, 25% |
| Normalized model given the wrong worker layout | 25%, 25%, 25% |
| Shuffled training labels | 50%, 0%, 25% |

The raw and normalized training losses both fell below 0.0002. Shuffled-label
held-out loss stayed near 1.39 with correct-class probability 0.238–0.264. Its
coarse accuracy varies because there are only four semantic paths. Repeated
executions are not independent program-understanding tasks.

The direct conclusion is narrow: a relative coordinate supplied from the launch
event lets this small exact-token embedding share learned positions across
relocation. Raw held-out PCs are all out of vocabulary by construction, so its
25% result is expected. This does not show that every raw-address model fails,
that the network discovered subtraction, or that arbitrary ASLR is solved.
Both models receive the same marker-based segmentation; only normalized address
features use the anchor. Metadata cannot identify labels in the balanced data.

[Result summaries](../example/normalization/data/results.json) preserve every
seed and backend. Full curves/checkpoints are in the Mac cache's `cpu/` and `mps/`
directories. The notebook reruns training and records its own curves and outputs.

## Packaging and checks

All 24 notebook cells executed without errors on CPU and MPS. The repository
contains the executed CPU notebook; the executed MPS copy is in the Mac cache.
The generated learning plot was visually checked. Native builds and CTest passed
on macOS, ARM Linux and x86 Linux. Independent review checked the new worker
scope/ordering path, model split/controls, recorded-data oracles and tutorial.

The regular wheel was installed and tested from outside the source checkout:
**104 passed, three explicit CUDA skips**. Package imports resolved to that
environment's `site-packages`. The suite includes the prior state-capture replays
and the new layout, collection and learning checks.

```text
cpu2tensor-0.5.0-cp310-cp310-macosx_15_0_arm64.whl
SHA256 fbe57cccb047985209b4788968f90584b51e0f750454df5559a4558a36422302
```

The [tutorial source manifest](normalization-source.sha256) identifies this
snapshot separately from the preceding state-correctness and throughput evidence.

## Cost and remaining coverage

The new event adds exactly 56 wire bytes per worker: 24 payload, 32 header.
It reads three getters once after loading, retains a translation-side once check,
and adds no work to block execution or memory callbacks. No timing speedup is
claimed; the benchmark can compare otherwise identical layout-off/on profiles.
Metadata contributes bytes/frames without pretending to be a CPU event.

The runtime exposes the initial executable span only; full load maps, unloads,
remaps, image identity, libraries, kernels and cross-worker synchronization are
not added. CUDA remains unvalidated. The example's eager Python sample reducer
is for this tiny tutorial, not a billion-event throughput implementation.
