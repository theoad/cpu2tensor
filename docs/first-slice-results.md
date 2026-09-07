# First-slice evidence

Recorded 2026-09-07. The source checkout has no initial commit yet. Runtime and
build file hashes are recorded in [the source manifest](first-slice-source.sha256).
This is correctness and portability evidence, not a performance report.

## Environment

- Worker: `trail-arm`, Ubuntu 24.04.2, AArch64 host and guest, GCC 13.3.0,
  GLib 2.80.0. QEMU is the separate authorized 11.0.3 development build with
  matching local API 7 header described in [the probe](qemu-probe.md). No QEMU
  binary or source is bundled with cpu2tensor.
- Learner: local macOS arm64, AppleClang 17.0.0, CPython 3.10.12, PyTorch 2.13.0,
  actual MPS available. Host-local `dev-python` and `wheel-python` environments
  inherit the existing operator Python's dependencies. They have separate package
  installations and do not share extension binaries with Linux.
- Signal configuration: block entry address only, 256 entries per batch, 256
  possible source indices, little-endian version 1 framing. Address bit patterns
  are preserved in signed int64 tensors. Memory values and register capture are
  not implemented in this slice.

## Checks and results

| Check | Evidence |
| --- | --- |
| Standalone native build | `cmake --preset portable`, build, and CTest passed from `native/` on macOS and ARM Linux; selecting the required worker profile on macOS fails with a Linux-toolchain diagnostic |
| Native contract | Exact encoding, malformed header/version, independent source progress, duplicate/gap/end checks passed |
| Real quickstart | `observe.py ... --device mps` read 19,169 block entries; worker output was exactly `bytes=11 sum=1055` |
| Address fidelity | ELF `main` at `0x400928` appeared in the real trace; CPU/MPS fixtures retained all 64 bits including bit 63 |
| Real threads | Three sources observed, with 20,570 / 10,214 / 10,219 entries in one run; each source's sequence checked independently |
| Real slow consumer | A 600,000-byte input produced 624,698 entries with contiguous sequences; while reading paused, QEMU's `/proc/PID/wchan` was `pipe_write`; target checksum matched |
| Finalization | Final partial batches, source ends and child exit were checked; a deliberate signal could not become successful completion |
| Cancellation | Mid-run disconnect reaped the target; a separate lifecycle fixture confirmed disconnect also cancels after the internal capture seal |
| Worker restart | A second one-shot worker can bind the same port immediately after the first run |
| Unsupported process behavior | Real fork/exec fixtures produced explicit errors; no successful partial trace claim |
| Native review fixes | Closed-stdin worker launch preserves prescribed input; teardown wait observes disconnect; both regression checks pass |
| Capture review | Independent review against the matching QEMU source found no blocking ownership, concurrency, pipe-publication, signal, or completion issue |
| Consumer tests | 11 passed, including actual MPS retention and model forward/backward, fragmented frames, malformed sequences, missing completion, nonzero exit and timeout |
| Remote suite | After the final worker change, all 9 real-worker/lifecycle tests passed with no skips; the unchanged consumer's 11 checks also pass |
| Regular wheel | Separate installed wheel passed all 11 consumer checks from `/tmp`; `_native`, Python modules, `_native.pyi`, and `py.typed` resolved inside that environment |
| Actual IDE navigation | VS Code opened each source root and navigated native definitions/declarations, Python imports, and native binding stubs; see [IDE evidence](ide-evidence.md) for exact actions and limitations |

The slow-consumer check establishes actual backpressure and no detected sequence
gaps; it is not a throughput comparison or proof that instrumentation preserves
guest timing. Block counts may change with binary layout, libraries, translation,
or thread scheduling. Test the independent output and sequence invariants instead
of treating one count as a universal reference.

Run commands and test configuration are in [validation](validation.md). Actual IDE
navigation was checked separately under [the IDE contract](ide.md). CLion startup
and PyCharm inspection did not succeed in this environment. Pylance retains a
compiled-source warning, while stub navigation and the actual native import pass.

## Implementation choices

The binding uses the CPython C API directly. It adds no C++ exception or RTTI
requirement to the core, reports failures as Python exceptions, and creates one
owned buffer per batch. PyTorch remains outside the native hot path. scikit-build-core
invokes the same standalone CMake project for editable installs and wheels.

Capture allocates its source storage before callbacks and publishes one atomic
pipe write per batch. Each vCPU owns its buffer and sequence. The worker forwards
bounded frames and verifies actual process exit. The client uses synchronous
transfers and no receive queue. These choices prioritize explicit ownership and
readable code; no zero-copy or GPU-bound throughput claim is made.

All project implementation is new code. QEMU headers and GLib are external build
inputs; no AlphaFlow implementation or QEMU source was copied into the package.
The developer QEMU source copy is outside this repository. The package's license
file is included with the wheel.
