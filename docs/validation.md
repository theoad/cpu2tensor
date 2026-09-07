# Validate observation instrumentation

All checks below exercise the observation path. Fixture-only tests do not prove
QEMU compatibility or real endpoint delivery.

## Portable build and client

From `native/`, run `cmake --preset portable`, `cmake --build --preset portable`,
and `ctest --preset portable`. Presets place each source checkout's build under
that host's `~/.cache/cpu2tensor/`; the source path distinguishes checkouts.
The worker profile requires Linux and an explicitly supplied QEMU header.

Install the root package into a host-local interpreter with `python -m pip install
-e .`. Run `python -m unittest discover -s python/tests -v`. Consumer fixtures check
fragmented framing, exact address bits, retained batches, independent source
sequences, final partial batches, failures, cancellation, and CPU/MPS model work.
Signal tests additionally check register baselines/deltas, memory columns, optional
values, exact wide bytes, malformed records and complete baselines. Device tests
report an explicit skip when the requested MPS or CUDA device is unavailable. Remote checks
are skipped unless explicitly configured below.

Build a regular wheel with `python -m pip wheel --no-deps -w /host/local/wheels .`.
Install that wheel into a separate environment and run the same consumer tests
from outside the checkout. Confirm package modules and `_native` resolve from
that environment, and that `py.typed` and `_native.pyi` are included. An editable
install alone can hide packaging mistakes.

## Real worker checks

After the [worker build](quickstart.md), set these variables on the learner:

```sh
export CPU2TENSOR_REMOTE=trail-arm
export CPU2TENSOR_REMOTE_ADDRESS=WORKER_PRIVATE_IP
export CPU2TENSOR_REMOTE_BUILD=/absolute/worker/build
export CPU2TENSOR_REMOTE_QEMU=/absolute/path/to/qemu-aarch64
# Optional x86 guest check on this worker:
export CPU2TENSOR_REMOTE_X86_QEMU=/absolute/path/to/qemu-x86_64
export CPU2TENSOR_REMOTE_X86_TARGET=/absolute/path/to/static-x86-signals-target
python -m unittest discover -s python/tests -v
```

The test harness uses the operator's SSH alias to start isolated one-shot workers;
the package itself never connects over SSH. Tests create unique input files on
the worker and remove them. They disable core dumps for deliberate signal checks.

Real checks include a prescribed checksum on MPS, an ELF `main` address match,
retained tensor storage, closed stdin at worker launch, source sequences across
pthread execution with independent register baselines, labeled scalar/vector
transactions on ARM and x86 guests, opt-in values, a delayed consumer that observes QEMU blocked in pipe writes,
disconnect cancellation with child reaping, failed exits, and explicit rejection
of fork/exec. One clearly named lifecycle fixture seals capture and deliberately
does not exit; it verifies cancellation during teardown rather than QEMU capture.

## Learning and stdin checks

`test_learning.py` checks exact sample counts, distinct train/test inputs and
histograms, model fitting, shuffled controls and checkpoint reload. The actual
CPU/MPS run is recorded in [trace learning](learn-trace.md).
`test_stdio.py` covers streaming reset/step and optional Gym behavior;
install `cpu2tensor[gym]` to include the wrapper checks.

With the same remote variables above, `test_remote_stdio.py` checks actual stopped
QEMU state, reset while paused, child reaping, retained observations, rich two-read
capture, leftover action rejection, stdin remapping, SIGCONT handler rejection and
input deadlines. Build `stdio_digits` and `stdio_test_target` along with the worker.
The real policy results are recorded in [stdin learning](stdio-example.md).

[Kernel guest checks](kernel-examples.md) use a separate static init and ordinary
operator kernel. They do not establish rich kernel capture or a kernel Gym barrier.

## Interpretation

A block event means entry, not completed instructions. A successful stream means
all recorded callbacks were delivered through QEMU's capture boundary, subject to
its documented teardown coverage limits. The worker combines the capture seal
with the actual child exit status. Killed targets cannot become successful traces.

Storage is bounded structurally: 256 preallocated source slots, at most 256
addresses in a block-only frame and at most 4,096 bytes per wire frame.
Register capture reserves bounded per-source metadata, previous values, and scratch
space at source initialization (up to 512 registers, 256 bytes per register).
There is one frame in the worker, a bounded pipe, and a 64 KiB requested
socket send buffer (Linux may double it). The pool has no receive queue and keeps
no history of consumed batches. QEMU's own translation cache and tensors retained
by the client are separate memory owners. Backpressure changes timing and does
not promise to preserve race probabilities or establish a total guest memory order.

Current register and memory semantics and limits are in
[instrumentation](instrumentation.md); checked host results are in
[instrumentation results](instrumentation-results.md).

These are correctness and portability checks. They are not throughput claims.
Performance comparisons need named x86-host measurements, as described in
[development](development.md).

## Version 0.3 example integration evidence

Recorded 2026-09-07. Source/build identity is in `learning-source.sha256`.
The macOS arm64 0.3.0 wheel was built and installed in a separate interpreter.
From `/tmp`, all five consumer/signal/learning/stdio/pretraining suites ran against
that installed package: 52 tests, 51 passed and one explicit CUDA-unavailable skip.
The native module, all four example modules, `py.typed`, and `_native.pyi` resolved
from the wheel environment. Gymnasium 1.3.0 was installed for wrapper checks.

The ARM worker passed all 12 real observation/lifecycle tests and all six real
stdin checks. Native CTest passed on macOS and ARM Linux. The x86 host separately
built the portable core and optional static kernel target through CMake and passed
CTest plus the benign host workload check. Real learning and guest-boot results
are linked above; no CUDA, rich kernel capture or kernel Gym success is claimed.
