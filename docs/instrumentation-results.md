# Instrumentation results

Recorded 2026-09-07 for version 0.2.0, wire version 2. Source files are recorded
in `instrumentation-source.sha256`. The original first-slice manifest is historical.

## Checked behavior

The learner was local macOS arm64, CPython 3.10.12, PyTorch 2.13.0, real MPS.
The worker was `trail-arm`, Ubuntu AArch64, GCC 13.3.0. ARM capture used the
separate development QEMU 11.0.3 build and matching API 7 header from the first
slice. The x86 guest ran in the existing operator QEMU 11.0.3 on the same ARM
host, using a statically cross-compiled fixture. These are cross-ISA correctness
checks, not x86-host performance measurements.

| Check | Result |
| --- | --- |
| Portable contract | Native CMake build and CTest pass on Mac and ARM Linux |
| Consumer regressions | 11 tests pass |
| Regular wheel | Version 0.2.0 installed separately; all 11 consumer and 13 signal checks pass from outside the checkout, with 1 explicit CUDA skip; native module, type stub and py.typed resolve from the wheel environment |
| Signal fixtures | 13 pass, 1 explicit CUDA-unavailable skip |
| Real remote worker | 12 tests pass, including ARM CPU/MPS signals and x86 all-register capture |
| Rich concurrent sources | Three real pthread sources each emitted 34 unique register baselines, independent sequences, memory values and clean ends |
| Rich fault path | A deliberate target signal produced an explicit killed-target error after capture sealing, not successful completion |
| Exact transaction bytes | Scalar 1/2/4/8-byte and vector 16-byte transactions matched labeled instructions and ELF data addresses; a signed byte load reported raw A5 |
| Register meaning | Initial zeros and high-bit values preserved; zero-to-high-to-unchanged-to-zero checkpoints emit changes without repeating unchanged values |
| Wide registers | x86 all profile: 67 registers, including 10-byte x87 and 16-byte values; CPU/MPS synthetic wide-column retention also passes |
| Opt-in values | Same ARM fixture with values off exposed addresses/sizes/flags and no value tensor; enabled values matched independently known bytes |
| Baseline completeness | Review exposed a missing decoder gate; regression tests now reject absent schemas and incomplete baselines before data/end, while allowing empty sources |

One ARM run produced 18,223 block entries, 51,648 register observations, and
30,653 memory transactions with 34 initial register baselines. The x86 all-profile
run produced 36,555 / 86,420 / 32,713 respectively, with 67 baselines. These counts
identify the checked runs, not universal reference counts or throughput claims.
The fixture output was exactly `signals: ok checksum=0123456751428186`.

## Reproduce

Use the environment and operator test commands in [validation](validation.md).
Set `CPU2TENSOR_REMOTE_X86_QEMU` and `CPU2TENSOR_REMOTE_X86_TARGET` to run the x86
check too. The latter is `signals_target.c` cross-compiled with
`x86_64-linux-gnu-gcc -std=c17 -O2 -static -fno-pie -no-pie` on the ARM worker.
Build and input artifacts remain in host-local cache directories. Private exact
paths are in `.local/environments.md`; the package does not manage SSH or hosts.

Controlled fixture symbols, expected register masks, reviewed reference pitfalls,
and QEMU API constraints are in [the probe](instrumentation-probe.md). AlphaFlow
implementation was not copied. In particular, this implementation uses public
register APIs, emits initial baselines, and obtains memory values from the actual
transaction callback rather than rereading guest memory.

## Limits

Register changes are checkpoint samples, not every register write, and no final
register state is promised at process atexit. General registers are the default.
ARM `all` explicitly rejects its 65,536-byte ZA register under the 256-byte limit.
Memory events cover successful CPU accesses, not faulting transactions, syscall
copies, DMA, or every other guest-memory writer. See [the contract](instrumentation.md).

CUDA transfer code is present but no CUDA device was available for execution.
Kernel instrumentation remains future work. Signal changes currently flush
homogeneous frames, so some batches are small; aggregate-throughput optimization
and x86-host performance measurements remain open. No GPU-bound claim is made.
