# Kernel learning examples

The two kernel examples share a small, benign C17 guest target at
[`native/examples/kernel_init.c`](../native/examples/kernel_init.c). It supplies
deterministic memory and pipe workloads for observation, and a fixed command
adapter for later interaction. Its host check and real ordinary guest runs pass.
**A captured kernel training run and a kernel Gym environment are not implemented
or validated yet.**

The checked x86 host currently provides QEMU 8.2.2. Its system emulator lacks the
register-read and memory-value APIs used by the current plugin; a compatible
operator-provided system emulator and matching header are required. The plugin
also currently rejects system emulation. See [dependency evidence](qemu-probe.md),
[the work board](backlog.md), and [capture efficiency](capture-efficiency.md).

## Build and check the guest target

Run from the repository root on Linux with CMake, Ninja, an x86-64 compiler, and
static libc. The optional target belongs to the same native CMake project and
does not require a QEMU header:

```sh
export C2T_KERNEL_BUILD="$HOME/.cache/cpu2tensor/kernel-example"
cmake -S native -B "$C2T_KERNEL_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCPU2TENSOR_BUILD_KERNEL_EXAMPLE=ON -DCPU2TENSOR_BUILD_WORKER=OFF \
  -DBUILD_TESTING=ON
cmake --build "$C2T_KERNEL_BUILD" -j2
ctest --test-dir "$C2T_KERNEL_BUILD" --output-on-failure
"$C2T_KERNEL_BUILD/kernel_init" --check
```

`--check` runs the fixed workload as an ordinary process. It never mounts a
filesystem or powers off the host. Invoking the program without this option is
allowed only when its process ID is 1. The check prints four `C2T ` lines, each
followed by a JSON object:

```text
C2T {"event":"result","step":0,"action":"getpid","value":PID}
C2T {"event":"result","step":1,"action":"memory","seed":17,"bytes":4096,"checksum":520419}
C2T {"event":"result","step":2,"action":"pipe","seed":17,"bytes":256,"checksum":31717}
C2T {"event":"complete","steps":3,"ok":true}
```

`PID` varies in the host check and is 1 when the target runs as guest init. This
check exercises the guest workload, not QEMU, trace capture, model training, or
multi-vCPU ordering.

On 2026-09-07, GCC 13.3.0 on the Linux x86-64 `trail-x86` host compiled this target
as C17 with `-O2 -static -Wall -Wextra -Wpedantic -Werror`. The resulting executable
is statically linked. Its host
check passed, both checksums matched an independent Python calculation, and an
ordinary launch without `--check` returned an error before mounting or poweroff.
The CMake recipe above also passed on that host: the kernel target and portable
core built, CTest passed its trace-contract test, and `kernel_init --check`
returned the same checksums. Its 19-file native source manifest has SHA-256
`dd4b17363bfc49c493efb8200fe555e5ed8544afc2ec9becd35367fef6824720`.
The two real guest checks used the same host, existing QEMU 8.2.2, TCG, two
vCPUs, 256 MiB RAM, and the operator's `/boot/vmlinuz-6.9.0-dirty`. Observation
mode produced all expected results and powered off with QEMU exit 0. Interactive
mode accepted one command after each `ready`, returned the same exact results,
and powered off after `quit`. No cpu2tensor plugin was loaded. These checks do
not establish captured trace completeness, active work on both vCPUs, training,
or a whole-machine action boundary. No performance number was measured.

## Guest boot contract

The operator supplies an ordinary x86-64 Linux kernel, a compatible plugin-enabled
`qemu-system-x86_64`, and its matching `qemu-plugin.h`. Detailed capture uses TCG.
KVM is a separate execution mode and does not provide these TCG plugin events.
The kernel needs initramfs, gzip, ELF execution, procfs, serial console, and SMP
support. The small [archive builder](../python/cpu2tensor/examples/build_initramfs.py)
uses Python's standard library and writes the guest console device entry into
the archive without creating a host device node. It does not fetch or build a
kernel or QEMU.

With cpu2tensor installed on the build host:

```sh
python -m cpu2tensor.examples.build_initramfs \
  --init "$C2T_KERNEL_BUILD/kernel_init" \
  --output "$C2T_KERNEL_BUILD/initramfs.cpio.gz"
```

The initramfs must contain:

| Guest path | Contents |
| --- | --- |
| `/init` | The static `kernel_init` executable, mode 0755 |
| `/dev` | Directory, mode 0755 |
| `/dev/console` | Character device, major 5, minor 1, mode 0600 |
| `/proc` | Directory, mode 0555 |

Connect the guest serial console to the operator's transport and use
`console=ttyS0 rdinit=/init` in its kernel command line. The target reads and writes
its inherited console descriptors. It mounts only procfs, to read its command
line. On completion or a fatal workload error it requests guest poweroff. It
stays alive if poweroff fails, so the worker still needs a timeout and explicit
incomplete-capture reporting.

To check only the ordinary guest workload with operator-supplied files:

```sh
export QEMU_SYSTEM=/path/to/qemu-system-x86_64
export KERNEL=/path/to/ordinary/x86_64/bzImage
"$QEMU_SYSTEM" -accel tcg -smp 2 -m 256M \
  -nographic -monitor none -nic none -no-reboot \
  -kernel "$KERNEL" -initrd "$C2T_KERNEL_BUILD/initramfs.cpio.gz" \
  -append 'console=ttyS0 rdinit=/init panic=-1 cpu2tensor.mode=observe'
```

This guest check works without the plugin APIs needed for rich capture; it is
not a cpu2tensor worker command. Replace `observe` with `interactive` to enter
the named commands on the console. The tested runs each had a 60-second deadline
and terminated only their own QEMU process if that deadline was reached.

The workload options are:

| Kernel command-line option | Default | Meaning |
| --- | --- | --- |
| `cpu2tensor.mode=observe` | `observe` | Run three fixed actions and power off |
| `cpu2tensor.mode=interactive` | | Read one named action per console line |
| `cpu2tensor.seed=17` | 17 | Unsigned 32-bit seed for the observation workload |
| `cpu2tensor.bytes=4096` | 4096 | Observation memory size, from 1 through 65536 bytes |

Duplicate or malformed workload options fail explicitly. Other kernel options
are ignored by the target. In interactive mode each memory or pipe command carries
its own seed and size; the command-line seed and size configure observation only.

## Observation-only pretraining

The [`example/kernel_pretraining/`](../example/kernel_pretraining/README.md)
example is intended to run the same ordinary kernel on several operator-started
workers, with different explicit seeds. Each guest executes `getpid`, fills and
reads bounded anonymous memory, and sends the same generated byte pattern through
a pipe. No action requests or policy are needed in observation mode.

Every byte comes from the following unsigned 32-bit recurrence:

```text
state = (1664525 * state + 1013904223) modulo 2^32
byte = state >> 24
```

The checksum is the unsigned 64-bit sum of the generated bytes. Memory and pipe
actions restart the generator with their supplied seed. This is an independent
semantic oracle for the workload, not a claim that two kernel traces are equal.
ASLR, scheduling, boot state, and instrumentation backpressure can change traces.

The planned learning check predicts held-out trace events from previous events,
retains recurrent state separately for each worker and vCPU, and records loss
against a simple baseline. Split complete seeded runs between training and
validation before constructing windows. Output JSON, workload parameters, and
future events must not leak into model inputs. Preserve per-vCPU sequence when
collating workers; arriving frames do not establish a global memory order.

Before this example can claim completion, the system backend needs real kernel
capture with at least two active vCPUs, source completion and fault checks, bounded
lossless backpressure, and a measured training run. CUDA and operator-provisioned
AWS workers remain separate validation requirements.

## Named syscall adapter

The [`example/kernel_gym/`](../example/kernel_gym/README.md) example starts the
same target with `cpu2tensor.mode=interactive`. It prints a `start` event and then
`C2T {"event":"ready","step":0}`. Send exactly one newline-terminated command
after each `ready` event:

| Command | Work performed |
| --- | --- |
| `getpid` | Read the guest process ID |
| `memory SEED BYTES` | Allocate, fill, read, and release 1–65536 anonymous bytes |
| `pipe SEED BYTES` | Write and verify a 1–256 byte pipe roundtrip |
| `quit` | Emit completion and request guest poweroff |

`SEED` is a decimal unsigned 32-bit integer. The command length is bounded to
127 bytes including its newline. These actions have fixed implementations: the
interface accepts no arbitrary syscall number, memory address, pathname, or guest
code. An action may execute several ordinary syscalls. No shell is started.

Successful actions emit a `result` JSON object, increment `step`, and emit the
next `ready` event. An invalid command emits an `error` and repeats the same step.
A failed system operation or closed input emits an error and unsuccessful
completion. Clients must inspect both completion and the worker's capture status.
The adapter supplies results; clients still define rewards, success, failure, and
episode limits.

**`ready` is a guest protocol boundary, not a whole-machine pause.** Other vCPUs
can still execute, and trace bytes can still be in producer or transport buffers.
The kernel Gym wrapper must add a tested stop/resume handshake and a trace-tail
barrier before it returns an observation for action selection. The existing
Linux-user syscall hooks do not observe syscalls within a system guest. Until
that lifecycle is implemented and checked, this target is a side-adapter building
block and should not be presented as a working kernel Gym environment.
