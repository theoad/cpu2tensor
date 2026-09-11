# Kernel learning examples

The two kernel examples use a small C17 guest target at
[`native/examples/kernel_init.c`](../native/examples/kernel_init.c). Rich x86
system capture, synchronized action boundaries, observation-only pretraining,
and a Gym client now run end to end. The workload accepts only named getpid,
memory, pipe, parallel-memory, and bounded compute commands.

Use [kernel pretraining](kernel-pretraining.md) for observation-only workers or
[Kernel Gym](kernel-gym.md) for streamed reset/step and client-defined rewards.
The [QEMU build and API evidence](kernel-qemu-build.md) records the compatible
external development dependency and its limitations. For current exact system
register capture, use the separate [state-hook build](qemu-state-hook.md). The
original unmodified build remains a block-only or public-memory backend. The
package installs no QEMU.

The [integration checks](kernel-integration-results.md) include full boot through
poweroff. The checked learning runs select an explicit postboot start marker. This excludes
boot events; it does not restrict subsequent capture to userspace. Kernel and
user execution on both vCPUs are observed after that marker. CPU memory accesses
include opt-in transaction values. Device-originated writes, architectural flags
unavailable through upstream QEMU, and a global memory order are not supplied.

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
allowed only when its process ID is 1. The check prints five `C2T ` lines, each
followed by a JSON object:

```text
C2T {"event":"result","step":0,"action":"getpid","value":PID}
C2T {"event":"result","step":1,"action":"memory","seed":17,"bytes":4096,"checksum":520419}
C2T {"event":"result","step":2,"action":"pipe","seed":17,"bytes":256,"checksum":31717}
C2T {"event":"result","step":3,"action":"parallel","seed":17,"bytes":4096,"cpu0":CPU_A,"cpu1":CPU_B,"checksum0":520419,"checksum1":522544}
C2T {"event":"complete","steps":4,"ok":true}
```

`PID` varies in the host check and is 1 in the guest. `CPU_A` and `CPU_B` are distinct allowed CPUs; the checked two-vCPU guest reports 0 and 1. The host check needs two available CPUs. This
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
uses Python's standard library and writes the guest console and adapter device
entries into the archive without creating host device nodes. It does not fetch
or build a kernel or QEMU.

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
| `/dev/ttyS1` | Character device, major 4, minor 65, mode 0600 |
| `/proc` | Directory, mode 0555 |

Connect the first ordered serial device to kernel console ttyS0 and the second
to the private adapter ttyS1. Use `console=ttyS0 rdinit=/init` in the kernel
command line. Workload stdout and stderr remain on ttyS0; only C2T events and
adapter commands use ttyS1. This prevents asynchronous kernel diagnostics or
future workload output from entering the protocol stream. The target mounts
only procfs, to read its command line. On completion or a fatal workload error
it requests guest poweroff. It stays alive if poweroff fails, so the worker
still needs a timeout and explicit incomplete-capture reporting.

To check only the ordinary guest workload with operator-supplied files:

```sh
export QEMU_SYSTEM=/path/to/qemu-system-x86_64
export KERNEL=/path/to/ordinary/x86_64/bzImage
"$QEMU_SYSTEM" -accel tcg -smp 2 -m 256M \
  -display none -monitor none -nic none -no-reboot \
  -chardev file,id=c2tconsole,path="$C2T_KERNEL_BUILD/console.log" \
  -serial chardev:c2tconsole \
  -chardev stdio,id=c2tadapter,signal=off \
  -serial chardev:c2tadapter \
  -kernel "$KERNEL" -initrd "$C2T_KERNEL_BUILD/initramfs.cpio.gz" \
  -append 'console=ttyS0 rdinit=/init panic=-1 cpu2tensor.mode=observe'
```

This guest check works without the plugin APIs needed for rich capture; it is
not a cpu2tensor worker command. C2T output appears on the second, stdio-backed
serial device; kernel and workload diagnostics go to `console.log`. Replace
`observe` with `interactive` to enter named commands on the adapter terminal.
The tested runs each had a 60-second deadline and terminated only their own QEMU
process if that deadline was reached.

The workload options are:

| Kernel command-line option | Default | Meaning |
| --- | --- | --- |
| `cpu2tensor.mode=observe` | `observe` | Run four fixed actions and power off |
| `cpu2tensor.mode=interactive` | | Read one named action per console line |
| `cpu2tensor.seed=17` | 17 | Unsigned 32-bit seed for the observation workload |
| `cpu2tensor.bytes=4096` | 4096 | Observation memory size, from 1 through 65536 bytes |

Duplicate or malformed workload options fail explicitly. Other kernel options
are ignored by the target. In interactive mode each memory or pipe command carries
its own seed and size; the command-line seed and size configure observation only.

When this guest runs through `cpu2tensor-worker`, select `--kernel-protocol on`
for observation-only `Pool` captures. Select `--kernel-adapter on` for
`KernelEnv`; it implies the same two-UART protocol and additionally enables
paused action boundaries. In both modes the worker owns QMP, display startup and
both serial devices, so do not pass `-serial`, `-monitor`, `-nographic` or `-S`.
An invalid record, unsuccessful guest completion or missing completion fails the
capture, and console diagnostics cannot enter protocol framing.

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

The learning check predicts held-out block tokens from the previous block and
keeps that previous token separately for each worker/vCPU. It compares initial
and trained model loss on the same held-out sample. Recurrent models remain a
separate example; their latent state must likewise belong to the original source. Split complete seeded runs between training and
validation before constructing windows. Output JSON, workload parameters, and
future events must not leak into model inputs. Preserve per-vCPU sequence when
collating workers; arriving frames do not establish a global memory order.

The real learning check uses two training workers and an independent test
worker on `trail-x86`, with the model on Mac MPS. It maintains separate previous
blocks per worker/vCPU and drains all streams through completion. See
[the training evidence](kernel-pretraining.md#real-kernel-evidence). AWS execution,
CUDA, multiple learner devices, and throughput optimization remain separate work.

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
| `parallel SEED BYTES` | Pin two processes to distinct CPUs and verify independent memory loops |
| `compute ITERATIONS PADDING` | Run a bounded CPU loop and add 0–64 result-padding bytes outside its action window |
| `quit` | Emit completion and request guest poweroff |

`SEED` is a decimal unsigned 32-bit integer. `ITERATIONS` is 1–65536; `PADDING`
is 0–64. The compute command exists as a deterministic action-window oracle.
Equivalent decimal spellings execute the same body while changing command size,
and padding changes only result transport. The command length is bounded to
127 bytes including its newline. These actions have fixed implementations: the
interface accepts no arbitrary syscall number, memory address, pathname, or guest
code. An action may execute several ordinary syscalls. No shell is started.

Successful actions emit a `result` JSON object, increment `step`, and emit the
next `ready` event. An invalid command emits an `error` and repeats the same step.
A failed system operation or closed input emits an error and unsuccessful
completion. Clients must inspect both completion and the worker's capture status.
The adapter supplies results; clients still define rewards, success, failure, and
episode limits.

`ready` requests a boundary. The worker issues QMP `stop` while continuing to
drain observations. Once QEMU confirms the world is stopped, a private control
pipe requests an explicit plugin drain. The plugin flushes every source and
publishes a kernel request after those tails in the same trace pipe. Only then
does `KernelEnv.reset()` or `.step()` finish yielding batches. The guest stays
paused while the client decides. One bounded command is delivered before QMP
`cont`; reset cancels and reaps the previous QEMU before another episode starts.

An idle callback or a QMP reply alone is insufficient: queued CPU work can release
QEMU's internal lock before an idle flush runs. Cold per-source mutexes protect
lifecycle/drain operations, and release/acquire publication covers both directions
of buffer ownership. Execution and memory callbacks take no global lock. These
rules and the independent repeated-fence probe are recorded in
[the API review](kernel-qemu-build.md).

The kernel worker exclusively owns QMP and serial control. Hotplug, migration,
external monitor controllers, and rebooted episodes are outside this version's
contract. A guest `complete(ok=true)`, plugin seal, closed channels, and successful
QEMU exit are all required before interactive completion. A kernel panic followed
by QEMU exit zero is not sufficient.

## Guest deadline under rich backpressure

The example's parallel startup/result deadline defaults to 30 guest-clock seconds.
For slow tensor consumers, set `cpu2tensor.parallel_timeout=300` in the kernel
command line (seconds, accepted range 1..3600). This changes only the example
workload's failure budget. It does not freeze guest clocks, alter the plugin's
scheduling, or remove backpressure cost. A missed deadline remains an explicit
unsuccessful workload, never a complete training sample.
