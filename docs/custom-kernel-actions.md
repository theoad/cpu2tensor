# Custom kernel actions

The custom action example shows the smallest supported client-owned extension:
a C17 guest defines a bounded command grammar and marker functions, while an
ordinary synchronous `KernelEnv` client sends commands and drains one trace
window per action. cpu2tensor transports actions and observations. It does not
choose mutations, rewards, rankings, or policy updates.

The example compares these two commands after one guest start:

```text
open plain 8
open cloexec 8
```

Both repeat the same `openat` → one-byte `read` → `close` sequence against
`/proc/version`. The second adds only `O_CLOEXEC` to the `openat` flags. Each
action accepts 1 through 64 repetitions, so clients can keep work bounded and
approximately matched while studying a local flag mutation. The guest reports
the selected variant, repetition count, open flags, and expected syscall count;
these result fields are an oracle and must not become model inputs.

The complete, modifiable guest is
[`kernel_custom_actions.c`](../native/examples/kernel_custom_actions.c). Its
controller is
[`custom_kernel_actions.py`](../python/cpu2tensor/examples/custom_kernel_actions.py).
They use the existing protocol and add no package API.

## Guest contract

A custom guest must follow this ordering:

1. Open `/dev/ttyS1` for both protocol events and commands, leaving stdout and
   stderr on the diagnostic console. Emit one
   `C2T {"event":"start","mode":"interactive",...}` after that private serial
   channel is usable.
2. Emit `ready` with step 0 before reading the first action.
3. For every received command, enter the globally visible begin marker exactly
   once. Execute only the intended action inside the window. Reach either the end
   marker or abort marker exactly once.
4. After the end marker, emit a deterministic `result` naming the executed action
   and all arguments. Then increment the step and emit the next `ready`.
5. For a malformed command, use begin followed by abort, emit `error`, and request
   another action. An accepted action that fails also uses abort, then emits an
   unsuccessful `complete` event.
6. Treat shutdown as an action too: begin, end, then emit
   `complete(ok=true)`. The process must exit successfully after the guest powers
   off.

`ready` asks the worker to stop every vCPU. `KernelEnv.reset()` or `.step()` does
not return until QMP has confirmed the stop and the plugin has drained every
source. The begin/end functions delimit model observations; serial output alone
does not delimit a trace. Source sequence remains per vCPU, and no arrival order
between vCPUs is implied.

Keep marker functions non-inline and resolve their addresses from the exact ELF
used in the initramfs. Copying marker addresses from another build is invalid.
The example uses distinct no-op marker bodies. They change no guest data and the
plugin excludes their blocks from the compared action.

## Build and run

Build both kernel examples on Linux using the existing option, then package this
guest as `/init`:

```sh
export C2T_BUILD="$HOME/.cache/cpu2tensor/custom-actions"
cmake -S native -B "$C2T_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCPU2TENSOR_BUILD_KERNEL_EXAMPLE=ON
cmake --build "$C2T_BUILD" --target kernel_custom_actions
python -m cpu2tensor.examples.build_initramfs \
  --init "$C2T_BUILD/kernel_custom_actions" \
  --output "$C2T_BUILD/custom-actions.cpio.gz"

export WINDOW_BEGIN="$(nm "$C2T_BUILD/kernel_custom_actions" | awk '$3 == "cpu2tensor_action_begin" {print "0x" $1}')"
export WINDOW_END="$(nm "$C2T_BUILD/kernel_custom_actions" | awk '$3 == "cpu2tensor_action_end" {print "0x" $1}')"
export WINDOW_ABORT="$(nm "$C2T_BUILD/kernel_custom_actions" | awk '$3 == "cpu2tensor_action_abort" {print "0x" $1}')"
```

Start an operator-provisioned full-system worker. This block-only configuration
keeps rich registers and memory disabled while retaining raw blocks and bounded
transition counts:

```sh
cpu2tensor-worker \
  --qemu "$QEMU_SYSTEM" --plugin "$C2T_PLUGIN" \
  --system on --kernel-adapter on \
  --window-start-pc "$WINDOW_BEGIN" \
  --window-end-pc "$WINDOW_END" \
  --window-abort-pc "$WINDOW_ABORT" \
  --reducer block-transitions --transition-capacity 4096 \
  --blocks on --registers none --memory off \
  --host 127.0.0.1 --port 9400 --episodes 1 --timeout-ms 120000 -- \
  -accel tcg,thread=multi -smp 2 -m 256M -nic none -no-reboot \
  -kernel "$KERNEL" -initrd "$C2T_BUILD/custom-actions.cpio.gz" \
  -append 'console=ttyS0 rdinit=/init panic=-1'
```

The operator owns QEMU, kernel, networking, and process placement. Run the
controller wherever its endpoint can reach the worker:

```sh
python -m cpu2tensor.examples.custom_kernel_actions \
  --endpoint tcp://127.0.0.1:9400 --repetitions 8
```

The controller drains observations incrementally from each synchronous step. It
requires exactly one complete transition window, checks the result oracle, and
then sends the next action. Clients can copy these two small sources and replace
the grammar, syscall body, and verifier together. Preserve the marker and event
ordering so reset, timeout, all-source drain, and result semantics remain intact.

## Validation and limits

The socket fixture sends both variants after one reset, preserves block-only
columns, checks one ended window per action, and completes through `quit`. The CI
system check builds the static C17 guest, runs both actions in one ordinary Linux
process, validates exact events and results, and confirms all three marker symbols
exist. Run them with:

```sh
python -m pytest python/tests/test_kernel.py -q
CPU2TENSOR_CI_BUILD=/path/to/build \
  python -m pytest python/system_tests/test_custom_kernel_guest.py -q
```

The host check exercises the guest grammar and oracle but cannot prove QEMU's
stopped-world boundary. The generic action-window implementation has separate
native and socket-fixture coverage. The pinned CI image passed 172 unit tests
with 86 environment-dependent skips, 96% Python line coverage, and 96.3% native
line coverage in 28 seconds. Its system gate passed seven native and six Python
checks in 28 seconds, including fail-closed protocol emission coverage.

On 2026-09-11, commit `104975c06cfd9bd7055f51622643fee170e06e14`
was built and run on `trail-x86`: Linux x86-64, QEMU 11.0.3 TCG multi-thread
mode, Linux 6.9.0-dirty, and two guest vCPUs. Marker addresses read from the
matching static guest were begin `0x402270`, end `0x402280`, and abort
`0x402290`. One `KernelEnv` reset was followed by `open plain 8`,
`open cloexec 8`, and `quit`. Both actions returned their exact result oracle
and one ended, zero-overflow window. Raw block/transition counts were
20,162/20,161 for plain and 19,400/19,398 for cloexec. The worker reaped QEMU,
closed its listener, and left no matching process behind.

The checked SHA-256 values are
`2676047693b7a002803a23e11eea708c08d84fbf33f8d5d9de55afb2a3099c97`
for the guest,
`9c4806cda13f04128f79101794eab0aa3f03e7a256afd09a7d689d768d43f60b`
for the worker,
`06a78719f0206a81b3ae4234fbbb5ba63d58a0dcf803ddfe2dbf1d2ffc3933b0`
for the plugin, and
`11e92e1e13aa1842a20beb476725ab37fb5378d1ccbd0801b8aca1e00c4888e4`
for the initramfs. These identify correctness evidence, not a performance
measurement or a trace-distinguishability result. The full windows include
natural concurrent kernel work. A useful learning experiment still needs
held-parent and held-worker controls; the example supplies the mechanics for
constructing those inputs.
