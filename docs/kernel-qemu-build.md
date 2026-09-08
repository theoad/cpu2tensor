# Developer QEMU for kernel examples

On 2026-09-07 a separate QEMU 11.0.3 system emulator was built on `trail-x86`
for kernel integration. It uses unmodified upstream source, GCC 13.3.0, and native
x86-64 TCG. This is a developer dependency in the operator's cache. cpu2tensor
package installation does not build, install, or bundle QEMU.

## Source and installed files

The archive came from the [official release directory](https://download.qemu.org/).
Its detached signature was verified with the release-key fingerprint published
on the [QEMU download page](https://www.qemu.org/download/):
`CEACC9E15534EBABB82D3FA03353C9CEF108B584`. GPG reported a good cryptographic
signature and noted that the signing key is now expired. The verification used
a separate keyring in the build directory and did not change the operator's
personal keyring.

All paths below are on `trail-x86`, under:

```text
/home/user/.cache/cpu2tensor/qemu-system-x86_64/11.0.3-20260907
```

| Relative path | Contents |
| --- | --- |
| `qemu-11.0.3.tar.xz` | Official source archive |
| `qemu-11.0.3.tar.xz.sig`, `qemu-release-key.asc`, `signature.log` | Signature evidence |
| `source/qemu-11.0.3/` | Extracted source; no QEMU source patches applied |
| `source/qemu-11.0.3/include/plugins/qemu-plugin.h` | Source plugin header |
| `dependencies/` | Private Python build environment |
| `build/` | Native x86-64 system build |
| `install/bin/qemu-system-x86_64` | Installed emulator, reporting 11.0.3 |
| `install/include/qemu-plugin.h` | Matching installed plugin header, API **6** |
| `install/share/qemu/` | QEMU's firmware and runtime data |

SHA-256 values:

| File | SHA-256 |
| --- | --- |
| Source archive | `da5fcffc32762820568b828ed430a728864d34d50b6d2f30358597760cbb0523` |
| Installed emulator | `ada293226e56a252eda8dbfd0cdc8f82b984449d9e6c16e4fe161ed930e986dc` |
| Source and installed plugin header | `9592f519d7344b1e38b0123b92c28c4e688faedf41263984910353736bb0b7be` |

The ARM development installation uses a locally extended API 7 source tree.
Its header cannot be substituted for this upstream API 6 header merely because
both emulators report version 11.0.3. Build the plugin against the header belonging
to its actual emulator.

## Build record

The host already provided GCC/C++, Python 3.12.3, Ninja 1.11.1, GLib 2.80.0,
pixman 0.42.2, and zlib 1.3. No host packages or system QEMU files were replaced.
The first offline configure attempt lacked the Python `wheel` package. Installing
it into QEMU's generated `build/pyvenv` did not persist across configure, because
configure recreates that environment. A separate build environment resolved this:

```sh
export C2T_QEMU="$HOME/.cache/cpu2tensor/qemu-system-x86_64/11.0.3-20260907"
cd "$C2T_QEMU"
python3 -m venv --system-site-packages dependencies
dependencies/bin/python3 -m pip install --no-deps wheel==0.45.1
dependencies/bin/python3 -m pip install --no-deps --no-index \
  --find-links source/qemu-11.0.3/python/wheels \
  meson==1.10.0 pycotap==1.3.1 qemu.qmp==0.0.5
```

Only `wheel` was fetched from PyPI; the other named build packages came from
QEMU's release archive. Exact package versions are in `python-dependencies.txt`;
installation output is in `python-dependencies.log`. Configure then ran from
`build/` with:

```sh
"$C2T_QEMU/source/qemu-11.0.3/configure" \
  --target-list=x86_64-softmmu --enable-plugins --disable-download \
  --disable-docs --disable-tools --disable-guest-agent \
  --disable-gtk --disable-sdl --disable-vnc --disable-opengl \
  --disable-virglrenderer --disable-spice --disable-debug-info \
  --disable-werror --disable-rust \
  --prefix="$C2T_QEMU/install" \
  --python="$C2T_QEMU/dependencies/bin/python3"
ninja -j4 qemu-system-x86_64 trace/trace-events-all
```

The build completed all 1841 requested steps. Installation initially reported
missing expanded EDK2 firmware files. The twelve `pc-bios/*.fd` targets listed in
the generated Ninja graph were built, then this completed successfully:

```sh
"$C2T_QEMU/build/pyvenv/bin/meson" install --no-rebuild
```

Exact configure, compilation, firmware expansion, and installation commands are
stored in `configure-command.txt`, `build-command.txt`, `firmware-command.txt`, and
`install-command.txt`. Their output is in the corresponding `.log` files.
The configure summary confirms system emulation, x86-64 TCG, plugin support,
and disabled dependency downloads. This build step is not a performance benchmark.

## Public API check and the flags limitation

The installed binary exports register discovery/read, memory transaction values,
physical-address lookup, and vCPU idle/resume callbacks. A small standalone probe
was compiled against its installed header and loaded by the emulator. It used
two vCPUs and a fixed arithmetic boot sector; it loaded no kernel or application
under investigation. The probe halted after a few arithmetic operations and was
terminated through QEMU's monitor within a 15-second deadline.

Both vCPUs exposed 66 register descriptors, and all descriptor reads succeeded
at initialization. The first vCPU made 5 idle and 4 resume callbacks; the second
made 17 idle and 16 resume callbacks. Register reads succeeded in all those idle
callbacks. These counts describe that single probe, not an ordering guarantee or
a stable workload metric.

The probe also exposed a correctness limitation. It executed `XOR AX, AX`, then
used `PUSHF` and `POP BX` to store the actual flags in RAM. The memory transaction
reported value `0x46`, width 2, physical address `0x500`, and RAM rather than MMIO.
Reading `eflags` with `qemu_plugin_read_register` in that same memory callback,
registered with `QEMU_PLUGIN_CB_R_REGS`, returned **`0x2`**. Architectural ZF and PF
were absent from that register result.

The source explains the discrepancy:

- `plugins/api.c:qemu_plugin_read_register` delegates to `gdb_read_register`.
- `target/i386/gdbstub.c:x86_cpu_gdb_read_register` returns `env->eflags`.
- `target/i386/tcg/tcg-cpu.c:x86_cpu_exec_enter` moves arithmetic flags and DF
  into TCG's lazy representation; `x86_cpu_exec_exit` restores them afterward.
- The read-access callback flag flushes TCG globals for the callback; it does not
  reconstruct architectural EFLAGS from that lazy representation.

For this backend, a successful register read is therefore not sufficient evidence
of architectural flag correctness during guest execution. The full-system x86
profile must omit `eflags` with a visible explanation until a supported, validated
API supplies the correct value. Consumers should use the actual register schema;
they should not assume `general` or `all` means that unavailable registers are
present. The separate operator API 7 user-mode checks remain separate evidence.
No QEMU patch was applied to hide or repair this limitation.

Probe source and evidence are preserved alongside the build:
`kernel-api-probe.c`, `kernel-api-probe.so`, `kernel-flags-probe.S`,
`kernel-flags-probe.bin`, `api-probe-command.json`, and `api-probe.log`.

## Stop and trace-drain semantics

The matching source contains these relevant boundaries:

- `system/cpus.c:qemu_process_cpu_events` invokes idle/resume callbacks while
  holding QEMU's big lock, around the CPU's condition-variable wait. A callback
  must not wait for a QMP command that needs this lock. A blocking publication
  here can delay the other vCPUs and the QMP thread.
- `pause_all_vcpus` waits for every CPU's stopped flag. `do_vm_stop` calls it
  before emitting QMP's `STOP` event.
- The stopped flag is set before `process_queued_cpu_work`. That work can release
  the big lock for an exclusive callback. Consequently, successful QMP stop does
  **not** prove every vCPU has subsequently entered its idle callback.

A kernel action boundary must continue draining the trace pipe while QMP stop is
pending. After QMP establishes guest execution quiescence, a plugin control thread
can drain remaining partial buffers without calling vCPU-only QEMU APIs. A cold
per-source mutex can serialize that drain with source initialization, idle flush,
and source exit. It does not replace the publication needed between execution
callbacks and the control thread: captured writes need a release/acquire handoff,
and any drain mutations need a corresponding handoff before execution resumes.

Only after all source tails are published should the plugin append the shared
kernel-request boundary to the trace pipe. The worker can then expose an action
request after reading that boundary. A `READY` console line or a QMP `STOP` event
alone says nothing about trace bytes still buffered upstream. Idle callback
counts alone are also insufficient for this protocol.

These are source findings and implementation requirements. The independent
plugin check below does not establish that cpu2tensor's full worker/client
handshake or a kernel learning example has passed. Current integration evidence belongs in the
[kernel example guide](kernel-examples.md) and [work board](backlog.md).

## Independent plugin drain check

A private snapshot of cpu2tensor's plugin and core was compiled against the new
API 6 header. Its four-file manifest has SHA-256
`81a23ba8f5d5ae2849cbc39e82cc753f8c72461190b5482957b48ac1f0c69318`.
The snapshot includes a release publication at the end of each capture callback,
an acquire before control-thread drain, and the reverse release/acquire handoff
before a source reuses its drained buffer. Cold mutexes cover lifecycle, idle,
and control-thread buffer access; execution callbacks take no mutex.

On the same x86 host, an independent selector-based harness booted the ordinary
Linux 6.9.0-dirty guest with two vCPUs and 256 MiB RAM. It continuously read the
trace, QMP, and console pipes. Capture enabled general registers and memory values,
starting at `cpu2tensor_capture_begin` after boot. For each guest `ready` event it
sent QMP `stop`, waited for the command response, sent the plugin's drain command,
and read through the resulting kernel-request frame before sending an action and
resuming the guest.

Two runs passed six stop/drain/resume boundaries each. The second deliberately
held each completed boundary for at least 50 milliseconds while continuing to
read all channels; no trace data arrived after the fence while execution remained
stopped. Both runs checked:

- Exact source sequence continuity through every frame and final source completion.
- A complete 25-register baseline on each vCPU, with `eflags` absent.
- Two pinned parallel-memory actions, with independently calculated checksums.
- The first 16 one-byte stores inside `cpu2tensor_parallel_memory` on each vCPU,
  matching that CPU's expected seed-derived byte sequence.
- Block, register-change, and memory activity on both vCPUs, followed by clean
  guest completion and QEMU poweroff.

The harness retained summaries, register metadata, and sixteen checked bytes per
CPU rather than keeping the full trace. The final run read 308701708 trace bytes;
this is a captured byte count, not a throughput result. Probe source, its copied
plugin/core source, hashes, command, console log, and result JSON are under
`/home/user/.cache/cpu2tensor/kernel-drain-probe/` on `trail-x86`. This checks the
plugin fence with an independent driver. The packaged worker, Python environment,
and model training still need their own integration checks.
