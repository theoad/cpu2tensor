# QEMU probe

Checked 2026-09-07 for C2T-01. These are dependency and source findings,
not a completed cpu2tensor capture test. The initial inventory made no changes.
The user then explicitly authorized a separate development AArch64 QEMU build:
"It's ok build aarch64 qemu if needed. I just meant we shouldn't ship qemu with
our package to clients". Package installation still does not supply QEMU.

## Installed dependencies

| Host | Evidence | Result |
| --- | --- | --- |
| `trail-arm`, Linux AArch64 | `/usr/bin/qemu-aarch64 --version` | Ubuntu 8.2.2, package revision `1:8.2.2+ds-0ubuntu1.17` |
| `trail-arm` | `/usr/bin/qemu-aarch64 -plugin /nonexistent /bin/true` | `unknown option 'plugin'`; this binary cannot perform the required capture |
| `trail-arm` | `~/.cache/alphaflow/qemu/11.0.3/install/bin/qemu-x86_64 --version` | Existing operator build, 11.0.3, with plugin support; x86 guest only |
| `trail-arm` | Existing build's `config-host.mak` and `config-host.h` | `TARGET_DIRS=x86_64-linux-user`; `CONFIG_PLUGIN` defined |
| `trail-arm` | `~/.cache/alphaflow/qemu/11.0.3/install/include/qemu-plugin.h` | Operator header exists, reports plugin API **7**, includes local x86 register API additions; do not infer upstream compatibility from version text |
| `trail-x86`, Linux x86-64 | `/usr/bin/qemu-system-x86_64 --version` | Ubuntu 8.2.2, same package revision |
| `trail-x86` | `qemu-system-x86_64 -accel help` | Both TCG and KVM listed; instrumentation needs TCG |
| `trail-x86` | `qemu-system-x86_64 -machine none -display none -plugin /nonexistent` | Reaches plugin loader and reports missing shared library; no guest launched |
| `trail-x86` | `/usr/bin/qemu-x86_64 -plugin /nonexistent /bin/true` | User-mode binary rejects `-plugin` |
| Both workers | `pkg-config --modversion glib-2.0` | 2.80.0 |

The existing ARM source header is also at
`~/.cache/alphaflow/qemu/11.0.3/source/qemu-11.0.3/include/plugins/qemu-plugin.h`.
Its license identifier is `GPL-2.0-or-later`. It is an external build input;
no header or implementation was copied into this repository.

Searched `/home/user`, `/usr/local`, and `/opt` on ARM for additional AArch64
QEMU binaries, excluding the shared source mount. None were found. The existing
install contains only `qemu-x86_64`. On x86, the same locations contained no
`qemu-plugin.h`; `~/.cache/alphaflow` is absent. Older AlphaFlow environment
notes describing that x86 cache are stale for this host.

The initial missing dependency was a plugin-enabled AArch64 Linux user-mode
QEMU and matching header. A working x86 guest on the ARM host would not satisfy
this requirement. Under the user's subsequent authorization, a separate
development build completed under
`~/.cache/cpu2tensor/qemu-aarch64/11.0.3-20260907/`. It uses a private copy of the
existing operator source; the AlphaFlow source, build, and installation stay
unchanged. Downloads were disabled and the build used at most two jobs. The
plugin loader and an unmodified AArch64 program passed the smoke check below.
Real cpu2tensor pipeline acceptance remains separate work.

## Authorized development build

The private source copy configured successfully with these options:

```text
--target-list=aarch64-linux-user --enable-plugins --disable-download
--disable-docs --disable-tools --disable-system --disable-werror
--disable-debug-info --disable-install-blobs
--prefix=/home/user/.cache/cpu2tensor/qemu-aarch64/11.0.3-20260907/install
--python=/home/user/.cache/alphaflow/qemu/11.0.3/build/pyvenv/bin/python3
```

The existing Python environment supplies Meson without downloading dependencies.
Configure creates its own build environment. It reports native AArch64 TCG,
plugins enabled, GCC 13.3.0, GLib 2.80.0, and QEMU 11.0.3. Source, `build/`,
`install/`, `configure-command.txt`, `configure.log`, `build.log`, and
`install.log` are kept beneath the private development root above. Build command:
`ninja -j2 qemu-aarch64`; installation uses `meson install --no-rebuild`.
The first install attempt reported a missing `trace/trace-events-all`; generating
that target with `ninja -j2 trace/trace-events-all` and repeating installation
succeeded. The final build directory uses 39 MiB; the copied source uses 897 MiB.

The copied plugin header matches the existing operator-installed header:
SHA-256 `9e25ef7ea9d311532e8bfbe791ba8f450895324988c86f3c9fb35d69c8041a26`.
This preserves the source's local API 7 extensions; it is not an unmodified
upstream-build claim. Runtime capture checks must use that matching header.

Usable installed paths on `trail-arm`:

```text
/home/user/.cache/cpu2tensor/qemu-aarch64/11.0.3-20260907/install/bin/qemu-aarch64
/home/user/.cache/cpu2tensor/qemu-aarch64/11.0.3-20260907/install/include/qemu-plugin.h
```

The binary reports `qemu-aarch64 version 11.0.3`; its SHA-256 is
`c0b6a45ddd0671f0fc2076112112d730a3083be1f268d6e5d517cc99bc9e6601`.
`-plugin /nonexistent /bin/true` reaches the plugin loader and reports a missing
library. For a successful load check, the existing QEMU
`source/tests/tcg/plugins/empty.c` was compiled as a shared library against the
installed header and GLib include flags. Running the new binary with that plugin
and unmodified `/bin/cat` echoed exactly `cpu2tensor\n`, returned 0, and produced
no stderr. Full command and result are in `smoke.log` in the private build root.
This QEMU-owned test source remains outside the package. No performance claim
or complete cpu2tensor trace claim follows from this smoke test.

## Event meaning and finalization

For the first contract, call the event a **block entry**. QEMU inserts the TB
callback at translated-block entry, before its instructions. A translated
instruction count describes the block, not how many instructions completed.
Translation is not execution, and one address can be translated again.
See the [8.2 generator](https://github.com/qemu/qemu/blob/v8.2.2/accel/tcg/plugin-gen.c)
(`plugin_gen_tb_start`, `PLUGIN_GEN_FROM_TB`) and
[plugin API](https://github.com/qemu/qemu/blob/v8.2.2/include/qemu/qemu-plugin.h).

Instruction callbacks run before instructions; a synchronous fault can prevent
completion. Memory callbacks describe successful accesses, not failed accesses.
Do not infer completed blocks or faulting memory transactions from those
callbacks. Modern QEMU documents these distinctions explicitly in its
[TCG plugin semantics](https://www.qemu.org/docs/master/devel/tcg-plugins.html#exposure-of-qemu-internals).

Execution callbacks may run on different vCPU threads concurrently. Use a
separate sequence and bounded producer storage for each source. The 8.2 inline
counter operation is not atomic; it cannot supply a shared exact counter.
Registration uses QEMU locks, while callback dispatch avoids that global lock.
See the [8.2 plugin guide](https://github.com/qemu/qemu/blob/v8.2.2/docs/devel/tcg-plugins.rst)
and API above. A collector may interleave source batches without claiming a
global guest order. Any backpressure still perturbs timing.

Normal Linux-user cleanup calls `qemu_plugin_user_exit`: it makes vCPUs exclusive,
removes callbacks, flushes translations, then calls plugin exit callbacks.
The API warns that child threads can still execute a few uninstrumented
instructions during host teardown. Source completion must therefore mean
**all recorded callbacks were delivered through the capture boundary**, not
that every target instruction through process destruction was observed.
See [8.2 plugin cleanup](https://github.com/qemu/qemu/blob/v8.2.2/plugins/core.c)
and [Linux-user exit](https://github.com/qemu/qemu/blob/v8.2.2/linux-user/exit.c).

The handled fatal-target-signal path also calls pre-exit cleanup before
terminating, as shown in
[8.2 signal handling](https://github.com/qemu/qemu/blob/v8.2.2/linux-user/signal.c).
That does not guarantee cleanup for host crashes, forced kills, or broken
transport. A worker must combine plugin completion with the actual child exit
status; missing final records mean incomplete capture. Sealing final partial
batches belongs in capture shutdown. Real normal-exit, fault, slow-consumer,
and disconnect checks remain C2T-03/05 work.

## Stdin and future kernel boundaries

A small existing-program probe ran `/usr/bin/qemu-aarch64 -strace /bin/cat`
on `trail-arm`, with stdin held open and initially empty. A Python subprocess
probe waited at most three seconds for `read(0,...)`, confirmed the process
was still running, then sent `cpu2tensor\n` and closed stdin. The output was
exactly those bytes, exit status 0, and the trace showed reads returning 11
and then 0 bytes. This verifies the ordinary pipe-input path. It does not
verify a plugin callback or whole-world pause.

QEMU 8.2 Linux-user wraps syscall dispatch with entry and return hooks;
non-returning exit calls do not produce a normal return hook. See
[`do_syscall`](https://github.com/qemu/qemu/blob/v8.2.2/linux-user/syscall.c).
This can identify an attempted stdin read when a suitable plugin build exists.
An attempted read is not proof that it would block: buffered data, zero-length
reads, descriptor changes, and other input operations matter. A visible prompt
is not a reliable input boundary either. The eventual stdio adapter needs an
explicit supported syscall/descriptor contract and tests.

Linux-user syscall hooks do not observe syscalls inside a full-system guest.
A future kernel adapter needs guest-aware events or a side adapter.
QMP provides machine-level control, including `stop`, in system emulation;
see the [QMP specification](https://www.qemu.org/docs/master/interop/qmp-spec.html).
The plugin API is not itself a general whole-world pause interface. Do not
block a callback waiting for a pause operation that needs that vCPU to return.
A future pause handshake must prove all relevant streams reached the boundary,
including idle or I/O-waiting vCPUs, before reporting an actionable observation.
No such handshake or kernel adapter has been implemented here.

## Existing x86 guest artifacts

Read-only `ls -l`, `file`, and `qemu-img info` found these operator artifacts:

| Path on `trail-x86` | Metadata |
| --- | --- |
| `/boot/vmlinuz-6.9.0-dirty` | Readable x86 bzImage, Linux 6.9.0-dirty; matching readable initrd exists |
| `/home/user/workspace/linux-hacking-workshop/arch/x86/boot/bzImage` | Readable x86 bzImage, version 6.18.0-lab0+ |
| `/home/user/workspace/linux/arch/x86/boot/bzImage` | Readable x86 bzImage, version 7.2.0-rc6-00247-gc0a27675eaf0 |
| `/home/user/tinycore.qcow2` | qcow2, 2 GiB virtual disk, no backing file reported |
| `/home/user/tinycore1.qcow2` | qcow2, 1 GiB virtual disk, no backing file reported |

No image was mounted or modified, no guest was booted, and no artifact has been
selected as the kernel example. Its workload, boot configuration, provenance,
and compatible plugin header must be checked before that later slice.
