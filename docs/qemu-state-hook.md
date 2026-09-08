# Optional x86 state hook for development

The [QEMU 11.0.3 patch](../native/qemu/patches/qemu-11.0.3-x86-state.patch)
adds a small optional x86 system API and fixes the public API's MMIO
I/O dispatch flag. It is a separate developer dependency. The cpu2tensor package
does not build, download, install, or bundle QEMU.

This hook exists to expose execution width, CS base, and raw general-register
storage together, and to avoid a separate GDB reader call for each field.
Upstream general-register readers mask or zero values outside a 64-bit code
segment. **Upstream x86-64 control-register readers already return full-width
values**, including CR3; the hook is not a fix for control-register truncation.

Only the three patch hunks together define this capability. The hook's presence
also means that `qemu_plugin_get_hwaddr()` incorporates the MMIO slow flag.
Clients discover the symbol once at initialization with `dlsym`; they must not
infer its presence from a generic QEMU release or plugin API number.

## Contract

```c
bool qemu_plugin_cpu2tensor_x86_state_v1(uint64_t *out, size_t count);
```

The hook is exported by the patched **x86-64 system** build. Call it only from
a current vCPU callback that requested `QEMU_PLUGIN_CB_R_REGS` (or RW_REGS).
TCG performs the required register synchronization before that callback.
The caller owns an array of exactly 8 or 28 `uint64_t` elements. Values are
host-native integers, so callers serialize them explicitly for the wire.

Null output, an unsupported count, no current vCPU, or a NO_REGS callback returns
false without writing output. As with the public register API, calling from
atexit, translation, or an arbitrary control thread is outside the supported
contract. The hook adds no remote reads, allocation, lock, stop-world operation,
or lazy-`eflags` materialization.

| Index | Meaning |
| --- | --- |
| 0 | raw CR0 |
| 1 | raw CR2 |
| 2 | raw CR3 |
| 3 | raw CR4 |
| 4 | reserved zero; **not a CR8 sample** |
| 5 | raw EFER |
| 6 | current CS base |
| 7 | current code-segment execution width: 16, 32, or 64 |
| 8–23 | raw GPR storage in order RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8–R15 |
| 24 | raw EIP/RIP offset in CS |
| 25 | FS base |
| 26 | GS base |
| 27 | kernel GS base |

Count 8 writes only context fields; count 28 includes the register fields. The
first eight values are identical when no guest execution occurs between calls.
Raw storage deliberately retains high bits and inaccessible registers when
executing 16-bit or 32-bit code. Clients combine those values with execution
width instead of assuming every stored bit is accessible to that instruction.
EIP and CS base are separate; a segmented instruction's linear PC need not equal
its EIP offset.

Field 4 is reserved to keep the bulk path free of implicit APIC/VAPIC work.
Upstream CR8 reads call `cpu_get_apic_tpr`, which can synchronize a virtual APIC.
Clients that explicitly select CR8 use its public reader and measure that cost
separately. The reserved zero must never appear as a plausible CR8 value.

The hook does not make a checkpoint a log of every register write or a final
state snapshot. It does not capture RAM snapshots, hidden page-table updates,
DMA, every segment attribute, or a total ordering across vCPUs. Physical mapping
coverage still follows [the memory contract evidence](system-memory-probe.md).

## Apply and build separately

Use a pristine upstream 11.0.3 source tree and the operator's own configured
build environment. The patch changes only:

- `target/i386/gdbstub.c`: optional raw-state hook;
- `include/plugins/qemu-plugin.h`: optional prototype, used by QEMU's existing
  script to generate `build/plugins/qemu-plugin.symbols`;
- `accel/tcg/cputlb.c`: read `TLB_MMIO` from the full entry's slow flags.

There is no checked-in `plugins/qemu-plugins.symbols` file in this release.
Editing a generated linker file would be fragile; adding the prototype uses
the normal export generation path instead. The extension does not change the
upstream numeric plugin API version.

```sh
patch --dry-run -p1 < /path/to/qemu-11.0.3-x86-state.patch
patch -p1 < /path/to/qemu-11.0.3-x86-state.patch
ninja -C /path/to/configured/build -j 4 qemu-system-x86_64
nm -D /path/to/configured/build/qemu-system-x86_64 \
    | grep qemu_plugin_cpu2tensor_x86_state_v1
```

Run the build-directory binary explicitly. Do not replace the ordinary installed
binary used for baseline measurements. Keep both binary hashes in reports.

## Development validation

On 2026-09-08, the three files in the recorded `trail-x86` source tree were
compared byte-for-byte with the verified release archive before modification.
Pristine copies are under
`~/.cache/cpu2tensor/state-correctness-probe/pristine/`. The patch dry-run succeeds
against those originals. The existing developer build was rebuilt with four
jobs; no install command was run.

The original installed binary remains:

```text
/home/user/.cache/cpu2tensor/qemu-system-x86_64/11.0.3-20260907/install/bin/qemu-system-x86_64
SHA256 ada293226e56a252eda8dbfd0cdc8f82b984449d9e6c16e4fe161ed930e986dc
```

The patched binary is at the corresponding `build/qemu-system-x86_64` path.
Final SHA-256 values:

```text
patched QEMU  2b74dee41743b0fc390fbfb489d4bfafdf800b07bf54c748c41b955122c76b6e
patch         aaf0536dd9ae8c207b8054a3e9fa7a6295f6de6fd2466fc970e223965030da94
guest.img     cd908ea06ee3f527ee6ee94ffe924d0ead5d4aa6971deee4cd657bbf9f2d0d88
probe.so      38c10fd3489b464fcd682af857fd1cfdd68bca1d4ee412e07422363df210839d
hook log      348967c9748866eab11edd6ad5ad63dea1d7239317030c829094150ae78ea6fd
upstream log  c1e509959a40f8c09dbc839d24deae078514cb1db8f40321704c7b149454a089
```

The [paging probe](../native/tests/system_memory_probe.cpp) resolves the hook
optionally, so the same probe also runs against upstream. The
[512-byte boot guest](../native/tests/system_memory_guest.S) now transitions
16-bit → 32-bit → 64-bit → 32-bit compatibility mode after its original ten
selected memory operations. Both binary variants complete through the intentional
debug exit with process status 33 and empty stderr; no guests remain running.

The patched run verifies:

- the same ten memory addresses, values and paging controls as the original;
- the local APIC version read is now classified as MMIO, while the RAM accesses
  remain non-I/O;
- count 8 equals the context prefix of count 28 at every observed boot block;
- raw context controls equal the public readers at selected memory callbacks;
- null output, invalid count, and an explicit NO_REGS callback are rejected;
- all three execution widths are observed;
- at compatibility-mode PC `0x7dab`, raw RAX remains `0x1122334455667788`
  and raw R8 remains `0x8877665544332211`.

This is bounded functional evidence, not a throughput benchmark. Nonzero CS/FS/
GS bases, CR8, all GPR patterns, nested translation, and concurrent vCPUs are not
validated by this particular fixture. No upstream patch or issue was submitted.

The I/O flag describes QEMU dispatch, not backing storage type. Subpage wrappers
can dispatch RAM through I/O handlers; ROMD reads can use a direct path. The
flag applies to the callback address, not every byte of a physical prefix.
