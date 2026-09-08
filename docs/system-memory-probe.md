# System memory mapping probe

On 2026-09-08, the unmodified upstream QEMU 11.0.3 development build on
`trail-x86` passed the address/value/context cases below, but **misclassified a
local APIC read as non-I/O**. This is a correctness probe, not a benchmark. It
does not establish a general permission to query arbitrary virtual addresses
with a callback's memory handle.

The dependency and source identity are recorded in
[kernel dependency evidence](kernel-qemu-build.md). Artifacts are under
`~/.cache/cpu2tensor/state-correctness-probe/` on that host. The final run used
one TCG vCPU and 16 MiB RAM. No QEMU changes were made. The guest exited through
`isa-debug-exit` with the expected process status 33; no guest was left running.

## Reproduce

The two standalone sources are
[the boot guest](../native/tests/system_memory_guest.S) and
[the diagnostic plugin](../native/tests/system_memory_probe.cpp). Copy both into
an empty Linux development directory. With an operator-provided matching QEMU
and its public plugin header, run:

```sh
qemu_prefix=/path/to/matching/qemu/install
as --32 system_memory_guest.S -o guest.o
ld -m elf_i386 -Ttext 0x7c00 --oformat binary guest.o -o guest.img
g++ -std=c++20 -shared -fPIC -fno-exceptions -fno-rtti \
    $(pkg-config --cflags glib-2.0) -I "$qemu_prefix/include" \
    system_memory_probe.cpp -o probe.so $(pkg-config --libs glib-2.0) -ldl
timeout 20s "$qemu_prefix/bin/qemu-system-x86_64" \
    -accel tcg -smp 1 -m 16M \
    -drive file=guest.img,format=raw,if=floppy \
    -display none -serial none -monitor none -no-reboot \
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
    -plugin ./probe.so > probe.log 2> probe.err
```

The native probe intentionally reads four registers at every selected memory
callback and writes diagnostics synchronously. Do not use its timing as capture
overhead. It requires one system vCPU. It samples only translated boot-sector
code at linear addresses `0x7c00 <= PC < 0x7e00`; BIOS memory is excluded.

## Exact memory oracle

The boot guest enables 32-bit protected-mode paging. Root `0x1000` maps virtual
pages `0x8000`, `0x9000`, and `0xa000` to physical `0x20000`, `0x30000`, and
`0x20000`. Root `0x4000` maps the first two to `0x40000` and `0x50000`.
Both roots identity-map the executing boot-sector page.

The probe observes exactly ten selected callbacks, in the following order.
Numbers in the table are hexadecimal except event numbers and widths. The
first/second addresses are immediate public API lookup results. The second
lookup is an experimental call for `callback_va + 2`, not shipped behavior.

| Event | Operation | VA | Width | Value | First GPA | Second GPA | CR3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | store | 8ffe | 4 | 11223344 | 20ffe | 30000 | 1000 |
| 1 | load | 8ffe | 4 | 11223344 | 20ffe | 30000 | 1000 |
| 2 | load, virtual alias | affe | 2 | 3344 | 20ffe | not queried | 1000 |
| 3 | store, second root | 8ffe | 4 | 55667788 | 40ffe | 50000 | 4000 |
| 4 | load | 8ffe | 4 | 55667788 | 40ffe | 50000 | 4000 |
| 5 | load, original root | 8ffe | 4 | 11223344 | 20ffe | 30000 | 1000 |
| 6 | load, remap + INVLPG | 8ffe | 4 | 11227788 | 40ffe | 30000 | 1000 |
| 7 | load, CR4 changed | 8ffe | 4 | 11227788 | 40ffe | 30000 | 1000 |
| 8 | load, EFER changed | 8ffe | 4 | 11227788 | 40ffe | 30000 | 1000 |
| 9 | load, local APIC version | b030 | 4 | 50014 | fee00030 | not queried | 1000 |

CR0 was `0x80000011` at these ten accesses. CR4 was zero through event 6 and
`0x80` thereafter. EFER was zero through event 7 and `0x800` thereafter.
Memory-callback CR0/CR3/CR4/EFER matched the last block-entry readings for every
selected access (`context_mismatches=0`). Guest values follow directly from the
constant stores and page-table contents; event 6 intentionally combines bytes
from two earlier stores in different address spaces.

To check the capture plugin separately, collect this same guest without the
diagnostic plugin and select memory rows with VA `0x8ffe`, `0xaffe`, or `0xb030`
and an instruction PC in the boot-sector range. Expect these ten rows and
context values. This document records the diagnostic run only; it does not claim
that the package runtime has passed that separate check.

Original ten-access fixture artifact SHA-256 values (before adding the optional
hook's mode-transition checks; see [hook evidence](qemu-state-hook.md) for the
current sources and artifacts):

```text
guest.img  a3063aee21c013c4be4427dee5869f37cbafb40f4116f2740e50f2479680d68a
probe.so   9ddcc2c30a462b946b1d73cdb68c93f57e08d29e72cdfe945d30d8d8a691e4db
probe.log  c1e509959a40f8c09dbc839d24deae078514cb1db8f40321704c7b149454a089
```

## Safe mapping semantics

`include/plugins/qemu-plugin.h:692` defines the `vaddr` argument as the virtual
address of the memory operation. It promises a callback-scoped handle. The
probe's second-page query succeeds on this exact implementation, but the public
contract does not explicitly support an offset or a complete list of segments.
Do not promote this observation into a portable segmentation guarantee.

`plugins/api-system.c:49` delegates to `tlb_plugin_lookup`, which looks in the
current soft TLB rather than walking current page tables. In
`accel/tcg/cputlb.c:1578`, lookup copies the physical page base and adds the
virtual page offset. It does not refill or check the victim TLB. A failed
lookup returns a null handle with a diagnostic; it must become unavailable
mapping, not GPA zero. Copy the result immediately: the returned handle is
thread-local scratch reused by another lookup.

For this x86 backend, a conservative supported representation is the first GPA
plus a valid physical-address prefix ending at the next 4 KiB boundary:
`min(access_size, 4096 - (virtual_address & 4095))`. Thus each `0x8ffe` row has
two mapped bytes and two bytes whose mapping is unavailable, even though all
four value bytes are known. A larger guest page does not invalidate the smaller
prefix. This prefix describes address translation; it is not evidence that all
bytes share a RAM/device classification or a universal storage identity.

The API itself warns that GPA is not unique across all QEMU address spaces.
CR3 is a paging control value, not a PID or an address-space lifetime identity.
The probe does not cover nested translation, SMM address-space changes,
concurrent vCPUs, subpage device regions, faulting/partial transactions, or all
x86 execution modes. Initial memory and implicit page-table/device writes are
also outside this probe's coverage.

## MMIO classification failure

The final read accesses the local APIC version register at physical
`0xfee00030`. It returns `0x50014`, proving the configured device responded, but
`qemu_plugin_hwaddr_is_io()` returns false. All ten probe lines therefore have
`io=0`. This field reports the observed API boolean; it must not be interpreted
as proof of RAM.

Source explains this discrepancy:

- `include/exec/tlb-flags.h:39–61` defines `TLB_MMIO` as a **slow flag**.
- `accel/tcg/cputlb.c:998–1014` puts slow flags in
  `CPUTLBEntryFull.slow_flags[access_type]`, not the fast TLB address bits.
- Normal memory lookup merges those slow flags at `cputlb.c:1665–1666`.
- Plugin lookup checks only `tlb_addr & TLB_MMIO` at `cputlb.c:1595`.

Until a corrected build and independent RAM/MMIO oracle are validated, publish
classification as **unknown** on this backend. Physical address validity and
classification validity must be separate. This probe deliberately makes the
failure visible; its successful guest exit is not a successful classification
acceptance test. No upstream report or patch was submitted during this task.

## A block-entry context cache is insufficient in general

The ordinary control-write cases pass because translation ends the block after
`MOV CRn` (`target/i386/tcg/emit.c.inc:368–369`) and `WRMSR` (`:4701–4702`).
That does not prove that paging context is constant during every instruction's
internal memory accesses.

The hardware task-switch helper is a source-level counterexample.
`target/i386/tcg/seg_helper.c:510–519` changes CR0 and CR3, then reads LDT/segment
descriptors through guest-memory helpers (`:565–566` and following code) before
the instruction returns. `accel/tcg/plugin-gen.c:50–66` explicitly enables
memory callbacks for accesses made inside instruction helpers. A block-entry
context may therefore precede the translation context used by those accesses.
This path was source-reviewed, not exercised by the boot fixture.

For exact callback context, sample the selected paging controls in the memory
callback with register-read access, and publish a context record only when its
value changes. This costs four public register reads per memory callback even
when no context record is emitted. Measure that cost before proposing a cache
optimization; retain the exact path or explicitly weaken the context contract.

Finally, `target/i386/gdbstub.c` uses `gdb_read_reg_cs64` for these controls.
On this x86-64 target, that helper returns a full 64-bit value regardless of
code-segment width. An earlier review incorrectly generalized the separate
GPR reader's mode masking to the control reader. CR3 truncation is therefore
not a reason to replace this public control API. The optional
[state hook](qemu-state-hook.md) adds execution width and CS base, preserves raw
GPR storage outside 64-bit code, and samples in bulk; its MMIO fix is separately
validated by this fixture.

## Dispatch is not storage classification

Follow-up source review: `system/physmem.c:1360–1388` installs an I/O wrapper for
subpage mappings; `:2913–2927` and `:2980–2993` can dispatch through that wrapper
to RAM. ROMD reads can use direct access (`accel/tcg/cputlb.c:1056–1075`). Even
a corrected `qemu_plugin_hwaddr_is_io` therefore describes QEMU's dispatch path,
not a universal RAM/device classification. Current mapping flags explicitly
carry dispatch information only. The APIC probe still establishes the original
slow-flag bug and its correction.
